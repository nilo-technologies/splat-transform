import type { TypedArray } from '../data-table';
import type { GaussianBVH } from '../spatial';
import type { CollisionColorMode } from '../types';

/**
 * Splat color columns used to colorize mesh vertices. The `f_dc_*` columns hold
 * spherical harmonics DC coefficients and `opacity` holds logit-encoded opacity.
 */
type SplatColorColumns = {
    /** SH DC coefficient for the red channel */
    f_dc_0: TypedArray;

    /** SH DC coefficient for the green channel */
    f_dc_1: TypedArray;

    /** SH DC coefficient for the blue channel */
    f_dc_2: TypedArray;

    /** Logit-encoded opacity */
    opacity: TypedArray;
};

const SH_C0 = 0.28209479177387814;

// Candidate pipeline thresholds, in multiples of voxelResolution. The query
// box (2x) is wider than the distance gate (1.5x) so the BVH's coarse
// AABB-overlap results never pre-filter a splat whose center would still
// pass the gate. The inward margin (0.5x) tolerates splats sitting slightly
// outside the surface.
const QUERY_RADIUS = 2;
const DISTANCE_GATE = 1.5;
const INWARD_MARGIN = 0.5;
const FALLBACK_RADIUS = 4;

const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v));

const srgbToLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

const displayColor = (f_dc: TypedArray, idx: number): number => Math.min(Math.max(0.5 + SH_C0 * f_dc[idx], 0), 1);

// Reusable scratch buffers. colorizeVertices runs once per mesh vertex, so on a
// multi-million-vertex mesh anything allocated per vertex (a candidate array
// from the BVH, one object per candidate per channel) dominates both time and
// GC pressure. Everything below is grown on demand and then reused.
let queryBuf = new Uint32Array(256);
let gatedBuf = new Uint32Array(256);
let inwardBuf = new Uint32Array(256);
let valueBuf = new Float64Array(256);
let weightBuf = new Float64Array(256);
let orderBuf = new Int32Array(256);
const orderList: number[] = [];

// Above this candidate count the insertion sort below is replaced by a native
// sort. Candidate sets are small in practice — a few dozen — but a dense region
// can produce thousands, where an O(n^2) sort would dominate.
const INSERTION_SORT_LIMIT = 96;

/**
 * Grow the per-candidate scratch buffers to hold at least `n` entries.
 *
 * @param n - Required capacity.
 */
const ensureCandidateScratch = (n: number): void => {
    if (n <= valueBuf.length) return;
    let cap = valueBuf.length;
    while (cap < n) cap *= 2;
    gatedBuf = new Uint32Array(cap);
    inwardBuf = new Uint32Array(cap);
    valueBuf = new Float64Array(cap);
    weightBuf = new Float64Array(cap);
    orderBuf = new Int32Array(cap);
};

/**
 * Collect the splats whose AABB overlaps a box, into the shared query buffer.
 *
 * @param bvh - BVH over the gaussian AABBs.
 * @param cx - Box centre X.
 * @param cy - Box centre Y.
 * @param cz - Box centre Z.
 * @param radius - Box half-size.
 * @returns Number of matches, written to `queryBuf[0..n)`.
 */
const queryBox = (
    bvh: GaussianBVH,
    cx: number, cy: number, cz: number,
    radius: number
): number => {
    let n = bvh.queryOverlappingRawInto(
        cx - radius, cy - radius, cz - radius,
        cx + radius, cy + radius, cz + radius,
        queryBuf, 0
    );
    if (n > queryBuf.length) {
        let cap = queryBuf.length;
        while (cap < n) cap *= 2;
        queryBuf = new Uint32Array(cap);
        n = bvh.queryOverlappingRawInto(
            cx - radius, cy - radius, cz - radius,
            cx + radius, cy + radius, cz + radius,
            queryBuf, 0
        );
    }
    return n;
};

/**
 * Opacity-weighted median of one display-color channel over the candidate
 * splats: order candidates by channel value, accumulate `sigmoid(opacity)` in
 * that order, and take the first value whose cumulative weight reaches at least
 * half the total.
 *
 * Candidates of equal value are interchangeable here — whichever of them the
 * cumulative sum crosses on, the value reported is the same — so no tie-break
 * on splat index is needed and the ordering can be done in place.
 *
 * @param n - Number of candidates, with values in `valueBuf` and weights in
 * `weightBuf`.
 * @param totalWeight - Sum of `weightBuf[0..n)`.
 * @returns The weighted-median display color in [0, 1].
 */
const weightedMedianOfScratch = (n: number, totalWeight: number): number => {
    for (let i = 0; i < n; i++) orderBuf[i] = i;

    if (n <= INSERTION_SORT_LIMIT) {
        for (let a = 1; a < n; a++) {
            const t = orderBuf[a];
            const tv = valueBuf[t];
            let b = a - 1;
            while (b >= 0 && valueBuf[orderBuf[b]] > tv) {
                orderBuf[b + 1] = orderBuf[b];
                b--;
            }
            orderBuf[b + 1] = t;
        }
    } else {
        orderList.length = n;
        for (let i = 0; i < n; i++) orderList[i] = i;
        orderList.sort((a, b) => valueBuf[a] - valueBuf[b]);
        for (let i = 0; i < n; i++) orderBuf[i] = orderList[i];
    }

    const half = totalWeight / 2;
    let cum = 0;
    for (let i = 0; i < n; i++) {
        const o = orderBuf[i];
        cum += weightBuf[o];
        if (cum >= half) return valueBuf[o];
    }
    return valueBuf[orderBuf[n - 1]];
};

/**
 * Compute a linear-space RGB color for every mesh vertex from the gaussian
 * splats near it, restricting candidates structurally so unrelated splats
 * cannot bleed color across surfaces.
 *
 * Candidate selection per vertex:
 * 1. Query splat AABBs overlapping a box of half-size `2 * voxelResolution`.
 * 2. Distance gate: keep splats whose center is within
 *    `1.5 * voxelResolution` of the vertex.
 * 3. Inward filter: keep gated splats with
 *    `dot(center - vertex, normal) <= 0.5 * voxelResolution`, so splats in
 *    front of the surface (e.g. a poster touching a wall) cannot tint it.
 *    Vertices with a zero normal skip this filter.
 * 4. Fallback ladder: if the inward+gated set is empty use the gated set;
 *    if that is empty use all candidates in the 2x box; if that is empty
 *    re-query at `4 * voxelResolution`; if still empty use mid-grey.
 *
 * How the final candidate set combines is selected by `mode`:
 * - `average` (default): display colors weighted by `sigmoid(opacity)`
 *   (smooth mean).
 * - `solid`: per-channel opacity-weighted median, snapping to the majority
 *   color instead of blending (crisp, leak-resistant).
 *
 * Display color per splat is `clamp(0.5 + SH_C0 * f_dc, 0, 1)` with
 * `SH_C0 = 0.28209479177387814`.
 *
 * @param positions - Vertex positions as packed xyz triplets.
 * @param normals - Vertex normals as packed xyz triplets (zero vectors
 * allowed; they disable the inward filter for that vertex).
 * @param bvh - BVH over the gaussian AABBs used for overlap queries.
 * @param columns - Splat color columns (`f_dc_0/1/2` and `opacity`).
 * @param voxelResolution - Size of one voxel in world units; scales all
 * candidate-pipeline thresholds.
 * @param mode - Coloring algorithm. Defaults to `'average'`.
 * @returns Linear RGB colors, 3 floats per vertex, same length as `positions`.
 */
const colorizeVertices = (
    positions: Float32Array,
    normals: Float32Array,
    bvh: GaussianBVH,
    columns: SplatColorColumns,
    voxelResolution: number,
    mode: CollisionColorMode = 'average'
): Float32Array => {
    const { f_dc_0, f_dc_1, f_dc_2, opacity } = columns;

    const result = new Float32Array(positions.length);

    const queryRadius = QUERY_RADIUS * voxelResolution;
    const gate = DISTANCE_GATE * voxelResolution;
    const gate2 = gate * gate;
    const margin = INWARD_MARGIN * voxelResolution;
    const fallbackRadius = FALLBACK_RADIUS * voxelResolution;
    const isAverage = mode === 'average';

    const bx = bvh.x;
    const by = bvh.y;
    const bz = bvh.z;

    for (let i = 0; i < positions.length; i += 3) {
        const px = positions[i];
        const py = positions[i + 1];
        const pz = positions[i + 2];
        const nx = normals[i];
        const ny = normals[i + 1];
        const nz = normals[i + 2];
        const hasNormal = nx * nx + ny * ny + nz * nz > 1e-24;

        let candidateCount = queryBox(bvh, px, py, pz, queryRadius);
        ensureCandidateScratch(candidateCount);

        // distance gate
        let gatedCount = 0;
        for (let c = 0; c < candidateCount; c++) {
            const idx = queryBuf[c];
            const dx = bx[idx] - px;
            const dy = by[idx] - py;
            const dz = bz[idx] - pz;
            if (dx * dx + dy * dy + dz * dz <= gate2) {
                gatedBuf[gatedCount++] = idx;
            }
        }

        // inward filter (skipped for vertices without a usable normal)
        let inwardCount = 0;
        if (hasNormal) {
            for (let c = 0; c < gatedCount; c++) {
                const idx = gatedBuf[c];
                const dx = bx[idx] - px;
                const dy = by[idx] - py;
                const dz = bz[idx] - pz;
                if (dx * nx + dy * ny + dz * nz <= margin) {
                    inwardBuf[inwardCount++] = idx;
                }
            }
        }

        // fallback ladder
        let selected: Uint32Array;
        let selectedCount: number;
        if (inwardCount > 0) {
            selected = inwardBuf;
            selectedCount = inwardCount;
        } else if (gatedCount > 0) {
            selected = gatedBuf;
            selectedCount = gatedCount;
        } else if (candidateCount > 0) {
            selected = queryBuf;
            selectedCount = candidateCount;
        } else {
            candidateCount = queryBox(bvh, px, py, pz, fallbackRadius);
            ensureCandidateScratch(candidateCount);
            selected = queryBuf;
            selectedCount = candidateCount;
        }

        let cr = 0.5;
        let cg = 0.5;
        let cb = 0.5;

        if (selectedCount > 0) {
            // opacity weights are shared by all three channels, so the sigmoid
            // is evaluated once per candidate rather than once per channel
            let totalWeight = 0;
            for (let j = 0; j < selectedCount; j++) {
                const w = sigmoid(opacity[selected[j]]);
                weightBuf[j] = w;
                totalWeight += w;
            }

            if (isAverage) {
                let sumR = 0;
                let sumG = 0;
                let sumB = 0;
                for (let j = 0; j < selectedCount; j++) {
                    const idx = selected[j];
                    const w = weightBuf[j];
                    sumR += w * displayColor(f_dc_0, idx);
                    sumG += w * displayColor(f_dc_1, idx);
                    sumB += w * displayColor(f_dc_2, idx);
                }
                cr = sumR / totalWeight;
                cg = sumG / totalWeight;
                cb = sumB / totalWeight;
            } else {
                for (let j = 0; j < selectedCount; j++) {
                    valueBuf[j] = displayColor(f_dc_0, selected[j]);
                }
                cr = weightedMedianOfScratch(selectedCount, totalWeight);
                for (let j = 0; j < selectedCount; j++) {
                    valueBuf[j] = displayColor(f_dc_1, selected[j]);
                }
                cg = weightedMedianOfScratch(selectedCount, totalWeight);
                for (let j = 0; j < selectedCount; j++) {
                    valueBuf[j] = displayColor(f_dc_2, selected[j]);
                }
                cb = weightedMedianOfScratch(selectedCount, totalWeight);
            }
        }

        result[i] = srgbToLinear(cr);
        result[i + 1] = srgbToLinear(cg);
        result[i + 2] = srgbToLinear(cb);
    }

    return result;
};

export { colorizeVertices, srgbToLinear };
export type { SplatColorColumns };
