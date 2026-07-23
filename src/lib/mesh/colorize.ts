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

/**
 * Opacity-weighted median of one display-color channel over the candidate
 * splats: sort candidates by channel value (ties broken by splat index for
 * determinism), accumulate `sigmoid(opacity)` in that order, and take the
 * first value whose cumulative weight reaches at least half the total.
 *
 * @param f_dc - SH DC column for the channel.
 * @param indices - Candidate splat indices.
 * @param opacity - Logit-encoded opacity column.
 * @returns The weighted-median display color in [0, 1].
 */
const weightedMedianColor = (f_dc: TypedArray, indices: number[], opacity: TypedArray): number => {
    const pairs = indices.map(idx => ({
        v: displayColor(f_dc, idx),
        w: sigmoid(opacity[idx]),
        i: idx
    }));
    pairs.sort((a, b) => (a.v - b.v) || (a.i - b.i));

    let total = 0;
    for (const p of pairs) total += p.w;

    const half = total / 2;
    let cum = 0;
    for (const p of pairs) {
        cum += p.w;
        if (cum >= half) return p.v;
    }
    return pairs[pairs.length - 1].v;
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

    // scratch buffers for the per-vertex filter stages
    const gated: number[] = [];
    const inward: number[] = [];

    for (let i = 0; i < positions.length; i += 3) {
        const px = positions[i];
        const py = positions[i + 1];
        const pz = positions[i + 2];
        const nx = normals[i];
        const ny = normals[i + 1];
        const nz = normals[i + 2];
        const hasNormal = nx * nx + ny * ny + nz * nz > 1e-24;

        const candidates = bvh.queryOverlappingRaw(
            px - queryRadius, py - queryRadius, pz - queryRadius,
            px + queryRadius, py + queryRadius, pz + queryRadius
        );

        // distance gate
        gated.length = 0;
        for (const idx of candidates) {
            const dx = bvh.x[idx] - px;
            const dy = bvh.y[idx] - py;
            const dz = bvh.z[idx] - pz;
            if (dx * dx + dy * dy + dz * dz <= gate2) {
                gated.push(idx);
            }
        }

        // inward filter (skipped for vertices without a usable normal)
        inward.length = 0;
        if (hasNormal) {
            for (const idx of gated) {
                const dx = bvh.x[idx] - px;
                const dy = bvh.y[idx] - py;
                const dz = bvh.z[idx] - pz;
                if (dx * nx + dy * ny + dz * nz <= margin) {
                    inward.push(idx);
                }
            }
        }

        // fallback ladder
        let selected: number[];
        if (inward.length > 0) {
            selected = inward;
        } else if (gated.length > 0) {
            selected = gated;
        } else if (candidates.length > 0) {
            selected = candidates;
        } else {
            selected = bvh.queryOverlappingRaw(
                px - fallbackRadius, py - fallbackRadius, pz - fallbackRadius,
                px + fallbackRadius, py + fallbackRadius, pz + fallbackRadius
            );
        }

        let cr = 0.5;
        let cg = 0.5;
        let cb = 0.5;

        if (selected.length > 0) {
            if (mode === 'average') {
                let sumR = 0;
                let sumG = 0;
                let sumB = 0;
                let sumW = 0;

                for (const idx of selected) {
                    const w = sigmoid(opacity[idx]);
                    sumR += w * displayColor(f_dc_0, idx);
                    sumG += w * displayColor(f_dc_1, idx);
                    sumB += w * displayColor(f_dc_2, idx);
                    sumW += w;
                }

                cr = sumR / sumW;
                cg = sumG / sumW;
                cb = sumB / sumW;
            } else {
                cr = weightedMedianColor(f_dc_0, selected, opacity);
                cg = weightedMedianColor(f_dc_1, selected, opacity);
                cb = weightedMedianColor(f_dc_2, selected, opacity);
            }
        }

        result[i] = srgbToLinear(cr);
        result[i + 1] = srgbToLinear(cg);
        result[i + 2] = srgbToLinear(cb);
    }

    return result;
};

export { colorizeVertices };
export type { SplatColorColumns };
