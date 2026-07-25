import { buildVertexHash, forEachNeighbor, majorityFilterIndices } from './color-spatial';

// Oklab (Björn Ottosson). Converts natively from linear RGB, so no gamma
// round-trip is needed for clustering.
const linearToOklab = (r: number, g: number, b: number): [number, number, number] => {
    const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    // cbrt, not ** (1 / 3): the latter is NaN for the marginally negative
    // values that upstream colour clamping can leave behind.
    const l_ = Math.cbrt(l);
    const m_ = Math.cbrt(m);
    const s_ = Math.cbrt(s);

    return [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    ];
};

const oklabToLinear = (L: number, A: number, B: number): [number, number, number] => {
    const l_ = L + 0.3963377774 * A + 0.2158037573 * B;
    const m_ = L - 0.1055613458 * A - 0.0638541728 * B;
    const s_ = L - 0.0894841775 * A - 1.2914855480 * B;

    const l = l_ * l_ * l_;
    const m = m_ * m_ * m_;
    const s = s_ * s_ * s_;

    // A mean of in-gamut colours can land just outside sRGB, so clamp.
    return [
        Math.min(Math.max(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, 0), 1),
        Math.min(Math.max(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, 0), 1),
        Math.min(Math.max(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s, 0), 1)
    ];
};

// Lightness counts for less than chroma so palette slots are spent on hue
// variety rather than on lighting: a same-material shadow-to-highlight swing
// otherwise measures ~16x larger than a genuine cross-hue difference. Not
// zero, so pure black and pure white stay distinguishable.
const LIGHTNESS_WEIGHT = 0.2;

// Candidate bin size in Oklab. Deliberately below one just-noticeable
// difference, so averaging within a bin denoises without blending hues.
const L_BIN_STEP = 0.05;
const AB_BIN_STEP = 0.02;

// A vertex counts as spatially supported when at least this many neighbours
// within SUPPORT_RADIUS voxels sit within SUPPORT_COLOR_EPS of its colour.
// Matching by perceptual radius rather than by exact bin is what lets a small
// jittered region (whose vertices scatter across several bins) still qualify.
const SUPPORT_RADIUS = 1.5;
const SUPPORT_MIN_NEIGHBORS = 2;
const SUPPORT_COLOR_EPS = 0.06;

// A colour earns a palette slot when most of its vertices are supported, plus
// a small absolute floor. Deliberately a ratio, not an absolute count: an
// absolute threshold scales with mesh size and so rejects exactly the small
// solid features this is meant to protect, while still admitting a large but
// entirely scattered noise colour.
const SUPPORT_MIN_COUNT = 3;
const SUPPORT_MIN_RATIO = 0.5;

const MAX_ITERS = 25;
const CONVERGENCE_EPS2 = 1e-10;

type CandidateBin = {
    id: number;
    count: number;
    supported: number;
    sumL: number;
    sumA: number;
    sumB: number;
    L: number;
    A: number;
    B: number;
};

type PalettizeOptions = {
    /** Per-vertex XYZ triplets, enabling the spatial-coherence gate. */
    positions?: Float32Array;

    /** Size of one voxel in world units; required alongside `positions`. */
    voxelResolution?: number;

    /** Majority-filter radius in voxel units applied after assignment. */
    coherentRadius?: number;
};

const dist2 = (
    aL: number, aA: number, aB: number,
    bL: number, bA: number, bB: number
): number => {
    const dL = aL - bL;
    const dA = aA - bA;
    const dB = aB - bB;
    return LIGHTNESS_WEIGHT * dL * dL + dA * dA + dB * dB;
};

/**
 * Group vertices into sub-perceptual Oklab bins and count, per bin, how many
 * of its vertices sit in a spatially coherent patch of similar colour.
 *
 * Without `positions` every vertex counts as supported, which degrades the
 * gate to a plain colour-space population threshold.
 *
 * @param oklab - Per-vertex Oklab triplets.
 * @param vertexCount - Number of vertices.
 * @param positions - Per-vertex XYZ triplets, or undefined to skip the gate.
 * @param voxelResolution - Size of one voxel in world units.
 * @returns Candidate bins in first-encounter order.
 */
const buildCandidates = (
    oklab: Float32Array,
    vertexCount: number,
    positions: Float32Array | undefined,
    voxelResolution: number | undefined
): CandidateBin[] => {
    const bins = new Map<string, CandidateBin>();
    const binOf = new Int32Array(vertexCount);
    const order: CandidateBin[] = [];

    for (let v = 0; v < vertexCount; v++) {
        const L = oklab[v * 3];
        const A = oklab[v * 3 + 1];
        const B = oklab[v * 3 + 2];
        const key = `${Math.floor(L / L_BIN_STEP)}_${Math.floor(A / AB_BIN_STEP)}_${Math.floor(B / AB_BIN_STEP)}`;
        let bin = bins.get(key);
        if (!bin) {
            bin = { id: order.length, count: 0, supported: 0, sumL: 0, sumA: 0, sumB: 0, L: 0, A: 0, B: 0 };
            bins.set(key, bin);
            order.push(bin);
        }
        binOf[v] = bin.id;
        bin.count++;
        bin.sumL += L;
        bin.sumA += A;
        bin.sumB += B;
    }

    for (const bin of order) {
        bin.L = bin.sumL / bin.count;
        bin.A = bin.sumA / bin.count;
        bin.B = bin.sumB / bin.count;
    }

    if (positions && voxelResolution !== undefined) {
        const hash = buildVertexHash(positions, voxelResolution);
        const radius = SUPPORT_RADIUS * voxelResolution;
        const eps2 = SUPPORT_COLOR_EPS * SUPPORT_COLOR_EPS;
        for (let v = 0; v < vertexCount; v++) {
            const vL = oklab[v * 3];
            const vA = oklab[v * 3 + 1];
            const vB = oklab[v * 3 + 2];
            let similar = 0;
            forEachNeighbor(hash, positions, v, radius, (n) => {
                if (n === v) return;
                if (dist2(vL, vA, vB, oklab[n * 3], oklab[n * 3 + 1], oklab[n * 3 + 2]) <= eps2) {
                    similar++;
                }
            });
            if (similar >= SUPPORT_MIN_NEIGHBORS) {
                order[binOf[v]].supported++;
            }
        }
    } else {
        for (const bin of order) {
            bin.supported = bin.count;
        }
    }

    return order;
};

/**
 * Bins that represent real content: spatially coherent, and not a bare handful
 * of vertices. Both seeding and refinement work from this set, so scattered
 * noise can neither claim a slot nor pull a centroid off a real colour.
 *
 * @param candidates - Candidate bins from `buildCandidates`.
 * @returns Eligible bins, or all of them when too few qualify.
 */
const selectPool = (candidates: CandidateBin[]): CandidateBin[] => {
    const isCoherent = (c: CandidateBin): boolean => {
        return c.supported >= SUPPORT_MIN_COUNT && c.supported / c.count >= SUPPORT_MIN_RATIO;
    };
    const eligible = candidates.filter(isCoherent);
    return eligible.length >= 2 ? eligible : candidates;
};

/**
 * Pick centroids by farthest-first traversal over the candidate pool.
 *
 * @param pool - Eligible candidate bins from `selectPool`.
 * @param effK - Maximum number of centroids to select.
 * @returns Seed centroids as Oklab triplets, at most `effK` of them.
 */
const seedCentroids = (pool: CandidateBin[], effK: number): number[][] => {
    let meanL = 0, meanA = 0, meanB = 0;
    for (const c of pool) {
        meanL += c.L;
        meanA += c.A;
        meanB += c.B;
    }
    meanL /= pool.length;
    meanA /= pool.length;
    meanB /= pool.length;

    let firstIdx = 0;
    let firstBest = Infinity;
    for (let i = 0; i < pool.length; i++) {
        const d = dist2(pool[i].L, pool[i].A, pool[i].B, meanL, meanA, meanB);
        if (d < firstBest) {
            firstBest = d;
            firstIdx = i;
        }
    }

    const centroids = [[pool[firstIdx].L, pool[firstIdx].A, pool[firstIdx].B]];
    const taken = new Uint8Array(pool.length);
    taken[firstIdx] = 1;

    const minDist = new Float64Array(pool.length);
    for (let i = 0; i < pool.length; i++) {
        minDist[i] = dist2(pool[i].L, pool[i].A, pool[i].B, centroids[0][0], centroids[0][1], centroids[0][2]);
    }

    while (centroids.length < effK) {
        let bestIdx = -1;
        let bestDist = -1;
        for (let i = 0; i < pool.length; i++) {
            if (!taken[i] && minDist[i] > bestDist) {
                bestDist = minDist[i];
                bestIdx = i;
            }
        }
        if (bestIdx < 0) break;

        taken[bestIdx] = 1;
        const picked = [pool[bestIdx].L, pool[bestIdx].A, pool[bestIdx].B];
        centroids.push(picked);
        for (let i = 0; i < pool.length; i++) {
            const d = dist2(pool[i].L, pool[i].A, pool[i].B, picked[0], picked[1], picked[2]);
            if (d < minDist[i]) minDist[i] = d;
        }
    }

    return centroids;
};

const assignNearest = (
    oklab: Float32Array,
    vertexCount: number,
    centroids: number[][],
    out: Uint32Array
): void => {
    for (let v = 0; v < vertexCount; v++) {
        const vL = oklab[v * 3];
        const vA = oklab[v * 3 + 1];
        const vB = oklab[v * 3 + 2];
        let bestDist = Infinity;
        let bestC = 0;
        for (let c = 0; c < centroids.length; c++) {
            const d = dist2(vL, vA, vB, centroids[c][0], centroids[c][1], centroids[c][2]);
            if (d < bestDist) {
                bestDist = d;
                bestC = c;
            }
        }
        out[v] = bestC;
    }
};

/**
 * Snap vertex colours to a deterministic palette of at most `k` colours.
 *
 * Clustering happens in Oklab with lightness weighted below chroma, so slots
 * are spent on hue variety rather than on lighting. Centroids are seeded by
 * farthest-first traversal over binned candidate colours that clear a
 * spatial-coherence gate — without that gate, clamped reconstruction noise at
 * the corners of the RGB cube captures most of the palette and collapses the
 * scene towards its mean colour. Refinement then runs over those bins with each
 * bin counting once, rather than over vertices, so a colour family covering most
 * of the mesh cannot outvote small features; it stops at convergence or
 * `MAX_ITERS` and reseeds any cluster that empties. A final per-vertex pass
 * guarantees every vertex receives its nearest palette entry. No step is
 * randomised.
 *
 * @param linearColors - Per-vertex linear-space RGB triplets.
 * @param k - Maximum number of palette colours.
 * @param opts - Optional spatial behaviour.
 * @returns Linear-space quantized colours, same length as `linearColors`.
 */
const palettizeColors = (linearColors: Float32Array, k: number, opts?: PalettizeOptions): Float32Array => {
    const vertexCount = linearColors.length / 3;
    const effK = Math.min(k, vertexCount);

    if (vertexCount <= 1 || effK >= vertexCount) {
        return new Float32Array(linearColors);
    }

    const oklab = new Float32Array(linearColors.length);
    for (let v = 0; v < vertexCount; v++) {
        const [L, A, B] = linearToOklab(linearColors[v * 3], linearColors[v * 3 + 1], linearColors[v * 3 + 2]);
        oklab[v * 3] = L;
        oklab[v * 3 + 1] = A;
        oklab[v * 3 + 2] = B;
    }

    const result = new Float32Array(linearColors.length);

    if (effK < 2) {
        // single-colour palette: snap every vertex to the global mean
        let sumL = 0, sumA = 0, sumB = 0;
        for (let v = 0; v < vertexCount; v++) {
            sumL += oklab[v * 3];
            sumA += oklab[v * 3 + 1];
            sumB += oklab[v * 3 + 2];
        }
        const [r, g, b] = oklabToLinear(sumL / vertexCount, sumA / vertexCount, sumB / vertexCount);
        for (let v = 0; v < vertexCount; v++) {
            result[v * 3] = r;
            result[v * 3 + 1] = g;
            result[v * 3 + 2] = b;
        }
        return result;
    }

    const positions = opts?.positions;
    const voxelResolution = opts?.voxelResolution;
    const candidates = buildCandidates(oklab, vertexCount, positions, voxelResolution);
    const pool = selectPool(candidates);
    const centroids = seedCentroids(pool, effK);

    // Refine over candidate bins, each counting once, rather than over vertices.
    // Weighting by vertex population is what collapses the palette: a colour
    // family covering most of the mesh outvotes every small feature and drags
    // the centroids seeded on those features into its own mean.
    const binAssign = new Int32Array(pool.length);
    const accum = new Float64Array(centroids.length * 3);
    const counts = new Uint32Array(centroids.length);
    const reseeded = new Uint8Array(pool.length);

    for (let iter = 0; iter < MAX_ITERS; iter++) {
        for (let b = 0; b < pool.length; b++) {
            const bin = pool[b];
            let bestDist = Infinity;
            let bestC = 0;
            for (let c = 0; c < centroids.length; c++) {
                const d = dist2(bin.L, bin.A, bin.B, centroids[c][0], centroids[c][1], centroids[c][2]);
                if (d < bestDist) {
                    bestDist = d;
                    bestC = c;
                }
            }
            binAssign[b] = bestC;
        }

        accum.fill(0);
        counts.fill(0);
        for (let b = 0; b < pool.length; b++) {
            const c = binAssign[b];
            const bin = pool[b];
            accum[c * 3] += bin.L;
            accum[c * 3 + 1] += bin.A;
            accum[c * 3 + 2] += bin.B;
            counts[c]++;
        }

        let changed = false;

        // Reseed empty clusters onto the worst-served bin instead of freezing
        // them, so a wasted slot gets another chance. Each bin can only be
        // claimed once per iteration.
        reseeded.fill(0);
        for (let c = 0; c < centroids.length; c++) {
            if (counts[c] > 0) continue;
            let worstBin = -1;
            let worstDist = -1;
            for (let b = 0; b < pool.length; b++) {
                if (reseeded[b]) continue;
                const bin = pool[b];
                const a = centroids[binAssign[b]];
                const d = dist2(bin.L, bin.A, bin.B, a[0], a[1], a[2]);
                if (d > worstDist) {
                    worstDist = d;
                    worstBin = b;
                }
            }
            if (worstBin < 0) continue;
            reseeded[worstBin] = 1;
            centroids[c] = [pool[worstBin].L, pool[worstBin].A, pool[worstBin].B];
            changed = true;
        }

        for (let c = 0; c < centroids.length; c++) {
            if (counts[c] === 0) continue;
            const nL = accum[c * 3] / counts[c];
            const nA = accum[c * 3 + 1] / counts[c];
            const nB = accum[c * 3 + 2] / counts[c];
            const cur = centroids[c];
            if (dist2(nL, nA, nB, cur[0], cur[1], cur[2]) > CONVERGENCE_EPS2) {
                changed = true;
            }
            centroids[c] = [nL, nA, nB];
        }

        if (!changed) break;
    }

    // final pass, so every vertex really does get its nearest palette entry
    const assignments = new Uint32Array(vertexCount);
    assignNearest(oklab, vertexCount, centroids, assignments);

    let finalAssignments: Uint32Array<ArrayBufferLike> = assignments;
    const coherentRadius = opts?.coherentRadius;
    if (coherentRadius !== undefined && coherentRadius > 0 && positions && voxelResolution !== undefined) {
        finalAssignments = majorityFilterIndices(
            assignments, centroids.length, positions, coherentRadius, voxelResolution
        );
    }

    const palette = centroids.map(c => oklabToLinear(c[0], c[1], c[2]));
    for (let v = 0; v < vertexCount; v++) {
        const entry = palette[finalAssignments[v]];
        result[v * 3] = entry[0];
        result[v * 3 + 1] = entry[1];
        result[v * 3 + 2] = entry[2];
    }

    return result;
};

export { palettizeColors };
export type { PalettizeOptions };
