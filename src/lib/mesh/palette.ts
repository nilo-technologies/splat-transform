import { buildVertexHash, forEachNeighbor, majorityFilterIndices } from './color-spatial';
import { srgbToLinear } from './colorize';
import { linearToOklab, oklabToLinear, toOklabArray } from './oklab';

// ---------------------------------------------------------------------------
// Tuning constants
//
// These trade colour fidelity against hue coverage, and they were calibrated
// against a single reference asset: an outdoor diorama whose collision mesh is
// ~92% red/orange by vertex count, with the remaining hues (olive foliage,
// stone, and a few percent of saturated accents) spread thinly. A scene with a
// very different colour balance is the useful next test — revisit these if
// output on such a scene looks either washed towards one tone or speckled.
//
// How to re-measure, since none of this is obvious from the values alone:
// `node --import tsx tools/color-noise-bench.mjs` builds a synthetic diorama
// with known per-material colours, runs this pipeline over it, and reports
// spatial noise, colour drift and small-feature survival side by side. Coverage
// and drift pull in opposite directions; the settings below sit deliberately
// towards coverage, per the feature's intent.
//
//   LIGHTNESS_WEIGHT     lower = more slots on hue, fewer on light/dark
//   L/AB_BIN_STEP        candidate granularity; smaller = more, finer bins
//   SUPPORT_*            what counts as a real region vs. scattered noise
//   MAX_ITERS            refinement budget
//
// One property worth knowing before touching any of it: because refinement
// gives every candidate bin one vote, a material's share of the palette tracks
// how far its colours *spread* in Oklab, not how much of the mesh it covers. A
// noisy material spreads across many bins and so collects many entries, and
// adjacent faces then alternate between them — quantized output that reads as
// noisier than the splats it came from. Two directions were measured and
// rejected as fixes: weighting bins by vertex population (much worse — it hands
// the palette to whichever material is largest) and using a different lightness
// weight for assignment than for clustering (no effect). The variance itself is
// the thing to remove, which is `smoothVertexColors`' job, upstream of here.
// ---------------------------------------------------------------------------

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

// Refinement stops early once no centroid moves, so the cap only bounds the
// pathological case; it is generous because reseeding an emptied cluster can
// perturb convergence late in the loop.
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
 * Give every vertex the palette entry nearest its colour, optionally smoothing
 * the result with a spatial majority filter.
 *
 * @param oklab - Per-vertex Oklab triplets.
 * @param vertexCount - Number of vertices.
 * @param centroids - Palette entries in Oklab, used for the nearest lookup.
 * @param palette - The same entries in linear RGB, written to the output.
 * @param opts - Optional spatial behaviour.
 * @returns Linear-space quantized colours, three per vertex.
 */
const applyPalette = (
    oklab: Float32Array,
    vertexCount: number,
    centroids: number[][],
    palette: number[][],
    opts?: PalettizeOptions
): Float32Array => {
    const assignments = new Uint32Array(vertexCount);
    assignNearest(oklab, vertexCount, centroids, assignments);

    let finalAssignments: Uint32Array<ArrayBufferLike> = assignments;
    const positions = opts?.positions;
    const voxelResolution = opts?.voxelResolution;
    const coherentRadius = opts?.coherentRadius;
    if (coherentRadius !== undefined && coherentRadius > 0 && positions && voxelResolution !== undefined) {
        finalAssignments = majorityFilterIndices(
            assignments, centroids.length, positions, coherentRadius, voxelResolution
        );
    }

    const result = new Float32Array(vertexCount * 3);
    for (let v = 0; v < vertexCount; v++) {
        const entry = palette[finalAssignments[v]];
        result[v * 3] = entry[0];
        result[v * 3 + 1] = entry[1];
        result[v * 3 + 2] = entry[2];
    }

    return result;
};

// `#rgb` / `#rrggbb`, with the hash optional so a comma-separated list only
// needs one to mark itself as colours rather than a count.
const HEX_COLOR = /^#?(?:[0-9a-f]{3}|[0-9a-f]{6})$/i;

/**
 * Convert sRGB hex colour specs into the linear-space triplets the colour
 * pipeline works in.
 *
 * @param specs - Hex colours, e.g. `['#3243aa', '4444ff']`. Both `#rgb` and
 * `#rrggbb` are accepted, with or without the leading hash.
 * @returns Linear-space RGB triplets, three entries per colour.
 * @throws If the list is empty or any entry is not a hex colour.
 */
const parsePaletteColors = (specs: string[]): Float32Array => {
    if (specs.length === 0) {
        throw new Error('palette colour list is empty');
    }

    const out = new Float32Array(specs.length * 3);
    for (let i = 0; i < specs.length; i++) {
        const spec = specs[i].trim();
        if (!HEX_COLOR.test(spec)) {
            throw new Error(`Invalid palette colour: ${specs[i]}. Expected a hex colour such as '#3243aa'.`);
        }
        const digits = spec.replace('#', '');
        const short = digits.length === 3;
        for (let ch = 0; ch < 3; ch++) {
            const hex = short ? digits[ch].repeat(2) : digits.slice(ch * 2, ch * 2 + 2);
            out[i * 3 + ch] = srgbToLinear(parseInt(hex, 16) / 255);
        }
    }

    return out;
};

/**
 * Snap vertex colours to a fixed palette supplied by the caller.
 *
 * Unlike `palettizeColors` nothing is clustered: the entries are given, so each
 * vertex simply takes its nearest one under the same lightness-weighted Oklab
 * metric, and output colours match the requested ones exactly. `coherentRadius`
 * still applies, since speckle removal is independent of how the palette was
 * chosen.
 *
 * @param linearColors - Per-vertex linear-space RGB triplets.
 * @param palette - Palette entries as linear-space RGB triplets, from
 * `parsePaletteColors`.
 * @param opts - Optional spatial behaviour.
 * @returns Linear-space quantized colours, same length as `linearColors`.
 */
const mapToPalette = (linearColors: Float32Array, palette: Float32Array, opts?: PalettizeOptions): Float32Array => {
    const entryCount = palette.length / 3;
    if (entryCount < 1) {
        throw new Error('mapToPalette requires at least one palette colour');
    }

    const vertexCount = linearColors.length / 3;
    const entries: number[][] = [];
    const centroids: number[][] = [];
    for (let e = 0; e < entryCount; e++) {
        const [r, g, b] = [palette[e * 3], palette[e * 3 + 1], palette[e * 3 + 2]];
        entries.push([r, g, b]);
        centroids.push(linearToOklab(r, g, b));
    }

    return applyPalette(toOklabArray(linearColors), vertexCount, centroids, entries, opts);
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

    const oklab = toOklabArray(linearColors);

    if (effK < 2) {
        // single-colour palette: snap every vertex to the global mean
        const result = new Float32Array(linearColors.length);
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
    return applyPalette(
        oklab, vertexCount, centroids, centroids.map(c => oklabToLinear(c[0], c[1], c[2])), opts
    );
};

export { palettizeColors, mapToPalette, parsePaletteColors };
export type { PalettizeOptions };
