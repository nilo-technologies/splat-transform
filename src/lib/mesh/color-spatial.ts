import { fromOklabArray, toOklabArray } from './oklab';
import { IntKeyMap } from '../utils/int-key-map';

/**
 * Spatial hash over mesh vertex positions, bucketed into cubic cells.
 *
 * Cell coordinates are stored relative to the position bounds so bucket keys
 * stay exact non-negative integers; `dim` is the per-axis cell count used to
 * pack a 3D cell into a single key.
 *
 * Occupancy is held in compressed-row form — `cellOf` maps a packed cell key to
 * a cell ordinal, `starts` gives that cell's slice of `entries` — rather than as
 * a `Map<number, number[]>`. A per-cell JS array costs an allocation and a
 * header per occupied cell, and on a fine grid the cell count approaches one
 * per vertex, which both dominates memory and runs into V8's 2^24 `Map` entry
 * cap. Vertices are filled in ascending order, so each cell's slice is ordered
 * exactly as the pushed arrays were.
 */
type VertexHash = {
    cellSize: number;
    minX: number;
    minY: number;
    minZ: number;
    dim: number;
    cellOf: IntKeyMap;
    starts: Int32Array;
    entries: Int32Array;
};

// Packing three cell coordinates into one exact double requires dim^3 to stay
// within Number.MAX_SAFE_INTEGER.
const MAX_HASH_DIM = 200000;

// Neighbourhood cost grows with the cube of the radius, so cap what callers
// may ask for.
const MAX_COLOR_RADIUS = 8;

/**
 * Bucket vertex positions into a cubic spatial hash.
 *
 * @param positions - Per-vertex XYZ triplets.
 * @param cellSize - Edge length of one hash cell in world units.
 * @returns Hash suitable for `forEachNeighbor`.
 */
const buildVertexHash = (positions: Float32Array, cellSize: number): VertexHash => {
    const vertexCount = positions.length / 3;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let v = 0; v < vertexCount; v++) {
        const x = positions[v * 3];
        const y = positions[v * 3 + 1];
        const z = positions[v * 3 + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }

    const rawSpan = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
    const span = Number.isFinite(rawSpan) ? rawSpan : 0;

    // Enlarge the cell rather than wrapping coordinates, so packed keys stay
    // collision-free for meshes spanning more cells than the key can encode.
    let cell = cellSize;
    let dim = Math.floor(span / cell) + 2;
    if (dim > MAX_HASH_DIM) {
        cell = span / (MAX_HASH_DIM - 2);
        dim = MAX_HASH_DIM;
    }

    const keyOf = (v: number): number => {
        const cx = Math.floor((positions[v * 3] - minX) / cell);
        const cy = Math.floor((positions[v * 3 + 1] - minY) / cell);
        const cz = Math.floor((positions[v * 3 + 2] - minZ) / cell);
        return cx + dim * (cy + dim * cz);
    };

    // pass 1: assign a dense ordinal to every occupied cell and count it
    const cellOf = new IntKeyMap(Math.ceil(vertexCount / 0.7));
    let cellCount = 0;
    let counts = new Int32Array(Math.max(16, vertexCount >> 3));
    for (let v = 0; v < vertexCount; v++) {
        const key = keyOf(v);
        const slot = cellOf.slot(key);
        let cell1;
        if (cellOf.keys[slot] === -1) {
            cell1 = cellCount++;
            if (cell1 >= counts.length) {
                const grown = new Int32Array(counts.length * 2);
                grown.set(counts);
                counts = grown;
            }
            cellOf.insertAt(slot, key, cell1);
        } else {
            cell1 = cellOf.values[slot];
        }
        counts[cell1]++;
    }

    // pass 2: prefix-sum into slice starts, then fill in ascending vertex order
    const starts = new Int32Array(cellCount + 1);
    for (let c = 0; c < cellCount; c++) {
        starts[c + 1] = starts[c] + counts[c];
    }
    const cursor = counts.subarray(0, cellCount);
    cursor.fill(0);
    const entries = new Int32Array(vertexCount);
    for (let v = 0; v < vertexCount; v++) {
        const cell1 = cellOf.get(keyOf(v));
        entries[starts[cell1] + cursor[cell1]++] = v;
    }

    return { cellSize: cell, minX, minY, minZ, dim, cellOf, starts, entries };
};

/**
 * Invoke `fn` for every vertex within `radius` world units of vertex `index`,
 * including `index` itself.
 *
 * @param hash - Hash built by `buildVertexHash`.
 * @param positions - The same positions used to build the hash.
 * @param index - Vertex whose neighbourhood is visited.
 * @param radius - Search radius in world units.
 * @param fn - Receives each neighbouring vertex index.
 */
const forEachNeighbor = (
    hash: VertexHash,
    positions: Float32Array,
    index: number,
    radius: number,
    fn: (neighbor: number) => void
): void => {
    const { cellSize, minX, minY, minZ, dim, cellOf, starts, entries } = hash;
    const px = positions[index * 3];
    const py = positions[index * 3 + 1];
    const pz = positions[index * 3 + 2];

    const cx = Math.floor((px - minX) / cellSize);
    const cy = Math.floor((py - minY) / cellSize);
    const cz = Math.floor((pz - minZ) / cellSize);
    const reach = Math.ceil(radius / cellSize);
    const radius2 = radius * radius;

    for (let dz = -reach; dz <= reach; dz++) {
        const kz = cz + dz;
        if (kz < 0 || kz >= dim) continue;
        for (let dy = -reach; dy <= reach; dy++) {
            const ky = cy + dy;
            if (ky < 0 || ky >= dim) continue;
            for (let dx = -reach; dx <= reach; dx++) {
                const kx = cx + dx;
                if (kx < 0 || kx >= dim) continue;
                const cell = cellOf.get(kx + dim * (ky + dim * kz));
                if (cell < 0) continue;
                const end = starts[cell + 1];
                for (let i = starts[cell]; i < end; i++) {
                    const n = entries[i];
                    const ex = positions[n * 3] - px;
                    const ey = positions[n * 3 + 1] - py;
                    const ez = positions[n * 3 + 2] - pz;
                    if (ex * ex + ey * ey + ez * ez <= radius2) {
                        fn(n);
                    }
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Denoising constants
//
// A plain neighbourhood mean removes noise and material edges in equal measure,
// which is the wrong trade for a voxel mesh: the edges are the content. So the
// mean is restricted to neighbours that already agree perceptually, and the
// pass is repeated instead of widened — each repeat lets a vertex re-choose
// which side of an edge it belongs to, which is what makes a textured region
// converge on its local colour while a boundary stays put.
//
// Measured on the synthetic diorama in `tools/color-noise-bench.mjs`: three
// passes at radius 2 cut same-material colour changes from ~22% of neighbour
// pairs to ~7% and isolated speckle from ~3.8% to ~0.7% of faces, while
// *lowering* colour error at material boundaries. One wide pass of equal cost
// (radius 3, one iteration) is worse on every one of those measures.
// ---------------------------------------------------------------------------

// How close two colours must be, as a plain Oklab distance, to be averaged
// together. Roughly two just-noticeable differences: comfortably above
// reconstruction noise, well below the gap between two materials.
const SMOOTH_COLOR_EPS = 0.06;

// Repeats of the pass. Convergence is quick; beyond three the return per unit
// of work drops off sharply.
const SMOOTH_ITERATIONS = 3;

// Neighbours (excluding the vertex itself) that must agree before the
// restricted mean is trusted. Below it the vertex is isolated noise rather
// than part of a region, and falls back to the plain neighbourhood mean so it
// still gets pulled towards its surroundings.
const SMOOTH_MIN_SUPPORT = 2;

/**
 * Denoise vertex colours with an edge-preserving spatial filter.
 *
 * Each vertex is replaced by the mean of the neighbours within `radiusVoxels`
 * whose colour is within `SMOOTH_COLOR_EPS` of its own in Oklab, repeated
 * `SMOOTH_ITERATIONS` times. Neighbours across a material boundary fail that
 * test, so boundaries survive while noise inside a region averages out. A
 * vertex with fewer than `SMOOTH_MIN_SUPPORT` agreeing neighbours is treated as
 * isolated noise and takes the plain neighbourhood mean instead.
 *
 * Applied before palette construction, this stops colour noise from claiming
 * palette slots. That matters more than it sounds: palette entries are seeded
 * from binned candidate colours, so a high-variance material spreads across
 * many bins and collects many entries, and neighbouring faces then alternate
 * between them. Denoising first is what keeps a quantized mesh from reading as
 * noisier than the splats it came from.
 *
 * @param colors - Per-vertex linear-space RGB triplets.
 * @param positions - Per-vertex XYZ triplets.
 * @param radiusVoxels - Neighbourhood radius in voxel units.
 * @param voxelResolution - Size of one voxel in world units.
 * @returns Smoothed linear-space colours, same length as `colors`.
 */
const smoothVertexColors = (
    colors: Float32Array,
    positions: Float32Array,
    radiusVoxels: number,
    voxelResolution: number
): Float32Array => {
    const vertexCount = colors.length / 3;
    const hash = buildVertexHash(positions, voxelResolution);
    const radius = radiusVoxels * voxelResolution;
    const eps2 = SMOOTH_COLOR_EPS * SMOOTH_COLOR_EPS;

    let src = toOklabArray(colors);

    for (let iter = 0; iter < SMOOTH_ITERATIONS; iter++) {
        const cur = src;
        const dst = new Float32Array(cur.length);
        for (let v = 0; v < vertexCount; v++) {
            const vL = cur[v * 3];
            const vA = cur[v * 3 + 1];
            const vB = cur[v * 3 + 2];

            // restricted mean (neighbours that agree) and plain mean (all
            // neighbours), accumulated in one traversal
            let nearL = 0, nearA = 0, nearB = 0, nearCount = 0;
            let allL = 0, allA = 0, allB = 0, allCount = 0;

            forEachNeighbor(hash, positions, v, radius, (n) => {
                const nL = cur[n * 3];
                const nA = cur[n * 3 + 1];
                const nB = cur[n * 3 + 2];
                allL += nL;
                allA += nA;
                allB += nB;
                allCount++;

                const dL = vL - nL;
                const dA = vA - nA;
                const dB = vB - nB;
                if (dL * dL + dA * dA + dB * dB <= eps2) {
                    nearL += nL;
                    nearA += nA;
                    nearB += nB;
                    nearCount++;
                }
            });

            // nearCount includes the vertex itself, so subtract it for support
            if (nearCount - 1 >= SMOOTH_MIN_SUPPORT) {
                dst[v * 3] = nearL / nearCount;
                dst[v * 3 + 1] = nearA / nearCount;
                dst[v * 3 + 2] = nearB / nearCount;
            } else {
                dst[v * 3] = allL / allCount;
                dst[v * 3 + 1] = allA / allCount;
                dst[v * 3 + 2] = allB / allCount;
            }
        }
        src = dst;
    }

    return fromOklabArray(src);
};

/**
 * Replace each vertex's palette index with the most common index among its
 * spatial neighbours, breaking ties towards the lower index.
 *
 * Applied after assignment, this removes isolated speckle while leaving the
 * palette entries themselves untouched, so colour boundaries stay crisp.
 *
 * @param indices - Per-vertex palette indices.
 * @param paletteSize - Number of palette entries.
 * @param positions - Per-vertex XYZ triplets.
 * @param radiusVoxels - Neighbourhood radius in voxel units.
 * @param voxelResolution - Size of one voxel in world units.
 * @returns Filtered indices, same length as `indices`.
 */
const majorityFilterIndices = (
    indices: Uint32Array,
    paletteSize: number,
    positions: Float32Array,
    radiusVoxels: number,
    voxelResolution: number
): Uint32Array => {
    const hash = buildVertexHash(positions, voxelResolution);
    const radius = radiusVoxels * voxelResolution;
    const result = new Uint32Array(indices.length);
    const counts = new Int32Array(paletteSize);
    const touched: number[] = [];

    for (let v = 0; v < indices.length; v++) {
        touched.length = 0;
        forEachNeighbor(hash, positions, v, radius, (n) => {
            const idx = indices[n];
            if (counts[idx] === 0) touched.push(idx);
            counts[idx]++;
        });

        let best = indices[v];
        let bestCount = -1;
        for (const idx of touched) {
            const c = counts[idx];
            if (c > bestCount || (c === bestCount && idx < best)) {
                bestCount = c;
                best = idx;
            }
            counts[idx] = 0;
        }
        result[v] = best;
    }

    return result;
};

export { MAX_COLOR_RADIUS, buildVertexHash, forEachNeighbor, smoothVertexColors, majorityFilterIndices };
export type { VertexHash };
