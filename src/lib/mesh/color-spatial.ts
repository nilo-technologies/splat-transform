/**
 * Spatial hash over mesh vertex positions, bucketed into cubic cells.
 *
 * Cell coordinates are stored relative to the position bounds so bucket keys
 * stay exact non-negative integers; `dim` is the per-axis cell count used to
 * pack a 3D cell into a single key.
 */
type VertexHash = {
    cellSize: number;
    minX: number;
    minY: number;
    minZ: number;
    dim: number;
    buckets: Map<number, number[]>;
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

    const buckets = new Map<number, number[]>();
    for (let v = 0; v < vertexCount; v++) {
        const cx = Math.floor((positions[v * 3] - minX) / cell);
        const cy = Math.floor((positions[v * 3 + 1] - minY) / cell);
        const cz = Math.floor((positions[v * 3 + 2] - minZ) / cell);
        const key = cx + dim * (cy + dim * cz);
        const bucket = buckets.get(key);
        if (bucket) {
            bucket.push(v);
        } else {
            buckets.set(key, [v]);
        }
    }

    return { cellSize: cell, minX, minY, minZ, dim, buckets };
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
    const { cellSize, minX, minY, minZ, dim, buckets } = hash;
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
                const bucket = buckets.get(kx + dim * (ky + dim * kz));
                if (!bucket) continue;
                for (const n of bucket) {
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

/**
 * Average each vertex colour with the colours of its spatial neighbours.
 *
 * Applied before palette construction, this suppresses isolated colour noise
 * so it cannot claim a palette slot, at the cost of softening colour edges.
 *
 * @param colors - Per-vertex RGB triplets, in any single colour space.
 * @param positions - Per-vertex XYZ triplets.
 * @param radiusVoxels - Neighbourhood radius in voxel units.
 * @param voxelResolution - Size of one voxel in world units.
 * @returns Smoothed colours, same length and space as `colors`.
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
    const result = new Float32Array(colors.length);

    for (let v = 0; v < vertexCount; v++) {
        let sum0 = 0, sum1 = 0, sum2 = 0, count = 0;
        forEachNeighbor(hash, positions, v, radius, (n) => {
            sum0 += colors[n * 3];
            sum1 += colors[n * 3 + 1];
            sum2 += colors[n * 3 + 2];
            count++;
        });
        result[v * 3] = sum0 / count;
        result[v * 3 + 1] = sum1 / count;
        result[v * 3 + 2] = sum2 / count;
    }

    return result;
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
