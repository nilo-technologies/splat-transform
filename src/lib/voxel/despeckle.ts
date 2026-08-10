import { SparseVoxelGrid } from './sparse-voxel-grid';

/**
 * Options for {@link despeckleGrid}.
 */
type DespeckleOptions = {
    /**
     * Components with fewer than this many voxels are removed. 64 is one 4x4x4
     * block, the grid's natural island unit. 0 disables removal. Default: 64
     */
    minVoxels?: number;
};

/**
 * Result of {@link despeckleGrid}.
 */
type DespeckleResult = {
    /** The despeckled grid — the same instance that was passed in. */
    grid: SparseVoxelGrid;
    /** Voxels removed. */
    removed: number;
    /** Connected components found. */
    components: number;
    /** Components removed for being under the threshold. */
    componentsRemoved: number;
};

/**
 * Remove small disconnected islands of voxels.
 *
 * Labels 6-connected components of the occupied set and clears every component
 * holding fewer than `minVoxels` voxels. One pass: each component is flooded
 * once, with members collected into a buffer capped at `minVoxels`, so a
 * component that exceeds the cap is recognised as a keeper without the buffer
 * ever growing. Every occupied voxel is visited exactly once.
 *
 * Unlike the other cleanup stages this mutates `grid` in place and returns that
 * same instance, because clearing needs no copy.
 *
 * @param grid - Grid to despeckle. **Mutated in place.**
 * @param options - Minimum component size to keep.
 * @returns The grid with counts of voxels and components removed.
 */
const despeckleGrid = (
    grid: SparseVoxelGrid,
    options: DespeckleOptions = {}
): DespeckleResult => {
    const { minVoxels = 64 } = options;
    if (minVoxels <= 0) {
        return { grid, removed: 0, components: 0, componentsRemoved: 0 };
    }

    const { nx, ny, nz } = grid;
    const visited = new SparseVoxelGrid(nx, ny, nz);

    // BFS ring buffer of packed voxel coordinates, grown geometrically. Packing
    // into one number keeps the queue a flat typed array.
    let queue = new Int32Array(1 << 12);
    // Members of the component in progress, capped: past the cap the component
    // is known to be a keeper and the list is no longer needed.
    const members = new Int32Array(minVoxels);

    const pack = (x: number, y: number, z: number): number => x + y * nx + z * nx * ny;

    let removed = 0;
    let components = 0;
    let componentsRemoved = 0;

    // Voxels of every sub-threshold component, cleared after the walk. Deferring
    // the clears keeps `grid` unmodified while `forEachOccupiedVoxel` iterates
    // its `types`/`masks`, and avoids materialising a seed list of every
    // occupied voxel -- which on a real scene is millions of entries.
    const toClear: number[] = [];

    grid.forEachOccupiedVoxel((sx, sy, sz) => {
        if (visited.getVoxel(sx, sy, sz)) return;

        components++;
        visited.setVoxel(sx, sy, sz);
        queue[0] = pack(sx, sy, sz);
        let head = 0;
        let tail = 1;
        let size = 0;

        while (head < tail) {
            const v = queue[head++];
            const x = v % nx;
            const y = ((v / nx) | 0) % ny;
            const z = (v / (nx * ny)) | 0;
            if (size < minVoxels) members[size] = v;
            size++;

            for (let d = 0; d < 6; d++) {
                const dx = d === 0 ? 1 : d === 1 ? -1 : 0;
                const dy = d === 2 ? 1 : d === 3 ? -1 : 0;
                const dz = d === 4 ? 1 : d === 5 ? -1 : 0;
                const ax = x + dx;
                const ay = y + dy;
                const az = z + dz;
                if (ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz) continue;
                if (!grid.getVoxel(ax, ay, az)) continue;
                if (visited.getVoxel(ax, ay, az)) continue;
                visited.setVoxel(ax, ay, az);
                if (tail === queue.length) {
                    const bigger = new Int32Array(queue.length * 2);
                    bigger.set(queue);
                    queue = bigger;
                }
                queue[tail++] = pack(ax, ay, az);
            }
        }

        if (size < minVoxels) {
            for (let i = 0; i < size; i++) toClear.push(members[i]);
            removed += size;
            componentsRemoved++;
        }
    });

    for (let i = 0; i < toClear.length; i++) {
        const v = toClear[i];
        grid.clearVoxel(v % nx, ((v / nx) | 0) % ny, (v / (nx * ny)) | 0);
    }

    visited.releaseStorage();
    return { grid, removed, components, componentsRemoved };
};

export { despeckleGrid, type DespeckleOptions, type DespeckleResult };
