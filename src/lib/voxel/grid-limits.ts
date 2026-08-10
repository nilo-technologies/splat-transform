/**
 * Block-count ceiling for the voxel grid path.
 *
 * Three separate 32-bit limits sit above the sparse grid, and this constant is
 * the tightest of them expressed as a power of two:
 *
 * - `IntKeyMap` (`../utils/int-key-map.ts`) sizes itself as
 *   `1 << (32 - Math.clz32(capacity - 1))`, which goes negative once capacity
 *   reaches 2^30. `block-cleanup.ts` builds one at `blocks / 0.7`, so blocks
 *   must stay under `0.7 * 2^30` (about 751.6e6).
 * - `BlockMaskMap` (`./block-mask-map.ts`) stores block indices in an
 *   `Int32Array`, so a key at or past 2^31 stores negative, never matches on
 *   lookup, and silently loses every MIXED block's mask — the surface blocks —
 *   while SOLID interiors survive.
 * - `SparseVoxelGrid` (`./sparse-voxel-grid.ts`) indexes its `types` words with
 *   `blockIdx >>> 4`, which aliases past 2^32.
 *
 * 2^29 clears all three. Only the first fails loudly, which is why the guard
 * exists rather than relying on the throw.
 */
const MAX_GRID_BLOCKS = 2 ** 29;

/**
 * Throws when a voxel grid holds more blocks than the sparse grid can index
 * without silently corrupting surface data.
 *
 * @param nbx - Grid block count along X.
 * @param nby - Grid block count along Y.
 * @param nbz - Grid block count along Z.
 * @param voxelResolution - Voxel size in world units, reported in the error so
 * the caller can see which resolution produced the grid.
 * @throws If the total block count exceeds {@link MAX_GRID_BLOCKS}.
 */
const assertGridFits = (
    nbx: number,
    nby: number,
    nbz: number,
    voxelResolution: number
): void => {
    const blocks = nbx * nby * nbz;
    if (blocks > MAX_GRID_BLOCKS) {
        throw new Error(
            `Voxel grid too large: ${nbx}x${nby}x${nbz} blocks (${blocks}) exceeds the ` +
            `${MAX_GRID_BLOCKS} block limit at voxel resolution ${voxelResolution}. ` +
            'Past this size the sparse grid loses surface blocks without reporting it. ' +
            'Either coarsen the resolution with --voxel-params or shrink the region with ' +
            '--filter-box.'
        );
    }
};

export { MAX_GRID_BLOCKS, assertGridFits };
