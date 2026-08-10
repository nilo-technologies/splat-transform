/**
 * Directional neighbour-occupancy masks for a 4x4x4 voxel block.
 *
 * Bit layout matches `SparseVoxelGrid`: `bitIdx = lx + ly*4 + lz*16`, with `lo`
 * holding `lz` 0-1 and `hi` holding `lz` 2-3.
 *
 * `block-cleanup.ts` carries an equivalent private copy of this arithmetic
 * because it reads from a `BlockMaskBuffer` through `IntKeyMap` side tables
 * rather than from a grid. Migrating it onto this module is a follow-up.
 */

/** `lx == 0` positions in each 32-bit word. */
const FACE_X0 = 0x11111111;
/** `lx == 3` positions in each 32-bit word. */
const FACE_X3 = 0x88888888;
/** `ly == 0` positions in each 32-bit word. */
const FACE_Y0 = 0x000F000F;
/** `ly == 3` positions in each 32-bit word. */
const FACE_Y3 = 0xF000F000;
/** `lz == 0` positions: `lo` bits 0-15. */
const FACE_Z0_LO = 0x0000FFFF;
/** `lz == 3` positions: `hi` bits 16-31. */
const FACE_Z3_HI = 0xFFFF0000 >>> 0;

/** Number of `Uint32Array` entries {@link sixNeighborMasks} writes. */
const NEIGHBOR_SCRATCH_LEN = 12;

/**
 * Reads one block's `[lo, hi]` voxel mask into `out[0]`, `out[1]`.
 *
 * Implementations must write `[0, 0]` for block coordinates outside the grid,
 * which is what makes out-of-grid read as empty.
 */
type BlockReader = (bx: number, by: number, bz: number, out: Uint32Array) => void;

const adj = new Uint32Array(2);

/**
 * Compute the six directional neighbour-occupancy masks for one block.
 *
 * For direction `D`, the returned mask has a bit set at voxel position `p` when
 * the voxel at `p + D` is occupied — whether that neighbour lies inside this
 * block or in the adjacent one.
 *
 * @param read - Callback supplying any block's `[lo, hi]` mask.
 * @param lo - The subject block's own low mask word.
 * @param hi - The subject block's own high mask word.
 * @param bx - Subject block X coordinate.
 * @param by - Subject block Y coordinate.
 * @param bz - Subject block Z coordinate.
 * @param out - Destination of length >= {@link NEIGHBOR_SCRATCH_LEN}, filled as
 * `[+xLo, +xHi, -xLo, -xHi, +yLo, +yHi, -yLo, -yHi, +zLo, +zHi, -zLo, -zHi]`.
 */
const sixNeighborMasks = (
    read: BlockReader,
    lo: number,
    hi: number,
    bx: number,
    by: number,
    bz: number,
    out: Uint32Array
): void => {
    // In-block shifts. Each masks off the face that would wrap around.
    // +X: position p sees p+1, valid while lx < 3
    out[0] = (lo >>> 1) & ~FACE_X3;
    out[1] = (hi >>> 1) & ~FACE_X3;
    // -X: position p sees p-1, valid while lx > 0
    out[2] = (lo << 1) & ~FACE_X0;
    out[3] = (hi << 1) & ~FACE_X0;
    // +Y: p sees p+4
    out[4] = (lo >>> 4) & ~FACE_Y3;
    out[5] = (hi >>> 4) & ~FACE_Y3;
    // -Y: p sees p-4
    out[6] = (lo << 4) & ~FACE_Y0;
    out[7] = (hi << 4) & ~FACE_Y0;
    // +Z: p sees p+16, which crosses lo->hi at lz 1->2
    out[8] = (lo >>> 16) | (hi << 16);
    out[9] = hi >>> 16;
    // -Z: p sees p-16, crossing hi->lo at lz 2->1
    out[10] = lo << 16;
    out[11] = (hi << 16) | (lo >>> 16);

    // Cross-block faces. Our lx=3 column sees the neighbour's lx=0 column,
    // shifted up by 3 lanes; and symmetrically for the other five directions.
    read(bx + 1, by, bz, adj);
    out[0] |= (adj[0] & FACE_X0) << 3;
    out[1] |= (adj[1] & FACE_X0) << 3;

    read(bx - 1, by, bz, adj);
    out[2] |= (adj[0] & FACE_X3) >>> 3;
    out[3] |= (adj[1] & FACE_X3) >>> 3;

    read(bx, by + 1, bz, adj);
    out[4] |= (adj[0] & FACE_Y0) << 12;
    out[5] |= (adj[1] & FACE_Y0) << 12;

    read(bx, by - 1, bz, adj);
    out[6] |= (adj[0] & FACE_Y3) >>> 12;
    out[7] |= (adj[1] & FACE_Y3) >>> 12;

    read(bx, by, bz + 1, adj);
    out[9] |= (adj[0] & FACE_Z0_LO) << 16;

    read(bx, by, bz - 1, adj);
    out[10] |= (adj[1] & FACE_Z3_HI) >>> 16;

    // Normalize: the shifts above can leave results as signed int32.
    for (let i = 0; i < NEIGHBOR_SCRATCH_LEN; i++) out[i] >>>= 0;
};

export {
    NEIGHBOR_SCRATCH_LEN,
    sixNeighborMasks,
    type BlockReader
};
