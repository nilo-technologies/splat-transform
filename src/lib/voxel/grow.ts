import {
    NEIGHBOR_SCRATCH_LEN,
    sixNeighborMasks,
    type BlockReader
} from './block-neighbors';
import { popcount } from './morton';
import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';

/**
 * Options for {@link growGrid}.
 */
type GrowOptions = {
    /** Minimum occupied face neighbours a candidate voxel needs. Default: 3 */
    minNeighbors?: number;
    /** Cap on passes; growing stops early once a pass adds nothing. Default: 4 */
    maxIterations?: number;
};

/**
 * Result of {@link growGrid}.
 */
type GrowResult = {
    /** The grown grid. */
    grid: SparseVoxelGrid;
    /** Voxels added across all passes. */
    added: number;
    /**
     * Voxels that met the neighbour threshold but were blocked by the candidate
     * mask. The audit trail for the anti-fabrication guarantee: these are the
     * voxels an ungated fill would have invented.
     */
    gateRejected: number;
    /** Passes actually run, including the final unproductive one. */
    iterations: number;
};

/**
 * Bit-sliced "at least `k` of the six masks set at this position".
 *
 * The three planes hold the neighbour count in binary: `c2 c1 c0`, maximum 6.
 *
 * @param k - Threshold, 1 to 6.
 * @param c0 - Count bit 0.
 * @param c1 - Count bit 1.
 * @param c2 - Count bit 2.
 * @returns Mask of positions whose count is at least `k`.
 */
const atLeast = (k: number, c0: number, c1: number, c2: number): number => {
    switch (k) {
        case 1: return c2 | c1 | c0;
        case 2: return c2 | c1;
        case 3: return c2 | (c1 & c0);
        case 4: return c2;
        case 5: return c2 & (c1 | c0);
        case 6: return c2 & c1;
        default: return 0;
    }
};

/**
 * Build a {@link BlockReader} over a grid's own blocks.
 *
 * @param g - Grid to read from.
 * @returns A reader returning `[0, 0]` for out-of-bounds coordinates.
 */
const makeReader = (g: SparseVoxelGrid): BlockReader => {
    const { nbx, nby, nbz, bStride, masks } = g;
    return (bx: number, by: number, bz: number, out: Uint32Array): void => {
        if (bx < 0 || by < 0 || bz < 0 || bx >= nbx || by >= nby || bz >= nbz) {
            out[0] = 0;
            out[1] = 0;
            return;
        }
        const bi = bx + by * nbx + bz * bStride;
        const bt = g.getBlockType(bi);
        if (bt === BLOCK_EMPTY) {
            out[0] = 0;
            out[1] = 0;
        } else if (bt === BLOCK_SOLID) {
            out[0] = SOLID_LO;
            out[1] = SOLID_HI;
        } else {
            const s = masks.slot(bi);
            out[0] = masks.lo[s];
            out[1] = masks.hi[s];
        }
    };
};

/**
 * Fill voxels that look like holes in an existing surface, restricted to a
 * candidate mask.
 *
 * A voxel is filled when it is empty, set in `candidate`, and has at least
 * `minNeighbors` of its six face neighbours occupied. Passes repeat until one
 * adds nothing or `maxIterations` is reached; each pass reads a snapshot, so the
 * result does not depend on block iteration order.
 *
 * Because every addition must be in `candidate`, this can never place a voxel
 * where the source data has no support — see the design spec's anti-fabrication
 * guarantee. The block-level skip that keeps this cheap is independent of
 * `candidate`: it looks only at occupancy, so `gateRejected` stays accurate
 * even where `candidate` and occupancy diverge.
 *
 * @param grid - Grid to grow. **Consumed**: do not reuse it after the call.
 * @param candidate - Voxels permitted to be filled. Not modified.
 * @param options - Threshold and iteration cap.
 * @returns The grown grid with the number of voxels added and passes run.
 */
const growGrid = (
    grid: SparseVoxelGrid,
    candidate: SparseVoxelGrid,
    options: GrowOptions = {}
): GrowResult => {
    const { minNeighbors = 3, maxIterations = 4 } = options;

    const { nbx, nby, nbz, bStride } = grid;
    const readCandidate = makeReader(candidate);
    const own = new Uint32Array(2);
    const cand = new Uint32Array(2);
    const nb = new Uint32Array(NEIGHBOR_SCRATCH_LEN);

    let current = grid;
    let added = 0;
    let gateRejected = 0;
    let iterations = 0;

    // Type-only check (no mask decode) used to skip a block without ever
    // consulting `candidate`. Skipping based on candidate presence instead
    // would undercount gateRejected: a block with real occupancy next door
    // can have eligible positions even while its own candidate mask is
    // empty, and that is exactly the boundary shell where the anti-
    // fabrication gate does its most visible work.
    const isEmptyType = (bx: number, by: number, bz: number): boolean => {
        if (bx < 0 || by < 0 || bz < 0 || bx >= nbx || by >= nby || bz >= nbz) return true;
        return current.getBlockType(bx + by * nbx + bz * bStride) === BLOCK_EMPTY;
    };

    for (let iter = 0; iter < maxIterations; iter++) {
        iterations++;
        const readCurrent = makeReader(current);
        const next = current.clone();
        let addedThisPass = 0;

        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                for (let bx = 0; bx < nbx; bx++) {
                    const bi = bx + by * nbx + bz * bStride;
                    const bt = current.getBlockType(bi);

                    // A fully solid block has nothing to gain.
                    if (bt === BLOCK_SOLID) continue;

                    if (bt === BLOCK_EMPTY) {
                        // Provably zero eligible voxels when this block and
                        // all six face neighbours are empty too: an eligible
                        // voxel needs at least one occupied neighbour, and
                        // there is none anywhere nearby. This is independent
                        // of `candidate` by construction -- see the comment
                        // on isEmptyType above.
                        if (isEmptyType(bx + 1, by, bz) && isEmptyType(bx - 1, by, bz) &&
                            isEmptyType(bx, by + 1, bz) && isEmptyType(bx, by - 1, bz) &&
                            isEmptyType(bx, by, bz + 1) && isEmptyType(bx, by, bz - 1)) {
                            continue;
                        }
                    }

                    readCurrent(bx, by, bz, own);
                    readCandidate(bx, by, bz, cand);

                    sixNeighborMasks(readCurrent, own[0], own[1], bx, by, bz, nb);

                    // Carry-save accumulate the six masks into a 3-bit count.
                    let lo0 = 0;
                    let lo1 = 0;
                    let lo2 = 0;
                    let hi0 = 0;
                    let hi1 = 0;
                    let hi2 = 0;
                    for (let d = 0; d < 6; d++) {
                        const mLo = nb[d * 2];
                        const cLo = lo0 & mLo;
                        lo0 ^= mLo;
                        const c1Lo = lo1 & cLo;
                        lo1 ^= cLo;
                        lo2 |= c1Lo;

                        const mHi = nb[d * 2 + 1];
                        const cHi = hi0 & mHi;
                        hi0 ^= mHi;
                        const c1Hi = hi1 & cHi;
                        hi1 ^= cHi;
                        hi2 |= c1Hi;
                    }

                    const eligibleLo = (~own[0] & atLeast(minNeighbors, lo0, lo1, lo2)) >>> 0;
                    const eligibleHi = (~own[1] & atLeast(minNeighbors, hi0, hi1, hi2)) >>> 0;
                    const newLo = (eligibleLo & cand[0]) >>> 0;
                    const newHi = (eligibleHi & cand[1]) >>> 0;
                    // Counted only on the first pass: a voxel the gate blocks
                    // stays eligible every pass, so summing would multiply it.
                    if (iter === 0) {
                        gateRejected += popcount((eligibleLo & ~cand[0]) >>> 0) +
                            popcount((eligibleHi & ~cand[1]) >>> 0);
                    }
                    if (newLo === 0 && newHi === 0) continue;

                    addedThisPass += popcount(newLo) + popcount(newHi);
                    next.orBlock(bi, newLo, newHi);
                }
            }
        }

        if (addedThisPass === 0) {
            next.releaseStorage();
            break;
        }
        added += addedThisPass;
        current.releaseStorage();
        current = next;
    }

    return { grid: current, added, gateRejected, iterations };
};

export { growGrid, type GrowOptions, type GrowResult };
