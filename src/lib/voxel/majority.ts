import { SparseVoxelGrid } from './sparse-voxel-grid';

/**
 * Options for {@link majorityFilterGrid}.
 */
type MajorityOptions = {
    /** Occupied voxels required among the 27, self included, to be on. Default: 14 */
    threshold?: number;
    /** Passes to run. Default: 2 */
    iterations?: number;
    /**
     * Inner chunk edge in voxels. Lower it to bound peak memory or to force many
     * chunks in tests; the result is identical either way. Default: 256
     */
    chunkInner?: number;
};

/**
 * Result of {@link majorityFilterGrid}.
 */
type MajorityResult = {
    /** The filtered grid. */
    grid: SparseVoxelGrid;
    /** Voxels turned on. */
    added: number;
    /** Voxels turned off. */
    removed: number;
    /**
     * Voxels that reached the threshold but were blocked by the candidate mask.
     * The audit trail for the anti-fabrication guarantee.
     */
    gateRejected: number;
};

/**
 * Regularize a voxel surface with a 3x3x3 majority filter.
 *
 * A voxel ends up occupied when at least `threshold` of the 27 voxels in its
 * neighbourhood — itself included — are occupied. Turning a voxel **on** also
 * requires it to be set in `candidate`; turning one **off** does not, since
 * removal cannot fabricate structure.
 *
 * Counting is separable, so each pass is three linear sweeps (X then Z then Y)
 * over a dense chunk rather than 27 taps per voxel. Chunks carry a halo equal to
 * the iteration count, which makes every chunk's interior exact and the result
 * independent of `chunkInner`.
 *
 * @param grid - Grid to filter. **Consumed**: do not reuse it after the call.
 * @param candidate - Voxels permitted to be turned on. Not modified.
 * @param options - Threshold, pass count and chunk size.
 * @returns The filtered grid with counts of voxels added, removed, and blocked
 * by the candidate gate.
 */
const majorityFilterGrid = (
    grid: SparseVoxelGrid,
    candidate: SparseVoxelGrid,
    options: MajorityOptions = {}
): MajorityResult => {
    const { threshold = 14, iterations = 2, chunkInner = 256 } = options;
    const { nx, ny, nz } = grid;

    let current = grid;
    let added = 0;
    let removed = 0;
    let gateRejected = 0;

    // Each pass reads the whole previous state, so passes are sequential and
    // every chunk of a pass sees the same input grid.
    for (let iter = 0; iter < iterations; iter++) {
        const next = new SparseVoxelGrid(nx, ny, nz);

        // One voxel of reach per pass; a single pass is applied per chunk here,
        // so a halo of 1 makes each chunk's interior exact.
        const halo = 1;
        const step = Math.max(4, chunkInner);

        for (let cz = 0; cz < nz; cz += step) {
            for (let cy = 0; cy < ny; cy += step) {
                for (let cx = 0; cx < nx; cx += step) {
                    const innerX = Math.min(step, nx - cx);
                    const innerY = Math.min(step, ny - cy);
                    const innerZ = Math.min(step, nz - cz);

                    // Outer region includes the halo, clamped to the grid.
                    const ox = Math.max(0, cx - halo);
                    const oy = Math.max(0, cy - halo);
                    const oz = Math.max(0, cz - halo);
                    const ex = Math.min(nx, cx + innerX + halo);
                    const ey = Math.min(ny, cy + innerY + halo);
                    const ez = Math.min(nz, cz + innerZ + halo);
                    const ow = ex - ox;
                    const oh = ey - oy;
                    const od = ez - oz;

                    const src = new Uint8Array(ow * oh * od);
                    let anyOccupied = false;
                    for (let z = 0; z < od; z++) {
                        for (let y = 0; y < oh; y++) {
                            const rowBase = y * ow + z * ow * oh;
                            for (let x = 0; x < ow; x++) {
                                if (current.getVoxel(ox + x, oy + y, oz + z)) {
                                    src[rowBase + x] = 1;
                                    anyOccupied = true;
                                }
                            }
                        }
                    }
                    if (!anyOccupied) continue;

                    // Separable counting. sumX[i] counts the 3-window along X,
                    // sumZ adds the Z window on top, sumY the Y window: 27 total.
                    const sumX = new Uint8Array(src.length);
                    for (let z = 0; z < od; z++) {
                        for (let y = 0; y < oh; y++) {
                            const base = y * ow + z * ow * oh;
                            for (let x = 0; x < ow; x++) {
                                let s = src[base + x];
                                if (x > 0) s += src[base + x - 1];
                                if (x + 1 < ow) s += src[base + x + 1];
                                sumX[base + x] = s;
                            }
                        }
                    }
                    const sumZ = new Uint8Array(src.length);
                    const zStride = ow * oh;
                    for (let z = 0; z < od; z++) {
                        for (let y = 0; y < oh; y++) {
                            const base = y * ow + z * zStride;
                            for (let x = 0; x < ow; x++) {
                                let s = sumX[base + x];
                                if (z > 0) s += sumX[base + x - zStride];
                                if (z + 1 < od) s += sumX[base + x + zStride];
                                sumZ[base + x] = s;
                            }
                        }
                    }

                    // Y pass folded into the decision, so no third buffer.
                    for (let z = 0; z < od; z++) {
                        const gz = oz + z;
                        if (gz < cz || gz >= cz + innerZ) continue;
                        for (let y = 0; y < oh; y++) {
                            const gy = oy + y;
                            if (gy < cy || gy >= cy + innerY) continue;
                            const base = y * ow + z * zStride;
                            for (let x = 0; x < ow; x++) {
                                const gx = ox + x;
                                if (gx < cx || gx >= cx + innerX) continue;

                                let count = sumZ[base + x];
                                if (y > 0) count += sumZ[base + x - ow];
                                if (y + 1 < oh) count += sumZ[base + x + ow];

                                const was = src[base + x] === 1;
                                const on = count >= threshold;
                                if (on) {
                                    if (was) {
                                        next.setVoxel(gx, gy, gz);
                                    } else if (candidate.getVoxel(gx, gy, gz)) {
                                        next.setVoxel(gx, gy, gz);
                                        added++;
                                    } else {
                                        gateRejected++;
                                    }
                                } else if (was) {
                                    removed++;
                                }
                            }
                        }
                    }
                }
            }
        }

        current.releaseStorage();
        current = next;
    }

    return { grid: current, added, removed, gateRejected };
};

export { majorityFilterGrid, type MajorityOptions, type MajorityResult };
