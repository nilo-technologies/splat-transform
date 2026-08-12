import { despeckleGrid } from './despeckle';
import { growGrid } from './grow';
import { majorityFilterGrid } from './majority';
import { SparseVoxelGrid } from './sparse-voxel-grid';

/**
 * Opacity threshold for the candidate mask.
 *
 * Far below any sensible solid cutoff, so the candidate set covers everywhere
 * the gaussian field has measurable presence. Every stage that adds voxels
 * intersects with it, which is what stops cleanup from inventing structure in
 * empty space: a real void has no density and is therefore untouchable at any
 * radius.
 */
const CANDIDATE_CUTOFF = 0.002;

/** Occupied voxels required among the 27, self included, for the majority filter. */
const MAJORITY_THRESHOLD = 14;

/** Majority filter passes. */
const MAJORITY_ITERATIONS = 2;

/**
 * Occupied face neighbours that spare an under-threshold voxel from the majority
 * filter. A 1-voxel-thick sheet tops out at 4, so 3 keeps sheet interiors,
 * straight edges and solid convex edges while still shaving bumps, scatter and
 * stick tips.
 *
 * Currently equal to {@link GROW_MIN_NEIGHBORS}; see the coupling note there.
 */
const MAJORITY_KEEP_FACE_NEIGHBORS = 3;

/** Components below this many voxels are removed. 64 is one 4x4x4 block. */
const DESPECKLE_MIN_VOXELS = 64;

/**
 * Face neighbours a candidate voxel needs before `grow` fills it.
 *
 * Equal to {@link MAJORITY_KEEP_FACE_NEIGHBORS}, which makes the two the same
 * predicate: every voxel `grow` adds has at least 3 occupied face neighbours by
 * construction, and `grow` never removes any, so the majority filter's first
 * pass cannot remove a grown voxel. Only the second pass can, and only where a
 * neighbour disappeared meanwhile. The majority filter has therefore stopped
 * acting as a check on `grow` over-filling: measured on the rooftops capture the
 * cleaned voxel count came out 1.1% *above* the no-cleanup baseline, where the
 * design predicted at or below it. The two decouple only if the keep gate rises
 * above this value, or this value falls below it. See
 * `specs/2026-08-10-voxel-cleanup-sheet-aware-design.md`, "Measured outcome".
 */
const GROW_MIN_NEIGHBORS = 3;

/**
 * Hole-filling algorithm.
 *
 * - `grow` — fill candidate voxels with enough occupied face neighbours.
 * - `close` — morphological closing intersected with the candidate mask.
 * - `both` — `grow`, then `close` on its result.
 * - `none` — skip hole filling; run only the majority filter and despeckle.
 */
type CleanupFillMode = 'none' | 'grow' | 'close' | 'both';

/**
 * Options for {@link cleanupGrid}.
 */
type CleanupOptions = {
    /** Scale of defect to remove, in world units. Must be > 0. */
    strength: number;
    /** Voxel size in world units, used to convert `strength` to voxels. */
    voxelResolution: number;
    /** Hole-filling algorithm. Default: `'grow'` */
    fill?: CleanupFillMode;
};

/**
 * Per-stage counts from {@link cleanupGrid}.
 */
type CleanupStats = {
    /** Defect scale in voxels, derived from the strength dial. */
    radius: number;
    /** Voxels added by the fill stage. */
    grown: number;
    /**
     * Voxels that an ungated pass would have added but the candidate mask
     * blocked. The audit trail for the anti-fabrication guarantee: a large
     * number here means the gate is doing real work.
     */
    gateRejected: number;
    /** Voxels added by the majority filter. */
    majorityAdded: number;
    /** Voxels removed by the majority filter. */
    majorityRemoved: number;
    /**
     * Voxels the majority filter spared for being part of a thin surface: what a
     * density-only filter would have deleted. Large on scenes with genuine
     * 1-voxel-thick structure such as roof decks and fences.
     * Accumulates over the filter's passes, like `majorityAdded` and
     * `majorityRemoved`, so it counts sparings rather than distinct voxels.
     */
    majorityKept: number;
    /** Voxels removed by despeckling. */
    despeckled: number;
    /** Connected components found while despeckling. */
    components: number;
    /** Components removed for being under the size threshold. */
    componentsRemoved: number;
};

/**
 * Result of {@link cleanupGrid}.
 */
type CleanupResult = {
    /** The cleaned grid. */
    grid: SparseVoxelGrid;
    /** What each stage did. */
    stats: CleanupStats;
};

/**
 * Convert the cleanup strength dial into a defect radius in whole voxels.
 *
 * Always at least 1: a strength below one voxel still means "clean up at the
 * finest scale available" rather than "do nothing".
 *
 * @param strength - Defect scale in world units.
 * @param voxelResolution - Voxel size in world units.
 * @returns Radius in voxels, at least 1.
 */
const cleanupRadius = (strength: number, voxelResolution: number): number => {
    return Math.max(1, Math.round(strength / voxelResolution));
};

/**
 * Clean up a voxel grid: fill sampling holes, regularize the surface, drop
 * floating debris.
 *
 * Every stage that adds voxels intersects with `candidate`, so cleanup can only
 * place a voxel where the source gaussians have measurable density. Stages that
 * remove voxels are ungated, since removal cannot fabricate structure.
 *
 * Runs fill (per `options.fill`), then the majority filter, then despeckling.
 * Despeckling is last because the majority filter both creates and destroys
 * islands.
 *
 * @param grid - Grid to clean. **Consumed**: do not reuse it after the call.
 * @param candidate - Voxels permitted to be added. Not modified.
 * @param options - Strength, voxel size and fill mode.
 * @returns The cleaned grid and per-stage counts.
 * @throws If `strength` or `voxelResolution` is not positive, or if the fill
 * mode is not implemented.
 */
const cleanupGrid = (
    grid: SparseVoxelGrid,
    candidate: SparseVoxelGrid,
    options: CleanupOptions
): CleanupResult => {
    const { strength, voxelResolution, fill = 'grow' } = options;

    if (!(strength > 0)) {
        throw new Error(`cleanupGrid: strength must be > 0, got ${strength}`);
    }
    if (!(voxelResolution > 0)) {
        throw new Error(`cleanupGrid: voxelResolution must be > 0, got ${voxelResolution}`);
    }
    if (fill === 'close' || fill === 'both') {
        throw new Error(
            `cleanup fill mode "${fill}" is not implemented yet; use "grow" or "none"`);
    }

    const radius = cleanupRadius(strength, voxelResolution);

    let current = grid;
    let grown = 0;
    let gateRejected = 0;

    if (fill === 'grow') {
        const res = growGrid(current, candidate, {
            minNeighbors: GROW_MIN_NEIGHBORS,
            // A gap of width w needs about w/2 passes to close from both ends,
            // and the dial's radius is half the widest gap we mean to fill.
            maxIterations: 2 * radius + 2
        });
        current = res.grid;
        grown = res.added;
        gateRejected += res.gateRejected;
    }

    const maj = majorityFilterGrid(current, candidate, {
        threshold: MAJORITY_THRESHOLD,
        iterations: MAJORITY_ITERATIONS,
        keepFaceNeighbors: MAJORITY_KEEP_FACE_NEIGHBORS
    });
    current = maj.grid;
    gateRejected += maj.gateRejected;

    const desp = despeckleGrid(current, { minVoxels: DESPECKLE_MIN_VOXELS });
    current = desp.grid;

    return {
        grid: current,
        stats: {
            radius,
            grown,
            gateRejected,
            majorityAdded: maj.added,
            majorityRemoved: maj.removed,
            majorityKept: maj.kept,
            despeckled: desp.removed,
            components: desp.components,
            componentsRemoved: desp.componentsRemoved
        }
    };
};

export {
    CANDIDATE_CUTOFF,
    MAJORITY_KEEP_FACE_NEIGHBORS,
    cleanupGrid,
    cleanupRadius,
    type CleanupFillMode,
    type CleanupOptions,
    type CleanupResult,
    type CleanupStats
};
