import { Quat, Vec3 } from 'playcanvas';

import type { DataTable } from '../data-table';

/** Axis the alignment yaw rotates about. */
type UpAxis = 'x' | 'y' | 'z';

/**
 * Options for {@link estimateAlignYaw}.
 */
type AlignYawOptions = {
    /** Axis to rotate about. Default: `'y'` */
    up?: UpAxis;
    /** Search resolution in degrees. Default: 0.125 */
    stepDegrees?: number;
    /** Minimum predicted improvement required to report a non-zero yaw. Default: 0.02 */
    minImprovement?: number;
    /** Linear opacity below which a Gaussian does not vote. Default: 0.1 */
    opacityCutoff?: number;
};

/**
 * Result of {@link estimateAlignYaw}.
 */
type AlignYawResult = {
    /** Yaw to apply, in degrees, normalized to [-45, 45]. Zero when no rotation is recommended. */
    yawDegrees: number;
    /** Predicted fraction of surface voxels saved, in [0, 1). */
    improvement: number;
    /** Cost at zero yaw. */
    cost0: number;
    /** Cost at the best yaw. */
    costBest: number;
    /** Number of Gaussians that contributed a vote. */
    votedCount: number;
    /** Sum of vote weights. */
    totalWeight: number;
    /** Cost sampled across [0, 90) degrees. */
    curve: Float64Array;
    /** Set when no rotation is recommended, explaining why. */
    reason?: string;
};

const QUARTER_TURN = Math.PI / 2;

const REQUIRED_COLUMNS = [
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'scale_0', 'scale_1', 'scale_2',
    'opacity'
];

// Horizontal component pair in right-handed cyclic order, which makes
// R_up(+theta) map atan2(b, a) to atan2(b, a) + theta for every axis.
const PAIRS: Record<UpAxis, [number, number]> = {
    x: [1, 2], // (y, z)
    y: [2, 0], // (z, x)
    z: [0, 1]  // (x, y)
};

// Voxels touched per unit area by a plane whose normal sits at angle `a` to the
// grid: 1 on an axis, sqrt(2) at 45 degrees.
const gridCost = (a: number): number => Math.abs(Math.cos(a)) + Math.abs(Math.sin(a));

/**
 * Estimates the yaw that best aligns a splat scene's dominant surfaces with the
 * voxel grid axes.
 *
 * Each Gaussian votes for the orientation of its flattest axis - its surface
 * normal - weighted by how flat, how large and how opaque it is. A candidate
 * yaw's cost is how many voxels the voted surfaces would occupy at that
 * orientation, so the minimum is the best-aligned yaw. Rotating a scene by the
 * returned angle before voxelizing removes staircase voxels.
 *
 * Normals are taken into the DataTable's output space using
 * `dataTable.transform.rotation`, so the yaw can be applied directly on top of
 * the table's own transform.
 *
 * @param dataTable - Splat data with rotation, scale and opacity columns.
 * @param options - Search and weighting options.
 * @returns The recommended yaw with the cost curve and diagnostics. When no
 * orientation is meaningfully better, `yawDegrees` is 0 and `reason` explains why.
 * @throws If a required column is missing or an option is out of range.
 *
 * @example
 * ```ts
 * const { yawDegrees, improvement, reason } = estimateAlignYaw(dataTable);
 * if (!reason) {
 *     console.log(`yaw ${yawDegrees.toFixed(2)} saves ${(improvement * 100).toFixed(0)}%`);
 * }
 * ```
 */
const estimateAlignYaw = (dataTable: DataTable, options: AlignYawOptions = {}): AlignYawResult => {
    const {
        up = 'y',
        stepDegrees = 0.125,
        minImprovement = 0.02,
        opacityCutoff = 0.1
    } = options;

    if (!PAIRS[up]) {
        throw new Error(`estimateAlignYaw: invalid up axis '${up}', expected 'x', 'y' or 'z'`);
    }
    if (!(stepDegrees > 0) || stepDegrees > 90) {
        throw new Error(`estimateAlignYaw: stepDegrees must be in (0, 90], got ${stepDegrees}`);
    }
    for (const name of REQUIRED_COLUMNS) {
        if (!dataTable.hasColumn(name)) {
            throw new Error(`estimateAlignYaw: missing required column '${name}'`);
        }
    }

    const rotW = dataTable.getColumnByName('rot_0')!.data;
    const rotX = dataTable.getColumnByName('rot_1')!.data;
    const rotY = dataTable.getColumnByName('rot_2')!.data;
    const rotZ = dataTable.getColumnByName('rot_3')!.data;
    const scale0 = dataTable.getColumnByName('scale_0')!.data;
    const scale1 = dataTable.getColumnByName('scale_1')!.data;
    const scale2 = dataTable.getColumnByName('scale_2')!.data;
    const opacity = dataTable.getColumnByName('opacity')!.data;

    const numBins = Math.max(1, Math.round(90 / stepDegrees));
    const weights = new Float64Array(numBins);
    const binsPerRadian = numBins / QUARTER_TURN;

    const [aAxis, bAxis] = PAIRS[up];
    const preRotation = dataTable.transform.rotation;
    const q = new Quat();
    const n = new Vec3();
    const components = [0, 0, 0];

    let votedCount = 0;
    let totalWeight = 0;

    for (let i = 0; i < dataTable.numRows; i++) {
        const s0 = Math.exp(scale0[i]);
        const s1 = Math.exp(scale1[i]);
        const s2 = Math.exp(scale2[i]);
        if (!Number.isFinite(s0) || !Number.isFinite(s1) || !Number.isFinite(s2)) continue;

        // Logit opacity: +Infinity means fully opaque; NaN fails the comparison.
        const alpha = 1 / (1 + Math.exp(-opacity[i]));
        if (!(alpha >= opacityCutoff)) continue;

        // The flattest local axis is the surface normal.
        let flat = 0;
        let sMin = s0;
        if (s1 < sMin) {
            sMin = s1; flat = 1;
        }
        if (s2 < sMin) {
            sMin = s2; flat = 2;
        }
        const otherA = flat === 0 ? s1 : s0;
        const otherB = flat === 2 ? s1 : s2;
        const sMid = Math.min(otherA, otherB);
        const sMax = Math.max(otherA, otherB);
        if (!(sMid > 0)) continue;

        // Flat discs vote, round blobs do not; larger patches vote louder.
        const weight = (1 - sMin / sMid) * (sMid * sMax) * alpha;
        if (!(weight > 0)) continue;

        q.set(rotX[i], rotY[i], rotZ[i], rotW[i]);
        if (!(Math.hypot(q.x, q.y, q.z, q.w) > 1e-8)) continue;
        q.normalize();

        n.set(flat === 0 ? 1 : 0, flat === 1 ? 1 : 0, flat === 2 ? 1 : 0);
        q.transformVector(n, n);
        preRotation.transformVector(n, n);

        components[0] = n.x;
        components[1] = n.y;
        components[2] = n.z;
        const a = components[aAxis];
        const b = components[bAxis];
        const h = Math.hypot(a, b);
        // A normal parallel to the up axis (floor, ceiling) carries no yaw signal.
        if (!(h > 1e-6)) continue;

        let phi = Math.atan2(b, a) % QUARTER_TURN;
        if (phi < 0) phi += QUARTER_TURN;
        const bin = Math.min(numBins - 1, (phi * binsPerRadian) | 0);

        const vote = weight * h;
        weights[bin] += vote;
        totalWeight += vote;
        votedCount++;
    }

    const curve = new Float64Array(numBins);

    if (votedCount === 0) {
        return {
            yawDegrees: 0,
            improvement: 0,
            cost0: 0,
            costBest: 0,
            votedCount,
            totalWeight,
            curve,
            reason: 'no eligible gaussians: none were flat, opaque and non-degenerate enough to vote'
        };
    }

    // Both phi and theta land on bin multiples, so the sweep is a cyclic
    // correlation against a precomputed table - no trigonometry in the loop.
    const costTable = new Float64Array(numBins);
    for (let m = 0; m < numBins; m++) {
        costTable[m] = gridCost((m + 0.5) * QUARTER_TURN / numBins);
    }
    for (let j = 0; j < numBins; j++) {
        let sum = 0;
        for (let b = 0; b < numBins; b++) {
            sum += weights[b] * costTable[(b + j) % numBins];
        }
        curve[j] = sum;
    }

    let best = 0;
    for (let j = 1; j < numBins; j++) {
        if (curve[j] < curve[best]) best = j;
    }

    // Parabolic fit across the cyclic neighbours for sub-bin precision.
    const prev = curve[(best - 1 + numBins) % numBins];
    const mid = curve[best];
    const next = curve[(best + 1) % numBins];
    const denom = prev - 2 * mid + next;
    const offset = denom > 0 ? 0.5 * (prev - next) / denom : 0;

    const cost0 = curve[0];
    const costBest = mid;
    const improvement = cost0 > 0 ? 1 - costBest / cost0 : 0;

    let yawDegrees = (best + offset) * (90 / numBins);
    if (yawDegrees > 45) yawDegrees -= 90;

    if (improvement < minImprovement) {
        return {
            yawDegrees: 0,
            improvement,
            cost0,
            costBest,
            votedCount,
            totalWeight,
            curve,
            reason: `no dominant alignment: best yaw saves ${(improvement * 100).toFixed(1)}%, below the ${(minImprovement * 100).toFixed(1)}% threshold`
        };
    }

    return { yawDegrees, improvement, cost0, costBest, votedCount, totalWeight, curve };
};

export { estimateAlignYaw, type AlignYawOptions, type AlignYawResult, type UpAxis };
