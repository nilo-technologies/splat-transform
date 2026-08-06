/**
 * Tests for voxel yaw auto-alignment estimation.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { Quat, Vec3 } from 'playcanvas';

import { Column, DataTable } from '../src/lib/index.js';
import { estimateAlignYaw } from '../src/lib/voxel/align-yaw.js';

const COLUMN_NAMES = [
    'x', 'y', 'z',
    'scale_0', 'scale_1', 'scale_2',
    'opacity',
    'rot_0', 'rot_1', 'rot_2', 'rot_3'
];

/**
 * Build a DataTable of flat gaussians standing as vertical walls, one per given
 * yaw. Local X is the flat axis, so a yaw about Y aims the wall's normal.
 *
 * @param {number[]} yawsDeg - Wall orientation per splat, in degrees.
 * @param {object} [opts] - Overrides.
 * @param {number} [opts.thin=0.01] - Linear size of the flat axis.
 * @param {number} [opts.wide=1] - Linear size of the two in-plane axes.
 * @param {number} [opts.alpha=0.9] - Linear opacity.
 * @returns {DataTable} Table with the standard voxel columns.
 */
function makeWalls(yawsDeg, opts = {}) {
    const { thin = 0.01, wide = 1, alpha = 0.9 } = opts;
    const count = yawsDeg.length;
    const data = {};
    for (const name of COLUMN_NAMES) {
        data[name] = new Float32Array(count);
    }

    const q = new Quat();
    for (let i = 0; i < count; i++) {
        q.setFromEulerAngles(0, yawsDeg[i], 0);
        data.rot_0[i] = q.w;
        data.rot_1[i] = q.x;
        data.rot_2[i] = q.y;
        data.rot_3[i] = q.z;
        data.scale_0[i] = Math.log(thin);
        data.scale_1[i] = Math.log(wide);
        data.scale_2[i] = Math.log(wide);
        data.opacity[i] = Math.log(alpha / (1 - alpha));
    }

    return new DataTable(COLUMN_NAMES.map(name => new Column(name, data[name])));
}

/**
 * Rotate a splat's flat-axis normal by a yaw and report how axis-aligned it is.
 * Returns (|x| + |z|) / hypot(x, z): 1 on a grid axis, sqrt(2) at 45 degrees.
 *
 * @param {DataTable} table - Table built by makeWalls.
 * @param {number} row - Row index.
 * @param {number} yawDeg - Yaw to apply about Y, in degrees.
 * @returns {number} Alignment ratio in [1, sqrt(2)].
 */
function alignmentRatio(table, row, yawDeg) {
    const get = name => table.getColumnByName(name).data[row];
    const q = new Quat(get('rot_1'), get('rot_2'), get('rot_3'), get('rot_0')).normalize();
    const n = new Vec3(1, 0, 0);
    q.transformVector(n, n);
    new Quat().setFromEulerAngles(0, yawDeg, 0).transformVector(n, n);
    return (Math.abs(n.x) + Math.abs(n.z)) / Math.hypot(n.x, n.z);
}

describe('estimateAlignYaw', function () {
    it('finds the yaw that axis-aligns a set of tilted walls', function () {
        // Four walls 90 degrees apart, the whole set tilted by 17 degrees.
        const table = makeWalls([17, 107, 197, 287]);

        const result = estimateAlignYaw(table);

        assert.strictEqual(result.reason, undefined, `unexpected guard: ${result.reason}`);
        // gridCost(17deg) = cos(17deg) + sin(17deg) ~= 1.24868, so the true
        // improvement for this exact fixture is 1 - 1/1.24868 ~= 0.19915 -
        // under the mathematically natural 0.2 threshold, independent of
        // implementation. 0.19 leaves margin below the verified value while
        // still asserting a large improvement.
        assert.ok(result.improvement > 0.19, `expected a large improvement, got ${result.improvement}`);
        for (let row = 0; row < 4; row++) {
            assert.ok(alignmentRatio(table, row, 0) > 1.2,
                `row ${row}: fixture should start misaligned`);
            const after = alignmentRatio(table, row, result.yawDegrees);
            // 17deg is an exact multiple of the default stepDegrees (0.125),
            // so this vote lands exactly on a histogram bin boundary - the
            // worst case for the bin-center-sampled cost table, bounded at
            // half a bin (0.0625deg, ratio ~1.00109). 1.0015 covers that with
            // margin while still requiring near-perfect alignment.
            assert.ok(after < 1.0015,
                `row ${row}: expected axis-aligned after yaw, ratio ${after}`);
        }
    });

    it('returns the smallest equivalent rotation, inside [-45, 45]', function () {
        const table = makeWalls([62, 152, 242, 332]);

        const result = estimateAlignYaw(table);

        assert.ok(result.yawDegrees >= -45 && result.yawDegrees <= 45,
            `expected a normalized yaw, got ${result.yawDegrees}`);
        for (let row = 0; row < 4; row++) {
            const after = alignmentRatio(table, row, result.yawDegrees);
            // See the tolerance note in the previous test: 62deg is also an
            // exact multiple of the default stepDegrees.
            assert.ok(after < 1.0015,
                `row ${row}: expected axis-aligned after yaw, ratio ${after}`);
        }
    });

    it('ignores isotropic blobs, which have no surface orientation', function () {
        const table = makeWalls([17, 107, 197, 287], { thin: 0.5, wide: 0.5 });

        const result = estimateAlignYaw(table);

        assert.strictEqual(result.yawDegrees, 0);
        assert.strictEqual(result.votedCount, 0);
        assert.match(result.reason, /no eligible gaussians/);
    });

    it('reports no dominant alignment when normals are spread evenly', function () {
        const yaws = [];
        for (let i = 0; i < 360; i++) yaws.push(i * 0.25);

        const result = estimateAlignYaw(makeWalls(yaws));

        assert.strictEqual(result.yawDegrees, 0);
        assert.ok(result.votedCount > 0, 'the splats should still have voted');
        assert.match(result.reason, /no dominant alignment/);
    });

    it('does not let faint gaussians outvote opaque ones', function () {
        // 200 faint walls at 30 degrees against 4 opaque walls on-axis. The
        // faint set is under the cutoff, so the opaque set must decide.
        const faint = makeWalls(new Array(200).fill(30), { alpha: 0.02 });
        const opaque = makeWalls([0, 90, 180, 270], { alpha: 0.95 });
        const merged = new DataTable(COLUMN_NAMES.map((name) => {
            const head = faint.getColumnByName(name).data;
            const tail = opaque.getColumnByName(name).data;
            const data = new Float32Array(head.length + tail.length);
            data.set(head, 0);
            data.set(tail, head.length);
            return new Column(name, data);
        }));

        const result = estimateAlignYaw(merged);

        assert.ok(Math.abs(result.yawDegrees) < 1,
            `expected the opaque on-axis walls to win, got ${result.yawDegrees}`);
    });

    it('counts fully opaque splats and skips NaN rows', function () {
        const table = makeWalls([17, 107, 197, 287]);
        const opacityData = table.getColumnByName('opacity').data;
        opacityData[0] = Infinity;   // alpha === 1, votes
        opacityData[1] = NaN;        // skipped

        const result = estimateAlignYaw(table);

        assert.strictEqual(result.votedCount, 3);
        assert.ok(Number.isFinite(result.costBest));
        // Ceiling for this fixture is ~0.19915 (see Task 1); leave margin below it.
        assert.ok(result.improvement > 0.19);
    });
});
