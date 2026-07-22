import assert from 'node:assert';
import { describe, it, before } from 'node:test';

import { Column, DataTable } from '../src/lib/index.js';
import { colorizeVertices } from '../src/lib/mesh/index.js';
import { GaussianBVH } from '../src/lib/spatial/index.js';

import { assertClose } from './helpers/summary-compare.mjs';

const SH_C0 = 0.28209479177387814;
const packClr = c => (c - 0.5) / SH_C0;
const packOpacity = (opacity) => {
    if (opacity <= 0) return -20;
    if (opacity >= 1) return 20;
    return -Math.log(1 / opacity - 1);
};
const sigmoid = v => 1 / (1 + Math.exp(-v));
const srgbToLinear = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

describe('colorizeVertices', () => {
    // Two splats:
    //   splat 0 at (0, 0, 0), extent 0.5, color (0.8, 0.2, 0.5), opacity logit 0
    //   splat 1 at (0.8, 0, 0), extent 0.5, color (0.1, 0.9, 0.3), opacity logit 2
    const color0 = [0.8, 0.2, 0.5];
    const color1 = [0.1, 0.9, 0.3];
    const logit0 = 0;
    const logit1 = 2;
    const voxelResolution = 0.5;

    let bvh;
    let columns;

    before(() => {
        const dataTable = new DataTable([
            new Column('x', new Float32Array([0, 0.8])),
            new Column('y', new Float32Array([0, 0])),
            new Column('z', new Float32Array([0, 0])),
            new Column('f_dc_0', new Float32Array([packClr(color0[0]), packClr(color1[0])])),
            new Column('f_dc_1', new Float32Array([packClr(color0[1]), packClr(color1[1])])),
            new Column('f_dc_2', new Float32Array([packClr(color0[2]), packClr(color1[2])])),
            new Column('opacity', new Float32Array([logit0, logit1]))
        ]);

        const extents = new DataTable([
            new Column('extent_x', new Float32Array([0.5, 0.5])),
            new Column('extent_y', new Float32Array([0.5, 0.5])),
            new Column('extent_z', new Float32Array([0.5, 0.5]))
        ]);

        bvh = new GaussianBVH(dataTable, extents);
        columns = {
            f_dc_0: dataTable.getColumnByName('f_dc_0').data,
            f_dc_1: dataTable.getColumnByName('f_dc_1').data,
            f_dc_2: dataTable.getColumnByName('f_dc_2').data,
            opacity: dataTable.getColumnByName('opacity').data
        };
    });

    it('returns one linear RGB triplet per vertex', () => {
        const positions = new Float32Array([0, 0, 0, 10, 10, 10]);
        const result = colorizeVertices(positions, bvh, columns, voxelResolution);
        assert.strictEqual(result.length, positions.length);
    });

    it('colors a vertex overlapping exactly one splat with that splat\'s color', () => {
        // vertex box [-0.9, 0.1] on x: overlaps splat 0 ([-0.5, 0.5]) only
        const positions = new Float32Array([-0.4, 0, 0]);
        const result = colorizeVertices(positions, bvh, columns, voxelResolution);

        assertClose(result[0], srgbToLinear(color0[0]), 1e-4, 'red channel');
        assertClose(result[1], srgbToLinear(color0[1]), 1e-4, 'green channel');
        assertClose(result[2], srgbToLinear(color0[2]), 1e-4, 'blue channel');
    });

    it('blends two overlapping splats weighted by sigmoid(opacity)', () => {
        // vertex box [-0.1, 0.9] on x: overlaps both splat AABBs
        const positions = new Float32Array([0.4, 0, 0]);
        const result = colorizeVertices(positions, bvh, columns, voxelResolution);

        const w0 = sigmoid(logit0);
        const w1 = sigmoid(logit1);
        const expected = color0.map((c, i) => srgbToLinear((w0 * c + w1 * color1[i]) / (w0 + w1)));

        assertClose(result[0], expected[0], 1e-4, 'red channel');
        assertClose(result[1], expected[1], 1e-4, 'green channel');
        assertClose(result[2], expected[2], 1e-4, 'blue channel');
    });

    it('falls back to mid-grey when no splat is found at any radius', () => {
        const positions = new Float32Array([100, 0, 0]);
        const result = colorizeVertices(positions, bvh, columns, voxelResolution);

        const grey = srgbToLinear(0.5);
        assertClose(grey, 0.21404114, 1e-6, 'linear grey sanity check');
        assertClose(result[0], grey, 1e-4, 'red channel');
        assertClose(result[1], grey, 1e-4, 'green channel');
        assertClose(result[2], grey, 1e-4, 'blue channel');
    });

    it('expands the query radius to find splats beyond the initial radius', () => {
        // splat 1 AABB max x is 1.3; vertex box at 1x is [1.5, 2.5] (no overlap),
        // at 2x is [1.0, 3.0] which overlaps splat 1 only
        const positions = new Float32Array([2.0, 0, 0]);
        const result = colorizeVertices(positions, bvh, columns, voxelResolution);

        assertClose(result[0], srgbToLinear(color1[0]), 1e-4, 'red channel');
        assertClose(result[1], srgbToLinear(color1[1]), 1e-4, 'green channel');
        assertClose(result[2], srgbToLinear(color1[2]), 1e-4, 'blue channel');
    });

    it('produces bit-identical output for the average mode through the new signature', () => {
        // exact Float32 values captured from the implementation before the mode
        // parameter was added (vertex [0.4, 0, 0], two-splat blend above)
        const expected = [
            0.10256420075893402,
            0.3755735456943512,
            0.11435376107692719
        ];

        const positions = new Float32Array([0.4, 0, 0]);

        // no mode arg (default) and explicit 'average' must both match exactly
        for (const result of [
            colorizeVertices(positions, bvh, columns, voxelResolution),
            colorizeVertices(positions, bvh, columns, voxelResolution, 'average')
        ]) {
            assert.strictEqual(result[0], expected[0], 'red channel bits');
            assert.strictEqual(result[1], expected[1], 'green channel bits');
            assert.strictEqual(result[2], expected[2], 'blue channel bits');
        }
    });
});

// build a BVH + full column set (including rot/scale) from plain-object splats:
// { center, extent, color, logit, quat: [w, x, y, z], logScale }
const makeSplatFixture = (splats) => {
    const arr = f => new Float32Array(splats.map(f));
    const dataTable = new DataTable([
        new Column('x', arr(s => s.center[0])),
        new Column('y', arr(s => s.center[1])),
        new Column('z', arr(s => s.center[2])),
        new Column('f_dc_0', arr(s => packClr(s.color[0]))),
        new Column('f_dc_1', arr(s => packClr(s.color[1]))),
        new Column('f_dc_2', arr(s => packClr(s.color[2]))),
        new Column('opacity', arr(s => s.logit)),
        new Column('rot_0', arr(s => s.quat[0])),
        new Column('rot_1', arr(s => s.quat[1])),
        new Column('rot_2', arr(s => s.quat[2])),
        new Column('rot_3', arr(s => s.quat[3])),
        new Column('scale_0', arr(s => s.logScale[0])),
        new Column('scale_1', arr(s => s.logScale[1])),
        new Column('scale_2', arr(s => s.logScale[2]))
    ]);
    const extents = new DataTable([
        new Column('extent_x', arr(s => s.extent[0])),
        new Column('extent_y', arr(s => s.extent[1])),
        new Column('extent_z', arr(s => s.extent[2]))
    ]);
    const names = ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
        'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2'];
    const columns = {};
    for (const name of names) {
        columns[name] = dataTable.getColumnByName(name).data;
    }
    return { bvh: new GaussianBVH(dataTable, extents), columns };
};

describe('colorizeVertices density modes', () => {
    const identity = [1, 0, 0, 0];
    const voxelResolution = 0.5;

    // Two splats, both overlapping a vertex at the origin:
    //   splat 0 "fat/far":  center (3, 0, 0), sigma 2, red, opacity logit 4
    //   splat 1 "tight/near": center (0.25, 0, 0), sigma 0.5, blue, opacity logit 0
    // Identity rotations, so u = d and m^2 = (d / sigma)^2.
    // Hand-computed weights w = sigmoid(logit) * exp(-0.5 * m^2):
    //   w0 = sigmoid(4) * exp(-0.5 * (3/2)^2)   = 0.98201379 * exp(-1.125)
    //   w1 = sigmoid(0) * exp(-0.5 * (0.5)^2)   = 0.5        * exp(-0.125)
    const nearFarSplats = [
        { center: [3, 0, 0], extent: [3, 0.5, 0.5], color: [1, 0, 0], logit: 4, quat: identity, logScale: [Math.log(2), Math.log(2), Math.log(2)] },
        { center: [0.25, 0, 0], extent: [0.5, 0.5, 0.5], color: [0, 0, 1], logit: 0, quat: identity, logScale: [Math.log(0.5), Math.log(0.5), Math.log(0.5)] }
    ];
    const wFar = 0.31881319991573137;
    const wNear = 0.44124845129229773;

    describe('gaussian mode', () => {
        it('weights candidates by sigmoid(opacity) * exp(-0.5 * m^2)', () => {
            const { bvh, columns } = makeSplatFixture(nearFarSplats);
            const result = colorizeVertices(new Float32Array([0, 0, 0]), bvh, columns, voxelResolution, 'gaussian');

            const sumW = wFar + wNear;
            const expected = [
                srgbToLinear(wFar / sumW),          // red from splat 0 only
                srgbToLinear(0),                    // neither splat has green
                srgbToLinear(wNear / sumW)          // blue from splat 1 only
            ];

            assertClose(result[0], expected[0], 1e-4, 'red channel');
            assertClose(result[1], expected[1], 1e-4, 'green channel');
            assertClose(result[2], expected[2], 1e-4, 'blue channel');
        });

        it('lets a tight near splat win over a fatter far splat with higher opacity', () => {
            const { bvh, columns } = makeSplatFixture(nearFarSplats);
            const positions = new Float32Array([0, 0, 0]);

            // plain average: sigmoid(4) > sigmoid(0), so red dominates
            const average = colorizeVertices(positions, bvh, columns, voxelResolution);
            assert(average[0] > average[2], 'average mode should favor the high-opacity red splat');

            // gaussian: wNear > wFar, so blue dominates
            const gaussian = colorizeVertices(positions, bvh, columns, voxelResolution, 'gaussian');
            assert(gaussian[2] > gaussian[0], 'gaussian mode should favor the tight near blue splat');
        });

        it('rotates the offset into the splat frame (u = R^T d)', () => {
            // Two splats at the same center with the same anisotropic scales,
            // differing only in rotation. Splat B is rotated 90 degrees about z,
            // so the world-x offset falls along its long (sigma = 2) axis:
            //   splat A (identity): m^2 = 0.9^2 / 0.5^2 = 3.24   -> w = 0.5 * exp(-1.62)
            //   splat B (rot 90 z): m^2 = 0.9^2 / 2^2   = 0.2025 -> w = 0.5 * exp(-0.10125)
            const halfSqrt2 = Math.SQRT1_2;
            const splats = [
                { center: [0.9, 0, 0], extent: [0.5, 0.5, 0.5], color: [1, 0, 0], logit: 0, quat: identity, logScale: [Math.log(0.5), Math.log(2), 0] },
                { center: [0.9, 0, 0], extent: [0.5, 0.5, 0.5], color: [0, 1, 0], logit: 0, quat: [halfSqrt2, 0, 0, halfSqrt2], logScale: [Math.log(0.5), Math.log(2), 0] }
            ];
            const wA = 0.098949349541807327;
            const wB = 0.45185353893659802;

            const { bvh, columns } = makeSplatFixture(splats);
            const positions = new Float32Array([0, 0, 0]);

            const gaussian = colorizeVertices(positions, bvh, columns, voxelResolution, 'gaussian');
            const sumW = wA + wB;
            assertClose(gaussian[0], srgbToLinear(wA / sumW), 1e-4, 'red channel');
            assertClose(gaussian[1], srgbToLinear(wB / sumW), 1e-4, 'green channel');
            assertClose(gaussian[2], srgbToLinear(0), 1e-4, 'blue channel');

            // the rotated splat has the higher density, so dominant picks green
            const dominant = colorizeVertices(positions, bvh, columns, voxelResolution, 'dominant');
            assertClose(dominant[0], srgbToLinear(0), 1e-4, 'red channel');
            assertClose(dominant[1], srgbToLinear(1), 1e-4, 'green channel');
            assertClose(dominant[2], srgbToLinear(0), 1e-4, 'blue channel');
        });
    });

    describe('dominant mode', () => {
        it('returns the argmax-weight splat color, not a blend', () => {
            const { bvh, columns } = makeSplatFixture(nearFarSplats);
            const result = colorizeVertices(new Float32Array([0, 0, 0]), bvh, columns, voxelResolution, 'dominant');

            // wNear > wFar, so the output is exactly splat 1's blue
            assertClose(result[0], srgbToLinear(0), 1e-4, 'red channel');
            assertClose(result[1], srgbToLinear(0), 1e-4, 'green channel');
            assertClose(result[2], srgbToLinear(1), 1e-4, 'blue channel');

            // a blend would contain a red component (wFar / (wFar + wNear) > 0.4)
            assertClose(result[0], 0, 1e-4, 'red channel carries no blend');
        });
    });

    describe('topk mode', () => {
        // Four splats along +x, identity rotation, sigma 1, logits 0..3,
        // colors red, green, blue, white. Hand-computed weights
        // w = sigmoid(logit) * exp(-0.5 * d^2):
        const topkSplats = [
            { center: [0.1, 0, 0], extent: [0.5, 0.5, 0.5], color: [1, 0, 0], logit: 0, quat: identity, logScale: [0, 0, 0] },
            { center: [0.2, 0, 0], extent: [0.5, 0.5, 0.5], color: [0, 1, 0], logit: 1, quat: identity, logScale: [0, 0, 0] },
            { center: [0.3, 0, 0], extent: [0.5, 0.5, 0.5], color: [0, 0, 1], logit: 2, quat: identity, logScale: [0, 0, 0] },
            { center: [0.4, 0, 0], extent: [0.5, 0.5, 0.5], color: [1, 1, 1], logit: 3, quat: identity, logScale: [0, 0, 0] }
        ];
        const w0 = 0.49750623959634116;
        const w1 = 0.71658264888265299;
        const w2 = 0.84203978855280803;
        const w3 = 0.87933674761476455;

        it('averages only the top 3 candidates by weight, renormalized', () => {
            const { bvh, columns } = makeSplatFixture(topkSplats);
            const result = colorizeVertices(new Float32Array([0, 0, 0]), bvh, columns, voxelResolution, 'topk');

            // w3 > w2 > w1 > w0, so splat 0 (red) is dropped
            const sumW = w1 + w2 + w3;
            const expected = [
                srgbToLinear(w3 / sumW),                  // red from the white splat only
                srgbToLinear((w1 + w3) / sumW),           // green splat + white splat
                srgbToLinear((w2 + w3) / sumW)            // blue splat + white splat
            ];

            assertClose(result[0], expected[0], 1e-4, 'red channel');
            assertClose(result[1], expected[1], 1e-4, 'green channel');
            assertClose(result[2], expected[2], 1e-4, 'blue channel');

            // sanity: including all 4 weights would noticeably change the red channel
            const allRed = srgbToLinear((w0 + w3) / (w0 + w1 + w2 + w3));
            assert(Math.abs(result[0] - allRed) > 1e-2, 'top-3 result must differ from full weighted average');
        });

        it('uses all candidates when fewer than 3 overlap', () => {
            const { bvh, columns } = makeSplatFixture(nearFarSplats);
            const positions = new Float32Array([0, 0, 0]);
            const result = colorizeVertices(positions, bvh, columns, voxelResolution, 'topk');

            // both candidates used: identical to the gaussian-mode average
            const sumW = wFar + wNear;
            assertClose(result[0], srgbToLinear(wFar / sumW), 1e-4, 'red channel');
            assertClose(result[1], srgbToLinear(0), 1e-4, 'green channel');
            assertClose(result[2], srgbToLinear(wNear / sumW), 1e-4, 'blue channel');
        });
    });

    describe('column requirements', () => {
        it('throws for non-average modes when rot/scale columns are missing', () => {
            const { bvh, columns } = makeSplatFixture(nearFarSplats);
            const positions = new Float32Array([0, 0, 0]);

            const colorOnly = {
                f_dc_0: columns.f_dc_0,
                f_dc_1: columns.f_dc_1,
                f_dc_2: columns.f_dc_2,
                opacity: columns.opacity
            };

            for (const mode of ['dominant', 'topk', 'gaussian']) {
                assert.throws(
                    () => colorizeVertices(positions, bvh, colorOnly, voxelResolution, mode),
                    new RegExp(`mode '${mode}'.*rot_0.*scale_2`),
                    `${mode} mode should throw naming the missing columns`
                );
            }

            // partially present columns throw too, naming only what is missing
            const rotOnly = { ...colorOnly, rot_0: columns.rot_0, rot_1: columns.rot_1, rot_2: columns.rot_2, rot_3: columns.rot_3 };
            assert.throws(
                () => colorizeVertices(positions, bvh, rotOnly, voxelResolution, 'gaussian'),
                /scale_0.*scale_1.*scale_2/,
                'gaussian mode should throw naming the missing scale columns'
            );

            // average mode never requires rot/scale columns
            assert.doesNotThrow(() => colorizeVertices(positions, bvh, colorOnly, voxelResolution));
        });
    });

    describe('zero weights', () => {
        it('falls back to mid-grey when all densities are zero (zero-length quaternion)', () => {
            const splats = [
                { center: [0, 0, 0], extent: [0.5, 0.5, 0.5], color: [1, 0, 0], logit: 0, quat: [0, 0, 0, 0], logScale: [0, 0, 0] }
            ];
            const { bvh, columns } = makeSplatFixture(splats);
            const positions = new Float32Array([0, 0, 0]);

            const grey = srgbToLinear(0.5);
            assertClose(grey, 0.21404114, 1e-6, 'linear grey sanity check');

            for (const mode of ['dominant', 'topk', 'gaussian']) {
                const result = colorizeVertices(positions, bvh, columns, voxelResolution, mode);
                assertClose(result[0], grey, 1e-4, `${mode} red channel`);
                assertClose(result[1], grey, 1e-4, `${mode} green channel`);
                assertClose(result[2], grey, 1e-4, `${mode} blue channel`);
            }
        });
    });
});
