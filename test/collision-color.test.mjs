import assert from 'node:assert';
import { describe, it, before } from 'node:test';

import { Column, DataTable } from '../src/lib/index.js';
import { colorizeVertices, computeVertexNormals } from '../src/lib/mesh/index.js';
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

// zero normals disable the inward filter, isolating distance-gate behavior
const noNormals = count => new Float32Array(count);

describe('computeVertexNormals', () => {
    it('computes a unit normal for a single triangle', () => {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
        const indices = new Uint32Array([0, 1, 2]);
        const normals = computeVertexNormals(positions, indices);

        for (let v = 0; v < 3; v++) {
            assertClose(normals[v * 3], 0, 1e-6, `vertex ${v} nx`);
            assertClose(normals[v * 3 + 1], 0, 1e-6, `vertex ${v} ny`);
            assertClose(normals[v * 3 + 2], 1, 1e-6, `vertex ${v} nz`);
        }
    });

    it('area-weights normals across triangles sharing a vertex', () => {
        // tri A (0,0,0),(2,0,0),(0,2,0) contributes cross (0,0,4)
        // tri B (0,0,0),(0,2,0),(0,0,-2) contributes cross (-4,0,0)
        // shared vertex 0 accumulates (-4,0,4) -> normalized (-sqrt(.5),0,sqrt(.5))
        const positions = new Float32Array([
            0, 0, 0,
            2, 0, 0,
            0, 2, 0,
            0, 0, -2
        ]);
        const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
        const normals = computeVertexNormals(positions, indices);

        const s = Math.SQRT1_2;
        assertClose(normals[0], -s, 1e-6, 'shared vertex nx');
        assertClose(normals[1], 0, 1e-6, 'shared vertex ny');
        assertClose(normals[2], s, 1e-6, 'shared vertex nz');
        // vertex 1 only belongs to tri A -> pure (0,0,1)
        assertClose(normals[3], 0, 1e-6, 'tri-A-only vertex nx');
        assertClose(normals[5], 1, 1e-6, 'tri-A-only vertex nz');
    });

    it('leaves a zero normal for a degenerate zero-area triangle', () => {
        const positions = new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]);
        const indices = new Uint32Array([0, 1, 2]);
        const normals = computeVertexNormals(positions, indices);

        for (let i = 0; i < normals.length; i++) {
            assert.strictEqual(normals[i], 0, `component ${i} must be zero`);
        }
    });
});

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
        const result = colorizeVertices(positions, noNormals(6), bvh, columns, voxelResolution);
        assert.strictEqual(result.length, positions.length);
    });

    it('colors a vertex near exactly one splat with that splat\'s color', () => {
        // vertex (-0.4,0,0): splat 0 center is 0.4 away (within the 0.75 gate),
        // splat 1 center is 1.2 away (gated out)
        const positions = new Float32Array([-0.4, 0, 0]);
        const result = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution);

        assertClose(result[0], srgbToLinear(color0[0]), 1e-4, 'red channel');
        assertClose(result[1], srgbToLinear(color0[1]), 1e-4, 'green channel');
        assertClose(result[2], srgbToLinear(color0[2]), 1e-4, 'blue channel');
    });

    it('blends two nearby splats weighted by sigmoid(opacity)', () => {
        // vertex (0.4,0,0): both centers are 0.4 away, inside the gate
        const positions = new Float32Array([0.4, 0, 0]);
        const result = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution);

        const w0 = sigmoid(logit0);
        const w1 = sigmoid(logit1);
        const expected = color0.map((c, i) => srgbToLinear((w0 * c + w1 * color1[i]) / (w0 + w1)));

        assertClose(result[0], expected[0], 1e-4, 'red channel');
        assertClose(result[1], expected[1], 1e-4, 'green channel');
        assertClose(result[2], expected[2], 1e-4, 'blue channel');
    });

    it('excludes a splat whose center is beyond the distance gate', () => {
        // splat 1 moved to (1.5,0,0) with extent 2: its AABB still overlaps the
        // query box, but its center is 1.1 from the vertex (> 0.75 gate)
        const dataTable = new DataTable([
            new Column('x', new Float32Array([0, 1.5])),
            new Column('y', new Float32Array([0, 0])),
            new Column('z', new Float32Array([0, 0])),
            new Column('f_dc_0', new Float32Array([packClr(color0[0]), packClr(color1[0])])),
            new Column('f_dc_1', new Float32Array([packClr(color0[1]), packClr(color1[1])])),
            new Column('f_dc_2', new Float32Array([packClr(color0[2]), packClr(color1[2])])),
            new Column('opacity', new Float32Array([logit0, logit1]))
        ]);
        const extents = new DataTable([
            new Column('extent_x', new Float32Array([0.5, 2])),
            new Column('extent_y', new Float32Array([0.5, 2])),
            new Column('extent_z', new Float32Array([0.5, 2]))
        ]);
        const gatedBvh = new GaussianBVH(dataTable, extents);
        const gatedColumns = {
            f_dc_0: dataTable.getColumnByName('f_dc_0').data,
            f_dc_1: dataTable.getColumnByName('f_dc_1').data,
            f_dc_2: dataTable.getColumnByName('f_dc_2').data,
            opacity: dataTable.getColumnByName('opacity').data
        };

        const positions = new Float32Array([0.4, 0, 0]);
        const result = colorizeVertices(positions, noNormals(3), gatedBvh, gatedColumns, voxelResolution);

        // ungated, splat 1's higher opacity would pull the blend toward color1;
        // gated, only splat 0 remains
        assertClose(result[0], srgbToLinear(color0[0]), 1e-4, 'red channel');
        assertClose(result[1], srgbToLinear(color0[1]), 1e-4, 'green channel');
        assertClose(result[2], srgbToLinear(color0[2]), 1e-4, 'blue channel');
    });

    it('falls back to all 2x-box candidates when none pass the distance gate', () => {
        // vertex (2,0,0): 2x box [1,3] overlaps splat 1's AABB, but splat 1's
        // center is 1.2 away (> 0.75 gate) -> falls back to the ungated set
        const positions = new Float32Array([2, 0, 0]);
        const result = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution);

        assertClose(result[0], srgbToLinear(color1[0]), 1e-4, 'red channel');
        assertClose(result[1], srgbToLinear(color1[1]), 1e-4, 'green channel');
        assertClose(result[2], srgbToLinear(color1[2]), 1e-4, 'blue channel');
    });

    it('falls back to a 4x query when the 2x box is empty', () => {
        // vertex (3,0,0): 2x box [2,4] does not reach splat 1's AABB (max 1.3);
        // the 4x box [1,5] does
        const positions = new Float32Array([3, 0, 0]);
        const result = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution);

        assertClose(result[0], srgbToLinear(color1[0]), 1e-4, 'red channel');
        assertClose(result[1], srgbToLinear(color1[1]), 1e-4, 'green channel');
        assertClose(result[2], srgbToLinear(color1[2]), 1e-4, 'blue channel');
    });

    it('falls back to mid-grey when no splat is found at any radius', () => {
        const positions = new Float32Array([100, 0, 0]);
        const result = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution);

        const grey = srgbToLinear(0.5);
        assertClose(grey, 0.21404114, 1e-6, 'linear grey sanity check');
        assertClose(result[0], grey, 1e-4, 'red channel');
        assertClose(result[1], grey, 1e-4, 'green channel');
        assertClose(result[2], grey, 1e-4, 'blue channel');
    });

    describe('inward filter', () => {
        // vertex (0.4,0,0) between splat 0 (at 0) and splat 1 (at 0.8);
        // inward margin is 0.5 * 0.5 = 0.25
        const positions = new Float32Array([0.4, 0, 0]);

        it('excludes a splat on the outward side of the surface normal', () => {
            // normal +x: splat 0 is behind (dot = -0.4 <= 0.25), splat 1 is
            // outward (dot = +0.4 > 0.25) -> only splat 0 remains
            const normals = new Float32Array([1, 0, 0]);
            const result = colorizeVertices(positions, normals, bvh, columns, voxelResolution);

            assertClose(result[0], srgbToLinear(color0[0]), 1e-4, 'red channel');
            assertClose(result[1], srgbToLinear(color0[1]), 1e-4, 'green channel');
            assertClose(result[2], srgbToLinear(color0[2]), 1e-4, 'blue channel');
        });

        it('mirrors the selection when the normal is flipped', () => {
            // normal -x: splat 0 is outward, splat 1 is behind
            const normals = new Float32Array([-1, 0, 0]);
            const result = colorizeVertices(positions, normals, bvh, columns, voxelResolution);

            assertClose(result[0], srgbToLinear(color1[0]), 1e-4, 'red channel');
            assertClose(result[1], srgbToLinear(color1[1]), 1e-4, 'green channel');
            assertClose(result[2], srgbToLinear(color1[2]), 1e-4, 'blue channel');
        });

        it('uses all gated splats when the vertex normal is zero', () => {
            const result = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution);

            const w0 = sigmoid(logit0);
            const w1 = sigmoid(logit1);
            assertClose(result[0], srgbToLinear((w0 * color0[0] + w1 * color1[0]) / (w0 + w1)), 1e-4, 'red channel');
        });

        it('falls back to the distance-gated set when every gated splat is outward', () => {
            // vertex (-0.4,0,0), normal +x: splat 0 (gated, dist 0.4) is outward
            // (dot = +0.4 > 0.25), splat 1 is beyond the gate. Inward set is
            // empty -> falls back to the gated set containing splat 0.
            const positions2 = new Float32Array([-0.4, 0, 0]);
            const normals = new Float32Array([1, 0, 0]);
            const result = colorizeVertices(positions2, normals, bvh, columns, voxelResolution);

            assertClose(result[0], srgbToLinear(color0[0]), 1e-4, 'red channel');
            assertClose(result[1], srgbToLinear(color0[1]), 1e-4, 'green channel');
            assertClose(result[2], srgbToLinear(color0[2]), 1e-4, 'blue channel');
        });
    });
});

// build a BVH + color columns from plain-object splats:
// { center, extent, color, logit }
const makeSplatFixture = (splats) => {
    const arr = f => new Float32Array(splats.map(f));
    const dataTable = new DataTable([
        new Column('x', arr(s => s.center[0])),
        new Column('y', arr(s => s.center[1])),
        new Column('z', arr(s => s.center[2])),
        new Column('f_dc_0', arr(s => packClr(s.color[0]))),
        new Column('f_dc_1', arr(s => packClr(s.color[1]))),
        new Column('f_dc_2', arr(s => packClr(s.color[2]))),
        new Column('opacity', arr(s => s.logit))
    ]);
    const extents = new DataTable([
        new Column('extent_x', arr(s => s.extent)),
        new Column('extent_y', arr(s => s.extent)),
        new Column('extent_z', arr(s => s.extent))
    ]);
    const columns = {};
    for (const name of ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity']) {
        columns[name] = dataTable.getColumnByName(name).data;
    }
    return { bvh: new GaussianBVH(dataTable, extents), columns };
};

describe('colorizeVertices solid mode', () => {
    const voxelResolution = 0.5;

    // Three splats along +x near the vertex at (0.2,0,0), all within the 0.75
    // gate (distances 0.2, 0, 0.2), extents 0.5 so their AABBs overlap the
    // 2x query box. Zero normals disable the inward filter.
    //
    //   splat 0: red   (0.9, 0.1, 0.2), logit 0 -> w0 = 0.5
    //   splat 1: green (0.1, 0.9, 0.3), logit 2 -> w1 = sigmoid(2)
    //   splat 2: blue  (0.5, 0.3, 0.8), logit 1 -> w2 = sigmoid(1)
    const splats = [
        { center: [0, 0, 0], extent: 0.5, color: [0.9, 0.1, 0.2], logit: 0 },
        { center: [0.2, 0, 0], extent: 0.5, color: [0.1, 0.9, 0.3], logit: 2 },
        { center: [0.4, 0, 0], extent: 0.5, color: [0.5, 0.3, 0.8], logit: 1 }
    ];
    const positions = new Float32Array([0.2, 0, 0]);

    it('picks the per-channel opacity-weighted median, not the mean', () => {
        const { bvh, columns } = makeSplatFixture(splats);
        const result = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution, 'solid');

        const w0 = sigmoid(0);
        const w1 = sigmoid(2);
        const w2 = sigmoid(1);
        const total = w0 + w1 + w2;
        const half = total / 2;

        // hand-computed per channel (sort values ascending, accumulate weights,
        // take the first value reaching >= half the total):
        //   red values:   0.1 (w1), 0.5 (w2), 0.9 (w0)
        //     cum: w1 = 0.881 < half = 1.056; w1+w2 = 1.612 >= half -> 0.5
        //   green values: 0.1 (w0), 0.3 (w2), 0.9 (w1)
        //     cum: w0 = 0.5 < half; w0+w2 = 1.231 >= half -> 0.3
        //   blue values:  0.2 (w0), 0.3 (w1), 0.8 (w2)
        //     cum: w0 = 0.5 < half; w0+w1 = 1.381 >= half -> 0.3
        assert(half > w1 - 1e-9, 'test arithmetic sanity: half exceeds first cumulant for red');

        assertClose(result[0], srgbToLinear(0.5), 1e-4, 'red channel median');
        assertClose(result[1], srgbToLinear(0.3), 1e-4, 'green channel median');
        assertClose(result[2], srgbToLinear(0.3), 1e-4, 'blue channel median');

        // the weighted mean differs clearly on the red channel:
        // (0.9*w0 + 0.1*w1 + 0.5*w2) / total ~= 0.4278
        const meanR = (0.9 * w0 + 0.1 * w1 + 0.5 * w2) / total;
        assert(Math.abs(meanR - 0.5) > 0.05, 'mean and median must diverge on red channel');
    });

    it('uses the higher-value candidate when its weight passes half the total', () => {
        // Two candidates, unequal opacities. Red channel values 0.2 (logit 0,
        // w = 0.5) and 0.8 (logit 2, w = 0.881): sorted ascending, the first
        // cumulant 0.5 < half = 0.6905, so the median is the SECOND value 0.8
        // even though sorting starts at 0.2.
        const twoSplats = [
            { center: [0, 0, 0], extent: 0.5, color: [0.2, 0.5, 0.5], logit: 0 },
            { center: [0.2, 0, 0], extent: 0.5, color: [0.8, 0.5, 0.5], logit: 2 }
        ];
        const { bvh, columns } = makeSplatFixture(twoSplats);
        const result = colorizeVertices(new Float32Array([0.1, 0, 0]), noNormals(3), bvh, columns, voxelResolution, 'solid');

        assertClose(result[0], srgbToLinear(0.8), 1e-4, 'red channel median');
        // green/blue are uniform 0.5 across both splats
        assertClose(result[1], srgbToLinear(0.5), 1e-4, 'green channel');
        assertClose(result[2], srgbToLinear(0.5), 1e-4, 'blue channel');
    });

    it('snaps to the majority color instead of blending unlike colors', () => {
        // one bright red splat vs two dim blue splats of higher total opacity:
        // the average blends toward purple, the median picks one side
        const mixed = [
            { center: [0, 0, 0], extent: 0.5, color: [0.9, 0.1, 0.1], logit: 0 },
            { center: [0.2, 0, 0], extent: 0.5, color: [0.1, 0.1, 0.9], logit: 1 },
            { center: [0.4, 0, 0], extent: 0.5, color: [0.1, 0.1, 0.9], logit: 1 }
        ];
        const { bvh, columns } = makeSplatFixture(mixed);

        const average = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution, 'average');
        const solid = colorizeVertices(positions, noNormals(3), bvh, columns, voxelResolution, 'solid');

        // red channel values: 0.1 (w1), 0.1 (w2), 0.9 (w0); total = 0.5 + 2*sigmoid(1);
        // half = 0.981; cum: sigmoid(1) = 0.731 < half; 2*sigmoid(1) = 1.462 >= half -> 0.1
        assertClose(solid[0], srgbToLinear(0.1), 1e-4, 'solid red snaps to majority');
        // average red = (0.9*0.5 + 0.1*2*sigmoid(1)) / total ~= 0.314
        assert(srgbToLinear(0.1) < average[0] - 0.02, 'average must blend toward red');
    });
});
