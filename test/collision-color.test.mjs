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
});
