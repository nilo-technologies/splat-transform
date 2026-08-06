import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Column, DataTable } from '../src/lib/index.js';
import { GaussianBVH } from '../src/lib/spatial/index.js';
import { colorizeVertices } from '../src/lib/mesh/colorize.js';
import { palettizeColors, mapToPalette } from '../src/lib/mesh/palette.js';
import { smoothVertexColors, majorityFilterIndices } from '../src/lib/mesh/color-spatial.js';

// Characterization tests. These pin the exact output of the spatial colour
// helpers so the data structures behind them (bin keys, spatial hash) can be
// swapped for cheaper ones without silently changing results. The expected
// digests were captured from the implementation they replaced.

/**
 * Deterministic pseudo-random generator, so the fixture never shifts.
 *
 * @param {number} seed - Initial state.
 * @returns {() => number} Generator yielding values in [0, 1).
 */
const rng = (seed) => () => {
    seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
    return seed / 0x100000000;
};

/**
 * A voxel-grid-like cloud of positions with several colour regions plus noise.
 *
 * @returns {{ positions: Float32Array, colors: Float32Array, voxelResolution: number }} Fixture.
 */
const makeFixture = () => {
    const next = rng(7);
    const n = 12 * 12 * 12;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const voxelResolution = 0.25;

    let v = 0;
    for (let z = 0; z < 12; z++) {
        for (let y = 0; y < 12; y++) {
            for (let x = 0; x < 12; x++) {
                positions[v * 3] = x * voxelResolution;
                positions[v * 3 + 1] = y * voxelResolution;
                positions[v * 3 + 2] = z * voxelResolution;

                // three spatial regions, each with its own base colour, plus
                // per-vertex noise so the coherence gate has something to do
                const region = x < 4 ? 0 : (x < 8 ? 1 : 2);
                const base = [[0.8, 0.2, 0.15], [0.15, 0.7, 0.25], [0.2, 0.25, 0.85]][region];
                for (let ch = 0; ch < 3; ch++) {
                    colors[v * 3 + ch] = Math.min(1, Math.max(0, base[ch] + (next() - 0.5) * 0.08));
                }
                v++;
            }
        }
    }
    return { positions, colors, voxelResolution };
};

/**
 * Compact, order-sensitive digest of a float array.
 *
 * @param {Float32Array} a - Array to digest.
 * @returns {string} Digest string.
 */
const digest = (a) => {
    let sum = 0;
    let weighted = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i];
        weighted += a[i] * ((i % 97) + 1);
    }
    return [
        a.length,
        sum.toFixed(6),
        weighted.toFixed(4),
        a[0].toFixed(6),
        a[(a.length >> 1) - ((a.length >> 1) % 3)].toFixed(6),
        a[a.length - 1].toFixed(6)
    ].join('|');
};

describe('colour spatial helpers are stable', () => {
    const { positions, colors, voxelResolution } = makeFixture();

    it('palettizeColors with the spatial coherence gate', () => {
        const out = palettizeColors(colors, 6, { positions, voxelResolution });
        assert.strictEqual(
            digest(out),
            '5184|2034.187827|99201.2972|0.801951|0.792196|0.837070');
    });

    it('palettizeColors with a coherence radius', () => {
        const out = palettizeColors(colors, 6, { positions, voxelResolution, coherentRadius: 2 });
        assert.strictEqual(
            digest(out),
            '5184|2029.290200|98965.9967|0.801951|0.792196|0.837070');
    });

    it('palettizeColors without positions falls back to a population gate', () => {
        const out = palettizeColors(colors, 6);
        assert.strictEqual(
            digest(out),
            '5184|2034.187827|99201.2972|0.801951|0.792196|0.837070');
    });

    it('smoothVertexColors', () => {
        const out = smoothVertexColors(colors, positions, 2, voxelResolution);
        assert.strictEqual(
            digest(out),
            '5184|2045.756095|99755.0628|0.803494|0.800199|0.842242');
    });

    it('mapToPalette with a coherence radius', () => {
        const palette = new Float32Array([
            0.8, 0.2, 0.15,
            0.15, 0.7, 0.25,
            0.2, 0.25, 0.85
        ]);
        const out = mapToPalette(colors, palette, { positions, voxelResolution, coherentRadius: 2 });
        assert.strictEqual(
            digest(out),
            '5184|2044.800024|99706.4512|0.800000|0.800000|0.850000');
    });

    it('majorityFilterIndices', () => {
        const next = rng(11);
        const indices = new Int32Array(positions.length / 3);
        for (let i = 0; i < indices.length; i++) indices[i] = (next() * 4) | 0;
        const out = majorityFilterIndices(indices, positions, 4, 2, voxelResolution);
        let sum = 0;
        for (let i = 0; i < out.length; i++) sum += out[i] * ((i % 97) + 1);
        assert.strictEqual(`${out.length}|${sum}`, '1728|122349');
    });
});

describe('colorizeVertices is stable', () => {
    // Guards the candidate-selection and weighted-median rewrite: the same
    // splats and vertices must keep producing byte-identical colours.
    const makeScene = () => {
        const next = rng(3);
        const n = 400;
        const arrays = {
            x: new Float32Array(n), y: new Float32Array(n), z: new Float32Array(n),
            f_dc_0: new Float32Array(n), f_dc_1: new Float32Array(n), f_dc_2: new Float32Array(n),
            opacity: new Float32Array(n)
        };
        const extentX = new Float32Array(n);
        const extentY = new Float32Array(n);
        const extentZ = new Float32Array(n);
        for (let i = 0; i < n; i++) {
            arrays.x[i] = next() * 4;
            arrays.y[i] = next() * 4;
            arrays.z[i] = next() * 4;
            arrays.f_dc_0[i] = (next() - 0.5) * 3;
            arrays.f_dc_1[i] = (next() - 0.5) * 3;
            arrays.f_dc_2[i] = (next() - 0.5) * 3;
            arrays.opacity[i] = (next() - 0.5) * 6;
            extentX[i] = 0.1 + next() * 0.5;
            extentY[i] = 0.1 + next() * 0.5;
            extentZ[i] = 0.1 + next() * 0.5;
        }
        const dataTable = new DataTable(Object.entries(arrays).map(([k, a]) => new Column(k, a)));
        const extents = new DataTable([
            new Column('extent_x', extentX),
            new Column('extent_y', extentY),
            new Column('extent_z', extentZ)
        ]);
        return {
            bvh: new GaussianBVH(dataTable, extents),
            columns: {
                f_dc_0: arrays.f_dc_0,
                f_dc_1: arrays.f_dc_1,
                f_dc_2: arrays.f_dc_2,
                opacity: arrays.opacity
            }
        };
    };

    const makeVertices = () => {
        const next = rng(5);
        const m = 900;
        const positions = new Float32Array(m * 3);
        const normals = new Float32Array(m * 3);
        for (let i = 0; i < m; i++) {
            positions[i * 3] = next() * 4;
            positions[i * 3 + 1] = next() * 4;
            positions[i * 3 + 2] = next() * 4;
            // a mix of unit normals and deliberate zero normals
            if (i % 7 === 0) continue;
            const nx = next() - 0.5, ny = next() - 0.5, nz = next() - 0.5;
            const len = Math.hypot(nx, ny, nz) || 1;
            normals[i * 3] = nx / len;
            normals[i * 3 + 1] = ny / len;
            normals[i * 3 + 2] = nz / len;
        }
        return { positions, normals };
    };

    const { bvh, columns } = makeScene();
    const { positions, normals } = makeVertices();

    it('average mode', () => {
        const out = colorizeVertices(positions, normals, bvh, columns, 0.25, 'average');
        assert.strictEqual(digest(out), '2700|697.406625|34506.2567|0.262473|0.649658|0.414040');
    });

    it('solid mode', () => {
        const out = colorizeVertices(positions, normals, bvh, columns, 0.25, 'solid');
        assert.strictEqual(digest(out), '2700|718.208567|35524.6220|0.299971|0.740151|0.440236');
    });
});
