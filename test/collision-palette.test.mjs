import assert from 'node:assert';
import { describe, it } from 'node:test';

import { palettizeColors } from '../src/lib/mesh/index.js';

import { assertClose } from './helpers/summary-compare.mjs';

const linearToSrgb = c => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const srgbToLinear = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

describe('palettizeColors', () => {
    it('returns the input unchanged when k >= vertexCount', () => {
        const colors = new Float32Array([srgbToLinear(0.2), srgbToLinear(0.5), srgbToLinear(0.8)]);
        const result = palettizeColors(colors, 1);
        assert.strictEqual(result.length, 3);
        assertClose(result[0], colors[0], 1e-6, 'k=1 should equal input when vertexCount=1');
    });

    it('snaps 4 vertices to 2 clusters', () => {
        // two tight groups of red-ish and blue-ish vertices
        const red = [0.9, 0.1, 0.1];
        const blue = [0.1, 0.1, 0.8];
        const colors = new Float32Array([
            srgbToLinear(red[0]), srgbToLinear(red[1]), srgbToLinear(red[2]),
            srgbToLinear(red[0] + 0.01), srgbToLinear(red[1] + 0.01), srgbToLinear(red[2]),
            srgbToLinear(blue[0]), srgbToLinear(blue[1]), srgbToLinear(blue[2]),
            srgbToLinear(blue[0] + 0.01), srgbToLinear(blue[1]), srgbToLinear(blue[2])
        ]);

        const result = palettizeColors(colors, 2);

        // each group should snap to a single centroid
        for (let c = 0; c < 3; c++) {
            assertClose(result[c], result[c + 3], 1e-4, `red group channel ${c}`);
            assertClose(result[c + 6], result[c + 9], 1e-4, `blue group channel ${c}`);
        }
        // the two groups must differ
        assert(Math.abs(result[0] - result[6]) > 0.05, 'red and blue clusters must diverge');
    });

    it('is deterministic across two runs', () => {
        const colors = new Float32Array(12);
        for (let i = 0; i < 12; i++) colors[i] = srgbToLinear(Math.random());
        // deterministic input: constant values
        const a = palettizeColors(new Float32Array([srgbToLinear(0.8), srgbToLinear(0.2), srgbToLinear(0.5)]), 2);
        const b = palettizeColors(new Float32Array([srgbToLinear(0.8), srgbToLinear(0.2), srgbToLinear(0.5)]), 2);
        for (let i = 0; i < 3; i++) assert.strictEqual(a[i], b[i], 'deterministic output');
    });

    it('reduces all vertices to a single colour with k=1', () => {
        const colors = new Float32Array([
            srgbToLinear(0.8), srgbToLinear(0.2), srgbToLinear(0.5),
            srgbToLinear(0.1), srgbToLinear(0.9), srgbToLinear(0.3)
        ]);
        const result = palettizeColors(colors, 1);

        // all vertices snap to the same global-mean centroid; each channel
        // must be identical across the two vertices
        for (let c = 0; c < 3; c++) {
            assertClose(result[c], result[3 + c], 1e-12, `channel ${c} across vertices`);
        }
    });

    it('does not alter a vertex with k >= vertexCount', () => {
        const colors = new Float32Array([srgbToLinear(0.3), srgbToLinear(0.6), srgbToLinear(0.9)]);
        const result = palettizeColors(colors, 5);
        for (let i = 0; i < 3; i++) assert.strictEqual(result[i], colors[i], 'k=5 >= 1 vertex');
    });
});