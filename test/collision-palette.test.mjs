import assert from 'node:assert';
import { describe, it } from 'node:test';

import { palettizeColors, smoothVertexColors } from '../src/lib/mesh/index.js';

import { assertClose } from './helpers/summary-compare.mjs';

const linearToSrgb = c => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const srgbToLinear = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

// Build a linear-space colour array by repeating each sRGB group colour.
const buildColors = (groups) => {
    const out = [];
    for (const { rgb, count } of groups) {
        for (let i = 0; i < count; i++) {
            out.push(srgbToLinear(rgb[0]), srgbToLinear(rgb[1]), srgbToLinear(rgb[2]));
        }
    }
    return new Float32Array(out);
};

// Lay vertices out along X, one unit apart, so groups form contiguous runs.
const buildLinePositions = (vertexCount) => {
    const p = new Float32Array(vertexCount * 3);
    for (let i = 0; i < vertexCount; i++) p[i * 3] = i;
    return p;
};

const srgbAt = (colors, index) => [
    linearToSrgb(colors[index * 3]),
    linearToSrgb(colors[index * 3 + 1]),
    linearToSrgb(colors[index * 3 + 2])
];

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

const distinctSrgb = (colors) => {
    const seen = new Map();
    for (let i = 0; i < colors.length / 3; i++) {
        const c = srgbAt(colors, i);
        seen.set(c.map(v => v.toFixed(5)).join(','), c);
    }
    return [...seen.values()];
};

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

    it('keeps rare accent hues when a dominant colour outnumbers them', () => {
        const accents = {
            red: [0.85, 0.15, 0.15],
            green: [0.15, 0.75, 0.20],
            blue: [0.15, 0.20, 0.80]
        };
        const colors = buildColors([
            { rgb: [0.55, 0.42, 0.28], count: 300 },
            { rgb: accents.red, count: 20 },
            { rgb: accents.green, count: 20 },
            { rgb: accents.blue, count: 20 }
        ]);

        const result = palettizeColors(colors, 4);
        const palette = distinctSrgb(result);
        const brown = srgbAt(result, 0);

        for (const [name, rgb] of Object.entries(accents)) {
            const nearest = palette.reduce((a, b) => (dist(b, rgb) < dist(a, rgb) ? b : a));
            assert(dist(nearest, rgb) < 0.1, `${name} accent must survive, got ${nearest}`);
            assert(dist(nearest, brown) > 0.15, `${name} accent must stay distinct from the dominant colour`);
        }
    });

    it('does not spend palette slots on scattered single-vertex noise', () => {
        // Clamped reconstruction noise lands on RGB cube corners; each corner is
        // maximally distant from real colours, so ungated farthest-first seeding
        // hands it the whole palette.
        const corners = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]];
        const groups = [
            { rgb: [0.55, 0.42, 0.28], count: 200 },
            { rgb: [0.20, 0.62, 0.30], count: 40 }
        ];
        for (const rgb of corners) groups.push({ rgb, count: 1 });

        const colors = buildColors(groups);
        const positions = buildLinePositions(colors.length / 3);
        const result = palettizeColors(colors, 4, { positions, voxelResolution: 1 });

        for (const entry of distinctSrgb(result)) {
            for (const corner of corners) {
                assert(dist(entry, corner) > 0.2, `palette entry ${entry} sits on noise corner ${corner}`);
            }
        }

        // the real minority colour still earns a slot
        const green = [0.20, 0.62, 0.30];
        const nearestGreen = distinctSrgb(result).reduce((a, b) => (dist(b, green) < dist(a, green) ? b : a));
        assert(dist(nearestGreen, green) < 0.1, 'a genuine minority colour must keep its slot');
    });

    it('assigns every vertex its nearest palette entry', () => {
        const colors = buildColors([
            { rgb: [0.80, 0.20, 0.20], count: 30 },
            { rgb: [0.20, 0.70, 0.25], count: 25 },
            { rgb: [0.25, 0.25, 0.85], count: 20 },
            { rgb: [0.90, 0.88, 0.80], count: 15 }
        ]);

        const result = palettizeColors(colors, 4);
        const palette = distinctSrgb(result);

        for (let v = 0; v < colors.length / 3; v++) {
            const input = srgbAt(colors, v);
            const assigned = srgbAt(result, v);
            const nearest = palette.reduce((a, b) => (dist(b, input) < dist(a, input) ? b : a));
            assertClose(dist(assigned, input), dist(nearest, input), 1e-6, `vertex ${v} must take its nearest entry`);
        }
    });

    it('absorbs an isolated speckle voxel when coherentRadius is set', () => {
        // one stray green voxel in the middle of a red run
        const groups = [
            { rgb: [0.80, 0.20, 0.20], count: 20 },
            { rgb: [0.20, 0.75, 0.25], count: 1 },
            { rgb: [0.80, 0.20, 0.20], count: 20 }
        ];
        const colors = buildColors(groups);
        const positions = buildLinePositions(colors.length / 3);
        const opts = { positions, voxelResolution: 1 };

        const plain = palettizeColors(colors, 2, opts);
        const filtered = palettizeColors(colors, 2, { ...opts, coherentRadius: 1 });

        assert(distinctSrgb(plain).length > 1, 'without the filter the speckle keeps its own colour');
        assert.strictEqual(distinctSrgb(filtered).length, 1, 'majority filter must absorb the lone speckle');
    });

    it('smoothVertexColors pulls an isolated voxel towards its neighbours', () => {
        const colors = buildColors([
            { rgb: [0.80, 0.20, 0.20], count: 5 },
            { rgb: [0.20, 0.75, 0.25], count: 1 },
            { rgb: [0.80, 0.20, 0.20], count: 5 }
        ]);
        const positions = buildLinePositions(colors.length / 3);

        const smoothed = smoothVertexColors(colors, positions, 1, 1);
        const red = [0.80, 0.20, 0.20];
        const before = dist(srgbAt(colors, 5), red);
        const after = dist(srgbAt(smoothed, 5), red);

        assert(after < before, `smoothing must move the outlier towards red (${after} < ${before})`);
    });

    it('is a no-op relative to omitting opts when no spatial option is set', () => {
        const colors = buildColors([
            { rgb: [0.70, 0.30, 0.20], count: 12 },
            { rgb: [0.20, 0.40, 0.80], count: 12 }
        ]);
        const positions = buildLinePositions(colors.length / 3);

        const withoutOpts = palettizeColors(colors, 3);
        const withPositions = palettizeColors(colors, 3, { positions, voxelResolution: 1 });

        for (let i = 0; i < withoutOpts.length; i++) {
            assert.strictEqual(withPositions[i], withoutOpts[i], `channel ${i} must match`);
        }
    });

    it('handles k larger than the number of distinct colours', () => {
        const colors = buildColors([{ rgb: [0.4, 0.5, 0.6], count: 10 }]);
        const result = palettizeColors(colors, 5);

        assert.strictEqual(result.length, colors.length);
        assert.strictEqual(distinctSrgb(result).length, 1, 'a single input colour yields a single entry');
    });
});