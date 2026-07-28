import assert from 'node:assert';
import { describe, it } from 'node:test';

import { mapToPalette, palettizeColors, parsePaletteColors, smoothVertexColors } from '../src/lib/mesh/index.js';

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

    it('smoothVertexColors keeps a boundary between two solid regions crisp', () => {
        // 12 red then 12 blue. A plain neighbourhood mean bleeds each colour
        // several vertices into the other; the range test rejects the far side,
        // so both regions stay exactly their own colour right up to the seam.
        const red = [0.80, 0.20, 0.20];
        const blue = [0.20, 0.30, 0.80];
        const colors = buildColors([{ rgb: red, count: 12 }, { rgb: blue, count: 12 }]);
        const positions = buildLinePositions(colors.length / 3);

        const smoothed = smoothVertexColors(colors, positions, 2, 1);

        for (const [index, truth] of [[11, red], [12, blue]]) {
            const drift = dist(srgbAt(smoothed, index), truth);
            assert(drift < 0.02, `vertex ${index} at the seam drifted by ${drift}`);
        }
    });

    it('smoothVertexColors collapses noise inside a region towards one colour', () => {
        // A single material sampled with per-vertex jitter, as reconstruction
        // noise produces. The filter should converge the run on its mean.
        const base = [0.45, 0.55, 0.35];
        const groups = [];
        for (let i = 0; i < 40; i++) {
            const jitter = ((i * 7919) % 17) / 17 * 0.08 - 0.04;
            groups.push({ rgb: base.map(c => c + jitter), count: 1 });
        }
        const colors = buildColors(groups);
        const positions = buildLinePositions(colors.length / 3);

        const spread = (c) => {
            let lo = Infinity;
            let hi = -Infinity;
            // interior only, so the shrinking neighbourhood at the ends does
            // not dominate the measurement
            for (let v = 4; v < c.length / 3 - 4; v++) {
                const l = srgbAt(c, v)[1];
                lo = Math.min(lo, l);
                hi = Math.max(hi, l);
            }
            return hi - lo;
        };

        const smoothed = smoothVertexColors(colors, positions, 2, 1);
        assert(spread(smoothed) < spread(colors) / 4,
            `noise spread must shrink markedly (${spread(smoothed)} vs ${spread(colors)})`);
    });

    it('smoothVertexColors leaves an already-uniform region untouched', () => {
        const rgb = [0.5, 0.4, 0.3];
        const colors = buildColors([{ rgb, count: 20 }]);
        const positions = buildLinePositions(colors.length / 3);

        const smoothed = smoothVertexColors(colors, positions, 2, 1);
        for (let i = 0; i < colors.length; i++) {
            assertClose(smoothed[i], colors[i], 1e-5, `channel ${i} must be unchanged`);
        }
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

describe('parsePaletteColors', () => {
    it('converts hex to linear with and without a leading hash', () => {
        const parsed = parsePaletteColors(['#3243aa', '4444ff']);
        assert.strictEqual(parsed.length, 6);

        const expected = [0x32, 0x43, 0xaa, 0x44, 0x44, 0xff].map(b => srgbToLinear(b / 255));
        for (let i = 0; i < 6; i++) {
            assertClose(parsed[i], expected[i], 1e-6, `channel ${i}`);
        }
    });

    it('expands #rgb shorthand and tolerates surrounding whitespace', () => {
        const short = parsePaletteColors([' #f0a ']);
        const long = parsePaletteColors(['#ff00aa']);
        for (let i = 0; i < 3; i++) {
            assert.strictEqual(short[i], long[i], `channel ${i} must match the long form`);
        }
    });

    it('is case insensitive', () => {
        const upper = parsePaletteColors(['#3243AA']);
        const lower = parsePaletteColors(['#3243aa']);
        for (let i = 0; i < 3; i++) assert.strictEqual(upper[i], lower[i], `channel ${i}`);
    });

    it('rejects malformed colours and an empty list', () => {
        for (const spec of ['#12345', 'nothex', '#gggggg', '#', '0x3243aa']) {
            assert.throws(() => parsePaletteColors([spec]), /Invalid palette colour/, `must reject ${spec}`);
        }
        assert.throws(() => parsePaletteColors([]), /empty/);
    });
});

describe('mapToPalette', () => {
    const palette = parsePaletteColors(['#3243aa', '4444ff']);

    it('emits only the given colours, exactly', () => {
        const colors = buildColors([
            { rgb: [0.80, 0.20, 0.20], count: 10 },
            { rgb: [0.20, 0.25, 0.90], count: 10 },
            { rgb: [0.50, 0.50, 0.50], count: 10 }
        ]);

        const result = mapToPalette(colors, palette);
        assert.strictEqual(result.length, colors.length);

        const entries = new Set();
        for (let e = 0; e < palette.length / 3; e++) {
            entries.add(`${palette[e * 3]},${palette[e * 3 + 1]},${palette[e * 3 + 2]}`);
        }
        for (let v = 0; v < result.length / 3; v++) {
            const key = `${result[v * 3]},${result[v * 3 + 1]},${result[v * 3 + 2]}`;
            assert.ok(entries.has(key), `vertex ${v} colour ${key} is not a palette entry`);
        }
    });

    it('gives every vertex its nearest entry', () => {
        // one vertex per palette colour, plus a mid-grey that must pick one
        const colors = new Float32Array([
            palette[0], palette[1], palette[2],
            palette[3], palette[4], palette[5],
            srgbToLinear(0.5), srgbToLinear(0.5), srgbToLinear(0.5)
        ]);

        const result = mapToPalette(colors, palette);

        for (let c = 0; c < 3; c++) {
            assertClose(result[c], palette[c], 1e-6, `first entry channel ${c} must be untouched`);
            assertClose(result[3 + c], palette[3 + c], 1e-6, `second entry channel ${c} must be untouched`);
        }

        const grey = srgbAt(colors, 2);
        const assigned = srgbAt(result, 2);
        const candidates = [srgbAt(palette, 0), srgbAt(palette, 1)];
        const nearest = candidates.reduce((a, b) => (dist(b, grey) < dist(a, grey) ? b : a));
        assertClose(dist(assigned, grey), dist(nearest, grey), 1e-6, 'grey must take its nearest entry');
    });

    it('collapses to a single colour when given one entry', () => {
        const single = parsePaletteColors(['#3243aa']);
        const colors = buildColors([
            { rgb: [0.80, 0.20, 0.20], count: 4 },
            { rgb: [0.20, 0.25, 0.90], count: 4 }
        ]);

        const result = mapToPalette(colors, single);
        for (let v = 0; v < result.length / 3; v++) {
            for (let c = 0; c < 3; c++) {
                assert.strictEqual(result[v * 3 + c], single[c], `vertex ${v} channel ${c}`);
            }
        }
    });

    it('absorbs an isolated speckle voxel when coherentRadius is set', () => {
        // the run sits on '#3243aa' and the speckle on '#4444ff', so the two
        // start out on different palette entries
        const colors = buildColors([
            { rgb: [0.196, 0.263, 0.667], count: 20 },
            { rgb: [0.267, 0.267, 1.000], count: 1 },
            { rgb: [0.196, 0.263, 0.667], count: 20 }
        ]);
        const positions = buildLinePositions(colors.length / 3);
        const opts = { positions, voxelResolution: 1 };

        const plain = mapToPalette(colors, palette, opts);
        const filtered = mapToPalette(colors, palette, { ...opts, coherentRadius: 1 });

        assert.strictEqual(distinctSrgb(plain).length, 2, 'without the filter the speckle keeps its own entry');
        assert.strictEqual(distinctSrgb(filtered).length, 1, 'majority filter must absorb the lone speckle');
    });

    it('rejects an empty palette', () => {
        assert.throws(() => mapToPalette(buildColors([{ rgb: [0.5, 0.5, 0.5], count: 1 }]), new Float32Array(0)),
            /at least one palette colour/);
    });
});