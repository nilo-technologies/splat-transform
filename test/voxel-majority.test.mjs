/**
 * Tests for the 3x3x3 majority filter.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { majorityFilterGrid } from '../src/lib/voxel/majority.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

const allCandidate = (n) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = 0; z < n; z++) {
        for (let y = 0; y < n; y++) {
            for (let x = 0; x < n; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

const countVoxels = (g) => {
    let n = 0;
    g.forEachOccupiedVoxel(() => n++);
    return n;
};

// A solid slab spanning y in [y0, y1] across the whole grid interior.
const slab = (n, y0, y1) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = 0; z < n; z++) {
        for (let y = y0; y <= y1; y++) {
            for (let x = 0; x < n; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

// A 1-voxel-thick sheet at height y, spanning [x0,x1] x [z0,z1].
const sheet = (n, y, x0, x1, z0, z1) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = z0; z <= z1; z++) {
        for (let x = x0; x <= x1; x++) g.setVoxel(x, y, z);
    }
    return g;
};

// A solid axis-aligned box, inclusive bounds.
const box = (n, x0, x1, y0, y1, z0, z1) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = z0; z <= z1; z++) {
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

describe('majorityFilterGrid', function () {
    it('removes an isolated voxel', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0);
        assert.strictEqual(res.removed, 1);
    });

    it('removes a 1-voxel bump on a slab', function () {
        const g = slab(16, 6, 9);
        g.setVoxel(8, 10, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 10, 8), 0, 'bump should go');
        assert.strictEqual(res.grid.getVoxel(8, 9, 8), 1, 'slab top should stay');
    });

    it('fills a 1-voxel dent in a slab', function () {
        const g = slab(16, 6, 9);
        g.clearVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 1, 'interior dent should fill');
    });

    it('preserves the interior of a thick slab', function () {
        // The slab spans the full x/z extent, so its edge columns border
        // out-of-grid (counted empty) and see only 12 of 27. They survive
        // anyway: the sheet-aware gate spares any voxel with >= 3 occupied face
        // neighbours, which 'keeps a slab edge that borders out-of-grid' pins
        // directly. Interior voxels are untouched under either rule.
        const res = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        for (let y = 5; y <= 10; y++) {
            assert.strictEqual(res.grid.getVoxel(8, y, 8), 1, `interior y=${y}`);
        }
        assert.strictEqual(res.grid.getVoxel(8, 11, 8), 1, 'top surface centre');
        assert.strictEqual(res.grid.getVoxel(8, 4, 8), 1, 'bottom surface centre');
        assert.strictEqual(res.added, 0, 'a solid slab needs no additions');
    });

    it('keeps a slab edge that borders out-of-grid', function () {
        // Inverted deliberately: this used to assert 'erodes a slab edge that
        // borders out-of-grid'. The density count at the grid corner (0,8,0) is
        // still 12 of 27 -- (2/3)*(2/3)*1*27, since X and Z each lose a third to
        // out-of-grid while Y stays interior to the slab -- but the voxel has 4
        // occupied face neighbours (+x, +z, and both Y), so the sheet-aware gate
        // spares it. No voxel of a solid slab has fewer than 3: the minimum sits
        // at the bottom grid corner (0,4,0), with +x, +z and +y.
        const res = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(0, 8, 0), 1, 'the edge must survive');
        assert.strictEqual(res.removed, 0, 'a solid slab has nothing removable');
        assert.ok(res.kept >= 1, 'the spared under-threshold voxels must be audited');
    });

    it('gates additions by the candidate mask', function () {
        const g = slab(16, 6, 9);
        g.clearVoxel(8, 8, 8);
        const empty = new SparseVoxelGrid(16, 16, 16);
        const res = majorityFilterGrid(g, empty, { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0, 'gate must block the fill');
        assert.strictEqual(res.added, 0);
        assert.ok(res.gateRejected >= 1, 'the blocked dent must be counted');
    });

    it('does not gate removals by the candidate mask', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const empty = new SparseVoxelGrid(16, 16, 16);
        const res = majorityFilterGrid(g, empty, { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0, 'removal needs no candidate');
        assert.strictEqual(res.removed, 1);
    });

    it('produces the same result whatever the chunk size', function () {
        const build = () => {
            const g = new SparseVoxelGrid(32, 32, 32);
            let seed = 999;
            const rnd = () => {
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
                return seed / 0x7FFFFFFF;
            };
            for (let z = 0; z < 32; z++) {
                for (let y = 0; y < 32; y++) {
                    for (let x = 0; x < 32; x++) {
                        if (rnd() < 0.4) g.setVoxel(x, y, z);
                    }
                }
            }
            return g;
        };
        const big = majorityFilterGrid(build(), allCandidate(32),
            { threshold: 14, iterations: 2, chunkInner: 256 });
        const small = majorityFilterGrid(build(), allCandidate(32),
            { threshold: 14, iterations: 2, chunkInner: 8 });

        const dump = (g) => {
            const out = [];
            g.forEachOccupiedVoxel((x, y, z) => out.push(`${x},${y},${z}`));
            return out.sort();
        };
        assert.deepStrictEqual(dump(small.grid), dump(big.grid),
            'chunked result must equal single-chunk result');
        assert.strictEqual(small.added, big.added);
        assert.strictEqual(small.removed, big.removed);
        assert.strictEqual(small.kept, big.kept,
            'the face taps must read the halo, not the chunk edge');
    });

    it('iterates: two passes differ from one on a noisy volume', function () {
        // A single-Y-layer sheet cannot show this: with the Y+-1 layers always
        // empty, no position can ever exceed 6 of 27 (a periodic 1/3 removal
        // pattern caps the in-plane 3x3 window at 6), so any threshold above
        // 6 collapses everything to nothing in exactly one pass, and a second
        // pass on an already-empty grid can never differ. A genuinely 3D
        // noisy block has room in Y too, so the real production threshold
        // (14) shows real progressive smoothing.
        const build = () => {
            const g = new SparseVoxelGrid(24, 24, 24);
            let seed = 42;
            const rnd = () => {
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
                return seed / 0x7FFFFFFF;
            };
            for (let z = 4; z < 20; z++) {
                for (let y = 4; y < 20; y++) {
                    for (let x = 4; x < 20; x++) {
                        if (rnd() < 0.6) g.setVoxel(x, y, z);
                    }
                }
            }
            return g;
        };
        const one = majorityFilterGrid(build(), allCandidate(24),
            { threshold: 14, iterations: 1 });
        const two = majorityFilterGrid(build(), allCandidate(24),
            { threshold: 14, iterations: 2 });
        assert.notStrictEqual(countVoxels(one.grid), countVoxels(two.grid));
    });

    it('keeps an interior 1-thick sheet, bevelling only its corners', function () {
        // A 10x10 sheet well inside the grid: 100 voxels, of which the 4 convex
        // corners have 2 face neighbours and go. The bevel then advances one
        // diagonal per pass, because each removal exposes two new 2-neighbour
        // voxels, so the loss per corner is 1 after one pass and 3 after two.
        const one = majorityFilterGrid(sheet(16, 8, 3, 12, 3, 12), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(countVoxels(one.grid), 96, '100 less 1 voxel per corner');
        for (const [x, z] of [[3, 3], [12, 3], [3, 12], [12, 12]]) {
            assert.strictEqual(one.grid.getVoxel(x, 8, z), 0, `corner ${x},${z}`);
        }
        assert.strictEqual(one.grid.getVoxel(3, 8, 8), 1, 'straight edge stays');
        assert.strictEqual(one.grid.getVoxel(8, 8, 8), 1, 'interior stays');

        const two = majorityFilterGrid(sheet(16, 8, 3, 12, 3, 12), allCandidate(16),
            { threshold: 14, iterations: 2 });
        assert.strictEqual(countVoxels(two.grid), 88, '3 voxels per corner after two passes');
        assert.strictEqual(two.grid.getVoxel(3, 8, 8), 1, 'straight edges never erode');
    });

    it('preserves a solid interior cube exactly', function () {
        // Today's rule rounds this to 136 over two passes: a convex edge sees 12
        // of 27 and a corner 8, both under threshold. With the gate, an edge has
        // 4 face neighbours and a corner 3, so the cube is untouched.
        const res = majorityFilterGrid(box(16, 5, 10, 5, 10, 5, 10), allCandidate(16),
            { threshold: 14, iterations: 2 });
        assert.strictEqual(countVoxels(res.grid), 216);
        assert.strictEqual(res.removed, 0);
        assert.strictEqual(res.grid.getVoxel(5, 5, 5), 1, 'corner');
        assert.strictEqual(res.grid.getVoxel(5, 8, 5), 1, 'edge');
    });

    it('audits the voxels it spared', function () {
        // `kept` accumulates over passes, like added and removed: the cube
        // presents the same 8 corners and 48 edge voxels to every pass.
        const one = majorityFilterGrid(box(16, 5, 10, 5, 10, 5, 10), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(one.kept, 56, '8 corners + 48 edge voxels');
        const two = majorityFilterGrid(box(16, 5, 10, 5, 10, 5, 10), allCandidate(16),
            { threshold: 14, iterations: 2 });
        assert.strictEqual(two.kept, 112, 'per-pass counter, summed over 2 passes');

        const slabRes = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(slabRes.kept, 144, 'the full-extent slab edge columns');
    });

    it('reports nothing kept when there is nothing to spare', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.kept, 0, '0 face neighbours is not a surface');
        assert.strictEqual(res.removed, 1);
    });

    it('keeps a 2x2 bump patch on a slab: the documented tradeoff', function () {
        // Each of the four voxels has 2 in-plane neighbours plus the slab below
        // = 3, so the patch survives where a single-voxel bump does not. This is
        // the price of keeping 1-thick sheets and it is intended: do not "fix"
        // it without re-reading the design spec's tradeoffs section.
        const g = slab(16, 6, 9);
        for (const [x, z] of [[6, 6], [7, 6], [6, 7], [7, 7]]) g.setVoxel(x, 10, z);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 2 });
        for (const [x, z] of [[6, 6], [7, 6], [6, 7], [7, 7]]) {
            assert.strictEqual(res.grid.getVoxel(x, 10, z), 1, `patch voxel ${x},${z}`);
        }
    });

    it('still shaves a stick off a slab', function () {
        // A 1x1x3 stick: the tip has 1 face neighbour and the shaft 2, so the
        // whole thing goes in one pass. Poles and railings are not protected.
        const g = slab(16, 6, 9);
        for (let y = 10; y <= 12; y++) g.setVoxel(6, y, 6);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        for (let y = 10; y <= 12; y++) {
            assert.strictEqual(res.grid.getVoxel(6, y, 6), 0, `stick voxel y=${y}`);
        }
        assert.strictEqual(res.grid.getVoxel(6, 9, 6), 1, 'the slab top stays');
    });

    it('treats out-of-grid as empty when adding', function () {
        // The convention the old erosion pin carried, moved to the addition
        // path, which this change does not touch. A dent at the grid corner sees
        // 11 of 27 -- two axes lose a third to out-of-grid and the dent itself
        // is empty -- so it stays open, while an interior dent at 26 fills.
        const g = slab(16, 4, 11);
        g.clearVoxel(0, 8, 0);
        g.clearVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(0, 8, 0), 0, 'grid-corner dent stays open');
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 1, 'interior dent fills');
        assert.strictEqual(res.added, 1, 'exactly the interior dent');
    });

    it('leaves the candidate grid untouched', function () {
        const cand = allCandidate(16);
        const before = [...cand.types];
        majorityFilterGrid(slab(16, 6, 9), cand, { threshold: 14, iterations: 1 });
        assert.deepStrictEqual([...cand.types], before);
    });

    it('keeps a 1-thick sheet at the grid floor', function () {
        // Inverted deliberately: this used to assert 'treats out-of-grid as
        // empty' by requiring the sheet be erased. The convention is unchanged
        // -- the below-neighbours are outside the grid and count as empty, so
        // the density count peaks at 9 of 27 -- but every interior voxel has 4
        // in-plane face neighbours and survives. Only the sheet's own 4 convex
        // corners, at 2 face neighbours, go. The convention itself is now pinned
        // through the addition path by 'treats out-of-grid as empty when adding'.
        const g = new SparseVoxelGrid(16, 16, 16);
        for (let z = 0; z < 16; z++) {
            for (let x = 0; x < 16; x++) g.setVoxel(x, 0, z);
        }
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 0, 8), 1, 'the surface must survive');
        assert.strictEqual(countVoxels(res.grid), 252, '256 less the 4 corners');
        assert.strictEqual(res.removed, 4);
        for (const [x, z] of [[0, 0], [15, 0], [0, 15], [15, 15]]) {
            assert.strictEqual(res.grid.getVoxel(x, 0, z), 0, `corner ${x},${z} has 2 faces`);
        }
    });
});
