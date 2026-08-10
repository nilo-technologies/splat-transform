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
        // out-of-grid (counted empty) and see only 12 of 27 -- those DO erode.
        // Interior voxels, well away from the boundary, must survive untouched.
        const res = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        for (let y = 5; y <= 10; y++) {
            assert.strictEqual(res.grid.getVoxel(8, y, 8), 1, `interior y=${y}`);
        }
        assert.strictEqual(res.grid.getVoxel(8, 11, 8), 1, 'top surface centre');
        assert.strictEqual(res.grid.getVoxel(8, 4, 8), 1, 'bottom surface centre');
        assert.strictEqual(res.added, 0, 'a solid slab needs no additions');
    });

    it('erodes a slab edge that borders out-of-grid', function () {
        // Documents the boundary convention that the previous test works
        // around. A single grid edge alone isn't enough here: (0,8,8) still
        // sees 18 of 27 (only the X face is out-of-grid; Y and Z are both
        // comfortably interior), which stays above threshold. (0,8,0) sits at
        // a true grid CORNER -- X and Z both border out-of-grid, Y stays
        // interior to the slab -- giving (2/3)*(2/3)*1*27 = 12, below 14.
        const res = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(0, 8, 0), 0,
            'x=0,z=0 corner sees 2/3 of X and 2/3 of Z slices: 12 of 27');
        assert.ok(res.removed > 0);
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

    it('leaves the candidate grid untouched', function () {
        const cand = allCandidate(16);
        const before = [...cand.types];
        majorityFilterGrid(slab(16, 6, 9), cand, { threshold: 14, iterations: 1 });
        assert.deepStrictEqual([...cand.types], before);
    });

    it('treats out-of-grid as empty', function () {
        // A slab touching y=0: the bottom layer's below-neighbours are outside
        // the grid and must count as empty, so a thin slab at the edge erodes.
        const g = new SparseVoxelGrid(16, 16, 16);
        for (let z = 0; z < 16; z++) {
            for (let x = 0; x < 16; x++) g.setVoxel(x, 0, z);
        }
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 0, 8), 0,
            'a 1-thick sheet at the grid floor has at most 9 of 27');
    });
});
