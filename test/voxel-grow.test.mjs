/**
 * Tests for density-gated region growing and its block-neighbour helper.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    NEIGHBOR_SCRATCH_LEN,
    sixNeighborMasks
} from '../src/lib/voxel/block-neighbors.js';
import { growGrid } from '../src/lib/voxel/grow.js';
import {
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from '../src/lib/voxel/sparse-voxel-grid.js';

// bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4)
const bit = (x, y, z) => {
    const i = (x & 3) + ((y & 3) << 2) + ((z & 3) << 4);
    return i < 32 ? [(1 << i) >>> 0, 0] : [0, (1 << (i - 32)) >>> 0];
};

const emptyReader = (bx, by, bz, out) => {
    out[0] = 0;
    out[1] = 0;
};

describe('sixNeighborMasks', function () {
    it('reports the in-block +X neighbour of a single voxel', function () {
        // voxel at (1,0,0) occupied; position (0,0,0) sees it in +X
        const [lo, hi] = bit(1, 0, 0);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [pLo] = bit(0, 0, 0);
        assert.strictEqual(out[0], pLo, '+X mask should mark position (0,0,0)');
        // The -X mask is NOT empty here: position (2,0,0)'s -X neighbour is the
        // occupied voxel at (1,0,0), so out[2] must carry that bit. Verified
        // fully in the next test; this just confirms +X and -X don't collide.
        assert.strictEqual(out[0] & out[2], 0, '+X and -X masks must not overlap');
    });

    it('reports the in-block -X neighbour of a single voxel', function () {
        const [lo, hi] = bit(1, 0, 0);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [mLo] = bit(2, 0, 0);
        assert.strictEqual(out[2], mLo, '-X mask should mark position (2,0,0)');
    });

    it('does not wrap across the lx=3 to lx=0 boundary', function () {
        // voxel at (0,0,0); position (3,0,0) must NOT see it in +X
        const [lo, hi] = bit(0, 0, 0);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [wrapLo] = bit(3, 0, 0);
        assert.strictEqual(out[0] & wrapLo, 0, '+X must not wrap into lx=3');
    });

    it('crosses the Z boundary between lo and hi words', function () {
        // voxel at (0,0,2) is in hi; position (0,0,1) is in lo and sees it in +Z
        const [lo, hi] = bit(0, 0, 2);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [seenLo] = bit(0, 0, 1);
        assert.strictEqual(out[8] & seenLo, seenLo, '+Z must cross lo/hi');
    });

    it('pulls the adjacent block face across +X', function () {
        // subject block empty; neighbour block at bx+1 is fully solid.
        // every position with lx=3 must see an occupied +X neighbour.
        const solidReader = (bx, by, bz, out) => {
            if (bx === 1 && by === 0 && bz === 0) {
                out[0] = SOLID_LO;
                out[1] = SOLID_HI;
            } else {
                out[0] = 0;
                out[1] = 0;
            }
        };
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(solidReader, 0, 0, 0, 0, 0, out);
        for (let z = 0; z < 4; z++) {
            for (let y = 0; y < 4; y++) {
                const [eLo, eHi] = bit(3, y, z);
                // >>> 0 after & is required: JS's bitwise & converts both
                // operands to signed Int32 before combining, so comparing a
                // raw `&` result against an unsigned value like 2147483648
                // (bit 31 set) fails even when the bits genuinely match.
                if (eLo) assert.strictEqual((out[0] & eLo) >>> 0, eLo, `+X face at y=${y} z=${z}`);
                if (eHi) assert.strictEqual((out[1] & eHi) >>> 0, eHi, `+X face at y=${y} z=${z}`);
            }
        }
    });

    it('agrees with a brute-force per-voxel neighbour check', function () {
        // Build a small grid, then for one block compare sixNeighborMasks
        // against getVoxel on every position and direction.
        const g = new SparseVoxelGrid(12, 12, 12);
        let seed = 12345;
        const rnd = () => {
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
            return seed / 0x7FFFFFFF;
        };
        for (let z = 0; z < 12; z++) {
            for (let y = 0; y < 12; y++) {
                for (let x = 0; x < 12; x++) {
                    if (rnd() < 0.3) g.setVoxel(x, y, z);
                }
            }
        }
        const read = (bx, by, bz, out) => {
            if (bx < 0 || by < 0 || bz < 0 || bx >= g.nbx || by >= g.nby || bz >= g.nbz) {
                out[0] = 0;
                out[1] = 0;
                return;
            }
            const bi = bx + by * g.nbx + bz * g.bStride;
            const bt = g.getBlockType(bi);
            if (bt === 0) {
                out[0] = 0;
                out[1] = 0;
            } else if (bt === 1) {
                out[0] = SOLID_LO;
                out[1] = SOLID_HI;
            } else {
                const s = g.masks.slot(bi);
                out[0] = g.masks.lo[s];
                out[1] = g.masks.hi[s];
            }
        };
        const own = new Uint32Array(2);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        const dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
        // block (1,1,1) is interior, so all six neighbour blocks exist
        read(1, 1, 1, own);
        sixNeighborMasks(read, own[0], own[1], 1, 1, 1, out);
        for (let lz = 0; lz < 4; lz++) {
            for (let ly = 0; ly < 4; ly++) {
                for (let lx = 0; lx < 4; lx++) {
                    const [pLo, pHi] = bit(lx, ly, lz);
                    for (let d = 0; d < 6; d++) {
                        const [dx, dy, dz] = dirs[d];
                        const expected = g.getVoxel(4 + lx + dx, 4 + ly + dy, 4 + lz + dz);
                        const got = pLo ? ((out[d * 2] & pLo) !== 0) : ((out[d * 2 + 1] & pHi) !== 0);
                        assert.strictEqual(
                            got, expected === 1,
                            `dir ${d} at local (${lx},${ly},${lz})`
                        );
                    }
                }
            }
        }
    });
});

// A grid where every voxel is a candidate, for tests not exercising the gate.
const allCandidate = (nx, ny, nz) => {
    const g = new SparseVoxelGrid(nx, ny, nz);
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

// A one-voxel-thick sheet at y === yPlane, with the listed (x,z) holes.
const sheet = (n, yPlane, holes) => {
    const g = new SparseVoxelGrid(n, n, n);
    const isHole = new Set(holes.map(([x, z]) => `${x},${z}`));
    for (let z = 0; z < n; z++) {
        for (let x = 0; x < n; x++) {
            if (!isHole.has(`${x},${z}`)) g.setVoxel(x, yPlane, z);
        }
    }
    return g;
};

describe('growGrid', function () {
    it('fills a 1x1 hole in a sheet in one iteration', function () {
        const g = sheet(8, 4, [[4, 4]]);
        const res = growGrid(g, allCandidate(8, 8, 8), { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
        assert.strictEqual(res.added, 1);
    });

    it('fills a 1x3 slit in exactly two iterations, not one', function () {
        const holes = [[3, 4], [4, 4], [5, 4]];
        const one = growGrid(sheet(12, 4, holes), allCandidate(12, 12, 12),
            { minNeighbors: 3, maxIterations: 1 });
        assert.strictEqual(one.grid.getVoxel(4, 4, 4), 0, 'middle needs a second pass');
        assert.strictEqual(one.grid.getVoxel(3, 4, 4), 1, 'ends fill first');

        const two = growGrid(sheet(12, 4, holes), allCandidate(12, 12, 12),
            { minNeighbors: 3, maxIterations: 2 });
        assert.strictEqual(two.grid.getVoxel(4, 4, 4), 1);
        assert.strictEqual(two.added, 3);
    });

    it('never fills a 3x3 square hole, at any iteration count', function () {
        const holes = [];
        for (let x = 3; x <= 5; x++) {
            for (let z = 3; z <= 5; z++) holes.push([x, z]);
        }
        const res = growGrid(sheet(12, 4, holes), allCandidate(12, 12, 12),
            { minNeighbors: 3, maxIterations: 32 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 0, 'centre must stay open');
        assert.strictEqual(res.added, 0, 'nothing in a 3x3 hole reaches 3 neighbours');
    });

    it('does not grow outward from a flat sheet face', function () {
        // A complete sheet: every voxel just above it has exactly 1 occupied
        // neighbour, so nothing should be added anywhere.
        const res = growGrid(sheet(8, 4, []), allCandidate(8, 8, 8),
            { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.added, 0);
    });

    it('respects the candidate gate: an empty candidate adds nothing', function () {
        const empty = new SparseVoxelGrid(8, 8, 8);
        const res = growGrid(sheet(8, 4, [[4, 4]]), empty,
            { minNeighbors: 3, maxIterations: 8 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 0, 'gate must block the fill');
        assert.strictEqual(res.added, 0);
    });

    it('reports gate-rejected voxels', function () {
        const empty = new SparseVoxelGrid(8, 8, 8);
        const res = growGrid(sheet(8, 4, [[4, 4]]), empty,
            { minNeighbors: 3, maxIterations: 8 });
        assert.strictEqual(res.gateRejected, 1,
            'the hole was eligible but blocked, and must be counted once');
    });

    it('reports zero gate-rejected when everything is a candidate', function () {
        const res = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.gateRejected, 0);
    });

    it('respects the candidate gate per voxel', function () {
        // Two 1x1 holes; only one is a candidate.
        const cand = new SparseVoxelGrid(12, 12, 12);
        cand.setVoxel(4, 4, 4);
        const res = growGrid(sheet(12, 4, [[4, 4], [8, 8]]), cand,
            { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
        assert.strictEqual(res.grid.getVoxel(8, 4, 8), 0);
        assert.strictEqual(res.added, 1);
    });

    it('terminates early when an iteration adds nothing', function () {
        const res = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 3, maxIterations: 16 });
        assert.strictEqual(res.iterations, 2,
            'one productive pass plus one that adds nothing');
    });

    it('honours a higher minNeighbors', function () {
        // A 1x1 hole in a sheet has 4 neighbours, so k=4 fills it but k=5 does not.
        const four = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 4, maxIterations: 4 });
        assert.strictEqual(four.grid.getVoxel(4, 4, 4), 1);

        const five = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 5, maxIterations: 4 });
        assert.strictEqual(five.grid.getVoxel(4, 4, 4), 0);
    });

    it('fills an interior void with 6 neighbours at k=6', function () {
        // A 3x3x3 solid cube with its centre missing: the centre has all 6.
        const g = new SparseVoxelGrid(8, 8, 8);
        for (let z = 2; z <= 4; z++) {
            for (let y = 2; y <= 4; y++) {
                for (let x = 2; x <= 4; x++) {
                    if (!(x === 3 && y === 3 && z === 3)) g.setVoxel(x, y, z);
                }
            }
        }
        const res = growGrid(g, allCandidate(8, 8, 8), { minNeighbors: 6, maxIterations: 2 });
        assert.strictEqual(res.grid.getVoxel(3, 3, 3), 1);
        assert.strictEqual(res.added, 1);
    });

    it('crosses block boundaries', function () {
        // Hole at (4,4,4) sits at a block corner (blocks are 4^3), so its
        // neighbours live in four different blocks.
        const g = sheet(12, 4, [[4, 4]]);
        const res = growGrid(g, allCandidate(12, 12, 12), { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
    });

    it('leaves the candidate grid untouched', function () {
        const cand = allCandidate(8, 8, 8);
        const before = [...cand.types];
        growGrid(sheet(8, 4, [[4, 4]]), cand, { minNeighbors: 3, maxIterations: 4 });
        assert.deepStrictEqual([...cand.types], before);
    });

    it('defaults to minNeighbors 3 and maxIterations 4', function () {
        const res = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8));
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
    });
});
