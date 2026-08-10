/**
 * Tests for density-gated region growing and its block-neighbour helper.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    NEIGHBOR_SCRATCH_LEN,
    sixNeighborMasks
} from '../src/lib/voxel/block-neighbors.js';
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
