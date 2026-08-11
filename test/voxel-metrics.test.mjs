/**
 * Tests for the voxel metrics tool's octree decoder and metric definitions.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Vec3 } from 'playcanvas';

import { buildSparseOctree } from '../src/lib/writers/sparse-octree.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';
import { decodeOctree, gridMetrics } from '../tools/voxel-metrics.mjs';

// buildSparseOctree needs world bounds; one voxel per unit keeps the mapping
// from grid indices to world coordinates trivial.
const boundsFor = (nx, ny, nz) => ({
    min: new Vec3(0, 0, 0),
    max: new Vec3(nx, ny, nz)
});

const roundtrip = (grid, nx, ny, nz, dense) => {
    // buildSparseOctree has two emitters -- Morton streams, and a dense mip
    // build chosen by shouldUseDenseMipBuild or forced with options.dense
    // (sparse-octree.ts:669). Both write the same node format, so the decoder
    // must handle either; every case below runs through both.
    const octree = buildSparseOctree(grid, boundsFor(nx, ny, nz), boundsFor(nx, ny, nz), 1,
        { dense });
    return decodeOctree({
        nodes: octree.nodes,
        leafData: octree.leafData,
        treeDepth: octree.treeDepth,
        nx,
        ny,
        nz
    });
};

const dump = (voxels) => [...voxels].sort((a, b) => a - b);

// Runs a case through both emitters. `build` must return a fresh grid each
// call: buildSparseOctree may consume the one it is given.
const bothPaths = (name, build, nx, ny, nz, check) => {
    for (const dense of [false, true]) {
        check(roundtrip(build(), nx, ny, nz, dense), `${name} (dense=${dense})`);
    }
};

describe('decodeOctree', function () {
    it('recovers a mixed leaf exactly', function () {
        const expected = new Set();
        const build = () => {
            const g = new SparseVoxelGrid(16, 16, 16);
            expected.clear();
            for (const [x, y, z] of [[0, 0, 0], [1, 2, 3], [15, 15, 15], [4, 0, 9], [7, 7, 7]]) {
                g.setVoxel(x, y, z);
                expected.add(x + y * 16 + z * 16 * 16);
            }
            return g;
        };
        bothPaths('mixed leaf', build, 16, 16, 16, (got, label) => {
            assert.strictEqual(got.nx, 16, label);
            assert.deepStrictEqual(dump(got.voxels), dump(expected), label);
        });
    });

    it('recovers solid leaves and whole solid subtrees', function () {
        // A solid 8x8x8 corner: whole blocks are SOLID and their parents
        // aggregate, so this exercises SOLID_LEAF_MARKER above the leaf level.
        const n = 16;
        const expected = new Set();
        const build = () => {
            const g = new SparseVoxelGrid(n, n, n);
            expected.clear();
            for (let z = 0; z < 8; z++) {
                for (let y = 0; y < 8; y++) {
                    for (let x = 0; x < 8; x++) {
                        g.setVoxel(x, y, z);
                        expected.add(x + y * n + z * n * n);
                    }
                }
            }
            return g;
        };
        bothPaths('solid subtree', build, n, n, n, (got, label) => {
            assert.strictEqual(got.voxels.size, 512, label);
            assert.deepStrictEqual(dump(got.voxels), dump(expected), label);
        });
    });

    it('recovers a sparse shell across many blocks', function () {
        const n = 32;
        const expected = new Set();
        const build = () => {
            const g = new SparseVoxelGrid(n, n, n);
            expected.clear();
            let seed = 7;
            const rnd = () => {
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
                return seed / 0x7FFFFFFF;
            };
            for (let z = 0; z < n; z++) {
                for (let y = 0; y < n; y++) {
                    for (let x = 0; x < n; x++) {
                        if (rnd() < 0.1) {
                            g.setVoxel(x, y, z);
                            expected.add(x + y * n + z * n * n);
                        }
                    }
                }
            }
            return g;
        };
        bothPaths('sparse shell', build, n, n, n, (got, label) => {
            assert.deepStrictEqual(dump(got.voxels), dump(expected), label);
        });
    });

    it('recovers an empty grid as no voxels', function () {
        bothPaths('empty', () => new SparseVoxelGrid(16, 16, 16), 16, 16, 16, (got, label) => {
            assert.strictEqual(got.voxels.size, 0, label);
        });
    });
});

describe('gridMetrics', function () {
    it('counts components and the largest share', function () {
        const n = 16;
        const voxels = new Set();
        const key = (x, y, z) => x + y * n + z * n * n;
        // Component A: a 3x3 plate (9 voxels). Component B: 2 voxels, apart.
        for (let z = 2; z <= 4; z++) {
            for (let x = 2; x <= 4; x++) voxels.add(key(x, 8, z));
        }
        voxels.add(key(12, 8, 12));
        voxels.add(key(12, 8, 13));
        const m = gridMetrics({ nx: n, ny: n, nz: n, voxels });
        assert.strictEqual(m.occupied, 11);
        assert.strictEqual(m.islands, 2);
        assert.strictEqual(m.largestShare, 9 / 11);
    });

    it('reports zero roughness for a flat surface and more for a jagged one', function () {
        const n = 16;
        const key = (x, y, z) => x + y * n + z * n * n;
        const flat = new Set();
        for (let z = 2; z <= 12; z++) {
            for (let x = 2; x <= 12; x++) flat.add(key(x, 8, z));
        }
        assert.strictEqual(gridMetrics({ nx: n, ny: n, nz: n, voxels: flat }).roughness, 0);

        // A checkerboard of 4-voxel steps. Roughness averages over the up-to-8
        // neighbouring columns, and a checkerboard's four diagonal neighbours
        // sit at its own height, so the neighbour mean lands midway between the
        // two levels: an interior column reads step/2 = 2, and grid-edge
        // columns push the mean a little above that. 2 is therefore the bound
        // this fixture can reach, against 0 for the flat surface above.
        const jagged = new Set();
        for (let z = 2; z <= 12; z++) {
            for (let x = 2; x <= 12; x++) jagged.add(key(x, 8 + ((x + z) % 2) * 4, z));
        }
        assert.ok(gridMetrics({ nx: n, ny: n, nz: n, voxels: jagged }).roughness > 2,
            'alternating 4-voxel steps must read as rough');
    });

    it('matches the scatter definition the writer logs', function () {
        // scatterFraction (write-voxel.ts:72) is the share of occupied voxels
        // with at most 2 of 6 face neighbours. A lone voxel has 0.
        const n = 16;
        const voxels = new Set([8 + 8 * n + 8 * n * n]);
        assert.strictEqual(gridMetrics({ nx: n, ny: n, nz: n, voxels }).scatter, 1);
    });
});
