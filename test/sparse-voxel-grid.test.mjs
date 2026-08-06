import { describe, it } from 'node:test';
import assert from 'node:assert';

import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

/**
 * Reference implementation: dense raster walk over every cell.
 *
 * @param {SparseVoxelGrid} grid - Grid to scan.
 * @returns {string[]} Sorted "x,y,z" keys of occupied voxels.
 */
const denseOccupied = (grid) => {
    const out = [];
    for (let iz = 0; iz < grid.nz; iz++) {
        for (let iy = 0; iy < grid.ny; iy++) {
            for (let ix = 0; ix < grid.nx; ix++) {
                if (grid.getVoxel(ix, iy, iz)) out.push(`${ix},${iy},${iz}`);
            }
        }
    }
    return out.sort();
};

/**
 * Collect occupied voxels via the sparse iterator.
 *
 * @param {SparseVoxelGrid} grid - Grid to scan.
 * @returns {string[]} Sorted "x,y,z" keys of visited voxels.
 */
const sparseOccupied = (grid) => {
    const out = [];
    grid.forEachOccupiedVoxel((ix, iy, iz) => out.push(`${ix},${iy},${iz}`));
    return out.sort();
};

describe('SparseVoxelGrid.forEachOccupiedVoxel', () => {
    it('should visit nothing in an empty grid', () => {
        const grid = new SparseVoxelGrid(8, 8, 8);
        assert.deepStrictEqual(sparseOccupied(grid), []);
    });

    it('should visit every voxel of a fully solid block', () => {
        const grid = new SparseVoxelGrid(4, 4, 4);
        for (let iz = 0; iz < 4; iz++) {
            for (let iy = 0; iy < 4; iy++) {
                for (let ix = 0; ix < 4; ix++) grid.setVoxel(ix, iy, iz);
            }
        }
        // the block must have collapsed to SOLID, exercising that branch
        assert.strictEqual(grid.getBlockType(0), 1);
        assert.strictEqual(sparseOccupied(grid).length, 64);
        assert.deepStrictEqual(sparseOccupied(grid), denseOccupied(grid));
    });

    it('should visit each single voxel of a mixed block, including bit 63', () => {
        const grid = new SparseVoxelGrid(4, 4, 4);
        // bit 0 lives in `lo`, bit 63 in `hi`; both halves must be read
        grid.setVoxel(0, 0, 0);
        grid.setVoxel(3, 3, 3);
        assert.deepStrictEqual(sparseOccupied(grid), ['0,0,0', '3,3,3']);
    });

    it('should visit voxels across every local bit position', () => {
        const grid = new SparseVoxelGrid(4, 4, 4);
        const expected = [];
        for (let lz = 0; lz < 4; lz++) {
            for (let ly = 0; ly < 4; ly++) {
                for (let lx = 0; lx < 4; lx++) {
                    // a sparse but bit-spanning subset
                    if ((lx + ly * 4 + lz * 16) % 3 !== 0) continue;
                    grid.setVoxel(lx, ly, lz);
                    expected.push(`${lx},${ly},${lz}`);
                }
            }
        }
        assert.deepStrictEqual(sparseOccupied(grid), expected.sort());
    });

    it('should agree with a dense walk over a multi-block grid', () => {
        const grid = new SparseVoxelGrid(16, 12, 20);
        // deterministic pseudo-random spread across blocks and bit positions
        let seed = 12345;
        const next = () => {
            seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
            return seed / 0x100000000;
        };
        let placed = 0;
        for (let iz = 0; iz < 20; iz++) {
            for (let iy = 0; iy < 12; iy++) {
                for (let ix = 0; ix < 16; ix++) {
                    if (next() < 0.2) {
                        grid.setVoxel(ix, iy, iz);
                        placed++;
                    }
                }
            }
        }
        assert.ok(placed > 100, 'test should place a meaningful number of voxels');
        assert.deepStrictEqual(sparseOccupied(grid), denseOccupied(grid));
    });

    it('should agree with a dense walk when solid and mixed blocks are mixed', () => {
        const grid = new SparseVoxelGrid(16, 8, 8);
        // one fully solid block
        for (let iz = 0; iz < 4; iz++) {
            for (let iy = 0; iy < 4; iy++) {
                for (let ix = 4; ix < 8; ix++) grid.setVoxel(ix, iy, iz);
            }
        }
        // scattered voxels in other blocks
        grid.setVoxel(0, 0, 0);
        grid.setVoxel(15, 7, 7);
        grid.setVoxel(9, 5, 2);
        grid.setVoxel(12, 4, 6);

        assert.deepStrictEqual(sparseOccupied(grid), denseOccupied(grid));
        assert.strictEqual(sparseOccupied(grid).length, 64 + 4);
    });
});
