/**
 * Tests for connected-component despeckling.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { despeckleGrid } from '../src/lib/voxel/despeckle.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

const countVoxels = (g) => {
    let n = 0;
    g.forEachOccupiedVoxel(() => n++);
    return n;
};

// A solid axis-aligned box, inclusive bounds.
const box = (g, x0, y0, z0, x1, y1, z1) => {
    for (let z = z0; z <= z1; z++) {
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) g.setVoxel(x, y, z);
        }
    }
};

describe('despeckleGrid', function () {
    it('removes a single isolated voxel', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0);
        assert.strictEqual(res.removed, 1);
        assert.strictEqual(res.componentsRemoved, 1);
    });

    it('removes a 27-voxel island at minVoxels 64', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        box(g, 4, 4, 4, 6, 6, 6);           // 3x3x3 = 27
        assert.strictEqual(countVoxels(g), 27);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(countVoxels(res.grid), 0);
        assert.strictEqual(res.removed, 27);
    });

    it('keeps a 125-voxel island at minVoxels 64', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        box(g, 4, 4, 4, 8, 8, 8);           // 5x5x5 = 125
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(countVoxels(res.grid), 125);
        assert.strictEqual(res.removed, 0);
        assert.strictEqual(res.componentsRemoved, 0);
    });

    it('keeps exactly at the threshold and removes one below', function () {
        const atLimit = new SparseVoxelGrid(16, 16, 16);
        box(atLimit, 0, 0, 0, 3, 3, 3);     // 4x4x4 = 64
        const keep = despeckleGrid(atLimit, { minVoxels: 64 });
        assert.strictEqual(countVoxels(keep.grid), 64, '64 is not below 64');

        const below = new SparseVoxelGrid(16, 16, 16);
        box(below, 0, 0, 0, 3, 3, 3);
        below.clearVoxel(3, 3, 3);          // 63
        const drop = despeckleGrid(below, { minVoxels: 64 });
        assert.strictEqual(countVoxels(drop.grid), 0);
    });

    it('removes small islands and keeps large ones together', function () {
        const g = new SparseVoxelGrid(32, 32, 32);
        box(g, 2, 2, 2, 8, 8, 8);           // 343, keep
        g.setVoxel(20, 20, 20);             // 1, drop
        box(g, 24, 24, 24, 25, 25, 25);     // 8, drop
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(countVoxels(res.grid), 343);
        assert.strictEqual(res.removed, 9);
        assert.strictEqual(res.components, 3);
        assert.strictEqual(res.componentsRemoved, 2);
    });

    it('counts a component spanning many blocks as one', function () {
        // A 1-voxel-wide bar 20 long crosses five 4-wide blocks.
        const g = new SparseVoxelGrid(32, 32, 32);
        for (let x = 4; x < 24; x++) g.setVoxel(x, 8, 8);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.components, 1);
        assert.strictEqual(res.removed, 20, 'a 20-voxel bar is below 64');
    });

    it('treats diagonal contact as disconnected', function () {
        // Two voxels touching only at a corner are two components.
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(4, 4, 4);
        g.setVoxel(5, 5, 5);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.components, 2);
        assert.strictEqual(res.removed, 2);
    });

    it('is a no-op at minVoxels 0', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = despeckleGrid(g, { minVoxels: 0 });
        assert.strictEqual(countVoxels(res.grid), 1);
        assert.strictEqual(res.removed, 0);
    });

    it('handles a fully solid grid as one component', function () {
        const g = new SparseVoxelGrid(8, 8, 8);
        box(g, 0, 0, 0, 7, 7, 7);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.components, 1);
        assert.strictEqual(countVoxels(res.grid), 512);
    });

    it('defaults minVoxels to 64', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        box(g, 4, 4, 4, 6, 6, 6);           // 27
        const res = despeckleGrid(g);
        assert.strictEqual(countVoxels(res.grid), 0);
    });
});
