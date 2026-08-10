/**
 * Tests for the voxel cleanup orchestrator and its dial mapping.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { CANDIDATE_CUTOFF, cleanupGrid, cleanupRadius } from '../src/lib/voxel/cleanup.js';
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

// A thick slab (y in [y0,y1]) spanning the grid's XZ interior, with full-depth
// column holes at the listed (x,z) positions. Deliberately NOT a single-Y-layer
// sheet: cleanupGrid always runs the majority filter at the real production
// threshold (14 of 27), and a single-voxel-thick sheet can never reach that --
// its Y+-1 neighbours are always empty, capping every position at 9 of 27 -- so
// a thin sheet gets erased wholesale by majority regardless of what grow does.
// A slab several voxels thick lets positions well inside it reach threshold.
const slab = (n, y0, y1, holes) => {
    const g = new SparseVoxelGrid(n, n, n);
    const isHole = new Set(holes.map(([x, z]) => `${x},${z}`));
    for (let z = 2; z < n - 2; z++) {
        for (let x = 2; x < n - 2; x++) {
            if (isHole.has(`${x},${z}`)) continue;
            for (let y = y0; y <= y1; y++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

describe('cleanupRadius', function () {
    it('maps strength to whole voxels', function () {
        assert.strictEqual(cleanupRadius(0.2, 0.1), 2);
        assert.strictEqual(cleanupRadius(0.5, 0.1), 5);
        assert.strictEqual(cleanupRadius(0.02, 0.01), 2);
    });

    it('rounds to the nearest voxel', function () {
        assert.strictEqual(cleanupRadius(0.24, 0.1), 2);
        assert.strictEqual(cleanupRadius(0.26, 0.1), 3);
    });

    it('clamps a sub-voxel strength up to 1', function () {
        assert.strictEqual(cleanupRadius(0.04, 0.1), 1);
        assert.strictEqual(cleanupRadius(0.0001, 0.1), 1);
    });
});

describe('CANDIDATE_CUTOFF', function () {
    it('is far below the default opacity cutoff', function () {
        assert.strictEqual(CANDIDATE_CUTOFF, 0.002);
        assert.ok(CANDIDATE_CUTOFF < 0.1, 'must admit voxels the solid pass rejects');
    });
});

describe('cleanupGrid', function () {
    it('fills a hole and reports it', function () {
        // A full-depth column hole through an 8-thick slab: every layer has 4
        // immediately-occupied lateral neighbours (>= minNeighbors 3), so all
        // 8 layers fill in the first pass regardless of iteration order.
        const n = 24;
        const res = cleanupGrid(slab(n, 4, 11, [[10, 10]]), allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(res.grid.getVoxel(10, 8, 10), 1);
        assert.strictEqual(res.stats.grown, 8);
        assert.strictEqual(res.stats.radius, 2);
    });

    it('removes an isolated speck', function () {
        const n = 24;
        const g = slab(n, 4, 11, []);
        g.setVoxel(2, 20, 2);
        const res = cleanupGrid(g, allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(res.grid.getVoxel(2, 20, 2), 0);
        // An isolated single voxel has only 1 occupied neighbour (itself),
        // far below the majority threshold, so it is typically eaten by the
        // majority stage before despeckle ever sees it -- hence checking the
        // sum rather than despeckled alone.
        assert.ok(res.stats.majorityRemoved + res.stats.despeckled >= 1);
        // The slab itself, well inside its own extent, must survive.
        assert.strictEqual(res.grid.getVoxel(10, 8, 10), 1);
    });

    it('never adds a voxel outside the candidate mask', function () {
        const n = 24;
        const empty = new SparseVoxelGrid(n, n, n);
        const res = cleanupGrid(slab(n, 4, 11, [[10, 10]]), empty,
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(res.grid.getVoxel(10, 8, 10), 0, 'the gate must hold');
        assert.strictEqual(res.stats.grown, 0);
        assert.strictEqual(res.stats.majorityAdded, 0);
    });

    it('reports what the gate blocked', function () {
        const n = 24;
        const empty = new SparseVoxelGrid(n, n, n);
        const res = cleanupGrid(slab(n, 4, 11, [[10, 10]]), empty,
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.ok(res.stats.gateRejected >= 1,
            'the blocked hole is the audit trail and must be reported');
    });

    it('runs majority and despeckle with fill mode none', function () {
        const n = 24;
        const g = slab(n, 4, 11, []);
        g.setVoxel(2, 20, 2);
        const res = cleanupGrid(g, allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'none' });
        assert.strictEqual(res.stats.grown, 0, 'no fill stage ran');
        assert.strictEqual(res.grid.getVoxel(2, 20, 2), 0, 'but the speck still goes');
    });

    it('defaults the fill mode to grow', function () {
        const n = 24;
        const res = cleanupGrid(slab(n, 4, 11, [[10, 10]]), allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1 });
        assert.strictEqual(res.stats.grown, 8);
    });

    it('reduces to a single component on a speckled slab', function () {
        // Twenty scattered single-voxel specks, well away from the slab.
        // Each speck has only 1 occupied neighbour, so majority typically
        // removes them itself before despeckle runs -- hence asserting the
        // final component count (1, the slab) rather than despeckle's own
        // componentsRemoved, which can legitimately be 0.
        const n = 24;
        const g = slab(n, 4, 11, []);
        for (let i = 0; i < 20; i++) g.setVoxel(2 + (i % 18), 18, 2 + ((i * 7) % 18));
        const before = countVoxels(g);
        const res = cleanupGrid(g, allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.ok(countVoxels(res.grid) < before, 'specks should be gone');
        assert.strictEqual(res.stats.components, 1, 'only the slab should remain');
    });

    it('rejects a non-positive strength', function () {
        assert.throws(
            () => cleanupGrid(slab(24, 4, 11, []), allCandidate(24),
                { strength: 0, voxelResolution: 0.1 }),
            /strength must be > 0/
        );
    });

    it('rejects a non-positive voxel resolution', function () {
        assert.throws(
            () => cleanupGrid(slab(24, 4, 11, []), allCandidate(24),
                { strength: 0.2, voxelResolution: 0 }),
            /voxelResolution must be > 0/
        );
    });

    it('throws a clear not-implemented error for close', function () {
        assert.throws(
            () => cleanupGrid(slab(24, 4, 11, []), allCandidate(24),
                { strength: 0.2, voxelResolution: 0.1, fill: 'close' }),
            /not implemented yet/
        );
    });

    it('throws a clear not-implemented error for both', function () {
        assert.throws(
            () => cleanupGrid(slab(24, 4, 11, []), allCandidate(24),
                { strength: 0.2, voxelResolution: 0.1, fill: 'both' }),
            /not implemented yet/
        );
    });

    it('leaves the candidate grid untouched', function () {
        const n = 24;
        const cand = allCandidate(n);
        const before = [...cand.types];
        cleanupGrid(slab(n, 4, 11, [[10, 10]]), cand,
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.deepStrictEqual([...cand.types], before);
    });
});
