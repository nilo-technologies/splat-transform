/**
 * Tests for the voxel grid block-count ceiling.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { MAX_GRID_BLOCKS, assertGridFits } from '../src/lib/voxel/grid-limits.js';

describe('assertGridFits', function () {
    it('accepts a grid well inside the limit', function () {
        // 512^3 = 134217728 blocks
        assert.doesNotThrow(() => assertGridFits(512, 512, 512, 0.01));
    });

    it('rejects exactly at the boundary', function () {
        // pins the strict >: MAX_GRID_BLOCKS itself passes, one block over fails
        assert.doesNotThrow(() => assertGridFits(MAX_GRID_BLOCKS, 1, 1, 0.01));
        assert.throws(
            () => assertGridFits(MAX_GRID_BLOCKS + 1, 1, 1, 0.01),
            /too large/
        );
    });

    it('accepts a realistic large scene', function () {
        // filtered landscape.spz at 0.01m: 939 x 285 x 338 = 90.4e6 blocks
        assert.doesNotThrow(() => assertGridFits(939, 285, 338, 0.01));
    });

    it('rejects a grid past the limit', function () {
        // 1484 x 927 x 2208 = 3.04e9 blocks, the unfiltered landscape.spz case
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            /too large/
        );
    });

    it('names the block count and the limit in the error', function () {
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            (err) => {
                assert.match(err.message, /3037474944/);
                assert.match(err.message, new RegExp(String(MAX_GRID_BLOCKS)));
                return true;
            }
        );
    });

    it('suggests both remedies in the error', function () {
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            (err) => {
                assert.match(err.message, /--voxel-params/);
                assert.match(err.message, /--filter-box/);
                return true;
            }
        );
    });

    it('reports the voxel resolution that produced the grid', function () {
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            /0\.1/
        );
    });

    it('exposes the limit as a power of two under the IntKeyMap ceiling', function () {
        assert.strictEqual(MAX_GRID_BLOCKS, 2 ** 29);
        assert.ok(MAX_GRID_BLOCKS < 0.7 * 2 ** 30);
    });
});
