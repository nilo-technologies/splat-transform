import { describe, it } from 'node:test';
import assert from 'node:assert';

import { IntKeyMap } from '../src/lib/utils/int-key-map.js';

describe('IntKeyMap', () => {
    it('should return -1 for a missing key', () => {
        const map = new IntKeyMap();
        assert.strictEqual(map.get(0), -1);
        assert.strictEqual(map.get(12345), -1);
    });

    it('should store and retrieve a value for key 0', () => {
        // key 0 must not be confused with the empty-slot sentinel
        const map = new IntKeyMap();
        map.set(0, 7);
        assert.strictEqual(map.get(0), 7);
        assert.strictEqual(map.size, 1);
    });

    it('should store and retrieve many distinct keys', () => {
        const map = new IntKeyMap(4);
        const n = 10000;
        for (let i = 0; i < n; i++) map.set(i * 7919, i);
        assert.strictEqual(map.size, n);
        for (let i = 0; i < n; i++) {
            assert.strictEqual(map.get(i * 7919), i, `key ${i * 7919}`);
        }
        assert.strictEqual(map.get(1), -1);
    });

    it('should answer has() for present and absent keys', () => {
        const map = new IntKeyMap();
        map.set(0, 5);
        map.set(2 ** 40, 6);
        assert.strictEqual(map.has(0), true);
        assert.strictEqual(map.has(2 ** 40), true);
        assert.strictEqual(map.has(1), false);
        assert.strictEqual(map.has(2 ** 40 + 1), false);
    });

    it('should overwrite an existing key without growing', () => {
        const map = new IntKeyMap();
        map.set(42, 1);
        map.set(42, 2);
        assert.strictEqual(map.get(42), 2);
        assert.strictEqual(map.size, 1);
    });

    it('should handle keys above 2^32 exactly', () => {
        // vertexKey() in voxel-faces grows as coordStride^2 * z, which exceeds
        // 32 bits on large grids; keys must stay exact, not be truncated.
        const map = new IntKeyMap();
        const a = 2 ** 33;
        const b = 2 ** 33 + 1;
        const c = 2 ** 40 + 12345;
        map.set(a, 1);
        map.set(b, 2);
        map.set(c, 3);
        assert.strictEqual(map.get(a), 1);
        assert.strictEqual(map.get(b), 2);
        assert.strictEqual(map.get(c), 3);
        assert.strictEqual(map.size, 3);
    });

    it('should report an accurate size while growing repeatedly', () => {
        const map = new IntKeyMap(2);
        for (let i = 0; i < 5000; i++) {
            map.set(i, i);
            assert.strictEqual(map.size, i + 1);
        }
        for (let i = 0; i < 5000; i++) assert.strictEqual(map.get(i), i);
    });

    it('should hold more entries than a JS Map allows', () => {
        // V8 caps Map at 2^24 entries and throws
        // "RangeError: Map maximum size exceeded" beyond that. A voxel-face
        // mesh of a large grid needs more unique vertices than that, so the
        // dedup table must not be a Map.
        const V8_MAP_LIMIT = 2 ** 24;
        const n = V8_MAP_LIMIT + 1000;
        const map = new IntKeyMap();
        for (let i = 0; i < n; i++) map.set(i, i);
        assert.strictEqual(map.size, n);
        // spot check rather than re-walk 16M keys
        for (const i of [0, 1, 12345, V8_MAP_LIMIT - 1, V8_MAP_LIMIT, n - 1]) {
            assert.strictEqual(map.get(i), i, `key ${i}`);
        }
        assert.strictEqual(map.get(n), -1);
    });
});
