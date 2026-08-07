/**
 * Tests for the collision GLB node transform.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { encodeGlb } from '../src/lib/writers/collision-glb.js';

/**
 * Extract and parse the JSON chunk of a GLB container.
 *
 * @param {Uint8Array} glb - Encoded GLB bytes.
 * @returns {object} Parsed glTF JSON.
 */
function readGltfJson(glb) {
    const view = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
    const jsonLength = view.getUint32(12, true);
    const jsonBytes = glb.subarray(20, 20 + jsonLength);
    return JSON.parse(new TextDecoder().decode(jsonBytes));
}

describe('encodeGlb node transform', function () {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const indices = new Uint32Array([0, 1, 2]);

    it('emits a bare mesh node when no rotation is given', function () {
        const gltf = readGltfJson(encodeGlb(positions, indices));

        assert.deepStrictEqual(gltf.nodes, [{ mesh: 0 }]);
    });

    it('emits the node rotation when one is given', function () {
        const rotation = [0, -0.2588190451, 0, 0.9659258263];

        const gltf = readGltfJson(encodeGlb(positions, indices, undefined, rotation));

        assert.deepStrictEqual(gltf.nodes, [{ mesh: 0, rotation }]);
        assert.deepStrictEqual(gltf.scenes, [{ nodes: [0] }]);
    });
});
