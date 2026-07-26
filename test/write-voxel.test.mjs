/**
 * Tests for voxel writer file contracts.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { Column, DataTable } from '../src/lib/index.js';
import { MemoryFileSystem } from '../src/lib/io/write/index.js';
import { writeOctreeFiles, writeVoxel } from '../src/lib/writers/write-voxel.js';

describe('writeOctreeFiles', function () {
    it('writes metadata and little-endian nodes followed by leafData', async function () {
        const fs = new MemoryFileSystem();
        const octree = {
            gridBounds: {
                min: new Vec3(0, 1, 2),
                max: new Vec3(3, 4, 5)
            },
            sceneBounds: {
                min: new Vec3(-1, -2, -3),
                max: new Vec3(6, 7, 8)
            },
            voxelResolution: 0.25,
            leafSize: 4,
            treeDepth: 2,
            numInteriorNodes: 1,
            numMixedLeaves: 1,
            nodes: new Uint32Array([0x11223344, 0xAABBCCDD]),
            leafData: new Uint32Array([0x01020304, 0xFFEEDDCC])
        };

        await writeOctreeFiles(fs, 'scene.voxel.json', octree);

        const jsonBytes = fs.results.get('scene.voxel.json');
        assert.ok(jsonBytes, 'metadata file should be written');
        const metadata = JSON.parse(new TextDecoder().decode(jsonBytes));
        assert.strictEqual(metadata.nodeCount, 2);
        assert.strictEqual(metadata.leafDataCount, 2);
        assert.deepStrictEqual(metadata.gridBounds.min, [0, 1, 2]);
        assert.deepStrictEqual(metadata.sceneBounds.max, [6, 7, 8]);

        const binBytes = fs.results.get('scene.voxel.bin');
        assert.ok(binBytes, 'binary file should be written');
        assert.strictEqual(binBytes.byteLength, 16);
        assert.deepStrictEqual([...binBytes], [
            0x44, 0x33, 0x22, 0x11,
            0xDD, 0xCC, 0xBB, 0xAA,
            0x04, 0x03, 0x02, 0x01,
            0xCC, 0xDD, 0xEE, 0xFF
        ]);
    });
});

describe('writeVoxel collisionMesh validation', function () {
    // A dummy createDevice is enough: collisionMesh validation runs before
    // the device is created, so it is never invoked.
    const dummyCreateDevice = async () => ({});

    const run = (collisionMesh) => writeVoxel({
        filename: 'scene.voxel.json',
        dataTable: new DataTable([new Column('x', new Float32Array(1))]),
        createDevice: dummyCreateDevice,
        collisionMesh
    }, new MemoryFileSystem());

    it('rejects invalid collisionMesh values listing all supported shapes', async function () {
        await assert.rejects(
            run('invalid'),
            /^Error: Invalid collisionMesh value: invalid\. Expected true, false, "smooth", "faces", "voxel", or "tris"$/
        );
    });

    it('accepts voxel and tris and requires color columns for them', async function () {
        for (const shape of ['voxel', 'tris']) {
            await assert.rejects(
                run(shape),
                (err) => {
                    assert.match(err.message, /^writeVoxel: missing required column\(s\): /);
                    assert.match(err.message, /f_dc_0/);
                    assert.match(err.message, /f_dc_1/);
                    assert.match(err.message, /f_dc_2/);
                    return true;
                },
                `shape '${shape}' should require color columns`
            );
        }
    });

    it('rejects invalid collisionColorPalette values', async function () {
        const runPalette = collisionColorPalette => writeVoxel({
            filename: 'scene.voxel.json',
            dataTable: new DataTable([new Column('x', new Float32Array(1))]),
            createDevice: dummyCreateDevice,
            collisionMesh: 'voxel',
            collisionColorPalette
        }, new MemoryFileSystem());

        await assert.rejects(runPalette([]), /at least one colour/, 'an empty colour list is not a palette');
        await assert.rejects(runPalette(['#3243aa', 'nothex']), /Invalid palette colour: nothex/);
        await assert.rejects(runPalette(0), /must be an integer >= 1 or a list of hex colours/);
        await assert.rejects(runPalette(2.5), /must be an integer >= 1 or a list of hex colours/);

        // a well-formed palette gets past validation and on to the missing
        // colour columns for the 'voxel' shape
        await assert.rejects(runPalette(['#3243aa', '4444ff']), /missing required column/);
        await assert.rejects(runPalette(8), /missing required column/);
    });

    it('does not require color columns for uncolored shapes', async function () {
        for (const shape of ['smooth', 'faces']) {
            await assert.rejects(
                run(shape),
                (err) => {
                    assert.match(err.message, /^writeVoxel: missing required column\(s\): /);
                    assert.doesNotMatch(err.message, /f_dc/);
                    return true;
                },
                `shape '${shape}' should not require color columns`
            );
        }
    });
});
