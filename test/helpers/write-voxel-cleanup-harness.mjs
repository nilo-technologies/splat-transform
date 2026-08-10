/**
 * Subprocess harness for the writeVoxel cleanup behaviour tests.
 *
 * Runs writeVoxel end-to-end with a real Dawn device over a small slab scene,
 * in four option configurations, and writes the emitted files (base64) plus
 * the close-mode error to a JSON file whose path is argv[2].
 *
 * This must be a subprocess: the cleanup phase sits after GPU voxelization,
 * and Dawn's node binding keeps the event loop alive natively after a device
 * session with nothing public to unref, so the only way out is a force exit
 * -- the same reason the CLI calls exit(0) (src/cli/index.ts). A test file
 * cannot force-exit without abandoning the remaining suites, so it spawns
 * this script instead.
 */

import { writeFile } from 'node:fs/promises';

import { Column, DataTable } from '../../src/lib/index.js';
import { MemoryFileSystem } from '../../src/lib/io/write/index.js';
import { writeVoxel } from '../../src/lib/writers/write-voxel.js';
import { createDevice } from '../../src/cli/node-device.js';

/**
 * A thick slab of splats plus a small satellite cluster well away from it.
 * The slab voxelizes into a solid blob; the satellite (a few dozen voxels,
 * under the despeckle minimum of one 4x4x4 block) must not survive cleanup,
 * which changes the octree's block-level structure and not just per-voxel
 * occupancy.
 *
 * @returns {DataTable} The scene.
 */
const makeSlab = () => {
    const centers = [];
    for (let ix = 0; ix < 10; ix++) {
        for (let iy = 0; iy < 4; iy++) {
            for (let iz = 0; iz < 10; iz++) {
                centers.push([ix * 0.03, iy * 0.03, iz * 0.03]);
            }
        }
    }
    // Satellite: 2x2x2 splats ~0.75m from the slab -- no field overlap at
    // sigma 0.02, so it voxelizes into its own disconnected component.
    for (let ix = 0; ix < 2; ix++) {
        for (let iy = 0; iy < 2; iy++) {
            for (let iz = 0; iz < 2; iz++) {
                centers.push([1.5 + ix * 0.03, iy * 0.03, 1.5 + iz * 0.03]);
            }
        }
    }
    const col = fn => Float32Array.from({ length: centers.length }, (_, i) => fn(i));
    return new DataTable([
        new Column('x', col(i => centers[i][0])),
        new Column('y', col(i => centers[i][1])),
        new Column('z', col(i => centers[i][2])),
        new Column('rot_0', col(() => 1)),
        new Column('rot_1', col(() => 0)),
        new Column('rot_2', col(() => 0)),
        new Column('rot_3', col(() => 0)),
        new Column('scale_0', col(() => Math.log(0.02))),
        new Column('scale_1', col(() => Math.log(0.02))),
        new Column('scale_2', col(() => Math.log(0.02))),
        new Column('opacity', col(() => 4))
    ]);
};

const main = async () => {
    const outPath = process.argv[2];
    const out = {};

    let device;
    try {
        device = await createDevice();
    } catch {
        // No WebGPU runtime on this machine: report unavailability and let
        // the test file skip rather than fail.
        out.unavailable = true;
        await writeFile(outPath, JSON.stringify(out));
        process.exit(0);
    }

    const run = async (options) => {
        const fs = new MemoryFileSystem();
        await writeVoxel({
            filename: 'scene.voxel.json',
            dataTable: makeSlab(),
            createDevice: async () => device,
            ...options
        }, fs);
        return {
            bin: Buffer.from(fs.results.get('scene.voxel.bin')).toString('base64'),
            meta: JSON.parse(new TextDecoder().decode(fs.results.get('scene.voxel.json')))
        };
    };

    out.plain = await run({});
    out.disabled = await run({ voxelCleanup: 0 });
    out.cleaned = await run({ voxelCleanup: 0.1 });

    try {
        await run({ voxelCleanup: 0.1, voxelCleanupFill: 'close' });
        out.closeError = null;
    } catch (e) {
        out.closeError = e.message;
    }

    device.destroy();

    await writeFile(outPath, JSON.stringify(out));
    process.exit(0);
};

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
