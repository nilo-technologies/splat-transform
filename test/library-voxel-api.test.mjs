import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

// Deliberately imports only from the package root, the way a library consumer
// would. If any of the voxel-generation pieces stop being exported, this fails.
import {
    assertVoxFits,
    buildCollisionMesh,
    buildCollisionVox,
    Column,
    computeGaussianExtents,
    countVoxModels,
    DataTable,
    downsampleGrid,
    enumerateOccupied,
    forEachExposedFace,
    GaussianBVH,
    MAX_VOX_DIM,
    MAX_VOX_MODELS,
    minVoxFactorForModels,
    SparseVoxelGrid,
    voxelFaces
} from '../src/lib/index.js';

const SH_C0 = 0.28209479177387814;
const packColor = c => (c - 0.5) / SH_C0;

/**
 * A handful of splats spanning a small box, as a DataTable.
 *
 * @returns {DataTable} Splat table with position, colour and opacity columns.
 */
const makeSplats = () => {
    const centers = [
        [1, 1, 1], [6, 1, 1], [1, 6, 1], [1, 1, 6], [6, 6, 6]
    ];
    const colors = [
        [0.9, 0.2, 0.2], [0.2, 0.9, 0.2], [0.2, 0.2, 0.9],
        [0.9, 0.9, 0.2], [0.5, 0.5, 0.5]
    ];
    const n = centers.length;
    const col = (fn) => new Float32Array(Array.from({ length: n }, (_, i) => fn(i)));
    return new DataTable([
        new Column('x', col(i => centers[i][0])),
        new Column('y', col(i => centers[i][1])),
        new Column('z', col(i => centers[i][2])),
        new Column('scale_0', col(() => Math.log(0.6))),
        new Column('scale_1', col(() => Math.log(0.6))),
        new Column('scale_2', col(() => Math.log(0.6))),
        new Column('rot_0', col(() => 1)),
        new Column('rot_1', col(() => 0)),
        new Column('rot_2', col(() => 0)),
        new Column('rot_3', col(() => 0)),
        new Column('f_dc_0', col(i => packColor(colors[i][0]))),
        new Column('f_dc_1', col(i => packColor(colors[i][1]))),
        new Column('f_dc_2', col(i => packColor(colors[i][2]))),
        new Column('opacity', col(() => 3))
    ]);
};

/**
 * Build the colour source `buildCollisionVox` and `buildCollisionMesh` expect.
 *
 * @param {DataTable} splats - Splat table.
 * @param {object} extra - Extra colour options.
 * @returns {object} Colour source.
 */
const makeColorSource = (splats, extra = {}) => {
    const { extents } = computeGaussianExtents(splats, 0.1);
    return {
        bvh: new GaussianBVH(splats, extents),
        columns: {
            f_dc_0: splats.getColumnByName('f_dc_0').data,
            f_dc_1: splats.getColumnByName('f_dc_1').data,
            f_dc_2: splats.getColumnByName('f_dc_2').data,
            opacity: splats.getColumnByName('opacity').data
        },
        mode: 'solid',
        ...extra
    };
};

/**
 * A solid box of voxels inside a grid.
 *
 * @param {number} nx - Grid X size (multiple of 4).
 * @param {number} ny - Grid Y size (multiple of 4).
 * @param {number} nz - Grid Z size (multiple of 4).
 * @returns {SparseVoxelGrid} Filled grid.
 */
const solidBox = (nx, ny, nz) => {
    const grid = new SparseVoxelGrid(nx, ny, nz);
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) grid.setVoxel(x, y, z);
        }
    }
    return grid;
};

const boundsOf = (nx, ny, nz, res) => ({
    min: new Vec3(0, 0, 0),
    max: new Vec3(nx * res, ny * res, nz * res)
});

describe('library voxel API', () => {
    it('exports everything the documented flow needs', () => {
        for (const [name, value] of Object.entries({
            SparseVoxelGrid,
            GaussianBVH,
            computeGaussianExtents,
            buildCollisionVox,
            buildCollisionMesh,
            enumerateOccupied,
            countVoxModels,
            minVoxFactorForModels,
            assertVoxFits,
            downsampleGrid,
            voxelFaces,
            forEachExposedFace
        })) {
            assert.strictEqual(typeof value, 'function', `${name} should be exported`);
        }
        assert.strictEqual(MAX_VOX_DIM, 256);
        assert.strictEqual(MAX_VOX_MODELS, 256);
    });

    it('builds a .vox straight from a grid, with no file system involved', () => {
        // mirrors the README's "Voxel and Collision Generation" snippet exactly:
        // 16^3 grid, 0.5 voxels, bounds 0..8, solid mode, 64-colour palette
        const res = 0.5;
        const grid = solidBox(16, 16, 16);
        const splats = makeSplats();

        assert.deepStrictEqual(
            [boundsOf(16, 16, 16, res).max.x, boundsOf(16, 16, 16, res).max.y, boundsOf(16, 16, 16, res).max.z],
            [8, 8, 8], 'README quotes max (8, 8, 8) for this grid and resolution');

        const bytes = buildCollisionVox(
            grid, boundsOf(16, 16, 16, res), res,
            makeColorSource(splats, { palette: 64 })
        );

        assert.ok(bytes instanceof Uint8Array, 'should return raw bytes');
        assert.strictEqual(String.fromCharCode(...bytes.subarray(0, 4)), 'VOX ');
        assert.ok(bytes.length > 0);
    });

    it('builds a collision mesh GLB straight from a grid', () => {
        const res = 0.5;
        const grid = solidBox(16, 16, 16);
        const splats = makeSplats();

        const glb = buildCollisionMesh(
            grid, boundsOf(16, 16, 16, res), res, 'voxel',
            makeColorSource(splats, { palette: 64, flatShade: true })
        );

        assert.ok(glb instanceof Uint8Array);
        assert.strictEqual(String.fromCharCode(...glb.subarray(0, 4)), 'glTF');
    });

    it('lets a caller check the .vox limits before doing any colouring', () => {
        const grid = solidBox(16, 16, 16);
        const occupied = enumerateOccupied(grid);

        assert.strictEqual(occupied.count, 16 ** 3);
        assert.strictEqual(occupied.maxIx - occupied.minIx + 1, 16);
        assert.strictEqual(countVoxModels(occupied, 1), 1);
        assert.doesNotThrow(() => assertVoxFits(occupied, 0.5, 1));
        assert.strictEqual(minVoxFactorForModels(occupied), 1);
    });

    it('reports the reduction a too-large region needs, without building it', () => {
        const grid = new SparseVoxelGrid(256 * 300, 4, 4);
        for (let t = 0; t <= MAX_VOX_MODELS + 4; t++) grid.setVoxel(t * 256, 0, 0);
        const occupied = enumerateOccupied(grid);

        assert.ok(countVoxModels(occupied, 1) > MAX_VOX_MODELS);
        const factor = minVoxFactorForModels(occupied);
        assert.ok(factor > 1);
        assert.ok(countVoxModels(occupied, factor) <= MAX_VOX_MODELS);
        assert.throws(() => assertVoxFits(occupied, 0.01, 1), /--collision-voxels-size/);
    });

    it('coarsens a grid for the model while leaving the original alone', () => {
        const res = 0.25;
        const grid = solidBox(16, 16, 16);
        const before = enumerateOccupied(grid).count;

        const plan = downsampleGrid(grid, boundsOf(16, 16, 16, res), res, 4);

        assert.strictEqual(plan.voxelResolution, 1);
        assert.strictEqual(enumerateOccupied(plan.grid).count, 4 ** 3);
        assert.strictEqual(enumerateOccupied(grid).count, before,
            'the source grid must not be modified');
    });

    it('exposes exposed-face iteration for custom meshing', () => {
        const grid = solidBox(4, 4, 4);
        let faces = 0;
        const buckets = new Set();
        forEachExposedFace(grid, (x, y, z, bucket) => {
            faces++;
            buckets.add(bucket);
        });
        assert.strictEqual(faces, 6 * 16, 'a solid 4x4x4 block has 96 exposed faces');
        assert.strictEqual(buckets.size, 6, 'all six face directions should appear');
    });

    it('extracts a voxel-face mesh without any colour source', () => {
        const res = 0.5;
        const mesh = voxelFaces(solidBox(4, 4, 4), boundsOf(4, 4, 4, res), res, { perVoxel: true });
        assert.strictEqual(mesh.indices.length / 3, 96 * 2);
        assert.strictEqual(mesh.positions.length % 3, 0);
    });
});
