import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Vec3 } from 'playcanvas';

import { Column, DataTable } from '../src/lib/index.js';
import { GaussianBVH } from '../src/lib/spatial/index.js';
import { BlockMaskBuffer } from '../src/lib/voxel/block-mask-buffer.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';
import { marchingCubes } from '../src/lib/mesh/marching-cubes.js';
import { coplanarMerge } from '../src/lib/mesh/coplanar-merge.js';
import { voxelFaces } from '../src/lib/mesh/voxel-faces.js';
import { buildCollisionMesh } from '../src/lib/writers/collision-glb.js';
import { assertVoxFits, buildCollisionVox, downsampleGrid, enumerateOccupied, minVoxFactor, minVoxelSizeForVox, voxFitsAt } from '../src/lib/writers/collision-vox.js';

// Linear block index: bx + by*nbx + bz*nbx*nby. The buffer stores blocks
// keyed on this linear index now (not morton).
function linearBlockIdx(bx, by, bz, nbx, nby) {
    return bx + by * nbx + bz * nbx * nby;
}

// Convert a BlockMaskBuffer to a SparseVoxelGrid for the new marchingCubes API.
// nx/ny/nz are the voxel grid dimensions (must match the bounds passed to
// marchingCubes).
function toGrid(buffer, nx, ny, nz) {
    return SparseVoxelGrid.fromBuffer(buffer, nx, ny, nz);
}

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

const makeGridBounds = (minX, minY, minZ, maxX, maxY, maxZ) => ({
    min: new Vec3(minX, minY, minZ),
    max: new Vec3(maxX, maxY, maxZ)
});

describe('marchingCubes', () => {
    it('should return empty mesh for empty buffer', () => {
        const buffer = new BlockMaskBuffer();
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);

        assert.strictEqual(mesh.positions.length, 0);
        assert.strictEqual(mesh.indices.length, 0);
    });

    it('should generate triangles for a single solid block', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);

        assert.ok(mesh.positions.length > 0, 'should have vertices');
        assert.ok(mesh.indices.length > 0, 'should have indices');
        assert.strictEqual(mesh.indices.length % 3, 0, 'indices should be multiple of 3');
        assert.strictEqual(mesh.positions.length % 3, 0, 'positions should be multiple of 3');
    });

    it('should generate a closed surface for a solid cube', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);

        const numTriangles = mesh.indices.length / 3;
        // A 4x4x4 solid cube produces boundary triangles on all 6 faces.
        assert.strictEqual(numTriangles, 188,
            `expected 188 triangles for solid 4x4x4 cube, got ${numTriangles}`);
    });

    it('should pre-merge exact flat face cells when requested', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const grid = toGrid(buffer, 4, 4, 4);
        const raw = marchingCubes(grid, bounds, 1.0);
        const fast = marchingCubes(grid, bounds, 1.0, { mergeFlatFaces: true });

        assert.strictEqual(raw.indices.length / 3, 188,
            'default marchingCubes output should remain the raw MC mesh');
        assert.ok(fast.indices.length < raw.indices.length,
            `expected flat-face pre-merge to reduce triangles; raw=${raw.indices.length / 3}, fast=${fast.indices.length / 3}`);

        const rawMerged = coplanarMerge(raw, 1.0);
        const fastMerged = coplanarMerge(fast, 1.0);
        const rawStats = meshStats(rawMerged);
        const fastStats = meshStats(fastMerged);

        assert.strictEqual(fastStats.tris, rawStats.tris,
            'final coplanar merge should reach the same triangle count');
        assert.strictEqual(fastStats.verts, rawStats.verts,
            'final coplanar merge should reach the same vertex count');
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(fastStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${fastStats.min[a]}`);
            assert.strictEqual(fastStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${fastStats.max[a]}`);
        }
    });

    it('should merge straight binary-MC bevel strips during extraction', () => {
        const nx = 8;
        const nz = 8;
        const nbx = nx / 4;
        const nby = 1;
        const nbz = nz / 4;
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;
        const buffer = new BlockMaskBuffer();
        for (let bz = 0; bz < nbz; bz++) {
            for (let bx = 0; bx < nbx; bx++) {
                buffer.addBlock(linearBlockIdx(bx, 0, bz, nbx, nby), slabLo, slabHi);
            }
        }

        const bounds = makeGridBounds(0, 0, 0, nx, 4, nz);
        const grid = toGrid(buffer, nx, 4, nz);
        const raw = marchingCubes(grid, bounds, 1.0);
        const fast = marchingCubes(grid, bounds, 1.0, { mergeFlatFaces: true });
        const rawMerged = coplanarMerge(raw, 1.0);

        const fastStats = meshStats(fast);
        const rawMergedStats = meshStats(rawMerged);
        assert.ok(fastStats.tris < raw.indices.length / 3 * 0.2,
            `expected direct MC merge to remove most slab tris; raw=${raw.indices.length / 3}, fast=${fastStats.tris}`);
        assert.strictEqual(fastStats.tris, rawMergedStats.tris);
        assert.strictEqual(fastStats.verts, rawMergedStats.verts);
        assertClosedTriangleEdges(fast);
    });

    it('should keep merged MC rectangles watertight next to raw feature triangles', () => {
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;
        const bumpLo = ((1 << 4) | (1 << 8) | (1 << 12)) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(0, 0, 1, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 1, 2, 1), (slabLo | bumpLo) >>> 0, slabHi);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 8);
        const grid = toGrid(buffer, 8, 4, 8);
        const raw = marchingCubes(grid, bounds, 1.0);
        const fast = marchingCubes(grid, bounds, 1.0, { mergeFlatFaces: true });
        const rawMerged = coplanarMerge(raw, 1.0);
        const fastMerged = coplanarMerge(fast, 1.0);

        assert.ok(fast.indices.length < raw.indices.length * 0.25,
            `expected direct MC merge to shrink slab+bump mesh; raw=${raw.indices.length / 3}, fast=${fast.indices.length / 3}`);
        assertClosedTriangleEdges(fast);
        assert.strictEqual(fastMerged.indices.length, rawMerged.indices.length);
        assert.strictEqual(fastMerged.positions.length, rawMerged.positions.length);
    });

    it('should place vertices within grid bounds', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const res = 0.5;
        const bounds = makeGridBounds(0, 0, 0, 2, 2, 2);
        // bounds 2x2x2 / res 0.5 = 4x4x4 voxels = 1x1x1 blocks
        const mesh = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, res);

        for (let i = 0; i < mesh.positions.length; i += 3) {
            const x = mesh.positions[i];
            const y = mesh.positions[i + 1];
            const z = mesh.positions[i + 2];
            assert.ok(x >= -res && x <= 2 + res, `x=${x} out of range`);
            assert.ok(y >= -res && y <= 2 + res, `y=${y} out of range`);
            assert.ok(z >= -res && z <= 2 + res, `z=${z} out of range`);
        }
    });

    it('should produce more triangles for multiple blocks than one', () => {
        // bounds 8x4x4 / res 1.0 = 8x4x4 voxels = 2x1x1 blocks
        const buffer1 = new BlockMaskBuffer();
        buffer1.addBlock(linearBlockIdx(0, 0, 0, 2, 1), SOLID_LO, SOLID_HI);
        const mesh1 = marchingCubes(toGrid(buffer1, 8, 4, 4), makeGridBounds(0, 0, 0, 8, 4, 4), 1.0);

        const buffer2 = new BlockMaskBuffer();
        buffer2.addBlock(linearBlockIdx(0, 0, 0, 2, 1), SOLID_LO, SOLID_HI);
        buffer2.addBlock(linearBlockIdx(1, 0, 0, 2, 1), SOLID_LO, SOLID_HI);
        const mesh2 = marchingCubes(toGrid(buffer2, 8, 4, 4), makeGridBounds(0, 0, 0, 8, 4, 4), 1.0);

        // Two adjacent blocks form an 8x4x4 solid. The shared face has no
        // boundary, so the total triangle count is less than 2x a single block.
        assert.ok(mesh2.indices.length > mesh1.indices.length,
            'two adjacent blocks should produce more triangles than one');
        assert.ok(mesh2.indices.length < mesh1.indices.length * 2,
            'adjacent blocks should share the internal face');
    });

    it('should handle a single-voxel mixed block', () => {
        const buffer = new BlockMaskBuffer();
        // Set only voxel (0,0,0): bitIdx = 0 + 0*4 + 0*16 = 0 → lo bit 0
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 1, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);

        assert.ok(mesh.positions.length > 0, 'should have vertices for single voxel');
        assert.ok(mesh.indices.length > 0, 'should have triangles for single voxel');
        const numTriangles = mesh.indices.length / 3;
        // A single isolated voxel produces triangles for each exposed face.
        // Marching cubes with binary fields may triangulate corners differently.
        assert.ok(numTriangles >= 6 && numTriangles <= 12,
            `single voxel should produce 6-12 triangles, got ${numTriangles}`);
    });

    it('should handle non-unit voxel resolution', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const res = 0.25;
        const bounds = makeGridBounds(0, 0, 0, 1, 1, 1);
        // bounds 1x1x1 / res 0.25 = 4x4x4 voxels = 1x1x1 blocks
        const mesh = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, res);

        assert.ok(mesh.positions.length > 0);

        for (let i = 0; i < mesh.positions.length; i += 3) {
            assert.ok(mesh.positions[i] >= -res && mesh.positions[i] <= 1 + res);
            assert.ok(mesh.positions[i + 1] >= -res && mesh.positions[i + 1] <= 1 + res);
            assert.ok(mesh.positions[i + 2] >= -res && mesh.positions[i + 2] <= 1 + res);
        }
    });
});

/**
 * Compute triangle count, vertex count, and AABB for a mesh.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - The mesh
 *   to scan.
 * @returns {{ tris: number, verts: number, min: number[], max: number[] }}
 *   Triangle count, vertex count, and AABB extremes.
 */
const meshStats = (mesh) => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < mesh.positions.length; i += 3) {
        for (let a = 0; a < 3; a++) {
            const v = mesh.positions[i + a];
            if (v < min[a]) min[a] = v;
            if (v > max[a]) max[a] = v;
        }
    }
    return {
        tris: mesh.indices.length / 3,
        verts: mesh.positions.length / 3,
        min,
        max
    };
};

/**
 * Count triangles whose face normal is not aligned with any cardinal axis
 * (i.e. bevel / corner-cutting triangles produced by marching cubes).
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - The mesh
 *   to scan.
 * @returns {number} Count of bevel triangles.
 */
const countBevelTris = (mesh) => {
    const { positions, indices } = mesh;
    let count = 0;
    for (let i = 0; i < indices.length; i += 3) {
        const ia = indices[i] * 3;
        const ib = indices[i + 1] * 3;
        const ic = indices[i + 2] * 3;
        const ex = positions[ib] - positions[ia];
        const ey = positions[ib + 1] - positions[ia + 1];
        const ez = positions[ib + 2] - positions[ia + 2];
        const fx = positions[ic] - positions[ia];
        const fy = positions[ic + 1] - positions[ia + 1];
        const fz = positions[ic + 2] - positions[ia + 2];
        const nx = ey * fz - ez * fy;
        const ny = ez * fx - ex * fz;
        const nz = ex * fy - ey * fx;
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len < 1e-6) continue;
        const ax = Math.abs(nx) / len;
        const ay = Math.abs(ny) / len;
        const az = Math.abs(nz) / len;
        const m = Math.max(ax, ay, az);
        if (m < 1 - 1e-3) count++;
    }
    return count;
};

/**
 * Assert every triangle edge has exactly two incident triangles.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - The mesh
 *   to scan.
 */
const assertClosedTriangleEdges = (mesh) => {
    const edgeCount = new Map();
    const indices = mesh.indices;
    for (let i = 0; i < indices.length; i += 3) {
        const a = indices[i];
        const b = indices[i + 1];
        const c = indices[i + 2];
        const addEdge = (u, v) => {
            const key = u < v ? `${u},${v}` : `${v},${u}`;
            edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
        };
        addEdge(a, b);
        addEdge(b, c);
        addEdge(c, a);
    }
    for (const [key, count] of edgeCount) {
        assert.strictEqual(count, 2,
            `edge ${key} has ${count} incident tris (T-junction, boundary, or non-manifold edge)`);
    }
};

/**
 * Assert every vertex lands exactly on a voxel-grid corner.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - The mesh
 *   to scan.
 * @param {{ min: Vec3, max: Vec3 }} bounds - Grid bounds.
 * @param {number} voxelResolution - Voxel resolution.
 */
const assertVoxelGridCorners = (mesh, bounds, voxelResolution) => {
    for (let i = 0; i < mesh.positions.length; i += 3) {
        const x = (mesh.positions[i] - bounds.min.x) / voxelResolution;
        const y = (mesh.positions[i + 1] - bounds.min.y) / voxelResolution;
        const z = (mesh.positions[i + 2] - bounds.min.z) / voxelResolution;
        assert.ok(Math.abs(x - Math.round(x)) < 1e-6,
            `x=${mesh.positions[i]} is not on a voxel grid corner`);
        assert.ok(Math.abs(y - Math.round(y)) < 1e-6,
            `y=${mesh.positions[i + 1]} is not on a voxel grid corner`);
        assert.ok(Math.abs(z - Math.round(z)) < 1e-6,
            `z=${mesh.positions[i + 2]} is not on a voxel grid corner`);
    }
};

describe('voxelFaces', () => {
    it('should return an empty mesh for an empty grid', () => {
        const buffer = new BlockMaskBuffer();
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = voxelFaces(toGrid(buffer, 4, 4, 4), bounds, 1.0);

        assert.strictEqual(mesh.positions.length, 0);
        assert.strictEqual(mesh.indices.length, 0);
    });

    it('should mesh a solid block as shared voxel-boundary faces', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = voxelFaces(toGrid(buffer, 4, 4, 4), bounds, 1.0);
        const stats = meshStats(mesh);

        assert.strictEqual(stats.verts, 8);
        assert.strictEqual(stats.tris, 12);
        assert.deepStrictEqual(stats.min, [0, 0, 0]);
        assert.deepStrictEqual(stats.max, [4, 4, 4]);
        assertClosedTriangleEdges(mesh);
    });

    it('should not emit faces between adjacent solid blocks', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 2, 1), SOLID_LO, SOLID_HI);
        buffer.addBlock(linearBlockIdx(1, 0, 0, 2, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 4);
        const mesh = voxelFaces(toGrid(buffer, 8, 4, 4), bounds, 1.0);
        const stats = meshStats(mesh);

        assert.strictEqual(stats.verts, 8);
        assert.strictEqual(stats.tris, 12);
        assert.deepStrictEqual(stats.min, [0, 0, 0]);
        assert.deepStrictEqual(stats.max, [8, 4, 4]);
        for (let i = 0; i < mesh.positions.length; i += 3) {
            assert.notStrictEqual(mesh.positions[i], 4,
                'shared block boundary at x=4 should not be emitted as surface geometry');
        }
        assertClosedTriangleEdges(mesh);
    });

    it('should mesh a single mixed-block voxel on voxel boundaries', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 1, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = voxelFaces(toGrid(buffer, 4, 4, 4), bounds, 1.0);
        const stats = meshStats(mesh);

        assert.strictEqual(stats.verts, 8);
        assert.strictEqual(stats.tris, 12);
        assert.deepStrictEqual(stats.min, [0, 0, 0]);
        assert.deepStrictEqual(stats.max, [1, 1, 1]);
        assertClosedTriangleEdges(mesh);
    });

    it('should split greedy rectangle edges to avoid T-junctions', () => {
        // One-voxel-thick L shape on z=0. The +Z plane greedily contains
        // a 3x1 rectangle adjacent to a 1x1 rectangle along only part of
        // the larger rectangle's edge; a naive greedy mesh leaves a
        // T-junction there.
        const lo = (
            (1 << 0) | // (0,0,0)
            (1 << 1) | // (1,0,0)
            (1 << 2) | // (2,0,0)
            (1 << 4)   // (0,1,0)
        ) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), lo, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = voxelFaces(toGrid(buffer, 4, 4, 4), bounds, 1.0);

        assert.ok(mesh.indices.length > 0);
        assertVoxelGridCorners(mesh, bounds, 1.0);
        assertClosedTriangleEdges(mesh);
    });
});

/**
 * Brute-force count of voxel faces exposed to empty space (or to outside the
 * grid), which is exactly the quad count `perVoxel` must emit.
 *
 * @param {SparseVoxelGrid} grid - Grid to scan.
 * @returns {number} Exposed face count.
 */
const countExposedFaces = (grid) => {
    const neighbours = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]];
    let n = 0;
    for (let iz = 0; iz < grid.nz; iz++) {
        for (let iy = 0; iy < grid.ny; iy++) {
            for (let ix = 0; ix < grid.nx; ix++) {
                if (!grid.getVoxel(ix, iy, iz)) continue;
                for (const [dx, dy, dz] of neighbours) {
                    const x = ix + dx, y = iy + dy, z = iz + dz;
                    const inside = x >= 0 && y >= 0 && z >= 0 &&
                        x < grid.nx && y < grid.ny && z < grid.nz;
                    if (!inside || !grid.getVoxel(x, y, z)) n++;
                }
            }
        }
    }
    return n;
};

/**
 * Assert the `perVoxel` output contract: triangles come in consecutive pairs,
 * each pair spans exactly 4 distinct vertices forming an axis-aligned unit
 * square, and both triangles of a pair share the same outward normal.
 *
 * `buildCollisionMesh` relies on the consecutive pairing to assign one flat
 * colour per voxel quad without building an edge map.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - Mesh to check.
 * @param {number} voxelResolution - Expected quad edge length.
 */
const assertPerVoxelQuads = (mesh, voxelResolution) => {
    const numTris = mesh.indices.length / 3;
    assert.strictEqual(numTris % 2, 0, 'perVoxel should emit an even number of triangles');

    const pos = (v, a) => mesh.positions[v * 3 + a];
    const normalOf = (t) => {
        const [a, b, c] = [0, 1, 2].map(k => mesh.indices[t * 3 + k]);
        const e = [0, 1, 2].map(k => pos(b, k) - pos(a, k));
        const f = [0, 1, 2].map(k => pos(c, k) - pos(a, k));
        return [
            e[1] * f[2] - e[2] * f[1],
            e[2] * f[0] - e[0] * f[2],
            e[0] * f[1] - e[1] * f[0]
        ];
    };

    for (let q = 0; q < numTris / 2; q++) {
        const t0 = q * 2;
        const t1 = q * 2 + 1;
        const verts = new Set();
        for (let k = 0; k < 6; k++) verts.add(mesh.indices[t0 * 3 + k]);
        assert.strictEqual(verts.size, 4,
            `quad ${q} spans ${verts.size} distinct vertices, expected 4`);

        // the 4 corners must be flat on one axis and unit-sized on the other two
        const list = [...verts];
        const extent = [0, 1, 2].map((a) => {
            const vals = list.map(v => pos(v, a));
            return Math.max(...vals) - Math.min(...vals);
        });
        const flatAxes = extent.filter(e => e < 1e-6).length;
        assert.strictEqual(flatAxes, 1, `quad ${q} is not planar on a single axis: ${extent}`);
        for (const e of extent) {
            assert.ok(e < 1e-6 || Math.abs(e - voxelResolution) < 1e-6,
                `quad ${q} edge ${e} is not ${voxelResolution}`);
        }

        // both triangles wind the same way
        const n0 = normalOf(t0);
        const n1 = normalOf(t1);
        const dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
        assert.ok(dot > 0, `quad ${q} triangles have opposing winding`);
    }
};

/**
 * Assert no emitted quad lies in the given axis-aligned plane.
 *
 * Unlike the merged path, `perVoxel` legitimately places vertices at interior
 * block boundaries (every quad is 1x1, so corners land on every grid line), so
 * only whole faces in the plane indicate a wrongly emitted interior surface.
 *
 * @param {{ positions: Float32Array, indices: Uint32Array }} mesh - Mesh to check.
 * @param {number} axis - Axis index the plane is normal to.
 * @param {number} value - Coordinate of the plane along `axis`.
 */
const assertNoQuadInPlane = (mesh, axis, value) => {
    for (let t = 0; t < mesh.indices.length / 3; t++) {
        let all = true;
        for (let k = 0; k < 3; k++) {
            if (mesh.positions[mesh.indices[t * 3 + k] * 3 + axis] !== value) {
                all = false;
                break;
            }
        }
        assert.ok(!all,
            `triangle ${t} lies in the interior plane axis${axis}=${value}`);
    }
};

describe('voxelFaces perVoxel', () => {
    it('should return an empty mesh for an empty grid', () => {
        const buffer = new BlockMaskBuffer();
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = voxelFaces(toGrid(buffer, 4, 4, 4), bounds, 1.0, { perVoxel: true });

        assert.strictEqual(mesh.positions.length, 0);
        assert.strictEqual(mesh.indices.length, 0);
    });

    it('should emit one unit quad per exposed face of a solid block', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const grid = toGrid(buffer, 4, 4, 4);
        const mesh = voxelFaces(grid, bounds, 1.0, { perVoxel: true });
        const stats = meshStats(mesh);

        // 6 sides of a 4x4x4 cube, 16 faces each
        assert.strictEqual(countExposedFaces(grid), 96);
        assert.strictEqual(stats.tris, 96 * 2);
        assert.deepStrictEqual(stats.min, [0, 0, 0]);
        assert.deepStrictEqual(stats.max, [4, 4, 4]);
        assertVoxelGridCorners(mesh, bounds, 1.0);
        assertPerVoxelQuads(mesh, 1.0);
        assertClosedTriangleEdges(mesh);
    });

    it('should deduplicate shared corner vertices', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const mesh = voxelFaces(toGrid(buffer, 4, 4, 4), bounds, 1.0, { perVoxel: true });

        // the surface of a 4x4x4 cube has 5^3 - 3^3 = 98 grid corners
        assert.strictEqual(meshStats(mesh).verts, 98);
    });

    it('should not emit faces between adjacent solid blocks', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 2, 1), SOLID_LO, SOLID_HI);
        buffer.addBlock(linearBlockIdx(1, 0, 0, 2, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 4);
        const grid = toGrid(buffer, 8, 4, 4);
        const mesh = voxelFaces(grid, bounds, 1.0, { perVoxel: true });

        assert.strictEqual(meshStats(mesh).tris, countExposedFaces(grid) * 2);
        assertNoQuadInPlane(mesh, 0, 4);
        assertPerVoxelQuads(mesh, 1.0);
        assertClosedTriangleEdges(mesh);
    });

    it('should match the exposed face count on a ragged mixed-block shape', () => {
        // face-connected L shape straddling a mixed and a solid block
        const lo = ((1 << 0) | (1 << 1) | (1 << 2) | (1 << 4) | (1 << 5)) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 2, 2), lo, 0);
        buffer.addBlock(linearBlockIdx(1, 1, 0, 2, 2), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 8, 8, 4);
        const grid = toGrid(buffer, 8, 8, 4);
        const mesh = voxelFaces(grid, bounds, 1.0, { perVoxel: true });

        assert.strictEqual(meshStats(mesh).tris, countExposedFaces(grid) * 2);
        assertVoxelGridCorners(mesh, bounds, 1.0);
        assertPerVoxelQuads(mesh, 1.0);
    });

    it('should honour a non-unit voxel resolution and grid origin', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 1, 0);

        const res = 0.25;
        const bounds = makeGridBounds(-1, 2, 3, -1 + 4 * res, 2 + 4 * res, 3 + 4 * res);
        const mesh = voxelFaces(toGrid(buffer, 4, 4, 4), bounds, res, { perVoxel: true });
        const stats = meshStats(mesh);

        assert.strictEqual(stats.tris, 12);
        assert.strictEqual(stats.verts, 8);
        assert.deepStrictEqual(stats.min, [-1, 2, 3]);
        assert.deepStrictEqual(stats.max, [-1 + res, 2 + res, 3 + res]);
        assertPerVoxelQuads(mesh, res);
    });

    it('should scale to a wide plate without losing the face count', () => {
        // exercises the growth paths of the vertex table and output buffers
        const n = 200;
        const nb = n / 4;
        const buffer = new BlockMaskBuffer();
        const slabLo = 0x000F000F >>> 0;
        for (let bz = 0; bz < nb; bz++) {
            for (let bx = 0; bx < nb; bx++) {
                buffer.addBlock(linearBlockIdx(bx, 0, bz, nb, 1), slabLo, 0);
            }
        }

        const bounds = makeGridBounds(0, 0, 0, n, 4, n);
        const grid = toGrid(buffer, n, 4, n);
        const mesh = voxelFaces(grid, bounds, 1.0, { perVoxel: true });

        assert.strictEqual(meshStats(mesh).tris, countExposedFaces(grid) * 2);
        assertPerVoxelQuads(mesh, 1.0);
        assertClosedTriangleEdges(mesh);
    });
});

describe('buildCollisionMesh', () => {
    it('should skip smooth GLB output for an empty grid', () => {
        const buffer = new BlockMaskBuffer();
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const bytes = buildCollisionMesh(toGrid(buffer, 4, 4, 4), bounds, 1.0, 'smooth');

        assert.strictEqual(bytes, null);
    });
});

const SH_C0 = 0.28209479177387814;
const packClr = c => (c - 0.5) / SH_C0;
const srgbToLinear = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

// Build a colorSource containing a single splat at (x, y, z) with the given
// AABB half-extent, display color and opacity logit.
const makeSingleSplatColorSource = (x, y, z, extent, color, logit) => {
    const dataTable = new DataTable([
        new Column('x', new Float32Array([x])),
        new Column('y', new Float32Array([y])),
        new Column('z', new Float32Array([z])),
        new Column('f_dc_0', new Float32Array([packClr(color[0])])),
        new Column('f_dc_1', new Float32Array([packClr(color[1])])),
        new Column('f_dc_2', new Float32Array([packClr(color[2])])),
        new Column('opacity', new Float32Array([logit]))
    ]);
    const extents = new DataTable([
        new Column('extent_x', new Float32Array([extent])),
        new Column('extent_y', new Float32Array([extent])),
        new Column('extent_z', new Float32Array([extent]))
    ]);
    return {
        bvh: new GaussianBVH(dataTable, extents),
        columns: {
            f_dc_0: dataTable.getColumnByName('f_dc_0').data,
            f_dc_1: dataTable.getColumnByName('f_dc_1').data,
            f_dc_2: dataTable.getColumnByName('f_dc_2').data,
            opacity: dataTable.getColumnByName('opacity').data
        },
        mode: 'average'
    };
};

// Build a colorSource from plain-object splats:
// { center, extent, color, logit }. Used by the coloring mode GLB tests.
const makeSplatColorSource = (splats, mode) => {
    const arr = f => new Float32Array(splats.map(f));
    const dataTable = new DataTable([
        new Column('x', arr(s => s.center[0])),
        new Column('y', arr(s => s.center[1])),
        new Column('z', arr(s => s.center[2])),
        new Column('f_dc_0', arr(s => packClr(s.color[0]))),
        new Column('f_dc_1', arr(s => packClr(s.color[1]))),
        new Column('f_dc_2', arr(s => packClr(s.color[2]))),
        new Column('opacity', arr(s => s.logit))
    ]);
    const extents = new DataTable([
        new Column('extent_x', arr(s => s.extent)),
        new Column('extent_y', arr(s => s.extent)),
        new Column('extent_z', arr(s => s.extent))
    ]);
    const columns = {};
    for (const name of ['f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity']) {
        columns[name] = dataTable.getColumnByName(name).data;
    }
    return { bvh: new GaussianBVH(dataTable, extents), columns, mode };
};

// Parse a GLB into its JSON chunk and BIN chunk bytes.
const parseGlb = (bytes) => {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const jsonLength = view.getUint32(12, true);
    const json = JSON.parse(new TextDecoder().decode(bytes.subarray(20, 20 + jsonLength)));
    const bin = bytes.subarray(20 + jsonLength + 8);
    return { json, bin };
};

const solidGrid = () => {
    const buffer = new BlockMaskBuffer();
    buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);
    return toGrid(buffer, 4, 4, 4);
};

describe('buildCollisionMesh vertex colors', () => {
    it('should emit no COLOR_0 or materials for faces and smooth shapes', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        for (const shape of ['faces', 'smooth']) {
            const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, shape);
            assert.ok(bytes, `${shape} should produce GLB output`);

            const { json } = parseGlb(bytes);
            const primitive = json.meshes[0].primitives[0];
            assert.ok(!('COLOR_0' in primitive.attributes),
                `${shape} should not declare COLOR_0`);
            assert.ok(!('material' in primitive),
                `${shape} should not reference a material`);
            assert.ok(!json.materials, `${shape} should not declare materials`);
            assert.strictEqual(json.accessors.length, 2,
                `${shape} should only have position and index accessors`);
            assert.strictEqual(json.bufferViews.length, 2,
                `${shape} should only have position and index bufferViews`);
        }
    });

    it('should color voxel vertices with the overlapping splat color', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const color = [0.8, 0.2, 0.5];
        // splat at the grid centre with an AABB covering the whole 4x4x4 grid
        const colorSource = makeSingleSplatColorSource(2, 2, 2, 4, color, 0);

        const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, 'voxel', colorSource);
        assert.ok(bytes, 'voxel should produce GLB output');

        const { json, bin } = parseGlb(bytes);

        const primitive = json.meshes[0].primitives[0];
        assert.strictEqual(primitive.attributes.COLOR_0, 2);
        assert.strictEqual(primitive.material, 0);
        assert.deepStrictEqual(json.materials, [{
            pbrMetallicRoughness: {
                baseColorFactor: [1, 1, 1, 1],
                metallicFactor: 0,
                roughnessFactor: 1
            },
            doubleSided: true
        }]);

        const colorAccessor = json.accessors[2];
        assert.strictEqual(colorAccessor.componentType, 5126, 'COLOR_0 must be FLOAT');
        assert.strictEqual(colorAccessor.type, 'VEC3');
        assert.strictEqual(colorAccessor.count, json.accessors[0].count,
            'COLOR_0 count must match POSITION count');
        assert.ok(!('min' in colorAccessor) && !('max' in colorAccessor),
            'COLOR_0 accessor must not declare min/max');

        const colorView = json.bufferViews[2];
        assert.strictEqual(colorView.target, 34962, 'COLOR_0 must target ARRAY_BUFFER');
        assert.strictEqual(colorView.byteOffset % 4, 0, 'COLOR_0 byteOffset must be 4-byte aligned');

        // every vertex overlaps only the single synthetic splat, so every
        // color must equal its linear-space display color
        const colors = new Float32Array(
            bin.buffer, bin.byteOffset + colorView.byteOffset, colorAccessor.count * 3);
        const expected = color.map(srgbToLinear);
        for (let i = 0; i < colorAccessor.count; i++) {
            for (let c = 0; c < 3; c++) {
                assert.ok(Math.abs(colors[i * 3 + c] - expected[c]) < 1e-4,
                    `vertex ${i} channel ${c}: expected ${expected[c]}, got ${colors[i * 3 + c]}`);
            }
        }
    });

    it('should declare COLOR_0 with finite colors for the tris shape', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const colorSource = makeSingleSplatColorSource(2, 2, 2, 4, [0.8, 0.2, 0.5], 0);

        const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, 'tris', colorSource);
        assert.ok(bytes, 'tris should produce GLB output');

        const { json, bin } = parseGlb(bytes);

        const primitive = json.meshes[0].primitives[0];
        assert.strictEqual(primitive.attributes.COLOR_0, 2);
        assert.strictEqual(primitive.material, 0);
        assert.ok(json.materials?.[0]?.doubleSided, 'material must be doubleSided');

        const colorAccessor = json.accessors[2];
        assert.strictEqual(colorAccessor.componentType, 5126);
        assert.strictEqual(colorAccessor.type, 'VEC3');
        assert.strictEqual(colorAccessor.count, json.accessors[0].count,
            'COLOR_0 count must match POSITION count');

        const colorView = json.bufferViews[2];
        const colors = new Float32Array(
            bin.buffer, bin.byteOffset + colorView.byteOffset, colorAccessor.count * 3);
        for (let i = 0; i < colors.length; i++) {
            assert.ok(Number.isFinite(colors[i]), `color component ${i} must be finite`);
            assert.ok(colors[i] >= 0 && colors[i] <= 1,
                `color component ${i}=${colors[i]} out of [0, 1] range`);
        }
    });

    it('should throw for voxel shape with a null colorSource', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        assert.throws(
            () => buildCollisionMesh(solidGrid(), bounds, 1.0, 'voxel'),
            /colorSource/
        );
    });

    it('should color solid-mode vertices with the majority splat color', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        // Two splats at the grid centre, both AABBs covering the whole grid.
        // Their centers are ~3.46 from the corner vertices, beyond the 1.5x
        // distance gate, so every vertex falls back to the ungated candidate
        // set containing both splats. Opacity weights are 0.5 (red, logit 0)
        // and sigmoid(2) ~= 0.881 (blue, logit 2).
        //
        // Red channel values 1 (w=0.5) and 0 (w=0.881): sorted ascending the
        // first cumulant 0.881 >= half (0.6905), so the weighted median red
        // is 0. Blue channel values 0 (w=0.5) and 1 (w=0.881): first cumulant
        // 0.5 < half, so the median blue is 1. Solid mode must produce pure
        // blue for every vertex (average mode would blend to ~0.36 red).
        const colorSource = makeSplatColorSource([
            { center: [2, 2, 2], extent: 4, color: [1, 0, 0], logit: 0 },
            { center: [2, 2, 2], extent: 4, color: [0, 0, 1], logit: 2 }
        ], 'solid');

        const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, 'voxel', colorSource);
        assert.ok(bytes, 'voxel should produce GLB output');

        const { json, bin } = parseGlb(bytes);

        const colorAccessor = json.accessors[2];
        const colorView = json.bufferViews[2];
        const colors = new Float32Array(
            bin.buffer, bin.byteOffset + colorView.byteOffset, colorAccessor.count * 3);

        // linear-space pure blue is exactly [0, 0, 1]
        for (let i = 0; i < colorAccessor.count; i++) {
            assert.ok(Math.abs(colors[i * 3 + 0] - 0) < 1e-4,
                `vertex ${i} red: expected 0 (majority splat only), got ${colors[i * 3 + 0]}`);
            assert.ok(Math.abs(colors[i * 3 + 1] - 0) < 1e-4,
                `vertex ${i} green: expected 0, got ${colors[i * 3 + 1]}`);
            assert.ok(Math.abs(colors[i * 3 + 2] - 1) < 1e-4,
                `vertex ${i} blue: expected 1, got ${colors[i * 3 + 2]}`);
        }
    });

    it('should quantize vertex colors to a palette when a colour count is set', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const colorSource = makeSingleSplatColorSource(2, 2, 2, 4, [0.8, 0.2, 0.5], 0);
        colorSource.palette = 1;

        const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, 'voxel', colorSource);
        assert.ok(bytes, 'voxel should produce GLB output');

        const { json, bin } = parseGlb(bytes);

        const colorAccessor = json.accessors[2];
        const colorView = json.bufferViews[2];
        const colors = new Float32Array(
            bin.buffer, bin.byteOffset + colorView.byteOffset, colorAccessor.count * 3);

        // k=1 palette snaps every vertex to the global mean; all triplets
        // must be identical (all channels equal across all vertices)
        const ref = [colors[0], colors[1], colors[2]];
        for (let i = 1; i < colorAccessor.count; i++) {
            for (let c = 0; c < 3; c++) {
                assert.strictEqual(colors[i * 3 + c], ref[c],
                    `vertex ${i} channel ${c}: all vertices must be identical with k=1`);
            }
        }
    });

    it('should stick to a fixed palette when one is given', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        // three splats of colours nowhere near the palette, so the output can
        // only match if every vertex was snapped to a supplied entry
        const colorSource = makeSplatColorSource([
            { center: [0, 0, 0], extent: 4, color: [1, 0, 0], logit: 0 },
            { center: [4, 0, 0], extent: 4, color: [0, 1, 0], logit: 0 },
            { center: [4, 4, 4], extent: 4, color: [0, 0, 1], logit: 0 }
        ], 'average');
        colorSource.palette = ['#3243aa', '4444ff'];

        const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, 'voxel', colorSource);
        assert.ok(bytes, 'voxel should produce GLB output');

        const { json, bin } = parseGlb(bytes);
        const colorAccessor = json.accessors[2];
        const colorView = json.bufferViews[2];
        const colors = new Float32Array(
            bin.buffer, bin.byteOffset + colorView.byteOffset, colorAccessor.count * 3);

        // linear-space values of the two requested sRGB colours
        const toLinear = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
        const entries = [[0x32, 0x43, 0xaa], [0x44, 0x44, 0xff]]
            .map(e => e.map(b => Math.fround(toLinear(b / 255))));

        for (let v = 0; v < colorAccessor.count; v++) {
            const c = [colors[v * 3], colors[v * 3 + 1], colors[v * 3 + 2]];
            const match = entries.some(e => e.every((ch, i) => Math.abs(ch - c[i]) < 1e-6));
            assert.ok(match, `vertex ${v} colour ${c} is not one of the requested palette colours`);
        }
    });

    it('should keep every flat-shaded triangle facing outward', () => {
        // Welding a quad's two triangles onto 4 shared vertices remaps the
        // partner's indices; if that remap were wrong the winding would flip,
        // which is invisible to a colour check but makes the collision mesh
        // inside-out.
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const colorSource = makeSplatColorSource([
            { center: [2, 2, 2], extent: 4, color: [1, 0, 0], logit: 0 }
        ], 'average');
        colorSource.flatShade = true;

        const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, 'voxel', colorSource);
        const { json, bin } = parseGlb(bytes);
        const posView = json.bufferViews[0];
        const idxView = json.bufferViews[1];
        const positions = new Float32Array(
            bin.buffer, bin.byteOffset + posView.byteOffset, json.accessors[0].count * 3);
        const indices = new Uint32Array(
            bin.buffer, bin.byteOffset + idxView.byteOffset, json.accessors[1].count);

        // the solid 4x4x4 block is centred at (2, 2, 2)
        const centre = [2, 2, 2];
        let checked = 0;
        for (let t = 0; t < indices.length / 3; t++) {
            const [a, b, c] = [0, 1, 2].map(k => indices[t * 3 + k]);
            const p = i => [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
            const pa = p(a), pb = p(b), pc = p(c);
            const e = [0, 1, 2].map(k => pb[k] - pa[k]);
            const f = [0, 1, 2].map(k => pc[k] - pa[k]);
            const n = [
                e[1] * f[2] - e[2] * f[1],
                e[2] * f[0] - e[0] * f[2],
                e[0] * f[1] - e[1] * f[0]
            ];
            const mid = [0, 1, 2].map(k => (pa[k] + pb[k] + pc[k]) / 3 - centre[k]);
            const dot = n[0] * mid[0] + n[1] * mid[1] + n[2] * mid[2];
            assert.ok(dot > 0,
                `triangle ${t} faces inward (dot ${dot}), winding was not preserved`);
            checked++;
        }
        assert.strictEqual(checked, 96 * 2, 'a solid 4x4x4 block has 96 faces');
    });

    it('should flat-shade each voxel quad to a uniform colour', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        // two splats at opposite corners produce distinct vertex colours before
        // flat-shading; after flat-shading every triangle must be a single
        // colour and both triangles of a voxel quad must agree
        const colorSource = makeSplatColorSource([
            { center: [0, 0, 0], extent: 4, color: [1, 0, 0], logit: 0 },
            { center: [4, 4, 4], extent: 4, color: [0, 0, 1], logit: 0 }
        ], 'average');
        colorSource.flatShade = true;

        const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, 'voxel', colorSource);
        assert.ok(bytes, 'voxel should produce GLB output');

        const { json, bin } = parseGlb(bytes);

        const posAccessor = json.accessors[0];
        const idxAccessor = json.accessors[1];
        const colorAccessor = json.accessors[2];
        const colorView = json.bufferViews[2];
        const idxView = json.bufferViews[1];

        assert.strictEqual(colorAccessor.count, posAccessor.count,
            'every vertex needs a colour');

        const colors = new Float32Array(
            bin.buffer, bin.byteOffset + colorView.byteOffset, colorAccessor.count * 3);
        const indices = new Uint32Array(
            bin.buffer, bin.byteOffset + idxView.byteOffset, idxAccessor.count);

        // Read through the index buffer rather than assuming a vertex layout:
        // voxel quads are welded to 4 vertices shared by their 2 triangles, so
        // vertices do not come in per-triangle groups of 3.
        const numTris = indices.length / 3;
        const colorOf = v => [colors[v * 3], colors[v * 3 + 1], colors[v * 3 + 2]];

        let quadPairsChecked = 0;
        for (let t = 0; t < numTris; t++) {
            const [a, b, c] = [0, 1, 2].map(k => indices[t * 3 + k]);
            const ca = colorOf(a);
            for (const v of [b, c]) {
                assert.deepStrictEqual(colorOf(v), ca,
                    `triangle ${t} is not a single colour`);
            }
            // consecutive triangles form quad pairs
            if (t % 2 === 1) {
                const prev = colorOf(indices[(t - 1) * 3]);
                assert.deepStrictEqual(ca, prev,
                    `quad pair triangle ${t} must match its partner`);
                quadPairsChecked++;
            }
        }

        assert.ok(numTris >= 8, `must check at least 8 triangles, got ${numTris}`);
        assert.ok(quadPairsChecked >= 4, `must check at least 4 quad pairs, got ${quadPairsChecked}`);

        // welding means 4 vertices per quad, not 6
        assert.strictEqual(posAccessor.count, (numTris / 2) * 4,
            'each voxel quad should contribute exactly 4 vertices');

        // more than one distinct colour, or the test proves nothing
        const distinct = new Set();
        for (let v = 0; v < posAccessor.count; v++) distinct.add(colorOf(v).join(','));
        assert.ok(distinct.size > 1, 'fixture should produce more than one quad colour');
    });

    it('should not exceed the palette colour count when flat-shading is also enabled', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        // Averaging a face that straddles two palette entries would invent a
        // blend that is not in the palette, silently pushing the output well
        // past the requested colour count.
        for (const shape of ['voxel', 'tris']) {
            const paletteK = 3;
            const colorSource = makeSplatColorSource([
                { center: [0, 0, 0], extent: 4, color: [1, 0, 0], logit: 0 },
                { center: [4, 0, 0], extent: 4, color: [0, 1, 0], logit: 0 },
                { center: [4, 4, 4], extent: 4, color: [0, 0, 1], logit: 0 }
            ], 'average');
            colorSource.palette = paletteK;
            colorSource.flatShade = true;

            const bytes = buildCollisionMesh(solidGrid(), bounds, 1.0, shape, colorSource);
            assert.ok(bytes, `${shape} should produce GLB output`);

            const { json, bin } = parseGlb(bytes);
            const colorAccessor = json.accessors[2];
            const colorView = json.bufferViews[2];
            const colors = new Float32Array(
                bin.buffer, bin.byteOffset + colorView.byteOffset, colorAccessor.count * 3);

            const distinct = new Set();
            for (let v = 0; v < colorAccessor.count; v++) {
                distinct.add(`${colors[v * 3]},${colors[v * 3 + 1]},${colors[v * 3 + 2]}`);
            }
            assert.ok(distinct.size <= paletteK,
                `${shape}: expected at most ${paletteK} colours, got ${distinct.size}`);
        }
    });
});

describe('coplanarMerge', () => {
    it('should return an empty mesh when given an empty mesh', () => {
        const empty = { positions: new Float32Array(0), indices: new Uint32Array(0) };
        const merged = coplanarMerge(empty, 1.0);

        assert.strictEqual(merged.positions.length, 0);
        assert.strictEqual(merged.indices.length, 0);
    });

    it('should fuse the flat faces of a fully-occupied 4x4x4 slab', () => {
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), SOLID_LO, SOLID_HI);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);
        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);
        const rawFaceTris = rawStats.tris - rawBevels;
        const mergedFaceTris = mergedStats.tris - mergedBevels;

        // Lossless edge-collapse removes the strictly interior face vertices
        // of each face (the 2x2 inner grid whose fan is purely axis-aligned).
        // Demand a substantial face-tri reduction.
        assert.ok(mergedFaceTris < rawFaceTris,
            `expected face-tri reduction; got raw=${rawFaceTris}, merged=${mergedFaceTris}`);
        assert.ok(mergedFaceTris <= rawFaceTris * 0.7,
            `expected >=30% face-tri reduction; got ${mergedFaceTris} of ${rawFaceTris}`);

        // K=2 edge-collinear collapse merges each long bevel ridge of the
        // 4x4x4 cube into a single quad: the 12 cube edges contribute 24
        // tris and the 8 K>=3 corners are non-removable, so the bevel
        // count drops from rawBevels to roughly 32. Demand a clear
        // reduction with the corner bevels still present.
        assert.ok(mergedBevels < rawBevels,
            `expected bevel-tri reduction from K=2 collapse; raw=${rawBevels}, merged=${mergedBevels}`);
        assert.ok(mergedBevels >= 8,
            `at least 8 corner bevels must survive (K>=3 corners); got ${mergedBevels}`);

        // Surface AABB must be preserved exactly (lossless).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }
    });

    it('should preserve bevel triangles around isolated voxels', () => {
        // A single voxel produces a marching-cubes surface with both
        // axis-aligned face triangles and corner / edge bevel triangles.
        // The merge pass must leave the bevels untouched.
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), 1, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);

        assert.ok(rawBevels > 0, 'single voxel should have bevel tris in raw MC output');
        assert.strictEqual(mergedBevels, rawBevels,
            `bevel tris must pass through unchanged: raw=${rawBevels}, merged=${mergedBevels}`);
    });

    it('should collapse the long bevel ridges of a thin column', () => {
        // A 1x8x1 column of voxels along Y has 4 vertical bevel ridges,
        // each running 8 voxel-steps tall with intermediate vertices
        // collinear along the ridge. The K=2 edge-collinear pass collapses
        // each ridge to a single quad; only the K>=3 endpoint corners
        // survive. Net effect: the bevel count drops dramatically while
        // the AABB and the corner geometry are preserved exactly.
        //
        // bitIdx = lx + ly*4 + lz*16; column at (lx=0, lz=0), ly=0..3
        // gives bits 0, 4, 8, 12 -> lo = 0x0000_1111.
        const colLo = ((1 << 0) | (1 << 4) | (1 << 8) | (1 << 12)) >>> 0;
        // bounds 4x8x4 / res 1.0 = 4x8x4 voxels = 1x2x1 blocks
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 2), colLo, 0);
        buffer.addBlock(linearBlockIdx(0, 1, 0, 1, 2), colLo, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 8, 4);
        const raw = marchingCubes(toGrid(buffer, 4, 8, 4), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);

        // Tri count must drop substantially via K=2 ridge collapse.
        assert.ok(mergedStats.tris < rawStats.tris * 0.5,
            `expected >=50% tri reduction from K=2 ridge collapse; raw=${rawStats.tris}, merged=${mergedStats.tris}`);

        // AABB preserved exactly (lossless).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }

        // No fabricated vertex positions (every output position must exist
        // verbatim in the raw input).
        const rawKeys = new Set();
        for (let i = 0; i < raw.positions.length; i += 3) {
            rawKeys.add(`${raw.positions[i]},${raw.positions[i + 1]},${raw.positions[i + 2]}`);
        }
        for (let i = 0; i < merged.positions.length; i += 3) {
            const key = `${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`;
            assert.ok(rawKeys.has(key),
                `merged vertex (${key}) was not in raw input (fabricated)`);
        }
    });

    it('should preserve the convex/concave bevel topology of twin columns', () => {
        // Two parallel 1x4x1 columns separated by a 1-voxel-wide empty
        // gap along X produce both convex bevels (on the outer corners
        // of each column) and concave bevels (in the gap between them).
        // The lossless edge-collapse pass must NOT confuse convex and
        // concave bevels even though they share a plane offset. K=2
        // collapse legitimately merges each long vertical bevel ridge
        // into a single quad; assert the AABB is preserved exactly and
        // no vertex is fabricated.
        //
        // Voxels at (0, 0..3, 0) and (2, 0..3, 0):
        //   col A bitIdx = 0 + ly*4 + 0  -> bits 0, 4, 8, 12
        //   col B bitIdx = 2 + ly*4 + 0  -> bits 2, 6, 10, 14
        const colMask = (
            (1 << 0) | (1 << 4) | (1 << 8) | (1 << 12) |
            (1 << 2) | (1 << 6) | (1 << 10) | (1 << 14)
        ) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), colMask, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);

        // K=2 collapses each vertical bevel ridge to a quad; tris reduce.
        assert.ok(mergedStats.tris < rawStats.tris,
            `expected tri reduction from K=2 collapse; raw=${rawStats.tris}, merged=${mergedStats.tris}`);

        // AABB preserved exactly (lossless).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }

        // No fabricated vertices: convex/concave bevels must not be
        // confused into emitting positions absent from the raw input.
        const rawKeys = new Set();
        for (let i = 0; i < raw.positions.length; i += 3) {
            rawKeys.add(`${raw.positions[i]},${raw.positions[i + 1]},${raw.positions[i + 2]}`);
        }
        for (let i = 0; i < merged.positions.length; i += 3) {
            const key = `${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`;
            assert.ok(rawKeys.has(key),
                `merged vertex (${key}) was not in raw input (fabricated)`);
        }
    });

    it('should not fuse rogue axis-diagonal triangles into shifted quads', () => {
        // Some MC cube configurations (e.g. cubeIndex 29 = c0+c2+c3+c4)
        // emit a triangle whose face normal is axis-diagonal but whose
        // vertices do NOT lie on the canonical 2-tri edge-bevel wedge
        // vertex set. Without a guard, coplanarMerge would bucket such
        // a triangle as if it were a wedge half and emit a fused quad
        // shifted by ~half a voxel from the original geometry, breaking
        // watertightness.
        //
        // Place voxels so the cube cell at (cellX=0, cellY=0, cellZ=0)
        // resolves to cubeIndex 29:
        //   c0 (0,0,0), c2 (1,1,0), c3 (0,1,0), c4 (0,0,1) all solid;
        //   c1, c5, c6, c7 all empty.
        // bitIdx = lx + ly*4 + lz*16 within the (0,0,0) block:
        //   (0,0,0) →  0 → lo bit 0
        //   (0,1,0) →  4 → lo bit 4
        //   (1,1,0) →  5 → lo bit 5
        //   (0,0,1) → 16 → lo bit 16
        const lo = ((1 << 0) | (1 << 4) | (1 << 5) | (1 << 16)) >>> 0;
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 1, 1), lo, 0);

        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const raw = marchingCubes(toGrid(buffer, 4, 4, 4), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);

        // AABB must be preserved exactly. A misaligned fused quad would
        // typically pull a +X extreme in from x=1 to x=0.5 (or similar).
        for (let a = 0; a < 3; a++) {
            assert.strictEqual(mergedStats.min[a], rawStats.min[a],
                `min[${a}] changed: ${rawStats.min[a]} -> ${mergedStats.min[a]}`);
            assert.strictEqual(mergedStats.max[a], rawStats.max[a],
                `max[${a}] changed: ${rawStats.max[a]} -> ${mergedStats.max[a]}`);
        }

        // The rogue triangle in cubeIndex 29 has normal (+X, -Y, 0)/sqrt(2)
        // and would be (incorrectly) bucketed as a (pair=XY, su=+1, sv=-1)
        // wedge in cell (cellU=0, cellV=0, cellE=0). The fused "phantom"
        // quad has corners at (0.5, 1, 0), (0, 0.5, 0), (0, 0.5, 1) and
        // (0.5, 1, 1). Three of those four positions cannot be vertices
        // of the genuine MC surface for this voxel set:
        //   (0.5, 1, 0): edge (0,1,0)-(1,1,0), both solid -> no MC vertex
        //   (0, 0.5, 0): edge (0,0,0)-(0,1,0), both solid -> no MC vertex
        //   (0.5, 1, 1): edge (0,1,1)-(1,1,1), both empty -> no MC vertex
        // If any of these positions appear in the merged mesh, the
        // rogue diagonal was fused into a shifted quad.
        const phantoms = [
            [0.5, 1, 0],
            [0, 0.5, 0],
            [0.5, 1, 1]
        ];
        for (const [px, py, pz] of phantoms) {
            for (let i = 0; i < merged.positions.length; i += 3) {
                const dx = Math.abs(merged.positions[i] - px);
                const dy = Math.abs(merged.positions[i + 1] - py);
                const dz = Math.abs(merged.positions[i + 2] - pz);
                assert.ok(dx > 1e-6 || dy > 1e-6 || dz > 1e-6,
                    `merged vertex at phantom position (${px},${py},${pz}) ` +
                    'indicates a rogue diagonal triangle was fused');
            }
        }
    });

    it('should not fuse butterfly diagonal pairs that share a wedge side', () => {
        // A canonical edge-bevel wedge in cell (cellU=0, cellV=0, cellE=0)
        // for pair=XY, su=+1, sv=+1 has 4 corners:
        //   A_lo = (0.5, 0,   0)  mask bit 0x1
        //   A_hi = (0.5, 0,   1)  mask bit 0x2
        //   B_lo = (0,   0.5, 0)  mask bit 0x4
        //   B_hi = (0,   0.5, 1)  mask bit 0x8
        // Two triangles with masks 0x7 = {A_lo, A_hi, B_lo} and 0xB =
        // {A_lo, A_hi, B_hi} both have the (+x, +y, 0)/sqrt(2) normal and
        // pass the per-tri vertex check. Their masks union to 0xF (full
        // coverage) but their intersection is 0x3 = {A_lo, A_hi} - they
        // share the A SIDE of the wedge, not a diagonal. Fusing them into
        // the canonical 4-corner quad would replace a butterfly-shaped
        // region with a parallelogram offset half a voxel from the
        // original surface. The merge must reject this and emit both
        // tris verbatim.
        const positions = new Float32Array([
            0.5, 0,   0,   // 0: A_lo
            0.5, 0,   1,   // 1: A_hi
            0,   0.5, 0,   // 2: B_lo
            0,   0.5, 1    // 3: B_hi
        ]);
        // Winding chosen so each tri's face normal is (+x, +y, 0)/sqrt(2):
        //   tri1: A_lo, B_lo, A_hi -> cross = (+0.5, +0.5, 0)
        //   tri2: A_lo, B_hi, A_hi -> cross = (+0.5, +0.5, 0)
        const indices = new Uint32Array([
            0, 2, 1,
            0, 3, 1
        ]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);

        // Both tris must survive verbatim: 2 tris, 4 distinct welded
        // vertices, exactly the input positions, no spurious geometry.
        const mergedStats = meshStats(merged);
        assert.strictEqual(mergedStats.tris, 2,
            `butterfly pair must not fuse; got ${mergedStats.tris} tris`);
        assert.strictEqual(mergedStats.verts, 4,
            `expected 4 welded verts; got ${mergedStats.verts}`);

        // The fused phantom quad would introduce no new vertex positions
        // (its corners are exactly A_lo/A_hi/B_lo/B_hi), so a positional
        // check alone is not enough. Instead, verify each input triangle
        // is preserved by checking that the merged mesh has triangles
        // covering both {A_lo, B_lo, A_hi} and {A_lo, B_hi, A_hi} corner
        // sets - the fused quad would only cover the {A_lo,A_hi,B_lo,B_hi}
        // parallelogram with two triangles sharing the A_lo-B_hi (or
        // A_hi-B_lo) diagonal, neither of which equals the input pair.
        const cornerSets = new Set();
        const keyOf = (i) => {
            const x = merged.positions[i];
            const y = merged.positions[i + 1];
            const z = merged.positions[i + 2];
            return `${x},${y},${z}`;
        };
        for (let i = 0; i < merged.indices.length; i += 3) {
            const k0 = keyOf(merged.indices[i] * 3);
            const k1 = keyOf(merged.indices[i + 1] * 3);
            const k2 = keyOf(merged.indices[i + 2] * 3);
            cornerSets.add([k0, k1, k2].sort().join('|'));
        }
        const tri1Key = ['0.5,0,0', '0,0.5,0', '0.5,0,1'].sort().join('|');
        const tri2Key = ['0.5,0,0', '0,0.5,1', '0.5,0,1'].sort().join('|');
        assert.ok(cornerSets.has(tri1Key),
            'tri1 {A_lo,B_lo,A_hi} must survive verbatim');
        assert.ok(cornerSets.has(tri2Key),
            'tri2 {A_lo,B_hi,A_hi} must survive verbatim');
    });

    it('should not fuse half-cell axis-aligned face triangles into shifted full quads', () => {
        // A canonical face quad in cell (cellU=0, cellV=0) on the z=0.5
        // plane has 4 corners:
        //   c0 = (0, 0, 0.5)  mask bit 0x1 (u_lo, v_lo)
        //   c1 = (1, 0, 0.5)  mask bit 0x2 (u_hi, v_lo)
        //   c2 = (0, 1, 0.5)  mask bit 0x4 (u_lo, v_hi)
        //   c3 = (1, 1, 0.5)  mask bit 0x8 (u_hi, v_hi)
        // MC configurations like cubeIndex 31 (c0+c1+c2+c3+c4 solid) emit a
        // single half-cell triangle covering only 3 of the 4 corners on
        // this plane - the 4th corner (vertex on edge 8 of the cube) is
        // suppressed because its endpoints are both solid. Naively
        // bucketing this single triangle would mark the entire 1x1 cell
        // occupied and emit a full face quad, fabricating the missing
        // corner vertex and doubling the surface area at that cell. The
        // merge must reject this and emit the half-cell tri verbatim.
        const positions = new Float32Array([
            1, 0, 0.5,   // 0: c1 (u_hi, v_lo)
            1, 1, 0.5,   // 1: c3 (u_hi, v_hi)
            0, 1, 0.5    // 2: c2 (u_lo, v_hi)
        ]);
        // Winding for upward-facing normal (+z): c1 -> c3 -> c2 has
        // cross = (+1, 0, 0) x (-1, 0, 0) ... actually compute explicitly:
        // e = c3 - c1 = (0, 1, 0); f = c2 - c1 = (-1, 1, 0);
        // n = e x f = (1*0 - 0*1, 0*(-1) - 0*0, 0*1 - 1*(-1)) = (0, 0, 1).
        const indices = new Uint32Array([0, 1, 2]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);

        // The single half-cell triangle must survive verbatim, NOT be
        // expanded into a full 1x1 quad with a fabricated 4th corner.
        const mergedStats = meshStats(merged);
        assert.strictEqual(mergedStats.tris, 1,
            `half-cell tri must not fuse to full quad; got ${mergedStats.tris} tris`);
        assert.strictEqual(mergedStats.verts, 3,
            `expected 3 welded verts (no fabricated 4th corner); got ${mergedStats.verts}`);

        // Verify the missing corner (0, 0, 0.5) was NOT introduced.
        const mergedKeys = new Set();
        for (let i = 0; i < merged.positions.length; i += 3) {
            mergedKeys.add(`${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`);
        }
        assert.ok(!mergedKeys.has('0,0,0.5'),
            'missing corner (0,0,0.5) must NOT be fabricated by the merger');
    });

    it('should preserve a feature bump while collapsing flat slab regions', () => {
        // 2x2 grid of blocks (8x4x8 voxels) whose ly=0 layer is a fully
        // solid 8x8 slab one voxel thick. Add a 3-voxel-tall "bump" column
        // poking up from the slab. The merge pass must fuse the slab's
        // many coplanar triangles into a handful of quads while leaving
        // the bump's bevels and side faces intact.
        //
        // ly=0 slab voxel bits, derived from `bitIdx = lx + ly*4 + lz*16`:
        //   lz=0: bits  0..3   → lo 0x0000_000F
        //   lz=1: bits 16..19  → lo 0x000F_0000
        //   lz=2: bits 32..35  → hi 0x0000_000F
        //   lz=3: bits 48..51  → hi 0x000F_0000
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;

        // Bump column inside block (1,0,1) at (lx=0, lz=0), ly = 1..3:
        //   ly=1: bitIdx =  4 → lo bit 4
        //   ly=2: bitIdx =  8 → lo bit 8
        //   ly=3: bitIdx = 12 → lo bit 12
        const bumpLo = ((1 << 4) | (1 << 8) | (1 << 12)) >>> 0;

        // bounds 8x4x8 / res 1.0 = 8x4x8 voxels = 2x1x2 blocks
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(0, 0, 1, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 1, 2, 1), (slabLo | bumpLo) >>> 0, slabHi);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 8);
        const raw = marchingCubes(toGrid(buffer, 8, 4, 8), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawStats = meshStats(raw);
        const mergedStats = meshStats(merged);
        const rawBevels = countBevelTris(raw);
        const mergedBevels = countBevelTris(merged);
        const rawFaces = rawStats.tris - rawBevels;
        const mergedFaces = mergedStats.tris - mergedBevels;

        // The flat axis-aligned face count is what the merge attacks first.
        // The 8x8 top face has a large interior (5x5 of strictly-inner
        // vertices, plus more outside the bump's hole) whose fans are
        // purely +Y and so collapse losslessly. Demand a deep reduction.
        assert.ok(mergedFaces <= rawFaces * 0.4,
            `expected >=60% face-triangle reduction; got ${mergedFaces} of ${rawFaces}`);

        // K=2 edge-collinear collapse merges long bevel ridges (slab
        // perimeter, bump vertical edges) into quads. Bevels reduce
        // significantly.
        assert.ok(mergedBevels < rawBevels,
            `expected bevel reduction from K=2 collapse; raw=${rawBevels}, merged=${mergedBevels}`);

        // Bump apex must survive losslessly. MC places the surface at
        // voxel-centre boundaries, so the bump apex sits at y=3.5 (midpoint
        // of the topmost in-corner and the empty corner above).
        assert.strictEqual(mergedStats.max[1], rawStats.max[1],
            `bump apex must be preserved exactly: raw=${rawStats.max[1]}, merged=${mergedStats.max[1]}`);
    });

    it('should produce a T-junction-free output', () => {
        // For a manifold mesh with no T-junctions, every undirected edge
        // appears in exactly 2 incident triangles. The lossless edge-collapse
        // pass is the inverse of vertex split, so it preserves manifoldness
        // by construction; assert this property end-to-end on a slab+bump
        // scene that exercises both flat-face collapses and bevel passthrough.
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;
        const bumpLo = ((1 << 4) | (1 << 8) | (1 << 12)) >>> 0;
        // bounds 8x4x8 / res 1.0 = 8x4x8 voxels = 2x1x2 blocks
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(0, 0, 1, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 1, 2, 1), (slabLo | bumpLo) >>> 0, slabHi);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 8);
        const raw = marchingCubes(toGrid(buffer, 8, 4, 8), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const edgeCount = new Map();
        const indices = merged.indices;
        for (let i = 0; i < indices.length; i += 3) {
            const a = indices[i];
            const b = indices[i + 1];
            const c = indices[i + 2];
            const addEdge = (u, v) => {
                const key = u < v ? `${u},${v}` : `${v},${u}`;
                edgeCount.set(key, (edgeCount.get(key) ?? 0) + 1);
            };
            addEdge(a, b);
            addEdge(b, c);
            addEdge(c, a);
        }
        for (const [key, count] of edgeCount) {
            assert.strictEqual(count, 2,
                `edge ${key} has ${count} incident tris (T-junction or boundary)`);
        }
    });

    it('should collapse collinear vertices on a 2-plane seam', () => {
        // Build a 90-degree wedge by hand: plane A on z=0 (+z normal) and
        // plane B on y=0 (+y normal), sharing the long edge x in [0..6],
        // y=z=0. Sub-divide the seam at x = 0,1,2,...,6 (5 strictly
        // interior vertices, all collinear with their direct seam
        // neighbours). Each interior seam vertex has K=2 with collinear
        // crease neighbours and so must be removed by the K=2 pass.
        //
        // After the worklist converges, both planes should fully simplify
        // to a single quad each (4 tris total, 8 verts total). The seam
        // becomes a single edge (0,0,0)-(6,0,0) with no interior breaks.
        const positions = new Float32Array([
            // 0..6: seam vertices
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            5, 0, 0,
            6, 0, 0,
            // 7..8: plane A far edge (y=1)
            0, 1, 0,
            6, 1, 0,
            // 9..10: plane B far edge (z=1)
            0, 0, 1,
            6, 0, 1
        ]);
        // Plane A (+z normal): fan-triangulate from the far-edge endpoints.
        //   (0,7,1), (1,7,8), (1,8,2)? -- need consistent CCW from +z view.
        //
        // Looking down +z axis: y goes "up", x goes "right". CCW order
        // around the +z plane normal is the usual x-right, y-up convention.
        // Plane A polygon in CCW order: 0,1,2,3,4,5,6,8,7 (seam left-to-
        // right along y=0, then far edge right-to-left along y=1).
        // Triangulate via fan from vertex 7 (top-left).
        //   tri (7, 0, 1): e=(0,-1,0), f=(1,-1,0); n=(-1*0-0*-1, 0*1-0*0, 0*-1-(-1)*1)=(0,0,1) +z OK
        //   tri (7, 1, 2): same pattern +z OK
        //   ...continue for (7, i, i+1) for i in 0..5, then (7, 6, 8).
        //
        // Plane B (+y normal): polygon CCW from +y is (x-right, z-into-screen).
        // Looking down +y axis at the (x,z) plane: CCW means x-right, z-up
        // is actually CW in standard convention. Let me derive winding by
        // requiring positive cross product = +y.
        //   tri (0, 1, 9) at (0,0,0)-(1,0,0)-(0,0,1): e=(1,0,0), f=(0,0,1)
        //     n = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0). Wrong; flip.
        //   tri (0, 9, 1): e=(0,0,1), f=(1,0,0); n=(0*0-1*0, 1*1-0*0, 0*0-0*1)=(0,1,0) +y OK
        // Plane B polygon CCW from +y: 0,9,10,6,5,4,3,2,1.
        // Fan from vertex 9: (9, 10, 6), (9, 6, 5), ..., (9, 1, 0).
        const indices = new Uint32Array([
            // Plane A fan from vertex 7
            7, 0, 1,
            7, 1, 2,
            7, 2, 3,
            7, 3, 4,
            7, 4, 5,
            7, 5, 6,
            7, 6, 8,
            // Plane B fan from vertex 9
            9, 10, 6,
            9, 6, 5,
            9, 5, 4,
            9, 4, 3,
            9, 3, 2,
            9, 2, 1,
            9, 1, 0
        ]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);
        const stats = meshStats(merged);

        // Each plane should collapse to a single quad (2 tris). Total: 4.
        assert.strictEqual(stats.tris, 4,
            `wedge with collinear seam should collapse to 4 tris; got ${stats.tris}`);

        // All 5 interior seam vertices (1, 2, 3, 4, 5 in the input) must
        // be absent from the merged mesh.
        const mergedKeys = new Set();
        for (let i = 0; i < merged.positions.length; i += 3) {
            mergedKeys.add(`${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`);
        }
        for (let x = 1; x <= 5; x++) {
            assert.ok(!mergedKeys.has(`${x},0,0`),
                `interior seam vertex (${x},0,0) should have been collapsed`);
        }

        // The two seam endpoints and the four far corners must survive.
        for (const key of ['0,0,0', '6,0,0', '0,1,0', '6,1,0', '0,0,1', '6,0,1']) {
            assert.ok(mergedKeys.has(key),
                `corner vertex (${key}) must survive`);
        }
    });

    it('should not collapse a kinked seam', () => {
        // Same wedge as the previous test, but with a kink in the middle
        // of the seam: vertex 4 is moved off the y=z=0 line. Verify the
        // K=2 collinearity check rejects the kink (and the two seam
        // vertices flanking the kink, since their direct neighbours are
        // no longer collinear), while the strictly-collinear vertices
        // away from the kink (1, 2, 6) still collapse.
        //
        // Seam x = 0,1,2,3, then kink at x=4 (y=0.5), then 5,6,7. So:
        //   0=(0,0,0)  1=(1,0,0)  2=(2,0,0)  3=(3,0,0)  KINK 4=(4,0.5,0)
        //   5=(5,0,0)  6=(6,0,0)  7=(7,0,0)
        //   8=(0,1,0)  9=(7,1,0)         <- plane A far edge
        //   10=(0,0,1) 11=(7,0,1)        <- plane B far edge
        const positions = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0.5, 0,
            5, 0, 0,
            6, 0, 0,
            7, 0, 0,
            0, 1, 0,
            7, 1, 0,
            0, 0, 1,
            7, 0, 1
        ]);
        // Plane A (+z): fan from vertex 8.
        //   (8, 0, 1), (8, 1, 2), ..., (8, 6, 7), (8, 7, 9).
        // Plane B (+y): fan from vertex 10.
        //   (10, 11, 7), (10, 7, 6), ..., (10, 1, 0).
        const indices = new Uint32Array([
            // Plane A fan from vertex 8
            8, 0, 1,
            8, 1, 2,
            8, 2, 3,
            8, 3, 4,
            8, 4, 5,
            8, 5, 6,
            8, 6, 7,
            8, 7, 9,
            // Plane B fan from vertex 10
            10, 11, 7,
            10, 7, 6,
            10, 6, 5,
            10, 5, 4,
            10, 4, 3,
            10, 3, 2,
            10, 2, 1,
            10, 1, 0
        ]);
        const input = { positions, indices };

        const merged = coplanarMerge(input, 1.0);

        const mergedKeys = new Set();
        for (let i = 0; i < merged.positions.length; i += 3) {
            mergedKeys.add(`${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`);
        }

        // The kink vertex (4, 0.5, 0) must survive: its seam neighbours
        // (3,0,0) and (5,0,0) flank it, but (3, kink, 5) is not collinear.
        assert.ok(mergedKeys.has('4,0.5,0'),
            'kink vertex (4, 0.5, 0) must survive (not collinear with its seam neighbours)');

        // The seam vertices DIRECTLY ADJACENT to the kink (vertices 3 and
        // 5) also fail the K=2 collinearity test (their other seam
        // neighbour through the kink is off-axis), so they must survive.
        assert.ok(mergedKeys.has('3,0,0'),
            'seam vertex (3,0,0) adjacent to kink must survive');
        assert.ok(mergedKeys.has('5,0,0'),
            'seam vertex (5,0,0) adjacent to kink must survive');

        // The strictly-collinear seam vertices away from the kink (1, 2,
        // 6) still satisfy K=2 once the worklist propagates and so should
        // be removed.
        for (const x of [1, 2, 6]) {
            assert.ok(!mergedKeys.has(`${x},0,0`),
                `collinear seam vertex (${x},0,0) should still collapse`);
        }
    });

    it('should not produce sliver triangles', () => {
        // The lossless collapse must not output near-degenerate triangles
        // for a structured 2-plane wedge. After convergence, every output
        // triangle should have area >= voxelResolution^2 * 1e-6.
        const positions = new Float32Array([
            0, 0, 0,
            1, 0, 0,
            2, 0, 0,
            3, 0, 0,
            4, 0, 0,
            0, 1, 0,
            4, 1, 0,
            0, 0, 1,
            4, 0, 1
        ]);
        const indices = new Uint32Array([
            // Plane A (+z) fan from vertex 5
            5, 0, 1,
            5, 1, 2,
            5, 2, 3,
            5, 3, 4,
            5, 4, 6,
            // Plane B (+y) fan from vertex 7
            7, 8, 4,
            7, 4, 3,
            7, 3, 2,
            7, 2, 1,
            7, 1, 0
        ]);
        const input = { positions, indices };
        const voxelResolution = 1.0;
        const merged = coplanarMerge(input, voxelResolution);

        const minArea = voxelResolution * voxelResolution * 1e-6;
        for (let i = 0; i < merged.indices.length; i += 3) {
            const ia = merged.indices[i] * 3;
            const ib = merged.indices[i + 1] * 3;
            const ic = merged.indices[i + 2] * 3;
            const ex = merged.positions[ib] - merged.positions[ia];
            const ey = merged.positions[ib + 1] - merged.positions[ia + 1];
            const ez = merged.positions[ib + 2] - merged.positions[ia + 2];
            const fx = merged.positions[ic] - merged.positions[ia];
            const fy = merged.positions[ic + 1] - merged.positions[ia + 1];
            const fz = merged.positions[ic + 2] - merged.positions[ia + 2];
            const cx = ey * fz - ez * fy;
            const cy = ez * fx - ex * fz;
            const cz = ex * fy - ey * fx;
            const area = 0.5 * Math.sqrt(cx * cx + cy * cy + cz * cz);
            assert.ok(area >= minArea,
                `tri ${i / 3} area ${area} < threshold ${minArea} (sliver)`);
        }
    });

    it('should never fabricate vertex positions', () => {
        // Lossless edge-collapse only re-triangulates among existing vertices;
        // it never moves a vertex or creates a new position. Assert that
        // every output vertex of the merged mesh corresponds bit-exactly to
        // a vertex that exists in the raw MC output.
        const slabLo = 0x000F_000F >>> 0;
        const slabHi = 0x000F_000F >>> 0;
        const bumpLo = ((1 << 4) | (1 << 8) | (1 << 12)) >>> 0;
        // bounds 8x4x8 / res 1.0 = 8x4x8 voxels = 2x1x2 blocks
        const buffer = new BlockMaskBuffer();
        buffer.addBlock(linearBlockIdx(0, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 0, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(0, 0, 1, 2, 1), slabLo, slabHi);
        buffer.addBlock(linearBlockIdx(1, 0, 1, 2, 1), (slabLo | bumpLo) >>> 0, slabHi);

        const bounds = makeGridBounds(0, 0, 0, 8, 4, 8);
        const raw = marchingCubes(toGrid(buffer, 8, 4, 8), bounds, 1.0);
        const merged = coplanarMerge(raw, 1.0);

        const rawKeys = new Set();
        for (let i = 0; i < raw.positions.length; i += 3) {
            rawKeys.add(`${raw.positions[i]},${raw.positions[i + 1]},${raw.positions[i + 2]}`);
        }
        let fabricated = 0;
        for (let i = 0; i < merged.positions.length; i += 3) {
            const key = `${merged.positions[i]},${merged.positions[i + 1]},${merged.positions[i + 2]}`;
            if (!rawKeys.has(key)) fabricated++;
        }
        assert.strictEqual(fabricated, 0,
            `merged mesh fabricated ${fabricated} vertex positions not present in raw input`);
    });
});

const parseVox = (bytes) => {
    const buf = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const chunks = [];
    const walk = (off, end) => {
        while (off < end) {
            const id = buf.toString('ascii', off, off + 4);
            const contentSize = buf.readInt32LE(off + 4);
            const childrenSize = buf.readInt32LE(off + 8);
            const contentStart = off + 12;
            chunks.push({ id, contentSize, childrenSize, contentStart });
            if (childrenSize > 0) walk(contentStart + contentSize, contentStart + contentSize + childrenSize);
            off = contentStart + contentSize + childrenSize;
        }
    };
    walk(8, buf.length);
    const size = chunks.find(c => c.id === 'SIZE');
    const xyzi = chunks.find(c => c.id === 'XYZI');
    const rgba = chunks.find(c => c.id === 'RGBA');
    const numVoxels = buf.readInt32LE(xyzi.contentStart);
    const voxels = [];
    for (let i = 0; i < numVoxels; i++) {
        const o = xyzi.contentStart + 4 + i * 4;
        voxels.push([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    }
    const palette = [];
    for (let i = 0; i < 256; i++) {
        const o = rgba.contentStart + i * 4;
        palette.push([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    }
    return {
        magic: buf.toString('ascii', 0, 4),
        version: buf.readInt32LE(4),
        chunks,
        dims: [
            buf.readInt32LE(size.contentStart),
            buf.readInt32LE(size.contentStart + 4),
            buf.readInt32LE(size.contentStart + 8)
        ],
        voxels,
        palette,
        byteLength: buf.length
    };
};

const linearToSrgb8 = (c) => {
    const s = c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055;
    return Math.min(255, Math.max(0, Math.round(s * 255)));
};

describe('buildCollisionVox', () => {
    const voxColorSource = (paletteK) => {
        const cs = makeSplatColorSource([
            { center: [0, 0, 0], extent: 4, color: [1, 0, 0], logit: 0 },
            { center: [4, 0, 0], extent: 4, color: [0, 1, 0], logit: 0 },
            { center: [4, 4, 4], extent: 4, color: [0, 0, 1], logit: 0 }
        ], 'average');
        if (paletteK !== undefined) cs.palette = paletteK;
        return cs;
    };

    it('should return null for an empty grid', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const grid = new SparseVoxelGrid(4, 4, 4);
        assert.strictEqual(buildCollisionVox(grid, bounds, 1.0, voxColorSource(4)), null);
    });

    it('should not require a collision mesh', () => {
        // the whole point of the direct path: colours come from the splats, so
        // no mesh, vertex normals or face attribution are involved
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const bytes = buildCollisionVox(solidGrid(), bounds, 1.0, voxColorSource(4));
        assert.ok(bytes, 'vox should be produced from the grid alone');
    });

    it('should write a structurally valid vox model', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const bytes = buildCollisionVox(solidGrid(), bounds, 1.0, voxColorSource(4));

        const vox = parseVox(bytes);
        assert.strictEqual(vox.magic, 'VOX ');
        assert.strictEqual(vox.version, 150);
        assert.deepStrictEqual(vox.chunks.map(c => c.id), ['MAIN', 'SIZE', 'XYZI', 'RGBA']);

        // MAIN declares every following byte as its children
        const main = vox.chunks[0];
        assert.strictEqual(main.contentSize, 0);
        assert.strictEqual(8 + 12 + main.childrenSize, vox.byteLength,
            'declared chunk sizes must account for the whole file');

        // the 4x4x4 solid block fills the grid
        assert.deepStrictEqual(vox.dims, [4, 4, 4]);
        assert.strictEqual(vox.voxels.length, 64);

        for (const [x, y, z, idx] of vox.voxels) {
            assert.ok(x < vox.dims[0] && y < vox.dims[1] && z < vox.dims[2],
                `voxel ${x},${y},${z} outside declared size`);
            // index 0 means empty, so a written voxel must never use it
            assert.ok(idx >= 1 && idx <= 255, `palette index ${idx} out of range`);
        }
    });

    it('should respect the requested palette size', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const paletteK = 3;
        const vox = parseVox(buildCollisionVox(solidGrid(), bounds, 1.0, voxColorSource(paletteK)));

        const used = new Set(vox.voxels.map(v => v[3]));
        assert.ok(used.size <= paletteK, `expected at most ${paletteK} colors, got ${used.size}`);
        for (const idx of used) {
            assert.strictEqual(vox.palette[idx - 1][3], 255, 'used palette entries must be opaque');
        }
    });

    it('should place every voxel inside the declared model size', () => {
        // a ragged shape, so the tight-bounds offsetting is actually exercised
        const grid = new SparseVoxelGrid(8, 8, 8);
        grid.setVoxel(5, 2, 3);
        grid.setVoxel(6, 2, 3);
        grid.setVoxel(5, 3, 3);
        grid.setVoxel(5, 2, 6);
        const bounds = makeGridBounds(0, 0, 0, 8, 8, 8);

        const vox = parseVox(buildCollisionVox(grid, bounds, 1.0, voxColorSource(4)));
        assert.strictEqual(vox.voxels.length, 4);
        // x spans 5..6, y spans 2..3, z spans 3..6 -> 2 x 2 x 4, written as
        // SIZE (dimX, dimZ, dimY)
        assert.deepStrictEqual(vox.dims, [2, 4, 2]);
        for (const [x, y, z] of vox.voxels) {
            assert.ok(x < vox.dims[0] && y < vox.dims[1] && z < vox.dims[2],
                `voxel ${x},${y},${z} outside declared size ${vox.dims}`);
        }
    });

    it('should reject a grid larger than the vox coordinate range', () => {
        // XYZI stores coordinates as single bytes, so 256 per axis is the ceiling
        const grid = new SparseVoxelGrid(260, 4, 4);
        grid.setVoxel(0, 0, 0);
        grid.setVoxel(259, 0, 0);
        const bounds = makeGridBounds(0, 0, 0, 260, 4, 4);

        assert.throws(
            () => buildCollisionVox(grid, bounds, 1.0, voxColorSource(4)),
            /exceeds the MagicaVoxel limit/,
            'must explain the per-axis limit rather than emit a corrupt file');
    });

    it('should name a voxel size that would fit when rejecting', () => {
        const grid = new SparseVoxelGrid(1024, 4, 4);
        grid.setVoxel(0, 0, 0);
        grid.setVoxel(1023, 0, 0);
        const bounds = makeGridBounds(0, 0, 0, 1024, 4, 4);

        // 1024 voxels at 0.02 spans 20.48 units, so 20.48/256 = 0.08 fits
        assert.throws(
            () => buildCollisionVox(grid, bounds, 0.02, voxColorSource(4)),
            /--collision-voxels-size 0\.08\b/,
            'the error should name a size the user can actually pass');
    });
});

describe('vox size suggestion', () => {
    // The suggested --collision-voxels-size has to be usable first time. Coarse
    // cells align to the grid origin rather than to the occupied region, so a
    // suggestion derived from span/256 alone can still be one cell too small.
    const occupiedSpanning = (minIx, maxIx) => {
        const nx = (((maxIx + 4) >> 2) << 2);
        const grid = new SparseVoxelGrid(Math.max(4, nx), 4, 4);
        grid.setVoxel(minIx, 0, 0);
        grid.setVoxel(maxIx, 0, 0);
        return enumerateOccupied(grid);
    };

    it('should suggest a factor that actually fits, for every offset', () => {
        // offsets make the occupied region straddle coarse cell boundaries
        for (let offset = 0; offset < 24; offset++) {
            for (const span of [257, 300, 512, 513, 1000, 1024, 2411]) {
                const occupied = occupiedSpanning(offset, offset + span - 1);
                const factor = minVoxFactor(occupied);
                assert.ok(voxFitsAt(occupied, factor),
                    `offset ${offset} span ${span}: factor ${factor} does not fit`);
                assert.ok(factor === 1 || !voxFitsAt(occupied, factor - 1),
                    `offset ${offset} span ${span}: factor ${factor} is not minimal`);
            }
        }
    });

    it('should suggest 1 when the region already fits', () => {
        const occupied = occupiedSpanning(0, 255);
        assert.strictEqual(minVoxFactor(occupied), 1);
        assert.strictEqual(minVoxelSizeForVox(occupied, 0.02), 0.02);
    });

    it('should accept the suggested size without a second failure', () => {
        // reproduces the case where following the advice failed again: a
        // 2411-voxel span at 0.02 whose span/256 factor leaves 257 coarse cells
        const grid = new SparseVoxelGrid(2412, 4, 4);
        grid.setVoxel(1, 0, 0);
        grid.setVoxel(2411, 0, 0);
        const bounds = makeGridBounds(0, 0, 0, 2412 * 0.02, 4 * 0.02, 4 * 0.02);
        const occupied = enumerateOccupied(grid);

        const suggested = minVoxelSizeForVox(occupied, 0.02);
        const factor = Math.max(1, Math.round(suggested / 0.02));

        // the suggestion must survive the round trip through the CLI's rounding
        assert.doesNotThrow(() => assertVoxFits(occupied, 0.02, factor),
            `suggested size ${suggested} still does not fit`);

        const plan = downsampleGrid(grid, bounds, 0.02, factor);
        const out = enumerateOccupied(plan.grid);
        assert.ok(out.maxIx - out.minIx + 1 <= 256,
            'the reduced grid must fit the per-axis limit');
    });
});

describe('minVoxelSizeForVox', () => {
    it('should return the span divided by the 256 axis limit', () => {
        const grid = new SparseVoxelGrid(1024, 4, 4);
        grid.setVoxel(0, 0, 0);
        grid.setVoxel(1023, 0, 0);
        const occupied = enumerateOccupied(grid);

        assert.strictEqual(occupied.count, 2);
        assert.strictEqual(occupied.maxIx - occupied.minIx + 1, 1024);
        assert.strictEqual(minVoxelSizeForVox(occupied, 0.02), 1024 * 0.02 / 256);
    });
});

describe('downsampleGrid', () => {
    it('should return the input unchanged for factor 1', () => {
        const bounds = makeGridBounds(0, 0, 0, 4, 4, 4);
        const grid = solidGrid();
        const out = downsampleGrid(grid, bounds, 1.0, 1);
        assert.strictEqual(out.grid, grid);
        assert.strictEqual(out.voxelResolution, 1.0);
    });

    it('should keep a voxel wherever any fine voxel was solid', () => {
        const grid = new SparseVoxelGrid(8, 8, 8);
        grid.setVoxel(0, 0, 0);
        grid.setVoxel(1, 1, 1); // same coarse cell as (0,0,0) at factor 2
        grid.setVoxel(6, 0, 0);
        const bounds = makeGridBounds(0, 0, 0, 8, 8, 8);

        const out = downsampleGrid(grid, bounds, 0.5, 2);
        assert.strictEqual(out.voxelResolution, 1.0);
        assert.strictEqual(out.grid.nx % 4, 0, 'axes must stay block-aligned');

        const seen = [];
        out.grid.forEachOccupiedVoxel((x, y, z) => seen.push(`${x},${y},${z}`));
        assert.deepStrictEqual(seen.sort(), ['0,0,0', '3,0,0']);
    });

    it('should shrink an over-large grid into the vox limit', () => {
        const grid = new SparseVoxelGrid(1024, 4, 4);
        grid.setVoxel(0, 0, 0);
        grid.setVoxel(1023, 0, 0);
        const bounds = makeGridBounds(0, 0, 0, 1024 * 0.02, 4 * 0.02, 4 * 0.02);

        const out = downsampleGrid(grid, bounds, 0.02, 4);
        const occupied = enumerateOccupied(out.grid);
        assert.strictEqual(occupied.maxIx - occupied.minIx + 1, 256);
        assert.ok(buildCollisionVox(out.grid, out.gridBounds, out.voxelResolution, voxColorSourceForDownsample()),
            'the reduced grid should now encode');
    });

    const voxColorSourceForDownsample = () => makeSplatColorSource([
        { center: [0, 0, 0], extent: 40, color: [1, 0, 0], logit: 0 }
    ], 'average');
});
