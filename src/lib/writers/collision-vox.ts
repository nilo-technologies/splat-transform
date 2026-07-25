import type { Bounds } from '../data-table';
import { palettizeColors, type Mesh } from '../mesh';
import { logger } from '../utils';
import { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

const VOX_VERSION = 150;

// XYZI stores each voxel coordinate as a single byte, so one model spans at
// most 256 voxels per axis regardless of what the grid holds.
const MAX_VOX_DIM = 256;

// Palette indices run 1..255 in XYZI; 0 means empty.
const MAX_VOX_COLORS = 255;

const CHUNK_HEADER_SIZE = 12;
const RGBA_CONTENT_SIZE = 4 * 256;

const linearToSrgb = (c: number): number => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/**
 * Convert one linear-light channel to the 8-bit sRGB value MagicaVoxel stores.
 *
 * @param c - Linear-light channel value.
 * @returns Channel as an 8-bit sRGB value.
 */
const toSrgb8 = (c: number): number => Math.min(255, Math.max(0, Math.round(linearToSrgb(c) * 255)));

const colorKey = (r: number, g: number, b: number): string => `${r},${g},${b}`;

/**
 * Locate the solid voxel a mesh face belongs to.
 *
 * A voxel-face centroid sits exactly on the boundary plane between the solid
 * voxel and its empty neighbour, so flooring it alone is ambiguous. Stepping a
 * quarter voxel along the inward normal resolves it; the outward step and the
 * unshifted point cover the marching-cubes case, whose triangles are not
 * axis-aligned.
 *
 * @param grid - Voxel grid to test occupancy against.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @param cx - Face centroid X in world space.
 * @param cy - Face centroid Y in world space.
 * @param cz - Face centroid Z in world space.
 * @param nx - Face normal X, normalized.
 * @param ny - Face normal Y, normalized.
 * @param nz - Face normal Z, normalized.
 * @returns Linear voxel index, or -1 when no solid voxel was found.
 */
const findOwnerVoxel = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    cx: number, cy: number, cz: number,
    nx: number, ny: number, nz: number
): number => {
    for (const step of [-0.25, 0.25, 0]) {
        const d = step * voxelResolution;
        const ix = Math.floor((cx + nx * d - gridBounds.min.x) / voxelResolution);
        const iy = Math.floor((cy + ny * d - gridBounds.min.y) / voxelResolution);
        const iz = Math.floor((cz + nz * d - gridBounds.min.z) / voxelResolution);
        if (ix < 0 || iy < 0 || iz < 0 || ix >= grid.nx || iy >= grid.ny || iz >= grid.nz) continue;
        if (grid.getVoxel(ix, iy, iz)) return ix + iy * grid.nx + iz * grid.nx * grid.ny;
    }
    return -1;
};

/**
 * Encode the collision voxels as a MagicaVoxel `.vox` model.
 *
 * The colours come from the finished collision mesh, so whatever the palette
 * and spatial options produced for the GLB is what lands in the `.vox` too. A
 * MagicaVoxel voxel carries a single colour while a mesh voxel has up to six
 * independently coloured faces, so each voxel takes the colour held by most of
 * its faces; interior voxels, which contribute no faces and cannot be seen,
 * inherit the model's most common colour.
 *
 * @param grid - Voxel grid the collision mesh was extracted from.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @param mesh - Final collision mesh, after any flat-shade un-indexing.
 * @param colors - Per-vertex linear-space colours for `mesh`.
 * @returns `.vox` bytes, or null when the grid holds no voxels.
 * @throws Error if the occupied region exceeds 256 voxels on any axis.
 */
const buildCollisionVox = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    mesh: Mesh,
    colors: Float32Array
): Uint8Array | null => {
    // tight bounds of the occupied region, so a block-aligned grid with empty
    // margins still fits inside the format's per-axis limit
    let minIx = Infinity, minIy = Infinity, minIz = Infinity;
    let maxIx = -Infinity, maxIy = -Infinity, maxIz = -Infinity;
    let voxelCount = 0;
    for (let iz = 0; iz < grid.nz; iz++) {
        for (let iy = 0; iy < grid.ny; iy++) {
            for (let ix = 0; ix < grid.nx; ix++) {
                if (!grid.getVoxel(ix, iy, iz)) continue;
                voxelCount++;
                if (ix < minIx) minIx = ix;
                if (iy < minIy) minIy = iy;
                if (iz < minIz) minIz = iz;
                if (ix > maxIx) maxIx = ix;
                if (iy > maxIy) maxIy = iy;
                if (iz > maxIz) maxIz = iz;
            }
        }
    }

    if (voxelCount === 0) return null;

    const dimX = maxIx - minIx + 1;
    const dimY = maxIy - minIy + 1;
    const dimZ = maxIz - minIz + 1;
    if (dimX > MAX_VOX_DIM || dimY > MAX_VOX_DIM || dimZ > MAX_VOX_DIM) {
        throw new Error(
            `Voxel grid is ${dimX}x${dimY}x${dimZ}, which exceeds the MagicaVoxel limit of ` +
            `${MAX_VOX_DIM} per axis. Increase the voxel size in --voxel-params.`);
    }

    // intern the mesh colours so face tallies can count small integers
    const vertexCount = colors.length / 3;
    const idOfKey = new Map<string, number>();
    const idColors: number[] = [];
    const idOfVertex = new Int32Array(vertexCount);
    for (let v = 0; v < vertexCount; v++) {
        const r = colors[v * 3], g = colors[v * 3 + 1], b = colors[v * 3 + 2];
        const key = colorKey(r, g, b);
        let id = idOfKey.get(key);
        if (id === undefined) {
            id = idColors.length / 3;
            idOfKey.set(key, id);
            idColors.push(r, g, b);
        }
        idOfVertex[v] = id;
    }

    // tally face colours per voxel
    const tally = new Map<number, Map<number, number>>();
    const numTris = mesh.indices.length / 3;
    for (let t = 0; t < numTris; t++) {
        const a = mesh.indices[t * 3], b = mesh.indices[t * 3 + 1], c = mesh.indices[t * 3 + 2];
        const ax = mesh.positions[a * 3], ay = mesh.positions[a * 3 + 1], az = mesh.positions[a * 3 + 2];
        const bx = mesh.positions[b * 3], by = mesh.positions[b * 3 + 1], bz = mesh.positions[b * 3 + 2];
        const cx = mesh.positions[c * 3], cy = mesh.positions[c * 3 + 1], cz = mesh.positions[c * 3 + 2];

        const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;
        const len = Math.hypot(nx, ny, nz);
        if (len > 0) {
            nx /= len;
            ny /= len;
            nz /= len;
        }

        const owner = findOwnerVoxel(
            grid, gridBounds, voxelResolution,
            (ax + bx + cx) / 3, (ay + by + cy) / 3, (az + bz + cz) / 3,
            nx, ny, nz);
        if (owner < 0) continue;

        let counts = tally.get(owner);
        if (!counts) {
            counts = new Map<number, number>();
            tally.set(owner, counts);
        }
        for (const v of [a, b, c]) {
            const id = idOfVertex[v];
            counts.set(id, (counts.get(id) ?? 0) + 1);
        }
    }

    // dominant colour per voxel, ties to the lower id for determinism
    const dominant = new Map<number, number>();
    const globalCounts = new Map<number, number>();
    for (const [owner, counts] of tally) {
        let bestId = -1;
        let bestCount = -1;
        for (const [id, n] of counts) {
            if (n > bestCount || (n === bestCount && id < bestId)) {
                bestCount = n;
                bestId = id;
            }
        }
        dominant.set(owner, bestId);
        globalCounts.set(bestId, (globalCounts.get(bestId) ?? 0) + 1);
    }

    let fallbackId = 0;
    let fallbackCount = -1;
    for (const [id, n] of globalCounts) {
        if (n > fallbackCount || (n === fallbackCount && id < fallbackId)) {
            fallbackCount = n;
            fallbackId = id;
        }
    }

    // per-voxel colour, in the grid order the XYZI chunk will be written in
    const voxX = new Uint8Array(voxelCount);
    const voxY = new Uint8Array(voxelCount);
    const voxZ = new Uint8Array(voxelCount);
    let voxColors: Float32Array<ArrayBufferLike> = new Float32Array(voxelCount * 3);
    let w = 0;
    for (let iz = 0; iz < grid.nz; iz++) {
        for (let iy = 0; iy < grid.ny; iy++) {
            for (let ix = 0; ix < grid.nx; ix++) {
                if (!grid.getVoxel(ix, iy, iz)) continue;
                const linear = ix + iy * grid.nx + iz * grid.nx * grid.ny;
                const id = dominant.get(linear) ?? fallbackId;
                // MagicaVoxel is Z-up with +Y running away from the viewer,
                // engine space is Y-up with +Z towards it; negating Z keeps the
                // model from coming out mirrored.
                voxX[w] = ix - minIx;
                voxY[w] = maxIz - iz;
                voxZ[w] = iy - minIy;
                voxColors[w * 3] = idColors[id * 3];
                voxColors[w * 3 + 1] = idColors[id * 3 + 1];
                voxColors[w * 3 + 2] = idColors[id * 3 + 2];
                w++;
            }
        }
    }

    // the format holds at most 255 colours, so reduce if the mesh had more
    const distinct = new Set<string>();
    for (let v = 0; v < voxelCount; v++) {
        distinct.add(colorKey(voxColors[v * 3], voxColors[v * 3 + 1], voxColors[v * 3 + 2]));
    }
    if (distinct.size > MAX_VOX_COLORS) {
        logger.info(`vox palette: reducing ${distinct.size} colours to ${MAX_VOX_COLORS}`);
        voxColors = palettizeColors(voxColors, MAX_VOX_COLORS);
    }

    // assign palette indices in first-seen order
    const indexOfKey = new Map<string, number>();
    const palette: number[] = [];
    const voxIndex = new Uint8Array(voxelCount);
    for (let v = 0; v < voxelCount; v++) {
        const r = voxColors[v * 3], g = voxColors[v * 3 + 1], b = voxColors[v * 3 + 2];
        const key = colorKey(r, g, b);
        let idx = indexOfKey.get(key);
        if (idx === undefined) {
            idx = palette.length / 3 + 1; // palette indices start at 1
            indexOfKey.set(key, idx);
            palette.push(r, g, b);
        }
        voxIndex[v] = idx;
    }

    const xyziContentSize = 4 + voxelCount * 4;
    const childrenSize =
        (CHUNK_HEADER_SIZE + 12) +
        (CHUNK_HEADER_SIZE + xyziContentSize) +
        (CHUNK_HEADER_SIZE + RGBA_CONTENT_SIZE);
    const total = 8 + CHUNK_HEADER_SIZE + childrenSize;

    const bytes = new Uint8Array(total);
    const view = new DataView(bytes.buffer);
    let o = 0;
    const writeTag = (tag: string): void => {
        for (let i = 0; i < 4; i++) bytes[o++] = tag.charCodeAt(i);
    };
    const writeInt = (value: number): void => {
        view.setInt32(o, value, true);
        o += 4;
    };

    writeTag('VOX ');
    writeInt(VOX_VERSION);

    writeTag('MAIN');
    writeInt(0);
    writeInt(childrenSize);

    writeTag('SIZE');
    writeInt(12);
    writeInt(0);
    writeInt(dimX);
    writeInt(dimZ);
    writeInt(dimY);

    writeTag('XYZI');
    writeInt(xyziContentSize);
    writeInt(0);
    writeInt(voxelCount);
    for (let v = 0; v < voxelCount; v++) {
        bytes[o++] = voxX[v];
        bytes[o++] = voxY[v];
        bytes[o++] = voxZ[v];
        bytes[o++] = voxIndex[v];
    }

    writeTag('RGBA');
    writeInt(RGBA_CONTENT_SIZE);
    writeInt(0);
    // XYZI index i reads RGBA slot i-1, so slot 0 holds palette index 1
    for (let i = 0; i < 256; i++) {
        const base = i * 3;
        const present = base < palette.length;
        bytes[o++] = present ? toSrgb8(palette[base]) : 0;
        bytes[o++] = present ? toSrgb8(palette[base + 1]) : 0;
        bytes[o++] = present ? toSrgb8(palette[base + 2]) : 0;
        bytes[o++] = present ? 255 : 0;
    }

    logger.info(`vox: ${dimX}x${dimZ}x${dimY}, ${voxelCount} voxels, ${palette.length / 3} colours`);

    return bytes;
};

export { buildCollisionVox };
