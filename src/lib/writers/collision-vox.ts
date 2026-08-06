import { Vec3 } from 'playcanvas';

import type { Bounds } from '../data-table';
import { colorizeVertices, mapToPalette, palettizeColors, parsePaletteColors, smoothVertexColors, type SplatColorColumns } from '../mesh';
import { forEachExposedFace } from '../mesh/voxel-faces';
import type { GaussianBVH } from '../spatial';
import type { CollisionColorMode, CollisionColorPalette } from '../types';
import { fmtCount, logger } from '../utils';
import { IntKeyMap } from '../utils/int-key-map';
import { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

const VOX_VERSION = 150;

// XYZI stores each voxel coordinate as a single byte, so one model spans at
// most 256 voxels per axis regardless of what the grid holds.
const MAX_VOX_DIM = 256;

// Palette indices run 1..255 in XYZI; 0 means empty.
const MAX_VOX_COLORS = 255;

const CHUNK_HEADER_SIZE = 12;
const RGBA_CONTENT_SIZE = 4 * 256;

// Outward unit normal of each face bucket, ordered -X, +X, -Y, +Y, -Z, +Z.
const FACE_NORMALS = [
    [-1, 0, 0], [1, 0, 0],
    [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]
];

const linearToSrgb = (c: number): number => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

/**
 * Convert one linear-light channel to the 8-bit sRGB value MagicaVoxel stores.
 *
 * @param c - Linear-light channel value.
 * @returns Channel as an 8-bit sRGB value.
 */
const toSrgb8 = (c: number): number => Math.min(255, Math.max(0, Math.round(linearToSrgb(c) * 255)));

/**
 * Colour inputs for the `.vox` model, taken straight from the splats.
 */
type VoxColorSource = {
    /** BVH over the gaussian AABBs. */
    bvh: GaussianBVH;

    /** Splat colour columns. */
    columns: SplatColorColumns;

    /** Colour combination algorithm. */
    mode: CollisionColorMode;

    /** Optional palette quantisation. */
    palette?: CollisionColorPalette;

    /** Optional edge-preserving denoise radius, in voxels. */
    smoothRadius?: number;

    /** Optional spatial coherence radius, in voxels. */
    coherentRadius?: number;
};

/**
 * Occupied voxels of a grid, enumerated once into flat arrays.
 */
type VoxelSet = {
    count: number;
    ix: Int32Array;
    iy: Int32Array;
    iz: Int32Array;
    minIx: number;
    minIy: number;
    minIz: number;
    maxIx: number;
    maxIy: number;
    maxIz: number;
};

/**
 * Enumerate the occupied voxels of a grid along with their tight bounds.
 *
 * Walks the sparse block structure twice rather than the `nx*ny*nz` cell space,
 * so cost tracks occupancy instead of grid volume.
 *
 * @param grid - Grid to enumerate.
 * @returns The occupied voxel set, or null when the grid is empty.
 */
const enumerateOccupied = (grid: SparseVoxelGrid): VoxelSet | null => {
    let count = 0;
    grid.forEachOccupiedVoxel(() => {
        count++;
    });
    if (count === 0) return null;

    const ix = new Int32Array(count);
    const iy = new Int32Array(count);
    const iz = new Int32Array(count);
    let minIx = Infinity, minIy = Infinity, minIz = Infinity;
    let maxIx = -Infinity, maxIy = -Infinity, maxIz = -Infinity;
    let w = 0;
    grid.forEachOccupiedVoxel((x, y, z) => {
        ix[w] = x;
        iy[w] = y;
        iz[w] = z;
        w++;
        if (x < minIx) minIx = x;
        if (y < minIy) minIy = y;
        if (z < minIz) minIz = z;
        if (x > maxIx) maxIx = x;
        if (y > maxIy) maxIy = y;
        if (z > maxIz) maxIz = z;
    });

    return { count, ix, iy, iz, minIx, minIy, minIz, maxIx, maxIy, maxIz };
};

/**
 * Reduce a voxel grid to a coarser resolution.
 *
 * A coarse voxel is solid when any fine voxel inside it is solid, which keeps
 * the silhouette rather than thinning it. Used to fit a `.vox` inside the
 * format's 256-per-axis limit without lowering the collision resolution.
 *
 * @param grid - Fine grid to reduce.
 * @param gridBounds - Bounds of `grid`, aligned to block boundaries.
 * @param voxelResolution - Fine voxel size in world units.
 * @param factor - Integer reduction factor per axis; 1 returns the input.
 * @returns Coarse grid, its bounds and its resolution.
 */
const downsampleGrid = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    factor: number
): { grid: SparseVoxelGrid; gridBounds: Bounds; voxelResolution: number } => {
    if (factor <= 1) return { grid, gridBounds, voxelResolution };

    // SparseVoxelGrid works in 4-voxel blocks, so each axis must stay a
    // multiple of 4.
    const roundUp4 = (n: number): number => (((n + 3) >> 2) << 2);
    const nx = Math.max(4, roundUp4(Math.ceil(grid.nx / factor)));
    const ny = Math.max(4, roundUp4(Math.ceil(grid.ny / factor)));
    const nz = Math.max(4, roundUp4(Math.ceil(grid.nz / factor)));

    const out = new SparseVoxelGrid(nx, ny, nz);
    grid.forEachOccupiedVoxel((ix, iy, iz) => {
        out.setVoxel((ix / factor) | 0, (iy / factor) | 0, (iz / factor) | 0);
    });

    const resolution = voxelResolution * factor;
    const min = gridBounds.min.clone();
    return {
        grid: out,
        gridBounds: {
            min,
            max: new Vec3(
                min.x + nx * resolution,
                min.y + ny * resolution,
                min.z + nz * resolution
            )
        },
        voxelResolution: resolution
    };
};

/**
 * Per-axis span, in coarse voxels, of the occupied region after reduction.
 *
 * Coarse cells are aligned to the grid origin, not to the occupied region, so
 * this is `floor(max / f) - floor(min / f) + 1` rather than `ceil(span / f)` —
 * the two differ by one whenever the region straddles a coarse cell boundary.
 *
 * @param voxels - Occupied voxel set.
 * @param factor - Whole reduction factor per axis.
 * @returns Coarse span on X, Y and Z.
 */
const coarseSpan = (voxels: VoxelSet, factor: number): [number, number, number] => [
    Math.floor(voxels.maxIx / factor) - Math.floor(voxels.minIx / factor) + 1,
    Math.floor(voxels.maxIy / factor) - Math.floor(voxels.minIy / factor) + 1,
    Math.floor(voxels.maxIz / factor) - Math.floor(voxels.minIz / factor) + 1
];

/**
 * Whether the occupied region fits a `.vox` model at the given reduction.
 *
 * @param voxels - Occupied voxel set.
 * @param factor - Whole reduction factor per axis.
 * @returns True when every axis is within the format limit.
 */
const voxFitsAt = (voxels: VoxelSet, factor: number): boolean => {
    const span = coarseSpan(voxels, factor);
    return span[0] <= MAX_VOX_DIM && span[1] <= MAX_VOX_DIM && span[2] <= MAX_VOX_DIM;
};

/**
 * Smallest whole reduction factor whose model fits the per-axis limit.
 *
 * Because coarse cells align to the grid origin, the ratio of span to limit is
 * only a lower bound — a region spanning exactly `256 * factor` fine voxels can
 * still straddle 257 coarse cells. The search therefore steps up from that bound
 * until the reduction genuinely fits, rather than trusting it.
 *
 * @param voxels - Occupied voxel set.
 * @returns Reduction factor, at least 1.
 */
const minVoxFactor = (voxels: VoxelSet): number => {
    const span = Math.max(
        voxels.maxIx - voxels.minIx + 1,
        voxels.maxIy - voxels.minIy + 1,
        voxels.maxIz - voxels.minIz + 1
    );
    let factor = Math.max(1, Math.ceil(span / MAX_VOX_DIM));
    while (!voxFitsAt(voxels, factor)) factor++;
    return factor;
};

/**
 * Smallest voxel size whose occupied region fits the `.vox` per-axis limit.
 *
 * The result is a whole multiple of `voxelResolution`, so passing it back as
 * `--collision-voxels-size` reproduces exactly the reduction it was derived
 * from.
 *
 * @param voxels - Occupied voxel set.
 * @param voxelResolution - Voxel size the set was measured at.
 * @returns Minimum viable voxel size in world units.
 */
const minVoxelSizeForVox = (voxels: VoxelSet, voxelResolution: number): number => {
    return minVoxFactor(voxels) * voxelResolution;
};

/**
 * Compute one linear-space colour per occupied voxel, sampled from the splats.
 *
 * Each voxel is sampled once at its centre using the average of its exposed
 * face normals, so the same candidate pipeline that colours mesh vertices
 * applies without a mesh existing. Interior voxels have no exposed face and so
 * get a zero normal, which disables the inward filter for them; they cannot be
 * seen in the model anyway.
 *
 * @param grid - Grid the voxels came from.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @param voxels - Occupied voxel set to colour.
 * @param source - Splat colour inputs.
 * @returns Linear RGB triplets, one per voxel in `voxels` order.
 */
const colorizeVoxels = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    voxels: VoxelSet,
    source: VoxColorSource
): Float32Array => {
    const { count, ix, iy, iz } = voxels;

    const positions = new Float32Array(count * 3);
    const half = voxelResolution * 0.5;
    for (let v = 0; v < count; v++) {
        positions[v * 3] = gridBounds.min.x + ix[v] * voxelResolution + half;
        positions[v * 3 + 1] = gridBounds.min.y + iy[v] * voxelResolution + half;
        positions[v * 3 + 2] = gridBounds.min.z + iz[v] * voxelResolution + half;
    }

    // exposed faces are reported by voxel coordinate, so map back to ordinals
    const nxny = grid.nx * grid.ny;
    const ordinalOf = new IntKeyMap(Math.ceil(count / 0.7));
    for (let v = 0; v < count; v++) {
        ordinalOf.set(ix[v] + iy[v] * grid.nx + iz[v] * nxny, v);
    }

    const normals = new Float32Array(count * 3);
    forEachExposedFace(grid, (x, y, z, bucket) => {
        const v = ordinalOf.get(x + y * grid.nx + z * nxny);
        if (v < 0) return;
        const n = FACE_NORMALS[bucket];
        normals[v * 3] += n[0];
        normals[v * 3 + 1] += n[1];
        normals[v * 3 + 2] += n[2];
    });
    ordinalOf.releaseStorage();

    for (let v = 0; v < count; v++) {
        const nx = normals[v * 3];
        const ny = normals[v * 3 + 1];
        const nz = normals[v * 3 + 2];
        const len = Math.hypot(nx, ny, nz);
        if (len > 0) {
            normals[v * 3] = nx / len;
            normals[v * 3 + 1] = ny / len;
            normals[v * 3 + 2] = nz / len;
        }
    }

    const sampleSub = logger.group('Sampling splats');
    let colors = colorizeVertices(
        positions, normals, source.bvh, source.columns, voxelResolution, source.mode
    );
    sampleSub.end();

    if (source.smoothRadius !== undefined && source.smoothRadius > 0) {
        const smoothSub = logger.group('Denoising');
        colors = smoothVertexColors(colors, positions, source.smoothRadius, voxelResolution);
        logger.info(`denoised: ${source.smoothRadius} voxel radius`);
        smoothSub.end();
    }

    const palette = source.palette;
    const paletteOpts = { positions, voxelResolution, coherentRadius: source.coherentRadius };
    if (Array.isArray(palette) && palette.length >= 1) {
        const paletteSub = logger.group('Palette');
        colors = mapToPalette(colors, parsePaletteColors(palette), paletteOpts);
        logger.info(`palette: ${palette.length} fixed colours`);
        paletteSub.end();
    } else if (typeof palette === 'number' && palette >= 1) {
        const paletteSub = logger.group('Palette');
        colors = palettizeColors(colors, palette, paletteOpts);
        logger.info(`palette: ${palette} colours`);
        paletteSub.end();
    }

    return colors;
};

/**
 * Fail before any colouring work when the region cannot be represented.
 *
 * @param voxels - Occupied voxel set, measured on the collision grid.
 * @param voxelResolution - Collision voxel size in world units.
 * @param factor - Whole reduction factor the `.vox` will use.
 * @throws Error naming a `--collision-voxels-size` that would fit.
 */
const assertVoxFits = (voxels: VoxelSet, voxelResolution: number, factor = 1): void => {
    if (voxFitsAt(voxels, factor)) return;

    const [dimX, dimY, dimZ] = coarseSpan(voxels, factor);
    const spanWorld = (lo: number, hi: number): string => ((hi - lo + 1) * voxelResolution).toFixed(1);
    // printed at a precision that still round-trips to the same whole factor
    const minSize = parseFloat(minVoxelSizeForVox(voxels, voxelResolution).toPrecision(6));
    const at = factor > 1 ? ` at a voxel size of ${parseFloat((voxelResolution * factor).toPrecision(6))}` : '';

    throw new Error(
        `The .vox model would be ${dimX}x${dimY}x${dimZ} voxels${at}, which exceeds the ` +
        `MagicaVoxel limit of ${MAX_VOX_DIM} per axis. The occupied region spans ` +
        `${spanWorld(voxels.minIx, voxels.maxIx)}x${spanWorld(voxels.minIy, voxels.maxIy)}x` +
        `${spanWorld(voxels.minIz, voxels.maxIz)} world units, so the .vox needs a voxel size of ` +
        `at least ${minSize}. Pass --collision-voxels-size ${minSize} to coarsen only the .vox and ` +
        'keep the collision grid, or raise --voxel-params to coarsen everything.');
};

/**
 * Encode the collision voxels as a MagicaVoxel `.vox` model.
 *
 * Colours come straight from the splats: each voxel is sampled once at its
 * centre, using the average of its exposed face normals as the surface normal,
 * then run through the same denoise and palette steps the collision mesh uses.
 * Sampling per voxel rather than per mesh vertex is both the natural
 * granularity for the format — a MagicaVoxel voxel carries a single colour —
 * and far cheaper, since it needs no mesh, no per-vertex normals and no
 * face-to-voxel attribution pass.
 *
 * @param grid - Voxel grid to export.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param voxelResolution - Size of each voxel in world units.
 * @param source - Splat colour inputs.
 * @returns `.vox` bytes, or null when the grid holds no voxels.
 * @throws Error if the occupied region exceeds 256 voxels on any axis.
 */
const buildCollisionVox = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    source: VoxColorSource
): Uint8Array | null => {
    const voxels = enumerateOccupied(grid);
    if (!voxels) return null;

    assertVoxFits(voxels, voxelResolution);

    const { count, ix, iy, iz, minIx, minIy, minIz, maxIx, maxIy, maxIz } = voxels;
    const dimX = maxIx - minIx + 1;
    const dimY = maxIy - minIy + 1;
    const dimZ = maxIz - minIz + 1;

    let voxColors: Float32Array<ArrayBufferLike> =
        colorizeVoxels(grid, gridBounds, voxelResolution, voxels, source);

    // the format holds at most 255 colours, so reduce if sampling produced more
    const distinct = new IntKeyMap();
    let distinctCount = 0;
    const packKey = (v: number): number => {
        return toSrgb8(voxColors[v * 3]) * 65536 +
            toSrgb8(voxColors[v * 3 + 1]) * 256 +
            toSrgb8(voxColors[v * 3 + 2]);
    };
    for (let v = 0; v < count; v++) {
        const key = packKey(v);
        if (distinct.get(key) === -1) {
            distinct.set(key, distinctCount++);
            if (distinctCount > MAX_VOX_COLORS) break;
        }
    }
    if (distinctCount > MAX_VOX_COLORS) {
        const reduceSub = logger.group('Reducing palette');
        logger.info(`over ${MAX_VOX_COLORS} colours, clustering down`);
        voxColors = palettizeColors(voxColors, MAX_VOX_COLORS);
        reduceSub.end();
    }
    distinct.releaseStorage();

    // assign palette indices in first-seen order
    const indexOfColor = new IntKeyMap();
    const palette: number[] = [];
    const voxIndex = new Uint8Array(count);
    for (let v = 0; v < count; v++) {
        const key = packKey(v);
        let idx = indexOfColor.get(key);
        if (idx === -1) {
            idx = palette.length / 3 + 1; // palette indices start at 1
            indexOfColor.set(key, idx);
            palette.push(voxColors[v * 3], voxColors[v * 3 + 1], voxColors[v * 3 + 2]);
        }
        voxIndex[v] = idx;
    }
    indexOfColor.releaseStorage();

    const xyziContentSize = 4 + count * 4;
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
    writeInt(count);
    for (let v = 0; v < count; v++) {
        // MagicaVoxel is Z-up with +Y running away from the viewer, engine
        // space is Y-up with +Z towards it; negating Z keeps the model from
        // coming out mirrored.
        bytes[o++] = ix[v] - minIx;
        bytes[o++] = maxIz - iz[v];
        bytes[o++] = iy[v] - minIy;
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

    logger.info(`vox: ${dimX}x${dimZ}x${dimY}, ${fmtCount(count)} voxels, ${palette.length / 3} colours`);

    return bytes;
};

export {
    buildCollisionVox,
    downsampleGrid,
    enumerateOccupied,
    minVoxFactor,
    minVoxelSizeForVox,
    voxFitsAt,
    assertVoxFits,
    MAX_VOX_DIM,
    type VoxColorSource,
    type VoxelSet
};
