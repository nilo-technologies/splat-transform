import { basename } from 'pathe';
import { Vec3 } from 'playcanvas';

import { buildCollisionMesh } from './collision-glb';
import { assertVoxFits, buildCollisionVox, downsampleGrid, enumerateOccupied } from './collision-vox';
import { logWrittenFile } from './utils';
import { Column, DataTable, computeGaussianExtents, computeWriteTransform, transformColumns, type Bounds } from '../data-table';
import { GpuDilation, GpuVoxelization } from '../gpu';
import { type FileSystem, writeFile } from '../io/write';
import { MAX_COLOR_RADIUS } from '../mesh/color-spatial';
import { parsePaletteColors } from '../mesh/palette';
import { GaussianBVH } from '../spatial';
import type { CollisionColorMode, CollisionColorPalette, CollisionMeshShape, DeviceCreator } from '../types';
import { fmtCount, logger, Transform } from '../utils';
import { version } from '../version';
import { buildSparseOctree, type SparseOctree } from './sparse-octree';
import {
    filterAndFillBlocks,
    alignGridBounds,
    carve,
    fillExterior,
    fillFloor,
    type NavSeed,
    voxelizeToBuffer
} from '../voxel';
import { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

// Denoising radius, in voxels, used when a palette is requested and no explicit
// radius is given. Two voxels is where the measured noise reduction flattens
// out (see tools/color-noise-bench.mjs); the neighbourhood cost grows with the
// cube of the radius, so wider buys little for a lot of time.
const DEFAULT_COLLISION_COLOR_SMOOTH = 2;

/**
 * Pick the colour-denoising radius actually used.
 *
 * Quantizing amplifies whatever colour noise the splats carry: a material whose
 * colours spread widely occupies many candidate bins, so it collects several
 * palette entries and neighbouring faces alternate between them. Denoising
 * first is what keeps a palettized mesh from reading as noisier than the splats
 * it came from, so it is on by default whenever a palette is requested. An
 * explicit radius always wins, including `0` to opt out.
 *
 * @param palette - The requested palette, or undefined for no quantization.
 * @param explicit - The caller's `collisionColorSmooth`, if any.
 * @returns The radius in voxels, or undefined for no denoising.
 */
const resolveColorSmoothRadius = (
    palette: CollisionColorPalette | undefined,
    explicit: number | undefined
): number | undefined => {
    if (explicit !== undefined) return explicit;
    return palette !== undefined ? DEFAULT_COLLISION_COLOR_SMOOTH : undefined;
};

/**
 * Options for writing a voxel octree file.
 */
type WriteVoxelOptions = {
    /** Output filename ending in .voxel.json */
    filename: string;

    /** Gaussian splat data to voxelize */
    dataTable: DataTable;

    /** Size of each voxel in world units. Default: 0.05 */
    voxelResolution?: number;

    /** Opacity threshold for solid voxels - voxels below this are considered empty. Default: 0.1 */
    opacityCutoff?: number;

    /** Optional function to create a GPU device for voxelization */
    createDevice?: DeviceCreator;

    /** Exterior fill radius in world units. Enables exterior fill when set. Requires navSeed; ignored without it. */
    navExteriorRadius?: number;

    /** Capsule dimensions for carve. Height of 0 disables carve. When height > 0, only voxels contactable from the seed are kept. Requires navSeed. */
    navCapsule?: { height: number; radius: number };

    /** Seed position in world space for exterior fill and carve flood fill. */
    navSeed?: NavSeed;

    /** Fill each voxel column upward from the bottom until hitting solid. Runs before carve so the carve's BFS is confined to the actual navigable bubble. Default: false */
    floorFill?: boolean;

    /** When `floorFill` is enabled, dilation radius in world units used to identify "interior" XZ columns to patch. Empty XZ areas larger than `2 * floorFillDilation` from any solid column are treated as exterior and left empty. Default: 0 (patch every empty column). */
    floorFillDilation?: number;

    /** When set, a collision mesh (.collision.glb) is generated alongside the voxel output. `true` is equivalent to `smooth`. The `voxel` and `tris` shapes bake splat colors into a COLOR_0 vertex attribute. */
    collisionMesh?: boolean | CollisionMeshShape;

    /** Vertex color algorithm for `voxel`/`tris` collision meshes. Ignored for grey shapes. Default: `'average'` */
    collisionColorMode?: CollisionColorMode;

    /** Quantize collision-mesh vertex colors. A number builds a k-means palette of that many colors (integer >= 1); a non-empty list of sRGB hex colors (e.g. `['#3243aa', '4444ff']`) snaps colors to exactly those instead. Default: off. */
    collisionColorPalette?: CollisionColorPalette;

    /** When true, average each triangle's vertex colors for a uniform per-face flat colour. Default: false. */
    collisionColorFlat?: boolean;

    /** Edge-preserving denoise radius in voxels, applied before building the palette: each vertex takes the mean of the neighbours within this radius whose colour is perceptually close to its own, so noise averages out while material boundaries stay crisp. Must be >= 0 and <= 8; 0 disables it. Default: 2 when `collisionColorPalette` is set, off otherwise. */
    collisionColorSmooth?: number;

    /** After palette assignment, snap each vertex to the dominant palette color within this many voxels. Must be >= 0 and <= 8; 0 disables it. Default: off. */
    collisionColorCoherent?: number;

    /** Path to also write the collision voxels to as a MagicaVoxel `.vox` model. Colours are sampled per voxel straight from the splats, so no collision mesh is required. Default: off. */
    collisionVoxels?: string;

    /** Voxel size in world units for the `.vox` model only, letting it stay inside the format's 256-per-axis limit while the octree and collision mesh keep a finer `voxelResolution`. Rounded to the nearest whole multiple of `voxelResolution`. Default: same as `voxelResolution`. */
    collisionVoxelsSize?: number;
};

/**
 * Metadata for a voxel octree file.
 */
interface VoxelMetadata {
    /** File format version */
    version: string;

    /** Asset metadata */
    asset: {
        /** Tool that generated the file */
        generator: string;
    };

    /** Grid bounds aligned to 4x4x4 block boundaries */
    gridBounds: { min: number[]; max: number[] };

    /** Scene bounds (in PlayCanvas coordinate space for v1.1+) */
    sceneBounds: { min: number[]; max: number[] };

    /** Size of each voxel in world units */
    voxelResolution: number;

    /** Voxels per leaf dimension (always 4) */
    leafSize: number;

    /** Maximum tree depth */
    treeDepth: number;

    /** Number of interior nodes */
    numInteriorNodes: number;

    /** Number of mixed leaf nodes */
    numMixedLeaves: number;

    /** Total number of Uint32 entries in the nodes array */
    nodeCount: number;

    /** Total number of Uint32 entries in the leafData array */
    leafDataCount: number;
}

/**
 * Crop a voxel grid and its grid bounds to the occupied block range.
 * Removes empty padding that arises from Gaussian 3-sigma extents being
 * much larger than the actual solid voxel footprint.
 *
 * @param grid - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Cropped grid and grid bounds.
 */
const cropToOccupied = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number
): { grid: SparseVoxelGrid; gridBounds: Bounds } => {
    const { nbx, nby, nbz } = grid;

    const boundsBar = logger.bar('Scanning bounds', grid.types.length);
    const occupiedBounds = grid.getOccupiedBlockBounds(done => boundsBar.update(done));
    boundsBar.end();

    if (!occupiedBounds) {
        return { grid, gridBounds };
    }

    const { minBx, minBy, minBz, maxBx, maxBy, maxBz } = occupiedBounds;
    const cropMaxBx = maxBx + 1;
    const cropMaxBy = maxBy + 1;
    const cropMaxBz = maxBz + 1;

    if (minBx === 0 && minBy === 0 && minBz === 0 &&
        cropMaxBx === nbx && cropMaxBy === nby && cropMaxBz === nbz) {
        return { grid, gridBounds };
    }

    const cropBar = logger.bar('Cropping grid', grid.types.length);
    const croppedGrid = grid.cropTo(
        minBx, minBy, minBz, cropMaxBx, cropMaxBy, cropMaxBz,
        done => cropBar.update(done)
    );
    cropBar.end();

    const blockSize = 4 * voxelResolution;
    const croppedMin = new Vec3(
        gridBounds.min.x + minBx * blockSize,
        gridBounds.min.y + minBy * blockSize,
        gridBounds.min.z + minBz * blockSize
    );
    const croppedBounds: Bounds = {
        min: croppedMin,
        max: new Vec3(
            croppedMin.x + (cropMaxBx - minBx) * blockSize,
            croppedMin.y + (cropMaxBy - minBy) * blockSize,
            croppedMin.z + (cropMaxBz - minBz) * blockSize
        )
    };

    return { grid: croppedGrid, gridBounds: croppedBounds };
};

/**
 * Crop a voxel grid to fit the navigable (non-fully-solid) region tightly.
 * Since the runtime treats outside-the-grid as solid, we only need to include
 * blocks that contain at least one empty voxel. Fully-solid blocks beyond the
 * navigable boundary are redundant.
 *
 * @param grid - Voxelized scene data.
 * @param gridBounds - Axis-aligned bounds of the voxel grid.
 * @param voxelResolution - Size of each voxel in world units.
 * @returns Cropped grid and grid bounds.
 */
const cropToNavigable = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number
): { grid: SparseVoxelGrid; gridBounds: Bounds } => {
    const { nbx, nby, nbz } = grid;

    const boundsBar = logger.bar('Scanning bounds', grid.types.length);
    const navBounds = grid.getNavigableBlockBounds(done => boundsBar.update(done));
    boundsBar.end();
    if (!navBounds) {
        return { grid, gridBounds };
    }

    const { minBx, minBy, minBz, maxBx, maxBy, maxBz } = navBounds;

    // Pad by 1 block on each side so the cropped grid retains the solid wall
    // blocks immediately surrounding the navigable cavity. Matches the
    // MARGIN = 1 pattern used by carve() before this re-crop strips it. The
    // collision-mesh extractors treat out-of-grid as empty, so without this
    // padding the mesh has holes wherever the cavity reaches the cropped
    // boundary; with it, the mesh extractor sees a real SOLID→EMPTY
    // transition at the cavity edge and emits a sealed wall there.
    const MARGIN = 1;
    const cropMinBx = Math.max(0, minBx - MARGIN);
    const cropMinBy = Math.max(0, minBy - MARGIN);
    const cropMinBz = Math.max(0, minBz - MARGIN);
    const cropMaxBx = Math.min(nbx, maxBx + 1 + MARGIN);
    const cropMaxBy = Math.min(nby, maxBy + 1 + MARGIN);
    const cropMaxBz = Math.min(nbz, maxBz + 1 + MARGIN);

    if (cropMinBx === 0 && cropMinBy === 0 && cropMinBz === 0 &&
        cropMaxBx === nbx && cropMaxBy === nby && cropMaxBz === nbz) {
        return { grid, gridBounds };
    }

    const cropBar = logger.bar('Cropping grid', grid.types.length);
    const croppedGrid = grid.cropTo(
        cropMinBx, cropMinBy, cropMinBz, cropMaxBx, cropMaxBy, cropMaxBz,
        done => cropBar.update(done)
    );
    cropBar.end();

    const blockSize = 4 * voxelResolution;
    const croppedMin = new Vec3(
        gridBounds.min.x + cropMinBx * blockSize,
        gridBounds.min.y + cropMinBy * blockSize,
        gridBounds.min.z + cropMinBz * blockSize
    );
    const croppedBounds: Bounds = {
        min: croppedMin,
        max: new Vec3(
            croppedMin.x + (cropMaxBx - cropMinBx) * blockSize,
            croppedMin.y + (cropMaxBy - cropMinBy) * blockSize,
            croppedMin.z + (cropMaxBz - cropMinBz) * blockSize
        )
    };

    return { grid: croppedGrid, gridBounds: croppedBounds };
};

/**
 * Write octree data to files.
 *
 * @param fs - File system for writing output files.
 * @param jsonFilename - Output filename for JSON metadata.
 * @param octree - Sparse octree structure to write.
 */
const writeOctreeFiles = async (
    fs: FileSystem,
    jsonFilename: string,
    octree: SparseOctree
): Promise<void> => {
    // Build metadata object
    const metadata: VoxelMetadata = {
        version: '1.1',
        asset: {
            generator: `splat-transform v${version}`
        },
        gridBounds: {
            min: [octree.gridBounds.min.x, octree.gridBounds.min.y, octree.gridBounds.min.z],
            max: [octree.gridBounds.max.x, octree.gridBounds.max.y, octree.gridBounds.max.z]
        },
        sceneBounds: {
            min: [octree.sceneBounds.min.x, octree.sceneBounds.min.y, octree.sceneBounds.min.z],
            max: [octree.sceneBounds.max.x, octree.sceneBounds.max.y, octree.sceneBounds.max.z]
        },
        voxelResolution: octree.voxelResolution,
        leafSize: octree.leafSize,
        treeDepth: octree.treeDepth,
        numInteriorNodes: octree.numInteriorNodes,
        numMixedLeaves: octree.numMixedLeaves,
        nodeCount: octree.nodes.length,
        leafDataCount: octree.leafData.length
    };

    const jsonBytes = (new TextEncoder()).encode(JSON.stringify(metadata, null, 2));
    await writeFile(fs, jsonFilename, jsonBytes);
    logWrittenFile(basename(jsonFilename), jsonBytes.byteLength);

    const binFilename = jsonFilename.replace('.voxel.json', '.voxel.bin');

    const binarySize = (octree.nodes.length + octree.leafData.length) * 4;
    const buffer = new ArrayBuffer(binarySize);
    const view = new Uint32Array(buffer);
    view.set(octree.nodes, 0);
    view.set(octree.leafData, octree.nodes.length);

    await writeFile(fs, binFilename, new Uint8Array(buffer));
    logWrittenFile(basename(binFilename), binarySize);
};

/**
 * Voxelizes Gaussian splat data and writes the result as a sparse voxel octree.
 *
 * This function performs GPU-accelerated voxelization of Gaussian splat data
 * and outputs two to four files:
 * - `filename` (.voxel.json) - JSON metadata including bounds, resolution, and array sizes
 * - Corresponding .voxel.bin - Binary octree data (nodes + leafData as Uint32 arrays)
 * - Corresponding .collision.glb - Triangle mesh extracted from the voxel output (GLB format, optional; the `voxel` and `tris` shapes include COLOR_0 vertex colors baked from the splats)
 * - `collisionVoxels` (.vox) - The same voxels as a MagicaVoxel model, carrying the collision mesh colors (optional)
 *
 * The binary file layout is:
 * - Bytes 0 to (nodeCount * 4 - 1): nodes array (Uint32, little-endian)
 * - Bytes (nodeCount * 4) to end: leafData array (Uint32, little-endian)
 *
 * @param options - Options including filename, data, and voxelization settings.
 * @param fs - File system for writing output files.
 *
 * @example
 * ```ts
 * import { writeVoxel, MemoryFileSystem } from '@playcanvas/splat-transform';
 *
 * const fs = new MemoryFileSystem();
 * await writeVoxel({
 *     filename: 'scene.voxel.json',
 *     dataTable: myDataTable,
 *     voxelResolution: 0.05,
 *     opacityCutoff: 0.1,
 *     collisionMesh: true,
 *     createDevice: async () => myGraphicsDevice
 * }, fs);
 * ```
 */
const writeVoxel = async (options: WriteVoxelOptions, fs: FileSystem): Promise<void> => {
    const {
        filename,
        dataTable,
        voxelResolution = 0.05,
        opacityCutoff = 0.1,
        createDevice,
        navExteriorRadius,
        floorFill = false,
        floorFillDilation = 0,
        navCapsule,
        navSeed,
        collisionMesh = false,
        collisionColorMode = 'average',
        collisionColorPalette,
        collisionColorFlat = false,
        collisionColorSmooth,
        collisionColorCoherent,
        collisionVoxels,
        collisionVoxelsSize
    } = options;

    if (!createDevice) {
        throw new Error('writeVoxel requires a createDevice function for GPU voxelization');
    }

    const collisionMeshShape = (() => {
        if (collisionMesh === false || collisionMesh === undefined) return null;
        if (collisionMesh === true) return 'smooth';
        if (collisionMesh === 'smooth' || collisionMesh === 'faces' ||
            collisionMesh === 'voxel' || collisionMesh === 'tris') return collisionMesh;
        throw new Error(`Invalid collisionMesh value: ${String(collisionMesh)}. Expected true, false, "smooth", "faces", "voxel", or "tris"`);
    })();
    const coloredCollisionMesh = collisionMeshShape === 'voxel' || collisionMeshShape === 'tris';
    const emitVox = collisionVoxels !== undefined;

    // The .vox colours voxels directly from the splats, so it needs the BVH and
    // colour columns just as the coloured mesh shapes do.
    const needsSplatColors = coloredCollisionMesh || emitVox;

    if (collisionVoxelsSize !== undefined && !(collisionVoxelsSize >= voxelResolution)) {
        throw new Error(
            `collisionVoxelsSize must be >= voxelResolution (${voxelResolution}), got ${collisionVoxelsSize}`);
    }

    if (Array.isArray(collisionColorPalette)) {
        if (collisionColorPalette.length === 0) {
            throw new Error('collisionColorPalette must list at least one colour');
        }
        // throws on a malformed entry, so a bad colour fails before voxelizing
        parsePaletteColors(collisionColorPalette);
    } else if (collisionColorPalette !== undefined &&
        (!Number.isInteger(collisionColorPalette) || collisionColorPalette < 1)) {
        throw new Error(`collisionColorPalette must be an integer >= 1 or a list of hex colours, got ${collisionColorPalette}`);
    }

    for (const [name, value] of [
        ['collisionColorSmooth', collisionColorSmooth],
        ['collisionColorCoherent', collisionColorCoherent]
    ] as const) {
        if (value !== undefined && (!(value >= 0) || value > MAX_COLOR_RADIUS)) {
            throw new Error(`${name} must be >= 0 and <= ${MAX_COLOR_RADIUS}, got ${value}`);
        }
    }

    const smoothRadius = resolveColorSmoothRadius(collisionColorPalette, collisionColorSmooth);

    if (navCapsule && !navSeed) {
        logger.warn('navCapsule requires navSeed for nav carving, skipping nav carving');
    }
    const hasNav = !!(navCapsule && navSeed && navCapsule.height > 0);
    const hasFillExterior = !!(navExteriorRadius && navSeed);
    const hasFloorFill = floorFill;

    // Build a DataTable in engine space containing only the columns needed
    // for voxelization (no SH, so SH rotation cost is never paid). Colored
    // collision meshes also need the SH DC color columns.
    const voxelColumns = [
        'x', 'y', 'z',
        'rot_0', 'rot_1', 'rot_2', 'rot_3',
        'scale_0', 'scale_1', 'scale_2',
        'opacity',
        ...(needsSplatColors ? ['f_dc_0', 'f_dc_1', 'f_dc_2'] : [])
    ];
    const missingColumns = voxelColumns.filter(name => !dataTable.hasColumn(name));
    if (missingColumns.length > 0) {
        throw new Error(`writeVoxel: missing required column(s): ${missingColumns.join(', ')}`);
    }
    const delta = computeWriteTransform(dataTable.transform, Transform.IDENTITY);
    let cols: ReturnType<typeof transformColumns> | null = transformColumns(dataTable, voxelColumns, delta);
    let pcDataTable: DataTable | null = new DataTable(voxelColumns.map(name => new Column(name, cols!.get(name)!)));

    let extentsResult: ReturnType<typeof computeGaussianExtents> | null = computeGaussianExtents(pcDataTable);
    const bounds = extentsResult.sceneBounds;

    const g = logger.group('Build voxels');

    // gpuVoxelization is the only resource not owned by a scope; its
    // destruction is the sole job of the finally below. Open scopes on the
    // error path are reaped by the embedder's logger.error() -> unwindAll.
    let gpuVoxelization: GpuVoxelization | null = null;
    let gpuDilation: GpuDilation | null = null;
    let bvh: GaussianBVH | null = null;
    try {
        const bvhSub = logger.group('Building BVH');
        logger.debug(`scene extents: (${bounds.min.x.toFixed(2)},${bounds.min.y.toFixed(2)},${bounds.min.z.toFixed(2)}) - (${bounds.max.x.toFixed(2)},${bounds.max.y.toFixed(2)},${bounds.max.z.toFixed(2)})`);

        bvh = new GaussianBVH(pcDataTable, extentsResult.extents);
        bvhSub.end();

        const device = await createDevice();
        gpuVoxelization = new GpuVoxelization(device);
        gpuVoxelization.uploadAllGaussians(pcDataTable, extentsResult.extents);

        // Align grid bounds to block boundaries BEFORE voxelization so the
        // block coordinates used during voxelization match what the reader
        // expects. fillExterior and fillFloor both need a margin of empty
        // voxels outside the splat's tight 3-sigma extents to do their job:
        // fillExterior so the boundary-face flood seeds survive its dilation
        // (notably below the floor), fillFloor so its column walk has empty
        // XZ columns to convert into wall pillars and the dilation halo to
        // extend the floor footprint outward.
        //
        // Lateral pad combines both as `dilation_radius + 1` voxels per side.
        // Vertical pad is only contributed by exteriorPad — fillFloor's
        // dilation is XZ-only, and Y padding would extend the wall pillars
        // above the splat's natural ceiling and below its floor.
        const exteriorPad = hasFillExterior ?
            (Math.ceil(navExteriorRadius! / voxelResolution) + 1) * voxelResolution :
            0;
        const floorPad = hasFloorFill ?
            (Math.ceil(floorFillDilation / voxelResolution) + 1) * voxelResolution :
            0;
        const padXZ = Math.max(exteriorPad, floorPad);
        const padY = exteriorPad;
        let gridBounds = alignGridBounds(
            bounds.min.x - padXZ, bounds.min.y - padY, bounds.min.z - padXZ,
            bounds.max.x + padXZ, bounds.max.y + padY, bounds.max.z + padXZ,
            voxelResolution
        );

        const buffer = await voxelizeToBuffer(
            bvh, gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
        );
        if (!needsSplatColors) {
            bvh = null;
            pcDataTable = null;
        }
        extentsResult = null;
        cols = null;

        gpuVoxelization.destroy();
        gpuVoxelization = null;

        const filterSub = logger.group('Filtering');
        const nbxInit = Math.round((gridBounds.max.x - gridBounds.min.x) / (4 * voxelResolution));
        const nbyInit = Math.round((gridBounds.max.y - gridBounds.min.y) / (4 * voxelResolution));
        const nbzInit = Math.round((gridBounds.max.z - gridBounds.min.z) / (4 * voxelResolution));
        const filteredBuffer = filterAndFillBlocks(buffer, nbxInit, nbyInit, nbzInit);
        buffer.clear();
        filterSub.end();

        // Buffer → grid: the single conversion in the pipeline. Every phase
        // beyond this point operates on SparseVoxelGrid directly.
        const loadSub = logger.group('Loading grid');
        const nxInit = nbxInit << 2;
        const nyInit = nbyInit << 2;
        const nzInit = nbzInit << 2;
        const loadBar = logger.bar('Loading grid', Math.max(1, filteredBuffer.count));
        let grid = SparseVoxelGrid.fromBuffer(
            filteredBuffer, nxInit, nyInit, nzInit,
            (done, total) => loadBar.update(Math.min(done, total))
        );
        loadBar.end();
        filteredBuffer.clear();
        loadSub.end();

        // Reuse the same device for GPU dilation across exterior, floor, carve.
        const needsGpuDilation = hasFillExterior || hasNav || (hasFloorFill && floorFillDilation > 0);
        if (needsGpuDilation) {
            gpuDilation = new GpuDilation(device);
        }

        if (hasFillExterior) {
            const sub = logger.group('Fill exterior');
            const fillResult = await fillExterior(
                grid, gridBounds, voxelResolution,
                navExteriorRadius!, navSeed!,
                gpuDilation!
            );
            grid = fillResult.grid;
            gridBounds = fillResult.gridBounds;
            sub.end();
        }

        if (hasFloorFill) {
            const sub = logger.group('Fill floor');
            const floorResult = await fillFloor(
                grid, gridBounds, voxelResolution, floorFillDilation, gpuDilation
            );
            grid = floorResult.grid;
            gridBounds = floorResult.gridBounds;
            sub.end();
        }

        if (hasNav) {
            const sub = logger.group('Carve');
            const navResult = await carve(
                grid, gridBounds, voxelResolution,
                navCapsule!.height, navCapsule!.radius,
                navSeed!,
                gpuDilation!
            );
            grid = navResult.grid;
            gridBounds = navResult.gridBounds;
            sub.end();
        }

        const cropSub = logger.group('Cropping');
        const finalCrop = hasFillExterior || hasFloorFill ?
            cropToNavigable(grid, gridBounds, voxelResolution) :
            cropToOccupied(grid, gridBounds, voxelResolution);
        grid = finalCrop.grid;
        gridBounds = finalCrop.gridBounds;
        logger.debug(`grid: ${grid.nx} x ${grid.ny} x ${grid.nz} voxels @ ${voxelResolution}`);
        cropSub.end();

        gpuDilation?.destroy();
        gpuDilation = null;

        // Colored shapes and the .vox both need the retained BVH and splat
        // color columns; release both once neither is outstanding.
        const splatColors = needsSplatColors ? {
            bvh: bvh!,
            columns: {
                f_dc_0: pcDataTable!.getColumnByName('f_dc_0')!.data,
                f_dc_1: pcDataTable!.getColumnByName('f_dc_1')!.data,
                f_dc_2: pcDataTable!.getColumnByName('f_dc_2')!.data,
                opacity: pcDataTable!.getColumnByName('opacity')!.data
            },
            mode: collisionColorMode,
            palette: collisionColorPalette,
            smoothRadius,
            coherentRadius: collisionColorCoherent
        } : null;

        // Validate the .vox up front: its 256-per-axis limit is decided by the
        // cropped grid, so checking here costs microseconds and saves building
        // a mesh and colouring millions of vertices only to fail afterwards.
        let voxPlan: ReturnType<typeof downsampleGrid> | null = null;
        if (emitVox) {
            const factor = collisionVoxelsSize === undefined ?
                1 :
                Math.max(1, Math.round(collisionVoxelsSize / voxelResolution));
            const occupied = enumerateOccupied(grid);
            if (!occupied) {
                logger.warn('no occupied voxels, skipping .vox output');
            } else {
                if (factor > 1) {
                    const size = voxelResolution * factor;
                    logger.info(`vox voxel size: ${size} (${factor}x the collision grid)`);
                }
                assertVoxFits(occupied, voxelResolution, factor);
                voxPlan = downsampleGrid(grid, gridBounds, voxelResolution, factor);
            }
        }

        const glbBytes = collisionMeshShape ?
            buildCollisionMesh(grid, gridBounds, voxelResolution, collisionMeshShape,
                coloredCollisionMesh ? { ...splatColors!, flatShade: collisionColorFlat } : null) :
            null;

        let voxBytes: Uint8Array | null = null;
        if (voxPlan) {
            const voxSub = logger.group('Collision voxels');
            voxBytes = buildCollisionVox(
                voxPlan.grid, voxPlan.gridBounds, voxPlan.voxelResolution, splatColors!);
            voxSub.end();
        }
        bvh = null;
        pcDataTable = null;

        const octree = buildSparseOctree(
            grid,
            gridBounds,
            bounds,
            voxelResolution,
            { consumeGrid: true }
        );

        logger.info(`octree depth: ${octree.treeDepth}`);
        logger.info(`interior nodes: ${fmtCount(octree.numInteriorNodes)}`);
        logger.info(`mixed leaves: ${fmtCount(octree.numMixedLeaves)}`);

        const writingSub = logger.group('Writing');
        await writeOctreeFiles(fs, filename, octree);

        if (glbBytes) {
            const glbFilename = filename.replace('.voxel.json', '.collision.glb');
            await writeFile(fs, glbFilename, glbBytes);
            logWrittenFile(basename(glbFilename), glbBytes.length);
        }
        if (voxBytes && collisionVoxels) {
            await writeFile(fs, collisionVoxels, voxBytes);
            logWrittenFile(basename(collisionVoxels), voxBytes.length);
        }
        writingSub.end();

        g.end();
    } finally {
        gpuVoxelization?.destroy();
        gpuDilation?.destroy();
    }
};

export { writeVoxel, writeOctreeFiles, resolveColorSmoothRadius, type WriteVoxelOptions, type VoxelMetadata };
