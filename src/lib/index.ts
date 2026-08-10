// Data table
export { Column, DataTable, combine, convertToSpace, computeSummary, sortMortonOrder, sortByVisibility, simplifyGaussians, getSHBands } from './data-table';
export { computeGaussianExtents } from './data-table';
export type { TypedArray, ColumnType, Row, ColumnStats, SummaryData, Bounds, GaussianExtentsResult } from './data-table';

// Utils
export {
    fmtBytes, fmtCount, fmtDistance, fmtTime,
    logger, TextRenderer, Transform, WebPCodec
} from './utils';
export type { Bar, Group, LogEvent, Logger, MessageKind, Renderer, TextRendererOptions, Verbosity } from './utils';

// High-level read/write
export { readFile, getInputFormat } from './read';
export type { InputFormat, ReadFileOptions } from './read';
export { writeFile, getOutputFormat } from './write';
export type { OutputFormat, WriteOptions } from './write';

// Processing
export { processDataTable } from './process';
export type {
    ProcessAction,
    ProcessOptions,
    Translate,
    Rotate,
    Scale,
    FilterNaN,
    FilterByValue,
    FilterBands,
    FilterBox,
    FilterSphere,
    FilterFloaters,
    FilterCluster,
    Param as ProcessParam,
    Lod,
    Summary,
    MortonOrder,
    Decimate
} from './process';

// Worker pool for CPU-heavy tasks
export { WorkerQueue } from './workers';

// File system abstractions
export { ReadStream, BufferedReadStream, MemoryReadFileSystem, UrlReadFileSystem, ZipReadFileSystem } from './io/read';
export type { ReadSource, ReadFileSystem, ProgressCallback, ZipEntry } from './io/read';
export { MemoryFileSystem, ZipFileSystem } from './io/write';
export type { FileSystem, Writer } from './io/write';

// Individual readers (for advanced use)
export { readKsplat, readLcc, readLcc2, readMjs, readPly, readSog, readSplat, readSpz } from './readers';

// Individual writers (for advanced use)
export { writeSog, writeSpz, writePly, writeCompressedPly, writeCsv, writeHtml, writeImage, writeLod, writeGlb, writeVoxel } from './writers';
export type { WriteImageOptions, WriteVoxelOptions, VoxelMetadata } from './writers';

// Collision / voxel-model generation from a voxel grid (for advanced use).
// These are the pieces `writeVoxel` composes, exposed so a caller can drive the
// same paths directly without going through file output.
export {
    assertVoxFits,
    buildCollisionMesh,
    buildCollisionVox,
    countVoxModels,
    downsampleGrid,
    enumerateOccupied,
    minVoxFactorForModels,
    buildSparseOctree,
    MAX_VOX_DIM,
    MAX_VOX_MODELS,
    SOLID_LEAF_MARKER
} from './writers';
export type { VoxColorSource, VoxelSet, SparseOctree } from './writers';

// Mesh extraction and vertex colouring (for advanced use)
export {
    marchingCubes, coplanarMerge, voxelFaces, forEachExposedFace,
    computeVertexNormals, colorizeVertices,
    palettizeColors, mapToPalette, parsePaletteColors,
    smoothVertexColors, majorityFilterIndices, MAX_COLOR_RADIUS
} from './mesh';
export type { Mesh, MarchingCubesMesh, MarchingCubesOptions, SplatColorColumns, PalettizeOptions } from './mesh';

// Spatial acceleration (for advanced use)
export { GaussianBVH, KdTree, BTree, kmeans, quantize1d } from './spatial';
export type { GaussianBVHNode, BVHBounds, KdTreeNode, BTreeNode } from './spatial';

// Renderer (for advanced use)
export { renderSplats, buildCameraBasis } from './render';
export type { Projection, RenderCamera, CameraBasis } from './render';

// Voxel
export {
    alignGridBounds, applyAlignYaw, carve, estimateAlignYaw, fillExterior, fillFloor,
    filterAndFillBlocks, filterCluster, filterFloaters, findClusterVoxelFlood, growGrid,
    majorityFilterGrid, voxelizeToBuffer,
    BlockMaskBuffer, SparseVoxelGrid, BLOCK_EMPTY, BLOCK_MIXED, BLOCK_SOLID
} from './voxel';
export type { AlignYawApplied, AlignYawOptions, AlignYawResult, GrowOptions, GrowResult, MajorityOptions, MajorityResult, NavSeed, NavSimplifyResult, UpAxis } from './voxel';

// Types
export type { CollisionMeshShape, CollisionColorMode, CollisionColorPalette, Options, Param, DeviceCreator } from './types';

// Version
export { version, revision } from './version';
