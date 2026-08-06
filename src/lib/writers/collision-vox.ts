import { Vec3 } from 'playcanvas';

import type { Bounds } from '../data-table';
import { colorizeVertices, mapToPalette, palettizeColors, parsePaletteColors, smoothVertexColors, type SplatColorColumns } from '../mesh';
import { forEachExposedFace } from '../mesh/voxel-faces';
import type { GaussianBVH } from '../spatial';
import type { CollisionColorMode, CollisionColorPalette } from '../types';
import { fmtBytes, fmtCount, logger } from '../utils';
import { IntKeyMap } from '../utils/int-key-map';
import { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

const VOX_VERSION = 150;

// XYZI stores each voxel coordinate as a single byte, so one model spans at
// most 256 voxels per axis regardless of what the grid holds.
const MAX_VOX_DIM = 256;

// Palette indices run 1..255 in XYZI; 0 means empty.
const MAX_VOX_COLORS = 255;

// Past this many voxels the file is large enough that MagicaVoxel will struggle
// to open it usefully, so the caller is told rather than left guessing. Purely
// advisory - the file is still written.
const LARGE_VOX_VOXELS = 8_000_000;

// A region wider than MAX_VOX_DIM is split into that many models, placed by the
// scene graph. The format itself only bounds model count by its int32 node ids,
// but MagicaVoxel is not usable with thousands of objects and the voxel payload
// grows with every tile, so this is a deliberate practical ceiling rather than a
// format one. 256 tiles of 256^3 still cover 4.3 billion cells.
const MAX_VOX_MODELS = 256;

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

// A dense cell->ordinal table costs 4 bytes per grid cell but removes all
// hashing from the three passes that need the mapping — and the interior flood
// alone does six lookups per occupied voxel. Above this many cells the table
// would cost more than the hash it replaces, so the hash is used instead.
const DENSE_ORDINAL_MAX_CELLS = 1 << 25;

/**
 * Maps a linear grid cell index to an occupied-voxel ordinal.
 *
 * Backed by a dense array on grids small enough to afford one, and by an
 * open-addressed hash otherwise. Both live behind one class so the call sites
 * stay monomorphic; the dense form stores `ordinal + 1` so that an untouched
 * zero reads back as absent without a fill pass.
 */
class OrdinalMap {
    private dense: Int32Array | null = null;
    private hash: IntKeyMap | null = null;

    constructor(cells: number, count: number) {
        if (cells <= DENSE_ORDINAL_MAX_CELLS) {
            this.dense = new Int32Array(cells);
        } else {
            this.hash = new IntKeyMap(Math.ceil(count / 0.7));
        }
    }

    get(cell: number): number {
        return this.dense !== null ? this.dense[cell] - 1 : this.hash!.get(cell);
    }

    set(cell: number, ordinal: number): void {
        if (this.dense !== null) this.dense[cell] = ordinal + 1;
        else this.hash!.set(cell, ordinal);
    }

    release(): void {
        this.dense = null;
        this.hash?.releaseStorage();
        this.hash = null;
    }
}

// Neighbour offsets used to flood surface colours into enclosed voxels.
const NEIGHBOUR_OFFSETS = [
    [-1, 0, 0], [1, 0, 0],
    [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]
];

/**
 * Compute one linear-space colour per occupied voxel, sampled from the splats.
 *
 * Only voxels with at least one exposed face are sampled. A voxel enclosed on
 * all six sides cannot be seen in the model, so querying the BVH for it buys
 * nothing — and on a solid volume those voxels are the large majority: a
 * 200x180x204 grid at 5 mm holds 3.07M occupied voxels of which ~0.4M are on
 * the surface. Sampling all of them made the `.vox` 8x slower than the
 * equivalent collision mesh, which only ever colours surface geometry.
 *
 * Surface voxels are sampled at their centre using the average of their exposed
 * face normals, then denoised and palettised as a set — so the palette is
 * chosen from what is actually visible. Enclosed voxels then take the colour of
 * the nearest surface voxel by breadth-first flood, which costs no BVH work and
 * keeps the interior sensible if the model is later sliced open in MagicaVoxel.
 * Because the flood copies already-palettised colours, it introduces no new
 * palette entries.
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
    const { nx: gnx, ny: gny, nz: gnz } = grid;
    const nxny = gnx * gny;

    // ordinal lookup, kept alive for the inward flood below
    const ordinalOf = new OrdinalMap(gnx * gny * gnz, count);
    for (let v = 0; v < count; v++) {
        ordinalOf.set(ix[v] + iy[v] * gnx + iz[v] * nxny, v);
    }

    // Accumulate exposed-face normals and flag which voxels are on the surface.
    // The flag cannot be derived from the summed normal: a one-voxel-thick wall
    // has opposing exposed faces that cancel to zero.
    const faceNormals = new Float32Array(count * 3);
    const reached = new Uint8Array(count);
    let surfaceCount = 0;
    forEachExposedFace(grid, (x, y, z, bucket) => {
        const v = ordinalOf.get(x + y * gnx + z * nxny);
        if (v < 0) return;
        const n = FACE_NORMALS[bucket];
        faceNormals[v * 3] += n[0];
        faceNormals[v * 3 + 1] += n[1];
        faceNormals[v * 3 + 2] += n[2];
        if (reached[v] === 0) {
            reached[v] = 1;
            surfaceCount++;
        }
    });

    const colors = new Float32Array(count * 3);
    if (surfaceCount === 0) {
        // unreachable for a non-empty grid: the outermost voxels always border
        // empty space or the grid edge
        ordinalOf.release();
        return colors;
    }

    // compact the surface set, so the sampling pass walks only visible voxels
    const surfaceOf = new Int32Array(surfaceCount);
    const positions = new Float32Array(surfaceCount * 3);
    const normals = new Float32Array(surfaceCount * 3);
    const half = voxelResolution * 0.5;
    let s = 0;
    for (let v = 0; v < count; v++) {
        if (reached[v] === 0) continue;
        surfaceOf[s] = v;
        positions[s * 3] = gridBounds.min.x + ix[v] * voxelResolution + half;
        positions[s * 3 + 1] = gridBounds.min.y + iy[v] * voxelResolution + half;
        positions[s * 3 + 2] = gridBounds.min.z + iz[v] * voxelResolution + half;
        const nx = faceNormals[v * 3];
        const ny = faceNormals[v * 3 + 1];
        const nz = faceNormals[v * 3 + 2];
        const len = Math.hypot(nx, ny, nz);
        if (len > 0) {
            normals[s * 3] = nx / len;
            normals[s * 3 + 1] = ny / len;
            normals[s * 3 + 2] = nz / len;
        }
        s++;
    }

    logger.debug(`surface voxels: ${fmtCount(surfaceCount)} of ${fmtCount(count)}`);

    const sampleSub = logger.group('Sampling splats');
    let surfaceColors = colorizeVertices(
        positions, normals, source.bvh, source.columns, voxelResolution, source.mode
    );
    sampleSub.end();

    if (source.smoothRadius !== undefined && source.smoothRadius > 0) {
        const smoothSub = logger.group('Denoising');
        surfaceColors = smoothVertexColors(surfaceColors, positions, source.smoothRadius, voxelResolution);
        logger.info(`denoised: ${source.smoothRadius} voxel radius`);
        smoothSub.end();
    }

    const palette = source.palette;
    const paletteOpts = { positions, voxelResolution, coherentRadius: source.coherentRadius };
    if (Array.isArray(palette) && palette.length >= 1) {
        const paletteSub = logger.group('Palette');
        surfaceColors = mapToPalette(surfaceColors, parsePaletteColors(palette), paletteOpts);
        logger.info(`palette: ${palette.length} fixed colours`);
        paletteSub.end();
    } else if (typeof palette === 'number' && palette >= 1) {
        const paletteSub = logger.group('Palette');
        surfaceColors = palettizeColors(surfaceColors, palette, paletteOpts);
        logger.info(`palette: ${palette} colours`);
        paletteSub.end();
    }

    for (let i = 0; i < surfaceCount; i++) {
        const v = surfaceOf[i];
        colors[v * 3] = surfaceColors[i * 3];
        colors[v * 3 + 1] = surfaceColors[i * 3 + 1];
        colors[v * 3 + 2] = surfaceColors[i * 3 + 2];
    }

    if (surfaceCount < count) {
        const floodSub = logger.group('Filling interior');
        // Breadth-first from the surface, so each enclosed voxel takes the
        // colour of the nearest surface voxel. Every occupied voxel is
        // 6-connected to a surface voxel of its own component — the voxel with
        // maximal X in any component necessarily has an exposed +X face — so a
        // single sweep assigns them all.
        const queue = new Int32Array(count);
        let tail = 0;
        for (let i = 0; i < surfaceCount; i++) queue[tail++] = surfaceOf[i];

        for (let head = 0; head < tail; head++) {
            const v = queue[head];
            const x = ix[v];
            const y = iy[v];
            const z = iz[v];
            for (let k = 0; k < 6; k++) {
                const o = NEIGHBOUR_OFFSETS[k];
                const ax = x + o[0];
                const ay = y + o[1];
                const az = z + o[2];
                if (ax < 0 || ay < 0 || az < 0 || ax >= gnx || ay >= gny || az >= gnz) continue;
                const u = ordinalOf.get(ax + ay * gnx + az * nxny);
                if (u < 0 || reached[u] !== 0) continue;
                reached[u] = 1;
                colors[u * 3] = colors[v * 3];
                colors[u * 3 + 1] = colors[v * 3 + 1];
                colors[u * 3 + 2] = colors[v * 3 + 2];
                queue[tail++] = u;
            }
        }
        floodSub.end();
    }

    ordinalOf.release();

    return colors;
};

/**
 * Number of models a region needs at a given reduction, and whether that fits.
 *
 * Tiles are aligned to the occupied region rather than to the grid origin, and
 * only non-empty tiles become models, so a sparse region needs far fewer than
 * the bounding-box product suggests.
 *
 * @param voxels - Occupied voxel set.
 * @param factor - Whole reduction factor per axis.
 * @returns Non-empty tile count.
 */
const countVoxModels = (voxels: VoxelSet, factor: number): number => {
    const { count, ix, iy, iz } = voxels;
    const minX = Math.floor(voxels.minIx / factor);
    const minY = Math.floor(voxels.minIy / factor);
    const minZ = Math.floor(voxels.minIz / factor);
    const spanX = Math.floor(voxels.maxIx / factor) - minX + 1;
    const spanY = Math.floor(voxels.maxIy / factor) - minY + 1;
    const ntx = Math.ceil(spanX / MAX_VOX_DIM);
    const nty = Math.ceil(spanY / MAX_VOX_DIM);

    const seen = new IntKeyMap();
    let models = 0;
    for (let v = 0; v < count; v++) {
        const tx = ((Math.floor(ix[v] / factor) - minX) / MAX_VOX_DIM) | 0;
        const ty = ((Math.floor(iy[v] / factor) - minY) / MAX_VOX_DIM) | 0;
        const tz = ((Math.floor(iz[v] / factor) - minZ) / MAX_VOX_DIM) | 0;
        const key = tx + ty * ntx + tz * ntx * nty;
        if (!seen.has(key)) {
            seen.set(key, 1);
            models++;
            if (models > MAX_VOX_MODELS) break;
        }
    }
    seen.releaseStorage();
    return models;
};

/**
 * Smallest whole reduction factor whose model count fits `MAX_VOX_MODELS`.
 *
 * @param voxels - Occupied voxel set.
 * @returns Reduction factor, at least 1.
 */
const minVoxFactorForModels = (voxels: VoxelSet): number => {
    let factor = 1;
    // Doubling first, so a wildly over-large grid does not walk every factor;
    // then a linear back-off to the smallest one that still fits.
    while (countVoxModels(voxels, factor) > MAX_VOX_MODELS) factor *= 2;
    let best = factor;
    for (let f = Math.max(1, factor >> 1) + 1; f < factor; f++) {
        if (countVoxModels(voxels, f) <= MAX_VOX_MODELS) {
            best = f;
            break;
        }
    }
    return best;
};

/**
 * Fail before any colouring work when the region cannot be represented.
 *
 * Exceeding 256 voxels on an axis is no longer fatal — the region is split into
 * tiled models — so the only hard failure left is needing more models than
 * `MAX_VOX_MODELS`.
 *
 * @param voxels - Occupied voxel set, measured on the collision grid.
 * @param voxelResolution - Collision voxel size in world units.
 * @param factor - Whole reduction factor the `.vox` will use.
 * @throws Error naming a `--collision-voxels-size` that would fit.
 */
const assertVoxFits = (voxels: VoxelSet, voxelResolution: number, factor = 1): void => {
    const models = countVoxModels(voxels, factor);
    if (models <= MAX_VOX_MODELS) return;

    const [dimX, dimY, dimZ] = coarseSpan(voxels, factor);
    const spanWorld = (lo: number, hi: number): string => ((hi - lo + 1) * voxelResolution).toFixed(1);
    const minSize = parseFloat((minVoxFactorForModels(voxels) * voxelResolution).toPrecision(6));
    const at = factor > 1 ? ` at a voxel size of ${parseFloat((voxelResolution * factor).toPrecision(6))}` : '';

    throw new Error(
        `The .vox model would be ${dimX}x${dimY}x${dimZ} voxels${at}, which needs more than ` +
        `${MAX_VOX_MODELS} models of ${MAX_VOX_DIM}x${MAX_VOX_DIM}x${MAX_VOX_DIM} to represent. ` +
        'The occupied region spans ' +
        `${spanWorld(voxels.minIx, voxels.maxIx)}x${spanWorld(voxels.minIy, voxels.maxIy)}x` +
        `${spanWorld(voxels.minIz, voxels.maxIz)} world units, so the .vox needs a voxel size of ` +
        `at least ${minSize}. Pass --collision-voxels-size ${minSize} to coarsen only the .vox and ` +
        'keep the collision grid, or raise --voxel-params to coarsen everything.');
};

/**
 * Encode the `nTRN`/`nGRP`/`nSHP` scene graph placing each tiled model.
 *
 * Shape, per the MagicaVoxel extension format:
 *
 *     nTRN(0) -> nGRP(1) -> nTRN(2+2i) -> nSHP(3+2i) -> model i
 *
 * `_t` is the position of a model's *centre*, so a tile whose contents start at
 * `origin` translates by `origin + floor(size / 2)`. The whole assembly is
 * shifted by `-floor(total / 2)` so it is centred on the origin, matching where
 * MagicaVoxel puts a single untransformed model.
 *
 * @param tileCount - Number of models.
 * @param tileMin - Per-tile minimum vox coordinate, 3 per tile.
 * @param tileMax - Per-tile maximum vox coordinate, 3 per tile.
 * @param totalX - Overall vox-space X extent.
 * @param totalY - Overall vox-space Y extent.
 * @param totalZ - Overall vox-space Z extent.
 * @returns Concatenated scene-graph chunks.
 */
const encodeSceneGraph = (
    tileCount: number,
    tileMin: Int32Array,
    tileMax: Int32Array,
    totalX: number,
    totalY: number,
    totalZ: number
): Uint8Array => {
    const parts: number[] = [];

    const putInt = (value: number): void => {
        parts.push(value & 0xff, (value >>> 8) & 0xff, (value >>> 16) & 0xff, (value >>> 24) & 0xff);
    };
    const putTag = (tag: string): void => {
        for (let i = 0; i < 4; i++) parts.push(tag.charCodeAt(i));
    };
    const putString = (text: string): void => {
        const encoded = new TextEncoder().encode(text);
        putInt(encoded.length);
        for (const b of encoded) parts.push(b);
    };
    // DICT: pair count, then STRING key/value pairs
    const putDict = (entries: [string, string][]): void => {
        putInt(entries.length);
        for (const [key, value] of entries) {
            putString(key);
            putString(value);
        }
    };
    // chunks carry their content size, so the body is measured then patched
    const putChunk = (tag: string, body: () => void): void => {
        putTag(tag);
        const sizeAt = parts.length;
        putInt(0);
        putInt(0);
        const from = parts.length;
        body();
        const size = parts.length - from;
        parts[sizeAt] = size & 0xff;
        parts[sizeAt + 1] = (size >>> 8) & 0xff;
        parts[sizeAt + 2] = (size >>> 16) & 0xff;
        parts[sizeAt + 3] = (size >>> 24) & 0xff;
    };

    const halfX = Math.floor(totalX / 2);
    const halfY = Math.floor(totalY / 2);
    const halfZ = Math.floor(totalZ / 2);

    // root transform, holding the group
    putChunk('nTRN', () => {
        putInt(0);          // node id
        putDict([]);        // node attributes
        putInt(1);          // child: the group
        putInt(-1);         // reserved
        putInt(-1);         // layer
        putInt(1);          // frame count
        putDict([]);        // frame attributes
    });

    putChunk('nGRP', () => {
        putInt(1);
        putDict([]);
        putInt(tileCount);
        for (let t = 0; t < tileCount; t++) putInt(2 + t * 2);
    });

    for (let t = 0; t < tileCount; t++) {
        const ox = tileMin[t * 3];
        const oy = tileMin[t * 3 + 1];
        const oz = tileMin[t * 3 + 2];
        const sx = tileMax[t * 3] - ox + 1;
        const sy = tileMax[t * 3 + 1] - oy + 1;
        const sz = tileMax[t * 3 + 2] - oz + 1;
        const tx = ox + Math.floor(sx / 2) - halfX;
        const ty = oy + Math.floor(sy / 2) - halfY;
        const tz = oz + Math.floor(sz / 2) - halfZ;

        putChunk('nTRN', () => {
            putInt(2 + t * 2);
            putDict([]);
            putInt(3 + t * 2);      // child: the shape
            putInt(-1);
            putInt(0);              // default layer
            putInt(1);
            putDict([['_t', `${tx} ${ty} ${tz}`]]);
        });

        putChunk('nSHP', () => {
            putInt(3 + t * 2);
            putDict([]);
            putInt(1);              // one model
            putInt(t);              // model id = index in stored order
            putDict([]);            // model attributes
        });
    }

    return new Uint8Array(parts);
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

    // Vox space is Z-up with +Y running away from the viewer; engine space is
    // Y-up with +Z towards it. Negating Z keeps the model from coming out
    // mirrored. Coordinates are relative to the occupied region's corner.
    const voxCoordX = (v: number): number => ix[v] - minIx;
    const voxCoordY = (v: number): number => maxIz - iz[v];
    const voxCoordZ = (v: number): number => iy[v] - minIy;

    // Split into tiles no larger than the format's per-axis limit. Tiles are
    // aligned to the region corner and only non-empty ones become models.
    const ntx = Math.ceil(dimX / MAX_VOX_DIM);
    const nty = Math.ceil(dimZ / MAX_VOX_DIM);
    const tileStride = ntx * nty;

    const tileOf = new IntKeyMap();
    let tileCount = 0;
    const tileMin = new Int32Array(MAX_VOX_MODELS * 3);
    const tileMax = new Int32Array(MAX_VOX_MODELS * 3);
    const tileVoxels = new Int32Array(MAX_VOX_MODELS);
    const tileIndexOf = new Int32Array(count);

    for (let v = 0; v < count; v++) {
        const x = voxCoordX(v);
        const y = voxCoordY(v);
        const z = voxCoordZ(v);
        const key = (x / MAX_VOX_DIM | 0) + (y / MAX_VOX_DIM | 0) * ntx +
            (z / MAX_VOX_DIM | 0) * tileStride;
        let t = tileOf.get(key);
        if (t === -1) {
            if (tileCount >= MAX_VOX_MODELS) {
                // assertVoxFits runs before any colouring, so reaching here
                // means the caller skipped it
                throw new Error(
                    `The .vox model needs more than ${MAX_VOX_MODELS} models to represent.`);
            }
            t = tileCount++;
            tileOf.set(key, t);
            tileMin[t * 3] = x;
            tileMin[t * 3 + 1] = y;
            tileMin[t * 3 + 2] = z;
            tileMax[t * 3] = x;
            tileMax[t * 3 + 1] = y;
            tileMax[t * 3 + 2] = z;
        } else {
            if (x < tileMin[t * 3]) tileMin[t * 3] = x;
            if (y < tileMin[t * 3 + 1]) tileMin[t * 3 + 1] = y;
            if (z < tileMin[t * 3 + 2]) tileMin[t * 3 + 2] = z;
            if (x > tileMax[t * 3]) tileMax[t * 3] = x;
            if (y > tileMax[t * 3 + 1]) tileMax[t * 3 + 1] = y;
            if (z > tileMax[t * 3 + 2]) tileMax[t * 3 + 2] = z;
        }
        tileIndexOf[v] = t;
        tileVoxels[t]++;
    }
    tileOf.releaseStorage();

    // group voxels by tile, so each model's XYZI is one contiguous run
    const tileStart = new Int32Array(tileCount + 1);
    for (let t = 0; t < tileCount; t++) tileStart[t + 1] = tileStart[t] + tileVoxels[t];
    const cursor = new Int32Array(tileCount);
    const order = new Int32Array(count);
    for (let v = 0; v < count; v++) {
        const t = tileIndexOf[v];
        order[tileStart[t] + cursor[t]++] = v;
    }

    // A single model needs no scene graph, which keeps the common case byte for
    // byte what a plain single-model exporter produces.
    const sceneGraph = tileCount > 1 ?
        encodeSceneGraph(tileCount, tileMin, tileMax, dimX, dimZ, dimY) :
        new Uint8Array(0);

    let modelChunksSize = 0;
    for (let t = 0; t < tileCount; t++) {
        modelChunksSize += (CHUNK_HEADER_SIZE + 12) +
            (CHUNK_HEADER_SIZE + 4 + tileVoxels[t] * 4);
    }

    const childrenSize = modelChunksSize + sceneGraph.length +
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

    for (let t = 0; t < tileCount; t++) {
        // Sizes are tight around each tile's own contents, not a full 256 cube,
        // so a tile holding a thin sliver stays small.
        const ox = tileMin[t * 3];
        const oy = tileMin[t * 3 + 1];
        const oz = tileMin[t * 3 + 2];
        const sx = tileMax[t * 3] - ox + 1;
        const sy = tileMax[t * 3 + 1] - oy + 1;
        const sz = tileMax[t * 3 + 2] - oz + 1;

        writeTag('SIZE');
        writeInt(12);
        writeInt(0);
        writeInt(sx);
        writeInt(sy);
        writeInt(sz);

        const tileCountT = tileVoxels[t];
        writeTag('XYZI');
        writeInt(4 + tileCountT * 4);
        writeInt(0);
        writeInt(tileCountT);
        const from = tileStart[t];
        const to = tileStart[t + 1];
        for (let i = from; i < to; i++) {
            const v = order[i];
            bytes[o++] = voxCoordX(v) - ox;
            bytes[o++] = voxCoordY(v) - oy;
            bytes[o++] = voxCoordZ(v) - oz;
            bytes[o++] = voxIndex[v];
        }
    }

    bytes.set(sceneGraph, o);
    o += sceneGraph.length;

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

    const models = tileCount > 1 ? `, ${tileCount} models` : '';
    logger.info(
        `vox: ${dimX}x${dimZ}x${dimY}, ${fmtCount(count)} voxels, ` +
        `${palette.length / 3} colours${models}`);

    if (count > LARGE_VOX_VOXELS) {
        logger.warn(
            `the .vox holds ${fmtCount(count)} voxels (${fmtBytes(total)}); MagicaVoxel is ` +
            'unlikely to open it usefully. Pass a larger --collision-voxels-size to coarsen ' +
            'the model without touching the collision grid.');
    }

    return bytes;
};

export {
    buildCollisionVox,
    countVoxModels,
    downsampleGrid,
    enumerateOccupied,
    minVoxFactorForModels,
    assertVoxFits,
    MAX_VOX_DIM,
    MAX_VOX_MODELS,
    type VoxColorSource,
    type VoxelSet
};
