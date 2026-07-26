import type { Bounds } from '../data-table';
import { colorizeVertices, computeVertexNormals, coplanarMerge, mapToPalette, marchingCubes, palettizeColors, parsePaletteColors, smoothVertexColors, voxelFaces, type Mesh, type SplatColorColumns } from '../mesh';
import type { GaussianBVH } from '../spatial';
import { buildCollisionVox } from './collision-vox';
import type { CollisionColorMode, CollisionColorPalette, CollisionMeshShape } from '../types';
import { fmtCount, logger } from '../utils';
import { SparseVoxelGrid } from '../voxel/sparse-voxel-grid';

/**
 * Build a minimal GLB (glTF 2.0 binary) file containing a single triangle mesh.
 *
 * The output contains only positions and triangle indices — no normals or
 * UVs — suitable for collision meshes. When `colors` is provided, a COLOR_0
 * vertex attribute (VEC3 float, linear space) and a double-sided default
 * material are added.
 *
 * @param positions - Vertex positions (3 floats per vertex)
 * @param indices - Triangle indices (3 per triangle, unsigned 32-bit)
 * @param colors - Optional linear-space vertex colors (3 floats per vertex)
 * @returns GLB file as a Uint8Array
 */
function encodeGlb(positions: Float32Array, indices: Uint32Array, colors?: Float32Array): Uint8Array {
    const vertexCount = positions.length / 3;
    const indexCount = indices.length;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < positions.length; i += 3) {
        const x = positions[i], y = positions[i + 1], z = positions[i + 2];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (z < minZ) minZ = z;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
        if (z > maxZ) maxZ = z;
    }

    const positionsByteLength = positions.byteLength;
    const indicesByteLength = indices.byteLength;
    const colorsByteLength = colors?.byteLength ?? 0;
    const totalBinSize = positionsByteLength + indicesByteLength + colorsByteLength;

    // positions (Float32) and indices (Uint32) byte lengths are always
    // multiples of 4, so the colors bufferView stays 4-byte aligned in the
    // natural positions -> indices -> colors order without any padding
    if (colors && (positionsByteLength + indicesByteLength) % 4 !== 0) {
        throw new Error('COLOR_0 bufferView byteOffset is not 4-byte aligned');
    }

    const primitive: {
        attributes: Record<string, number>;
        indices: number;
        material?: number;
    } = {
        attributes: { POSITION: 0 },
        indices: 1
    };

    const accessors: object[] = [
        {
            bufferView: 0,
            componentType: 5126, // FLOAT
            count: vertexCount,
            type: 'VEC3',
            min: [minX, minY, minZ],
            max: [maxX, maxY, maxZ]
        },
        {
            bufferView: 1,
            componentType: 5125, // UNSIGNED_INT
            count: indexCount,
            type: 'SCALAR'
        }
    ];

    const bufferViews: object[] = [
        {
            buffer: 0,
            byteOffset: 0,
            byteLength: positionsByteLength,
            target: 34962 // ARRAY_BUFFER
        },
        {
            buffer: 0,
            byteOffset: positionsByteLength,
            byteLength: indicesByteLength,
            target: 34963 // ELEMENT_ARRAY_BUFFER
        }
    ];

    const gltf: {
        asset: object;
        scene: number;
        scenes: object[];
        nodes: object[];
        meshes: object[];
        accessors: object[];
        bufferViews: object[];
        buffers: object[];
        materials?: object[];
    } = {
        asset: { version: '2.0', generator: 'splat-transform' },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0 }],
        meshes: [{
            primitives: [primitive]
        }],
        accessors,
        bufferViews,
        buffers: [{ byteLength: totalBinSize }]
    };

    if (colors) {
        primitive.attributes.COLOR_0 = 2;
        primitive.material = 0;
        accessors.push({
            bufferView: 2,
            componentType: 5126, // FLOAT
            count: vertexCount,
            type: 'VEC3'
        });
        bufferViews.push({
            buffer: 0,
            byteOffset: positionsByteLength + indicesByteLength,
            byteLength: colorsByteLength,
            target: 34962 // ARRAY_BUFFER
        });
        gltf.materials = [{
            pbrMetallicRoughness: {
                baseColorFactor: [1, 1, 1, 1],
                metallicFactor: 0,
                roughnessFactor: 1
            },
            doubleSided: true
        }];
    }

    const jsonString = JSON.stringify(gltf);
    const jsonEncoder = new TextEncoder();
    const jsonBytes = jsonEncoder.encode(jsonString);

    // JSON chunk must be padded to 4-byte alignment with spaces (0x20)
    const jsonPadding = (4 - (jsonBytes.length % 4)) % 4;
    const jsonChunkLength = jsonBytes.length + jsonPadding;

    // BIN chunk must be padded to 4-byte alignment with zeros
    const binPadding = (4 - (totalBinSize % 4)) % 4;
    const binChunkLength = totalBinSize + binPadding;

    // GLB layout: header (12) + JSON chunk header (8) + JSON data + BIN chunk header (8) + BIN data
    const totalLength = 12 + 8 + jsonChunkLength + 8 + binChunkLength;
    const buffer = new ArrayBuffer(totalLength);
    const view = new DataView(buffer);
    const byteArray = new Uint8Array(buffer);
    let offset = 0;

    // GLB header
    view.setUint32(offset, 0x46546C67, true); offset += 4; // magic: "glTF"
    view.setUint32(offset, 2, true); offset += 4;           // version: 2
    view.setUint32(offset, totalLength, true); offset += 4;  // total length

    // JSON chunk header
    view.setUint32(offset, jsonChunkLength, true); offset += 4;
    view.setUint32(offset, 0x4E4F534A, true); offset += 4; // type: "JSON"

    // JSON chunk data
    byteArray.set(jsonBytes, offset); offset += jsonBytes.length;
    for (let i = 0; i < jsonPadding; i++) {
        byteArray[offset++] = 0x20;
    }

    // BIN chunk header
    view.setUint32(offset, binChunkLength, true); offset += 4;
    view.setUint32(offset, 0x004E4942, true); offset += 4; // type: "BIN\0"

    // BIN chunk data: positions, then indices, then optional colors
    byteArray.set(new Uint8Array(positions.buffer, positions.byteOffset, positionsByteLength), offset);
    offset += positionsByteLength;
    byteArray.set(new Uint8Array(indices.buffer, indices.byteOffset, indicesByteLength), offset);
    offset += indicesByteLength;
    if (colors) {
        byteArray.set(new Uint8Array(colors.buffer, colors.byteOffset, colorsByteLength), offset);
    }

    return byteArray;
}

/**
 * Extract a collision mesh from voxel data and encode it as a GLB file.
 *
 * Generates collision geometry from voxel data using either the smooth
 * marching-cubes path or a direct watertight voxel-face mesh.
 *
 * @param grid - Voxel grid after filtering / nav phases
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param voxelResolution - Size of each voxel in world units
 * @param shape - Collision mesh shape to generate: `faces` and `voxel` use
 * the voxel-face mesh, `smooth` and `tris` use marching cubes with coplanar
 * merging. `voxel` and `tris` also bake splat colors into a COLOR_0 vertex
 * attribute.
 * @param colorSource - Splat BVH, color columns, coloring mode, optional
 * palette quantisation, optional spatial smooth/coherence radii, and optional
 * flat-shade setting used to colorize mesh vertices. Required for the `voxel`
 * and `tris` shapes, ignored otherwise.
 * @returns GLB bytes, or null if no triangles were generated
 * @throws Error if shape is `voxel` or `tris` and `colorSource` is null
 */
/**
 * Write one uniform colour for a face into `out`.
 *
 * Averaging is right for continuous vertex colours, but once colours have been
 * quantized to a palette an average across a face straddling two entries is a
 * blend that is not in the palette — so pick the face's dominant colour
 * instead, keeping the output a strict subset of the palette. Ties go to the
 * earliest vertex, which keeps the result deterministic.
 *
 * @param verts - Vertex indices making up the face (3 or 4).
 * @param src - Per-vertex colour triplets.
 * @param out - Destination colour array.
 * @param outOff - Offset in `out` to write the triplet at.
 * @param quantized - True when `src` holds palette colours.
 */
const writeFaceColor = (
    verts: number[],
    src: Float32Array,
    out: Float32Array,
    outOff: number,
    quantized: boolean
): void => {
    if (quantized) {
        let bestVert = verts[0];
        let bestCount = 0;
        for (const v of verts) {
            let count = 0;
            for (const w of verts) {
                if (src[w * 3] === src[v * 3] &&
                    src[w * 3 + 1] === src[v * 3 + 1] &&
                    src[w * 3 + 2] === src[v * 3 + 2]) {
                    count++;
                }
            }
            if (count > bestCount) {
                bestCount = count;
                bestVert = v;
            }
        }
        out[outOff] = src[bestVert * 3];
        out[outOff + 1] = src[bestVert * 3 + 1];
        out[outOff + 2] = src[bestVert * 3 + 2];
        return;
    }

    for (let ch = 0; ch < 3; ch++) {
        let sum = 0;
        for (const v of verts) sum += src[v * 3 + ch];
        out[outOff + ch] = sum / verts.length;
    }
};

const buildCollisionOutputs = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    shape: CollisionMeshShape = 'smooth',
    colorSource: {
        bvh: GaussianBVH;
        columns: SplatColorColumns;
        mode: CollisionColorMode;
        palette?: CollisionColorPalette;
        flatShade?: boolean;
        smoothRadius?: number;
        coherentRadius?: number;
    } | null = null,
    opts: { emitVox?: boolean } = {}
): { glb: Uint8Array | null; vox: Uint8Array | null } => {
    const g = logger.group('Collision mesh');

    const colored = shape === 'voxel' || shape === 'tris';

    let finalMesh: Mesh;
    if (shape === 'faces' || shape === 'voxel') {
        const extractSub = logger.group('Extracting voxel faces');
        finalMesh = voxelFaces(grid, gridBounds, voxelResolution, { perVoxel: shape === 'voxel' });
        logger.info(`vertices: ${fmtCount(finalMesh.positions.length / 3)}`);
        logger.info(`triangles: ${fmtCount(finalMesh.indices.length / 3)}`);
        extractSub.end();
    } else {
        const extractSub = logger.group('Extracting');
        const preMergedMesh = marchingCubes(grid, gridBounds, voxelResolution, { mergeFlatFaces: true });
        logger.info(`pre-merged vertices: ${fmtCount(preMergedMesh.positions.length / 3)}`);
        logger.info(`pre-merged triangles: ${fmtCount(preMergedMesh.indices.length / 3)}`);
        const preMergedIndexCount = preMergedMesh.indices.length;
        extractSub.end();

        if (preMergedIndexCount < 3) {
            finalMesh = preMergedMesh;
        } else {
            const mergeSub = logger.group('Merging coplanar faces');
            finalMesh = coplanarMerge(preMergedMesh, voxelResolution);

            const reduction = (1 - finalMesh.indices.length / preMergedIndexCount) * 100;
            logger.info(`merged vertices: ${fmtCount(finalMesh.positions.length / 3)}`);
            logger.info(`merged triangles: ${fmtCount(finalMesh.indices.length / 3)}`);
            logger.info(`reduction: ${reduction.toFixed(0)}%`);
            mergeSub.end();
        }
    }

    if (finalMesh.indices.length < 3) {
        logger.warn('no triangles generated, skipping GLB output');
        g.end();
        return { glb: null, vox: null };
    }

    let colors: Float32Array | undefined;
    if (colored) {
        if (!colorSource) {
            throw new Error(`colorSource is required for collision mesh shape '${shape}'`);
        }
        const colorSub = logger.group('Coloring vertices');
        const normals = computeVertexNormals(finalMesh.positions, finalMesh.indices);
        colors = colorizeVertices(finalMesh.positions, normals, colorSource.bvh, colorSource.columns, voxelResolution, colorSource.mode);

        if (colorSource.smoothRadius !== undefined && colorSource.smoothRadius > 0) {
            colors = smoothVertexColors(colors, finalMesh.positions, colorSource.smoothRadius, voxelResolution);
            logger.info(`smoothed: ${colorSource.smoothRadius} voxel radius`);
        }

        const palette = colorSource.palette;
        const paletteOpts = {
            positions: finalMesh.positions,
            voxelResolution,
            coherentRadius: colorSource.coherentRadius
        };

        let quantized = false;
        if (Array.isArray(palette) && palette.length >= 1) {
            // fixed palette: the colours are given, so there is nothing to
            // cluster — every vertex just takes its nearest entry
            colors = mapToPalette(colors, parsePaletteColors(palette), paletteOpts);
            logger.info(`palette: ${palette.length} fixed colours`);
            quantized = true;
        } else if (typeof palette === 'number' && palette >= 1) {
            colors = palettizeColors(colors, palette, paletteOpts);
            logger.info(`palette: ${palette} colours`);
            quantized = true;
        }

        if (quantized && colorSource.coherentRadius !== undefined && colorSource.coherentRadius > 0) {
            logger.info(`coherent: ${colorSource.coherentRadius} voxel radius`);
        }

        if (colorSource.flatShade) {
            const isVoxel = shape === 'voxel';
            const numTris = finalMesh.indices.length / 3;

            // per-quad colours for voxel faces (each quad = 2 triangles)
            let quadColor: Float32Array | null = null;
            if (isVoxel) {
                // build an edge map to find the two triangles forming each
                // voxel quad. The diagonal edge has length ≈ √2·voxelResolution
                // in world space; perimeter edges are exactly voxelResolution.
                const edgeKey = (a: number, b: number): number => (a < b ? a * 0x100000000 + b : b * 0x100000000 + a);

                interface EdgeInfo { tris: number[]; len2: number }

                const edgeToInfo = new Map<number, EdgeInfo>();
                for (let t = 0; t < numTris; t++) {
                    const a = finalMesh.indices[t * 3];
                    const b = finalMesh.indices[t * 3 + 1];
                    const c = finalMesh.indices[t * 3 + 2];
                    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
                        const k = edgeKey(u, v);
                        let info = edgeToInfo.get(k);
                        if (!info) {
                            const uOff = u * 3, vOff = v * 3;
                            const dx = finalMesh.positions[uOff] - finalMesh.positions[vOff];
                            const dy = finalMesh.positions[uOff + 1] - finalMesh.positions[vOff + 1];
                            const dz = finalMesh.positions[uOff + 2] - finalMesh.positions[vOff + 2];
                            info = { tris: [], len2: dx * dx + dy * dy + dz * dz };
                            edgeToInfo.set(k, info);
                        }
                        info.tris.push(t);
                    }
                }

                // pair triangles by the diagonal edge (the longer edge shared
                // by exactly two triangles)
                const quadPartner = new Int32Array(numTris).fill(-1);
                for (const info of edgeToInfo.values()) {
                    if (info.tris.length === 2 && info.tris[0] !== info.tris[1]) {
                        // diagonal is approx √2 * voxelResolution long
                        if (info.len2 > voxelResolution * voxelResolution * 1.5) {
                            const t = info.tris[0];
                            const s = info.tris[1];
                            quadPartner[t] = s;
                            quadPartner[s] = t;
                        }
                    }
                }

                // per-quad uniform colour: average the 4 vertex colours
                quadColor = new Float32Array(numTris * 3);
                const visited = new Uint8Array(numTris);
                for (let t = 0; t < numTris; t++) {
                    if (visited[t]) continue;
                    const partner = quadPartner[t];
                    let allVerts: number[];
                    if (partner !== -1 && !visited[partner]) {
                        // this triangle + its partner form a quad;
                        // collect the 4 unique vertex indices
                        const a = finalMesh.indices[t * 3];
                        const b = finalMesh.indices[t * 3 + 1];
                        const c = finalMesh.indices[t * 3 + 2];
                        const dSet = new Set([a, b, c]);
                        const partnerVerts = [
                            finalMesh.indices[partner * 3],
                            finalMesh.indices[partner * 3 + 1],
                            finalMesh.indices[partner * 3 + 2]
                        ];
                        const extra = partnerVerts.filter(v => !dSet.has(v));
                        allVerts = [a, b, c, ...extra];
                        visited[partner] = 1;
                    } else {
                        allVerts = [
                            finalMesh.indices[t * 3],
                            finalMesh.indices[t * 3 + 1],
                            finalMesh.indices[t * 3 + 2]
                        ];
                    }
                    visited[t] = 1;

                    // one uniform colour across the quad's vertices
                    const vOff = t * 3;
                    const partnerOff = partner !== -1 ? partner * 3 : -1;
                    writeFaceColor(allVerts, colors, quadColor, vOff, quantized);
                    if (partnerOff >= 0) {
                        quadColor[partnerOff] = quadColor[vOff];
                        quadColor[partnerOff + 1] = quadColor[vOff + 1];
                        quadColor[partnerOff + 2] = quadColor[vOff + 2];
                    }
                }

                // assign the quad colour to all 4 (or 3) vertices when
                // un-indexing below
                colors = quadColor;
            }

            const flatPositions = new Float32Array(numTris * 9);
            const flatIndices = new Uint32Array(numTris * 3);
            const flatColors = new Float32Array(numTris * 9);

            for (let t = 0; t < numTris; t++) {
                const a = finalMesh.indices[t * 3];
                const b = finalMesh.indices[t * 3 + 1];
                const c = finalMesh.indices[t * 3 + 2];

                // duplicate positions per triangle
                for (let k = 0; k < 3; k++) {
                    flatPositions[t * 9 + k] = finalMesh.positions[a * 3 + k];
                    flatPositions[t * 9 + 3 + k] = finalMesh.positions[b * 3 + k];
                    flatPositions[t * 9 + 6 + k] = finalMesh.positions[c * 3 + k];
                }

                // per-face uniform colour: for voxel the pre-computed quadColor
                // array already holds the value at t*3; for tris collapse the 3
                // vertex colours
                const tOff = t * 3;
                const face = [0, 0, 0];
                if (isVoxel) {
                    face[0] = colors[tOff];
                    face[1] = colors[tOff + 1];
                    face[2] = colors[tOff + 2];
                } else {
                    const tmp = new Float32Array(3);
                    writeFaceColor([a, b, c], colors, tmp, 0, quantized);
                    face[0] = tmp[0];
                    face[1] = tmp[1];
                    face[2] = tmp[2];
                }
                for (let ch = 0; ch < 3; ch++) {
                    flatColors[t * 9 + ch] = face[ch];
                    flatColors[t * 9 + 3 + ch] = face[ch];
                    flatColors[t * 9 + 6 + ch] = face[ch];
                }

                flatIndices[t * 3] = t * 3;
                flatIndices[t * 3 + 1] = t * 3 + 1;
                flatIndices[t * 3 + 2] = t * 3 + 2;
            }

            finalMesh = { positions: flatPositions, indices: flatIndices };
            colors = flatColors;
            logger.info(isVoxel ?
                'flat-shading: per-voxel-quad colours, un-indexed' :
                'flat-shading: per-triangle colours, un-indexed');
        }
        colorSub.end();
    }

    // built from the finished mesh and colours, so the .vox carries whatever
    // the palette and spatial options produced for the GLB
    const vox = opts.emitVox && colors ?
        buildCollisionVox(grid, gridBounds, voxelResolution, finalMesh, colors) :
        null;

    g.end();
    return { glb: encodeGlb(finalMesh.positions, finalMesh.indices, colors), vox };
};

/**
 * Extract a collision mesh from voxel data and encode it as a GLB file.
 *
 * Thin wrapper over `buildCollisionOutputs` for callers that only want the GLB.
 *
 * @param grid - Voxel grid after filtering / nav phases
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param voxelResolution - Size of each voxel in world units
 * @param shape - Collision mesh shape to generate
 * @param colorSource - Colour inputs; required for the `voxel` and `tris` shapes
 * @returns GLB bytes, or null if no triangles were generated
 */
const buildCollisionMesh = (
    grid: SparseVoxelGrid,
    gridBounds: Bounds,
    voxelResolution: number,
    shape: CollisionMeshShape = 'smooth',
    colorSource: Parameters<typeof buildCollisionOutputs>[4] = null
): Uint8Array | null => {
    return buildCollisionOutputs(grid, gridBounds, voxelResolution, shape, colorSource).glb;
};

export { buildCollisionMesh, buildCollisionOutputs };
