import type { Bounds } from '../data-table';
import { colorizeVertices, computeVertexNormals, coplanarMerge, mapToPalette, marchingCubes, palettizeColors, parsePaletteColors, smoothVertexColors, voxelFaces, type Mesh, type SplatColorColumns } from '../mesh';
import type { GaussianBVH } from '../spatial';
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
 * @param nodeRotation - Optional node rotation quaternion (x, y, z, w) applied
 * to the mesh node. Omitted (or null) leaves the node with no rotation.
 * @returns GLB file as a Uint8Array
 */
function encodeGlb(
    positions: Float32Array,
    indices: Uint32Array,
    colors?: Float32Array,
    nodeRotation?: [number, number, number, number] | null
): Uint8Array {
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
        nodes: [nodeRotation ? { mesh: 0, rotation: nodeRotation } : { mesh: 0 }],
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


/**
 * Write the flat colour of a single triangle into `out`.
 *
 * @param t - Triangle index.
 * @param indices - Mesh index buffer.
 * @param src - Per-vertex colour triplets.
 * @param out - Destination triplet, written at offset 0.
 * @param quantized - True when `src` holds palette colours.
 */
const writeFaceColorOfTri = (
    t: number,
    indices: Uint32Array,
    src: Float32Array,
    out: Float32Array,
    quantized: boolean
): void => {
    writeFaceColor([indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]], src, out, 0, quantized);
};

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
 * @param options - Extra options.
 * @param options.nodeRotation - Optional node rotation quaternion (x, y, z, w)
 * applied to the mesh node. Omitted (or null) leaves the node with no rotation.
 * @returns GLB bytes, or null if no triangles were generated
 * @throws Error if shape is `voxel` or `tris` and `colorSource` is null
 */
const buildCollisionMesh = (
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
    options: { nodeRotation?: [number, number, number, number] | null } = {}
): Uint8Array | null => {
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
        return null;
    }

    let colors: Float32Array | undefined;
    if (colored) {
        if (!colorSource) {
            throw new Error(`colorSource is required for collision mesh shape '${shape}'`);
        }
        const colorSub = logger.group('Coloring vertices');
        const normalsSub = logger.group('Vertex normals');
        const normals = computeVertexNormals(finalMesh.positions, finalMesh.indices);
        normalsSub.end();

        const sampleSub = logger.group('Sampling splats');
        colors = colorizeVertices(finalMesh.positions, normals, colorSource.bvh, colorSource.columns, voxelResolution, colorSource.mode);
        sampleSub.end();

        if (colorSource.smoothRadius !== undefined && colorSource.smoothRadius > 0) {
            const smoothSub = logger.group('Denoising');
            colors = smoothVertexColors(colors, finalMesh.positions, colorSource.smoothRadius, voxelResolution);
            logger.info(`denoised: ${colorSource.smoothRadius} voxel radius`);
            smoothSub.end();
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
            const paletteSub = logger.group('Palette');
            colors = mapToPalette(colors, parsePaletteColors(palette), paletteOpts);
            logger.info(`palette: ${palette.length} fixed colours`);
            paletteSub.end();
            quantized = true;
        } else if (typeof palette === 'number' && palette >= 1) {
            const paletteSub = logger.group('Palette');
            colors = palettizeColors(colors, palette, paletteOpts);
            logger.info(`palette: ${palette} colours`);
            paletteSub.end();
            quantized = true;
        }

        if (quantized && colorSource.coherentRadius !== undefined && colorSource.coherentRadius > 0) {
            logger.info(`coherent: ${colorSource.coherentRadius} voxel radius`);
        }

        if (colorSource.flatShade) {
            const flatSub = logger.group('Flat shading');
            const isVoxel = shape === 'voxel';
            const numTris = finalMesh.indices.length / 3;
            const srcPositions = finalMesh.positions;
            const srcIndices = finalMesh.indices;
            const srcColors = colors;

            // Flat shading needs a colour per face, so vertices shared between
            // faces of different colours have to be split. For voxel quads the
            // split unit is the quad, not the triangle: its two triangles share
            // one colour and an edge, so 4 vertices suffice where un-indexing
            // per triangle would emit 6. That is a third of the position and
            // colour data on a mesh where those arrays run to gigabytes.
            //
            // voxelFaces(perVoxel) emits the two triangles of every quad
            // consecutively, so pairing is positional. The edge map this
            // replaced needed an entry per mesh edge, which both blew V8's 2^24
            // Map cap and allocated an object plus an array per edge.
            const pairStride = isVoxel ? 2 : 1;

            // scratch for one face's vertex indices: up to 4 for a quad
            const faceVerts: number[] = [];

            /**
             * Vertex indices of the face starting at triangle `t`, and whether
             * its triangles could be welded into a single quad.
             *
             * @param t - First triangle of the face.
             * @returns True when `t` and `t + 1` form a 4-vertex quad.
             */
            const collectFace = (t: number): boolean => {
                const a = srcIndices[t * 3];
                const b = srcIndices[t * 3 + 1];
                const c = srcIndices[t * 3 + 2];
                faceVerts.length = 3;
                faceVerts[0] = a;
                faceVerts[1] = b;
                faceVerts[2] = c;
                if (pairStride !== 2 || t + 1 >= numTris) return false;
                let extras = 0;
                let extra = -1;
                for (let k = 0; k < 3; k++) {
                    const v = srcIndices[(t + 1) * 3 + k];
                    if (v !== a && v !== b && v !== c) {
                        extras++;
                        extra = v;
                    }
                }
                if (extras !== 1) return false;
                faceVerts.push(extra);
                return true;
            };

            // Pass 1: size the output exactly. Growing a multi-gigabyte typed
            // array by doubling would transiently hold both copies.
            let outVertexCount = 0;
            let outTriCount = 0;
            for (let t = 0; t < numTris; t += pairStride) {
                if (collectFace(t)) {
                    outVertexCount += 4;
                    outTriCount += 2;
                } else {
                    outVertexCount += 3;
                    outTriCount += 1;
                    if (pairStride === 2 && t + 1 < numTris) {
                        // a partner that would not weld is emitted on its own,
                        // so it needs its own three vertices
                        outVertexCount += 3;
                        outTriCount += 1;
                    }
                }
            }

            const flatPositions = new Float32Array(outVertexCount * 3);
            const flatColors = new Float32Array(outVertexCount * 3);
            const flatIndices = new Uint32Array(outTriCount * 3);

            const faceColor = new Float32Array(3);
            let vw = 0;
            let iw = 0;

            for (let t = 0; t < numTris; t += pairStride) {
                const welded = collectFace(t);
                const count = welded ? 4 : 3;
                const base = vw;

                writeFaceColor(faceVerts, srcColors, faceColor, 0, quantized);

                for (let k = 0; k < count; k++) {
                    const v = faceVerts[k];
                    flatPositions[(base + k) * 3] = srcPositions[v * 3];
                    flatPositions[(base + k) * 3 + 1] = srcPositions[v * 3 + 1];
                    flatPositions[(base + k) * 3 + 2] = srcPositions[v * 3 + 2];
                    flatColors[(base + k) * 3] = faceColor[0];
                    flatColors[(base + k) * 3 + 1] = faceColor[1];
                    flatColors[(base + k) * 3 + 2] = faceColor[2];
                }
                vw += count;

                // first triangle keeps its winding
                flatIndices[iw++] = base;
                flatIndices[iw++] = base + 1;
                flatIndices[iw++] = base + 2;

                if (welded) {
                    // remap the partner through the same 4 vertices, so its
                    // winding survives without duplicating them
                    for (let k = 0; k < 3; k++) {
                        const v = srcIndices[(t + 1) * 3 + k];
                        let local = 3;
                        for (let j = 0; j < 4; j++) {
                            if (faceVerts[j] === v) {
                                local = j;
                                break;
                            }
                        }
                        flatIndices[iw++] = base + local;
                    }
                } else if (pairStride === 2 && t + 1 < numTris) {
                    // an unpaired partner still needs emitting, on its own
                    const pBase = vw;
                    writeFaceColorOfTri(t + 1, srcIndices, srcColors, faceColor, quantized);
                    for (let k = 0; k < 3; k++) {
                        const v = srcIndices[(t + 1) * 3 + k];
                        flatPositions[(pBase + k) * 3] = srcPositions[v * 3];
                        flatPositions[(pBase + k) * 3 + 1] = srcPositions[v * 3 + 1];
                        flatPositions[(pBase + k) * 3 + 2] = srcPositions[v * 3 + 2];
                        flatColors[(pBase + k) * 3] = faceColor[0];
                        flatColors[(pBase + k) * 3 + 1] = faceColor[1];
                        flatColors[(pBase + k) * 3 + 2] = faceColor[2];
                    }
                    vw += 3;
                    flatIndices[iw++] = pBase;
                    flatIndices[iw++] = pBase + 1;
                    flatIndices[iw++] = pBase + 2;
                }
            }

            finalMesh = { positions: flatPositions, indices: flatIndices };
            colors = flatColors;
            logger.info(isVoxel ?
                `flat-shading: per-voxel-quad colours, ${fmtCount(outVertexCount)} vertices` :
                `flat-shading: per-triangle colours, ${fmtCount(outVertexCount)} vertices`);
            flatSub.end();
        }
        colorSub.end();
    }

    const encodeSub = logger.group('Encoding GLB');
    const glb = encodeGlb(finalMesh.positions, finalMesh.indices, colors, options.nodeRotation);
    encodeSub.end();

    g.end();
    return glb;
};

export { buildCollisionMesh, encodeGlb };
