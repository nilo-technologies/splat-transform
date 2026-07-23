/**
 * Compute per-vertex normals for a triangle mesh.
 *
 * Each triangle contributes its unnormalized cross product (magnitude = 2x
 * the triangle's area) to its three vertices, so the accumulated vertex
 * normals are area-weighted before normalization. Degenerate (zero-area)
 * triangles are skipped. Vertices whose accumulated normal is zero (e.g.
 * isolated vertices or perfectly cancelling triangles) keep a zero normal;
 * callers are expected to handle that case explicitly.
 *
 * @param positions - Vertex positions as packed xyz triplets.
 * @param indices - Triangle indices (3 per triangle).
 * @returns Unit vertex normals as packed xyz triplets, same length as
 * `positions`. Zero vectors where no valid normal could be computed.
 */
const computeVertexNormals = (positions: Float32Array, indices: Uint32Array): Float32Array => {
    const normals = new Float32Array(positions.length);

    for (let t = 0; t + 2 < indices.length; t += 3) {
        const a = indices[t] * 3;
        const b = indices[t + 1] * 3;
        const c = indices[t + 2] * 3;

        const abx = positions[b] - positions[a];
        const aby = positions[b + 1] - positions[a + 1];
        const abz = positions[b + 2] - positions[a + 2];
        const acx = positions[c] - positions[a];
        const acy = positions[c + 1] - positions[a + 1];
        const acz = positions[c + 2] - positions[a + 2];

        // cross(ab, ac); its magnitude is twice the triangle area
        const nx = aby * acz - abz * acy;
        const ny = abz * acx - abx * acz;
        const nz = abx * acy - aby * acx;
        if (nx === 0 && ny === 0 && nz === 0) continue;

        for (const v of [a, b, c]) {
            normals[v] += nx;
            normals[v + 1] += ny;
            normals[v + 2] += nz;
        }
    }

    for (let i = 0; i < normals.length; i += 3) {
        const nx = normals[i];
        const ny = normals[i + 1];
        const nz = normals[i + 2];
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-12) {
            normals[i] = nx / len;
            normals[i + 1] = ny / len;
            normals[i + 2] = nz / len;
        }
    }

    return normals;
};

export { computeVertexNormals };
