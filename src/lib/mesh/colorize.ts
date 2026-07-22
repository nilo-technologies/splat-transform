import type { TypedArray } from '../data-table';
import type { GaussianBVH } from '../spatial';

/**
 * Splat color columns used to colorize mesh vertices. The `f_dc_*` columns hold
 * spherical harmonics DC coefficients and `opacity` holds logit-encoded opacity.
 */
type SplatColorColumns = {
    /** SH DC coefficient for the red channel */
    f_dc_0: TypedArray;

    /** SH DC coefficient for the green channel */
    f_dc_1: TypedArray;

    /** SH DC coefficient for the blue channel */
    f_dc_2: TypedArray;

    /** Logit-encoded opacity */
    opacity: TypedArray;
};

const SH_C0 = 0.28209479177387814;

const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v));

const srgbToLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

/**
 * Compute a linear-space RGB color for every mesh vertex by averaging the display
 * colors of the gaussian splats overlapping the vertex neighborhood.
 *
 * Each vertex queries a box of half-size `voxelResolution` around its position,
 * doubling the radius twice (2x, 4x) if no splat overlaps. Overlapping splats
 * contribute their display color `clamp(0.5 + SH_C0 * f_dc, 0, 1)` weighted by
 * `sigmoid(opacity)`. Vertices with no overlapping splat at any radius fall back
 * to mid-grey. The resulting sRGB colors are converted to linear space.
 *
 * @param positions - Vertex positions as packed xyz triplets.
 * @param bvh - BVH over the gaussian AABBs used for overlap queries.
 * @param columns - Splat color columns (`f_dc_0/1/2` and `opacity`).
 * @param voxelResolution - Initial query box half-size in world units.
 * @returns Linear RGB colors, 3 floats per vertex, same length as `positions`.
 */
const colorizeVertices = (
    positions: Float32Array,
    bvh: GaussianBVH,
    columns: SplatColorColumns,
    voxelResolution: number
): Float32Array => {
    const { f_dc_0, f_dc_1, f_dc_2, opacity } = columns;
    const result = new Float32Array(positions.length);

    for (let i = 0; i < positions.length; i += 3) {
        const px = positions[i];
        const py = positions[i + 1];
        const pz = positions[i + 2];

        // query at 1x, then retry at 2x and 4x if nothing overlaps
        let indices: number[] = [];
        for (let mult = 1; mult <= 4; mult *= 2) {
            const r = voxelResolution * mult;
            indices = bvh.queryOverlappingRaw(px - r, py - r, pz - r, px + r, py + r, pz + r);
            if (indices.length > 0) {
                break;
            }
        }

        let cr = 0.5;
        let cg = 0.5;
        let cb = 0.5;

        if (indices.length > 0) {
            let sumR = 0;
            let sumG = 0;
            let sumB = 0;
            let sumW = 0;

            for (let j = 0; j < indices.length; j++) {
                const idx = indices[j];
                const w = sigmoid(opacity[idx]);
                sumR += w * Math.min(Math.max(0.5 + SH_C0 * f_dc_0[idx], 0), 1);
                sumG += w * Math.min(Math.max(0.5 + SH_C0 * f_dc_1[idx], 0), 1);
                sumB += w * Math.min(Math.max(0.5 + SH_C0 * f_dc_2[idx], 0), 1);
                sumW += w;
            }

            cr = sumR / sumW;
            cg = sumG / sumW;
            cb = sumB / sumW;
        }

        result[i] = srgbToLinear(cr);
        result[i + 1] = srgbToLinear(cg);
        result[i + 2] = srgbToLinear(cb);
    }

    return result;
};

export { colorizeVertices };
export type { SplatColorColumns };
