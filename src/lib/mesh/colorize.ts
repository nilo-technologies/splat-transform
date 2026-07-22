import type { TypedArray } from '../data-table';
import type { GaussianBVH } from '../spatial';
import type { CollisionColorMode } from '../types';

/**
 * Splat color columns used to colorize mesh vertices. The `f_dc_*` columns hold
 * spherical harmonics DC coefficients and `opacity` holds logit-encoded opacity.
 *
 * The `rot_*` (quaternion with `rot_0` = w) and `scale_*` (log-scale) columns are
 * optional. They are required at runtime for the `dominant`, `topk` and
 * `gaussian` coloring modes, which weight splats by their Gaussian density at
 * the vertex; the `average` mode never reads them.
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

    /** Rotation quaternion w component */
    rot_0?: TypedArray;

    /** Rotation quaternion x component */
    rot_1?: TypedArray;

    /** Rotation quaternion y component */
    rot_2?: TypedArray;

    /** Rotation quaternion z component */
    rot_3?: TypedArray;

    /** Log-scale along the splat's first local axis */
    scale_0?: TypedArray;

    /** Log-scale along the splat's second local axis */
    scale_1?: TypedArray;

    /** Log-scale along the splat's third local axis */
    scale_2?: TypedArray;
};

const SH_C0 = 0.28209479177387814;

// Power applied to the size-coverage factor in `gaussianWeightFactor` (see
// below). Empirically chosen: squaring the linear ratio makes the discount
// fall off quickly for splats well below the mesh's voxel resolution while
// leaving splats at or above it (ratio >= 1, clamped) fully weighted.
const SIZE_FACTOR_POWER = 2;

const sigmoid = (v: number): number => 1 / (1 + Math.exp(-v));

const srgbToLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

const displayColor = (f_dc: TypedArray, idx: number): number => Math.min(Math.max(0.5 + SH_C0 * f_dc[idx], 0), 1);

/**
 * Evaluate the weight factor of a splat at a world-space point, combining
 * its Gaussian density with a size-coverage discount.
 *
 * The density term rotates the offset from the point to the splat center
 * into the splat's local frame (`u = R^T d`) and computes the Mahalanobis
 * distance with per-axis variances `sigma_k^2 = exp(scale_k)^2`. This term
 * alone peaks at exactly 1.0 at the splat's own center *regardless of the
 * splat's size* -- real trained 3DGS scenes contain many tiny, near-opaque
 * "cleanup" artifact splats clustered right at surfaces, and without a size
 * discount such a splat can outscore the large splat that actually
 * represents the true surface color, simply by being closer and more
 * opaque. The size-coverage term counters this by discounting splats whose
 * geometric-mean sigma is far smaller than the mesh's own `voxelResolution`,
 * since such splats cannot meaningfully represent color at that resolution
 * regardless of proximity or opacity. Splats with a zero-length quaternion
 * have no orientation and contribute zero weight (unaffected by the size
 * term, checked first).
 *
 * @param columns - Splat columns including `rot_*` and `scale_*`.
 * @param bvh - BVH holding the splat center positions.
 * @param idx - Index of the splat.
 * @param px - Point x coordinate.
 * @param py - Point y coordinate.
 * @param pz - Point z coordinate.
 * @param voxelResolution - The mesh's voxel resolution, used as the size
 * reference for the coverage discount.
 * @returns `exp(-0.5 * m^2) * sizeFactor`, where `m^2` is the squared
 * Mahalanobis distance and `sizeFactor = min(1, geoMeanSigma /
 * voxelResolution) ** SIZE_FACTOR_POWER`.
 */
const gaussianWeightFactor = (
    columns: SplatColorColumns,
    bvh: GaussianBVH,
    idx: number,
    px: number,
    py: number,
    pz: number,
    voxelResolution: number
): number => {
    const { rot_0, rot_1, rot_2, rot_3, scale_0, scale_1, scale_2 } = columns;

    // normalize the quaternion (rot_0 = w)
    const qw = rot_0[idx];
    const qx = rot_1[idx];
    const qy = rot_2[idx];
    const qz = rot_3[idx];
    const qlen2 = qw * qw + qx * qx + qy * qy + qz * qz;
    if (qlen2 === 0) {
        return 0;
    }
    const invQ = 1 / Math.sqrt(qlen2);
    const w = qw * invQ;
    const x = qx * invQ;
    const y = qy * invQ;
    const z = qz * invQ;

    // rotation matrix, row-major (matches gpu/shaders/chunks/quat-rotation.ts)
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;
    const r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
    const r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
    const r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);

    // offset from the vertex to the splat center, in the splat frame (u = R^T d)
    const dx = bvh.x[idx] - px;
    const dy = bvh.y[idx] - py;
    const dz = bvh.z[idx] - pz;
    const u0 = r00 * dx + r10 * dy + r20 * dz;
    const u1 = r01 * dx + r11 * dy + r21 * dz;
    const u2 = r02 * dx + r12 * dy + r22 * dz;

    // mahalanobis distance with per-axis sigma = exp(log-scale)
    const s0 = Math.exp(scale_0[idx]);
    const s1 = Math.exp(scale_1[idx]);
    const s2 = Math.exp(scale_2[idx]);
    const m2 = u0 * u0 / (s0 * s0) + u1 * u1 / (s1 * s1) + u2 * u2 / (s2 * s2);

    // discount splats much smaller than the mesh's own voxel resolution: a
    // sub-voxel splat cannot meaningfully represent color at that
    // resolution, however close/opaque it is
    const geoMeanSigma = (s0 * s1 * s2) ** (1 / 3);
    const sizeFactor = Math.min(1, geoMeanSigma / voxelResolution) ** SIZE_FACTOR_POWER;

    return Math.exp(-0.5 * m2) * sizeFactor;
};

/**
 * Compute a linear-space RGB color for every mesh vertex from the gaussian
 * splats overlapping the vertex neighborhood.
 *
 * Each vertex queries a box of half-size `voxelResolution` around its position,
 * doubling the radius twice (2x, 4x) if no splat overlaps. Vertices with no
 * overlapping splat at any radius fall back to mid-grey, as do vertices whose
 * overlapping splats all have zero weight. The resulting sRGB colors are
 * converted to linear space.
 *
 * How overlapping splats combine is selected by `mode`:
 * - `average` (default): display colors weighted by `sigmoid(opacity)`.
 * - `gaussian`: display colors weighted by `sigmoid(opacity)` times the
 * Gaussian density of the splat at the vertex, discounted for splats whose
 * size is well below the mesh's `voxelResolution`.
 * - `dominant`: the display color of the single highest-weight splat.
 * - `topk`: the weighted average of the top 3 splats by weight, renormalized.
 *
 * Display color per splat is `clamp(0.5 + SH_C0 * f_dc, 0, 1)` with
 * `SH_C0 = 0.28209479177387814`.
 *
 * @param positions - Vertex positions as packed xyz triplets.
 * @param bvh - BVH over the gaussian AABBs used for overlap queries.
 * @param columns - Splat color columns (`f_dc_0/1/2` and `opacity`, plus
 * `rot_0..3` and `scale_0..2` for the non-average modes).
 * @param voxelResolution - Initial query box half-size in world units.
 * @param mode - Coloring algorithm. Defaults to `'average'`.
 * @returns Linear RGB colors, 3 floats per vertex, same length as `positions`.
 * @throws Error if a non-average mode is requested without the `rot_*` and
 * `scale_*` columns.
 */
const colorizeVertices = (
    positions: Float32Array,
    bvh: GaussianBVH,
    columns: SplatColorColumns,
    voxelResolution: number,
    mode: CollisionColorMode = 'average'
): Float32Array => {
    const { f_dc_0, f_dc_1, f_dc_2, opacity } = columns;

    if (mode !== 'average') {
        const densityColumns = ['rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2'] as const;
        const missing = densityColumns.filter(name => columns[name] === undefined);
        if (missing.length > 0) {
            throw new Error(`colorizeVertices: mode '${mode}' requires splat columns missing from input: ${missing.join(', ')}`);
        }
    }

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
            if (mode === 'average') {
                let sumR = 0;
                let sumG = 0;
                let sumB = 0;
                let sumW = 0;

                for (let j = 0; j < indices.length; j++) {
                    const idx = indices[j];
                    const w = sigmoid(opacity[idx]);
                    // keep inline clamp verbatim - average mode must remain bit-identical to the original implementation
                    sumR += w * Math.min(Math.max(0.5 + SH_C0 * f_dc_0[idx], 0), 1);
                    sumG += w * Math.min(Math.max(0.5 + SH_C0 * f_dc_1[idx], 0), 1);
                    sumB += w * Math.min(Math.max(0.5 + SH_C0 * f_dc_2[idx], 0), 1);
                    sumW += w;
                }

                cr = sumR / sumW;
                cg = sumG / sumW;
                cb = sumB / sumW;
            } else {
                // weight candidates by opacity x gaussian weight factor at the vertex
                const weights = new Float64Array(indices.length);
                for (let j = 0; j < indices.length; j++) {
                    weights[j] = sigmoid(opacity[indices[j]]) * gaussianWeightFactor(columns, bvh, indices[j], px, py, pz, voxelResolution);
                }

                if (mode === 'dominant') {
                    let best = 0;
                    for (let j = 1; j < indices.length; j++) {
                        if (weights[j] > weights[best]) {
                            best = j;
                        }
                    }
                    if (weights[best] > 0) {
                        const idx = indices[best];
                        cr = displayColor(f_dc_0, idx);
                        cg = displayColor(f_dc_1, idx);
                        cb = displayColor(f_dc_2, idx);
                    }
                } else {
                    // gaussian uses all candidates; topk only the top 3 by weight
                    let selected: number[] | null = null;
                    if (mode === 'topk' && indices.length > 3) {
                        selected = Array.from({ length: indices.length }, (v, j) => j);
                        selected.sort((a, b) => weights[b] - weights[a]);
                        selected.length = 3;
                    }

                    let sumR = 0;
                    let sumG = 0;
                    let sumB = 0;
                    let sumW = 0;

                    for (let k = 0; k < (selected ? selected.length : indices.length); k++) {
                        const j = selected ? selected[k] : k;
                        const idx = indices[j];
                        const w = weights[j];
                        sumR += w * displayColor(f_dc_0, idx);
                        sumG += w * displayColor(f_dc_1, idx);
                        sumB += w * displayColor(f_dc_2, idx);
                        sumW += w;
                    }

                    if (sumW > 0) {
                        cr = sumR / sumW;
                        cg = sumG / sumW;
                        cb = sumB / sumW;
                    }
                }
            }
        }

        result[i] = srgbToLinear(cr);
        result[i + 1] = srgbToLinear(cg);
        result[i + 2] = srgbToLinear(cb);
    }

    return result;
};

export { colorizeVertices };
export type { SplatColorColumns };
