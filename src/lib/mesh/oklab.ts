// Oklab (Björn Ottosson). Converts natively from linear RGB, so no gamma
// round-trip is needed by the colour-quantization and denoising passes that
// work in this space.

/**
 * Convert a linear-space RGB triplet to Oklab.
 *
 * @param r - Linear red.
 * @param g - Linear green.
 * @param b - Linear blue.
 * @returns Oklab `[L, a, b]`.
 */
const linearToOklab = (r: number, g: number, b: number): [number, number, number] => {
    const l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    const m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    const s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

    // cbrt, not ** (1 / 3): the latter is NaN for the marginally negative
    // values that upstream colour clamping can leave behind.
    const l_ = Math.cbrt(l);
    const m_ = Math.cbrt(m);
    const s_ = Math.cbrt(s);

    return [
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    ];
};

/**
 * Convert an Oklab triplet back to linear-space RGB, clamped to sRGB.
 *
 * @param L - Oklab lightness.
 * @param A - Oklab green/red axis.
 * @param B - Oklab blue/yellow axis.
 * @returns Linear `[r, g, b]` in [0, 1].
 */
const oklabToLinear = (L: number, A: number, B: number): [number, number, number] => {
    const l_ = L + 0.3963377774 * A + 0.2158037573 * B;
    const m_ = L - 0.1055613458 * A - 0.0638541728 * B;
    const s_ = L - 0.0894841775 * A - 1.2914855480 * B;

    const l = l_ * l_ * l_;
    const m = m_ * m_ * m_;
    const s = s_ * s_ * s_;

    // A mean of in-gamut colours can land just outside sRGB, so clamp.
    return [
        Math.min(Math.max(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, 0), 1),
        Math.min(Math.max(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, 0), 1),
        Math.min(Math.max(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s, 0), 1)
    ];
};

/**
 * Convert per-vertex linear RGB triplets to a packed Oklab array.
 *
 * @param linearColors - Per-vertex linear-space RGB triplets.
 * @returns Oklab triplets, same length as the input.
 */
const toOklabArray = (linearColors: Float32Array): Float32Array => {
    const oklab = new Float32Array(linearColors.length);
    for (let v = 0; v < linearColors.length / 3; v++) {
        const [L, A, B] = linearToOklab(linearColors[v * 3], linearColors[v * 3 + 1], linearColors[v * 3 + 2]);
        oklab[v * 3] = L;
        oklab[v * 3 + 1] = A;
        oklab[v * 3 + 2] = B;
    }
    return oklab;
};

/**
 * Convert a packed Oklab array back to per-vertex linear RGB triplets.
 *
 * @param oklab - Per-vertex Oklab triplets.
 * @returns Linear-space RGB triplets, same length as the input.
 */
const fromOklabArray = (oklab: Float32Array): Float32Array => {
    const out = new Float32Array(oklab.length);
    for (let v = 0; v < oklab.length / 3; v++) {
        const [r, g, b] = oklabToLinear(oklab[v * 3], oklab[v * 3 + 1], oklab[v * 3 + 2]);
        out[v * 3] = r;
        out[v * 3 + 1] = g;
        out[v * 3 + 2] = b;
    }
    return out;
};

export { linearToOklab, oklabToLinear, toOklabArray, fromOklabArray };
