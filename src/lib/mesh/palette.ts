const srgbToLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);

const linearToSrgb = (c: number): number => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);

const luminance = (r: number, g: number, b: number): number => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/**
 * Snap vertex colors to a deterministic k-means palette in perceptual sRGB
 * space, returning the quantized colors back in linear space.
 *
 * Centroids are initialized deterministically from luminance quantiles of the
 * sRGB vertex colors, and exactly 10 Lloyd iterations are run (no
 * randomisation at any step). When `k >= vertexCount` the input is returned
 * as-is (no-op; the palette would merely repeat the data).
 *
 * @param linearColors - Per-vertex linear-space RGB triplets.
 * @param k - Number of palette colours (at most `vertexCount`).
 * @returns Linear-space quantized colors, same length as `linearColors`.
 */
const palettizeColors = (linearColors: Float32Array, k: number): Float32Array => {
    const vertexCount = linearColors.length / 3;
    const effK = Math.min(k, vertexCount);
    if (effK < 2) {
        // single-colour palette: snap every vertex to the global mean
        let sumR = 0, sumG = 0, sumB = 0;
        for (let i = 0; i < linearColors.length; i += 3) {
            sumR += linearToSrgb(linearColors[i]);
            sumG += linearToSrgb(linearColors[i + 1]);
            sumB += linearToSrgb(linearColors[i + 2]);
        }
        const cr = sumR / vertexCount;
        const cg = sumG / vertexCount;
        const cb = sumB / vertexCount;
        const result = new Float32Array(linearColors.length);
        for (let i = 0; i < linearColors.length; i += 3) {
            result[i] = srgbToLinear(cr);
            result[i + 1] = srgbToLinear(cg);
            result[i + 2] = srgbToLinear(cb);
        }
        return result;
    }

    if (effK === vertexCount) {
        return new Float32Array(linearColors);
    }

    // convert to sRGB for perceptual clustering
    const srgb = new Float32Array(linearColors.length);
    for (let i = 0; i < linearColors.length; i++) {
        srgb[i] = linearToSrgb(linearColors[i]);
    }

    // deterministic centroid initialisation via luminance quantiles
    const centroids = new Float32Array(effK * 3);
    const lum = new Float64Array(vertexCount);
    for (let v = 0; v < vertexCount; v++) {
        lum[v] = luminance(srgb[v * 3], srgb[v * 3 + 1], srgb[v * 3 + 2]);
    }
    const order = Array.from({ length: vertexCount }, (v, i) => i);
    order.sort((a, b) => lum[a] - lum[b]);
    for (let c = 0; c < effK; c++) {
        const idx = order[Math.min(Math.floor((2 * c + 1) / (2 * effK) * vertexCount), vertexCount - 1)];
        centroids[c * 3] = srgb[idx * 3];
        centroids[c * 3 + 1] = srgb[idx * 3 + 1];
        centroids[c * 3 + 2] = srgb[idx * 3 + 2];
    }

    const assignments = new Uint32Array(vertexCount);
    const accum = new Float64Array(effK * 3);
    const counts = new Uint32Array(effK);

    for (let iter = 0; iter < 10; iter++) {
        accum.fill(0);
        counts.fill(0);

        // assign each vertex to its nearest centroid (Euclidean in sRGB)
        for (let v = 0; v < vertexCount; v++) {
            const vr = srgb[v * 3];
            const vg = srgb[v * 3 + 1];
            const vb = srgb[v * 3 + 2];
            let bestDist = Infinity;
            let bestC = 0;
            for (let c = 0; c < effK; c++) {
                const dr = vr - centroids[c * 3];
                const dg = vg - centroids[c * 3 + 1];
                const db = vb - centroids[c * 3 + 2];
                const d2 = dr * dr + dg * dg + db * db;
                if (d2 < bestDist) {
                    bestDist = d2;
                    bestC = c;
                }
            }
            assignments[v] = bestC;
            const base = bestC * 3;
            accum[base] += vr;
            accum[base + 1] += vg;
            accum[base + 2] += vb;
            counts[bestC]++;
        }

        // recompute centroids; empty clusters keep their previous centroid
        for (let c = 0; c < effK; c++) {
            if (counts[c] > 0) {
                const base = c * 3;
                centroids[base] = accum[base] / counts[c];
                centroids[base + 1] = accum[base + 1] / counts[c];
                centroids[base + 2] = accum[base + 2] / counts[c];
            }
        }
    }

    // snap every vertex to its final centroid and convert back to linear
    const result = new Float32Array(linearColors.length);
    for (let v = 0; v < vertexCount; v++) {
        const base = assignments[v] * 3;
        result[v * 3] = srgbToLinear(centroids[base]);
        result[v * 3 + 1] = srgbToLinear(centroids[base + 1]);
        result[v * 3 + 2] = srgbToLinear(centroids[base + 2]);
    }

    return result;
};

export { palettizeColors };
