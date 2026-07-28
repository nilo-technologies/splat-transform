#!/usr/bin/env node
// Colour-noise bench for the voxel collision-mesh pipeline.
//
// Quantizing splat colours to a palette does not hide colour noise, it hardens
// it: a material whose colours spread widely in Oklab occupies many candidate
// bins, so it collects several palette entries, and neighbouring voxel faces
// then alternate between them. This bench makes that measurable. It builds a
// synthetic diorama whose true per-material colour is known, runs the real
// pipeline (voxel faces -> vertex colouring -> denoise -> palette) over it, and
// reports the noise/fidelity/coverage trade-off for several configurations.
//
// Voxelization is done analytically from the scene's own height field, so the
// bench needs no GPU.
//
// Usage:
//   node --import tsx tools/color-noise-bench.mjs [name-prefix ...]
//
// Environment:
//   SEED=<n>   scene RNG seed. Default 1234
//   NOISE=<f>  scales all per-splat colour noise. Default 1
//   OUT=<dir>  where to write the PNGs. Default tools/.color-noise-out
//   DETAIL=1   also print the per-material breakdown
//
// Columns, all measured on a top-down map of the mesh's upward faces:
//   flip%     neighbouring face pairs of the SAME material that differ in
//             colour — the direct measure of the artefact (lower is better)
//   iso%      faces differing from every same-material neighbour, i.e. isolated
//             speckle (lower is better)
//   cols/mat  palette entries covering >= 1% of a material, averaged over
//             materials (lower is flatter)
//   matErr    mean Oklab distance to the material's true colour (lower is
//             better) — the fidelity side of the trade
//   edgeErr   the same, restricted to faces on a material boundary, which is
//             where over-smoothing shows up first
//   used      distinct palette entries appearing in the output
//   redΔ/blueΔ  Oklab error on the two tiny saturated accents, which exist to
//             check that small features are not averaged away

import { mkdirSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

import { Vec3 } from 'playcanvas';

import { computeGaussianExtents } from '../src/lib/data-table/gaussian-aabb.js';
import { Column, DataTable } from '../src/lib/data-table/index.js';
import {
    colorizeVertices, computeVertexNormals, mapToPalette, palettizeColors,
    parsePaletteColors, smoothVertexColors, voxelFaces
} from '../src/lib/mesh/index.js';
import { linearToOklab } from '../src/lib/mesh/oklab.js';
import { GaussianBVH } from '../src/lib/spatial/index.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

const SH_C0 = 0.28209479177387814;
const packClr = c => (c - 0.5) / SH_C0;
const packOpacity = (o) => {
    if (o <= 0) return -20;
    if (o >= 1) return 20;
    return -Math.log(1 / o - 1);
};

const srgbToLinear = c => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const linearToSrgb = c => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
const dE = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

const makeRng = (seed) => {
    let s = seed;
    return () => {
        s = (s + 0x6D2B79F5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

const makeGauss = rng => () => {
    const u = Math.max(rng(), 1e-9);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
};

// ---------------------------------------------------------------------------
// Scene
//
// `lumaSigma` is the shadow-to-highlight swing, which dominates real 3DGS
// foliage; `hueSigma` is per-channel chroma jitter; `outlier*` models the
// clamped-reconstruction speckle that survives in high-frequency regions. The
// two accents are deliberately tiny — they are the coverage test.
// ---------------------------------------------------------------------------

const MATERIALS = {
    grass: { rgb: [0.33, 0.47, 0.20], lumaSigma: 0.115, hueSigma: 0.035, outlierRate: 0.05, outlierSigma: 0.22 },
    grassDark: { rgb: [0.20, 0.31, 0.13], lumaSigma: 0.090, hueSigma: 0.030, outlierRate: 0.04, outlierSigma: 0.18 },
    plaza: { rgb: [0.68, 0.68, 0.66], lumaSigma: 0.030, hueSigma: 0.010, outlierRate: 0.008, outlierSigma: 0.10 },
    stoneLight: { rgb: [0.76, 0.76, 0.74], lumaSigma: 0.040, hueSigma: 0.012, outlierRate: 0.010, outlierSigma: 0.10 },
    stoneMid: { rgb: [0.60, 0.60, 0.59], lumaSigma: 0.040, hueSigma: 0.012, outlierRate: 0.010, outlierSigma: 0.10 },
    stoneDark: { rgb: [0.40, 0.41, 0.42], lumaSigma: 0.040, hueSigma: 0.014, outlierRate: 0.012, outlierSigma: 0.10 },
    sand: { rgb: [0.83, 0.76, 0.60], lumaSigma: 0.040, hueSigma: 0.015, outlierRate: 0.010, outlierSigma: 0.10 },
    tanRoof: { rgb: [0.64, 0.53, 0.40], lumaSigma: 0.045, hueSigma: 0.018, outlierRate: 0.012, outlierSigma: 0.10 },
    redAccent: { rgb: [0.74, 0.13, 0.10], lumaSigma: 0.035, hueSigma: 0.015, outlierRate: 0.008, outlierSigma: 0.08 },
    blueAccent: { rgb: [0.14, 0.26, 0.72], lumaSigma: 0.035, hueSigma: 0.015, outlierRate: 0.008, outlierSigma: 0.08 }
};

const EXTENT = 2.0;   // half-size in x/z, so a 4 m x 4 m footprint
const VOXEL = 0.02;   // 2 cm voxels, so 200 x 200 columns

// A ring of building slabs around a central plaza, plus two tiny accents.
const SLABS = (() => {
    const slabs = [];
    const roofMats = ['stoneLight', 'sand', 'stoneMid', 'tanRoof', 'stoneDark', 'sand', 'stoneLight', 'tanRoof'];
    const rng = makeRng(9001);
    let k = 0;

    for (let side = 0; side < 4; side++) {
        for (let i = 0; i < 5; i++) {
            const t0 = -1.85 + i * 0.74 + 0.06;
            const t1 = t0 + 0.62;
            const d0 = 0.85 + rng() * 0.15;
            const d1 = 1.85 - rng() * 0.15;
            const h = 0.10 + rng() * 0.28;
            const mat = roofMats[k++ % roofMats.length];
            const box = side === 0 ? [t0, d0, t1, d1] :
                side === 1 ? [t0, -d1, t1, -d0] :
                    side === 2 ? [d0, t0, d1, t1] : [-d1, t0, -d0, t1];
            slabs.push({ x0: box[0], z0: box[1], x1: box[2], z1: box[3], h, mat });
        }
    }

    slabs.push({ x0: -0.10, z0: -0.06, x1: 0.02, z1: 0.06, h: 0.06, mat: 'redAccent' });
    slabs.push({ x0: 0.40, z0: 0.55, x1: 0.48, z1: 0.63, h: 0.05, mat: 'blueAccent' });
    return slabs;
})();

const PLAZA = { x0: -0.78, z0: -0.78, x1: 0.78, z1: 0.78 };
const inBox = (x, z, b) => x >= b.x0 && x <= b.x1 && z >= b.z0 && z <= b.z1;

// Ground truth at a world (x, z): top height and the material of the top face.
const surfaceAt = (x, z) => {
    for (const s of SLABS) {
        if (inBox(x, z, s)) return { h: s.h, mat: s.mat };
    }
    if (inBox(x, z, PLAZA)) return { h: 0, mat: 'plaza' };
    // two greens in large-scale patches, so the grass has real structure to
    // preserve as well as noise to remove
    const n = Math.sin(x * 2.7 + 1.3) * Math.cos(z * 3.1 - 0.7) + 0.35 * Math.sin(x * 7.3 - z * 5.1);
    return { h: 0, mat: n > 0.25 ? 'grassDark' : 'grass' };
};

const sampleColor = (mat, gauss, rng, noiseScale) => {
    const m = MATERIALS[mat];
    const luma = m.lumaSigma * noiseScale * gauss();
    const outlier = rng() < m.outlierRate ? m.outlierSigma * noiseScale * gauss() : 0;
    return m.rgb.map(c => Math.min(Math.max(c + luma + m.hueSigma * noiseScale * gauss() + outlier, 0), 1));
};

// Scatter splats over every exposed surface — tops and building walls.
const buildScene = ({ seed, noiseScale, density = 3.2 }) => {
    const rng = makeRng(seed);
    const gauss = makeGauss(rng);
    const xs = [], ys = [], zs = [], r = [], g = [], b = [], op = [];

    const push = (x, y, z, mat) => {
        const c = sampleColor(mat, gauss, rng, noiseScale);
        xs.push(x); ys.push(y); zs.push(z);
        r.push(c[0]); g.push(c[1]); b.push(c[2]);
        op.push(0.75 + rng() * 0.24);
    };

    const step = VOXEL / Math.sqrt(density);
    for (let x = -EXTENT; x < EXTENT; x += step) {
        for (let z = -EXTENT; z < EXTENT; z += step) {
            const jx = x + (rng() - 0.5) * step;
            const jz = z + (rng() - 0.5) * step;
            const s = surfaceAt(jx, jz);
            push(jx, s.h + VOXEL * 0.3, jz, s.mat);
        }
    }

    for (const s of SLABS) {
        const perim = [
            [s.x0, s.z0, s.x1, s.z0], [s.x0, s.z1, s.x1, s.z1],
            [s.x0, s.z0, s.x0, s.z1], [s.x1, s.z0, s.x1, s.z1]
        ];
        for (const [ax, az, bx, bz] of perim) {
            const n = Math.max(2, Math.ceil(Math.hypot(bx - ax, bz - az) / step));
            const rows = Math.max(2, Math.ceil(s.h / step));
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < rows; j++) {
                    const t = (i + rng()) / n;
                    push(ax + (bx - ax) * t, ((j + rng()) / rows) * s.h, az + (bz - az) * t, s.mat);
                }
            }
        }
    }

    const count = xs.length;
    const f32 = a => Float32Array.from(a);
    const scale = new Float32Array(count).fill(Math.log(VOXEL * 0.85));

    return new DataTable([
        new Column('x', f32(xs)), new Column('y', f32(ys)), new Column('z', f32(zs)),
        new Column('scale_0', scale), new Column('scale_1', scale.slice()), new Column('scale_2', scale.slice()),
        new Column('f_dc_0', f32(r.map(packClr))),
        new Column('f_dc_1', f32(g.map(packClr))),
        new Column('f_dc_2', f32(b.map(packClr))),
        new Column('opacity', f32(op.map(packOpacity))),
        new Column('rot_0', new Float32Array(count).fill(1)),
        new Column('rot_1', new Float32Array(count)),
        new Column('rot_2', new Float32Array(count)),
        new Column('rot_3', new Float32Array(count))
    ]);
};

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

const align4 = n => Math.ceil(n / 4) * 4;

// Fill a solid grid from the height field, recording each column's material.
const buildGrid = () => {
    const n = align4(Math.round((2 * EXTENT) / VOXEL));
    const ny = align4(Math.round(0.5 / VOXEL)); // the tallest slab is under 0.4 m
    const grid = new SparseVoxelGrid(n, ny, n);
    const gridBounds = {
        min: new Vec3(-EXTENT, 0, -EXTENT),
        max: new Vec3(-EXTENT + n * VOXEL, ny * VOXEL, -EXTENT + n * VOXEL)
    };

    const colMat = new Array(n * n);
    const colTop = new Int32Array(n * n);
    for (let ix = 0; ix < n; ix++) {
        const x = gridBounds.min.x + (ix + 0.5) * VOXEL;
        for (let iz = 0; iz < n; iz++) {
            const s = surfaceAt(x, gridBounds.min.z + (iz + 0.5) * VOXEL);
            const top = Math.max(0, Math.round(s.h / VOXEL));
            for (let iy = 0; iy <= top; iy++) grid.setVoxel(ix, iy, iz);
            colMat[ix + iz * n] = s.mat;
            colTop[ix + iz * n] = top;
        }
    }
    return { grid, gridBounds, n, colMat, colTop };
};

// The scene, BVH, mesh and per-vertex colours are identical across palette
// configurations, so they are built once per (mode, scene).
const cache = new Map();

const referenceColors = (mode, scene) => {
    const key = `${mode}|${scene.seed}|${scene.noiseScale}`;
    const hit = cache.get(key);
    if (hit) return hit;

    const table = buildScene(scene);
    const bvh = new GaussianBVH(table, computeGaussianExtents(table).extents);
    const columns = {
        f_dc_0: table.getColumnByName('f_dc_0').data,
        f_dc_1: table.getColumnByName('f_dc_1').data,
        f_dc_2: table.getColumnByName('f_dc_2').data,
        opacity: table.getColumnByName('opacity').data
    };

    const { grid, gridBounds, n, colMat, colTop } = buildGrid();
    const mesh = voxelFaces(grid, gridBounds, VOXEL, { perVoxel: true });
    const normals = computeVertexNormals(mesh.positions, mesh.indices);
    const reference = colorizeVertices(mesh.positions, normals, bvh, columns, VOXEL, mode);

    const built = { mesh, reference, gridBounds, n, colMat, colTop };
    cache.set(key, built);
    return built;
};

const runPipeline = ({ mode = 'solid', palette = 64, smoothRadius = 0, coherentRadius, scene }) => {
    const built = referenceColors(mode, scene);
    const { mesh } = built;

    let colors = built.reference;
    if (smoothRadius > 0) {
        colors = smoothVertexColors(colors, mesh.positions, smoothRadius, VOXEL);
    }

    const opts = { positions: mesh.positions, voxelResolution: VOXEL, coherentRadius };
    const quantized = Array.isArray(palette) ?
        mapToPalette(colors, parsePaletteColors(palette), opts) :
        palette >= 1 ? palettizeColors(colors, palette, opts) : colors;

    return { ...built, quantized };
};

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

// Index vertices by quantized world position, to find a voxel's top corners.
const buildPositionIndex = (positions) => {
    const map = new Map();
    const q = v => Math.round(v / (VOXEL * 1e-3));
    for (let v = 0; v < positions.length / 3; v++) {
        map.set(`${q(positions[v * 3])}_${q(positions[v * 3 + 1])}_${q(positions[v * 3 + 2])}`, v);
    }
    return { map, key: (x, y, z) => `${q(x)}_${q(y)}_${q(z)}` };
};

// Dominant colour of a face, matching how --collision-color-flat collapses one.
const dominant = (verts, colors) => {
    let best = verts[0];
    let bestCount = 0;
    for (const v of verts) {
        let count = 0;
        for (const w of verts) {
            if (colors[w * 3] === colors[v * 3] &&
                colors[w * 3 + 1] === colors[v * 3 + 1] &&
                colors[w * 3 + 2] === colors[v * 3 + 2]) count++;
        }
        if (count > bestCount) {
            bestCount = count; best = v;
        }
    }
    return [colors[best * 3], colors[best * 3 + 1], colors[best * 3 + 2]];
};

// Collapse the mesh to a top-down map: one colour per voxel column.
const buildTopMap = (run) => {
    const { mesh, quantized, gridBounds, n, colMat, colTop } = run;
    const idx = buildPositionIndex(mesh.positions);
    const quant = new Float32Array(n * n * 3);
    const valid = new Uint8Array(n * n);

    for (let ix = 0; ix < n; ix++) {
        for (let iz = 0; iz < n; iz++) {
            const c = ix + iz * n;
            const y = (colTop[c] + 1) * VOXEL + gridBounds.min.y;
            const x0 = gridBounds.min.x + ix * VOXEL;
            const z0 = gridBounds.min.z + iz * VOXEL;
            const verts = [
                idx.map.get(idx.key(x0, y, z0)), idx.map.get(idx.key(x0 + VOXEL, y, z0)),
                idx.map.get(idx.key(x0, y, z0 + VOXEL)), idx.map.get(idx.key(x0 + VOXEL, y, z0 + VOXEL))
            ];
            if (verts.some(v => v === undefined)) continue;
            valid[c] = 1;
            const q = dominant(verts, quantized);
            for (let k = 0; k < 3; k++) quant[c * 3 + k] = q[k];
        }
    }

    return { quant, valid, n, colMat };
};

const colorKey = (arr, i) => `${arr[i * 3].toFixed(6)},${arr[i * 3 + 1].toFixed(6)},${arr[i * 3 + 2].toFixed(6)}`;

const metrics = ({ quant, valid, n, colMat }) => {
    const at = (ix, iz) => ix + iz * n;
    const neighbours = (ix, iz) => [[ix + 1, iz], [ix - 1, iz], [ix, iz + 1], [ix, iz - 1]];

    let pairs = 0, flips = 0, isolated = 0, cells = 0;
    const isEdge = new Uint8Array(n * n);

    for (let ix = 0; ix < n; ix++) {
        for (let iz = 0; iz < n; iz++) {
            const c = at(ix, iz);
            if (!valid[c]) continue;
            cells++;
            const key = colorKey(quant, c);
            let sameMat = 0, differing = 0;
            for (const [jx, jz] of neighbours(ix, iz)) {
                if (jx < 0 || jz < 0 || jx >= n || jz >= n) continue;
                const d = at(jx, jz);
                if (!valid[d]) continue;
                if (colMat[d] !== colMat[c]) {
                    isEdge[c] = 1; continue;
                }
                sameMat++;
                const differs = colorKey(quant, d) !== key;
                if (differs) differing++;
                if (jx > ix || jz > iz) {
                    pairs++; if (differs) flips++;
                }
            }
            if (sameMat >= 3 && differing === sameMat) isolated++;
        }
    }

    const perMat = new Map();
    let matErr = 0, edgeErr = 0, edgeCount = 0, total = 0, effSum = 0;

    for (let c = 0; c < n * n; c++) {
        if (!valid[c]) continue;
        const m = colMat[c];
        let e = perMat.get(m);
        if (!e) {
            e = { count: 0, colors: new Map(), matErr: 0 }; perMat.set(m, e);
        }
        e.count++;
        e.colors.set(colorKey(quant, c), (e.colors.get(colorKey(quant, c)) ?? 0) + 1);

        const truth = MATERIALS[m].rgb.map(srgbToLinear);
        const err = dE(
            linearToOklab(quant[c * 3], quant[c * 3 + 1], quant[c * 3 + 2]),
            linearToOklab(truth[0], truth[1], truth[2])
        );
        e.matErr += err;
        if (isEdge[c]) {
            edgeErr += err; edgeCount++;
        }
    }

    const matRows = [];
    for (const [mat, e] of perMat) {
        matErr += e.matErr;
        total += e.count;
        // entries covering at least 1% of the material, so a couple of stray
        // faces do not read the same as a genuine second tone
        const eff = [...e.colors.values()].filter(v => v / e.count >= 0.01).length;
        effSum += eff;
        matRows.push({
            mat,
            count: e.count,
            colors: e.colors.size,
            eff,
            matErr: e.matErr / e.count,
            top: [...e.colors.entries()].sort((a, b) => b[1] - a[1])[0]
        });
    }

    const accents = {};
    for (const name of ['redAccent', 'blueAccent']) {
        const row = matRows.find(r => r.mat === name);
        const truth = MATERIALS[name].rgb.map(srgbToLinear);
        accents[name] = row ?
            dE(linearToOklab(...row.top[0].split(',').map(Number)), linearToOklab(...truth)) :
            NaN;
    }

    const used = new Set();
    for (let c = 0; c < n * n; c++) if (valid[c]) used.add(colorKey(quant, c));

    return {
        flipRate: flips / pairs,
        isolated: isolated / cells,
        matErr: matErr / total,
        edgeErr: edgeErr / edgeCount,
        perMatEff: effSum / matRows.length,
        entriesUsed: used.size,
        accents,
        matRows: matRows.sort((a, b) => b.count - a.count)
    };
};

// ---------------------------------------------------------------------------
// PNG output
// ---------------------------------------------------------------------------

const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
        t[n] = c >>> 0;
    }
    return t;
})();

const chunk = (type, data) => {
    const out = Buffer.alloc(12 + data.length);
    out.writeUInt32BE(data.length, 0);
    out.write(type, 4, 'ascii');
    data.copy(out, 8);
    let c = 0xFFFFFFFF;
    for (const byte of out.subarray(4, 8 + data.length)) c = CRC_TABLE[(c ^ byte) & 0xFF] ^ (c >>> 8);
    out.writeUInt32BE((c ^ 0xFFFFFFFF) >>> 0, 8 + data.length);
    return out;
};

// Write the top-down map as a PNG, scaled up with nearest-neighbour.
const renderMap = (path, { quant, valid, n }, scale = 3) => {
    const w = n * scale;
    const rgb = Buffer.alloc(w * w * 3, 235);
    for (let iz = 0; iz < n; iz++) {
        for (let ix = 0; ix < n; ix++) {
            const c = ix + iz * n;
            if (!valid[c]) continue;
            const px = [0, 1, 2].map(k => Math.round(Math.min(Math.max(linearToSrgb(quant[c * 3 + k]), 0), 1) * 255));
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    const o = ((iz * scale + dy) * w + ix * scale + dx) * 3;
                    rgb[o] = px[0]; rgb[o + 1] = px[1]; rgb[o + 2] = px[2];
                }
            }
        }
    }

    const raw = Buffer.alloc(w * (1 + w * 3));
    for (let y = 0; y < w; y++) {
        rgb.copy(raw, y * (1 + w * 3) + 1, y * w * 3, (y + 1) * w * 3);
    }

    const ihdr = Buffer.alloc(13);
    ihdr.writeUInt32BE(w, 0);
    ihdr.writeUInt32BE(w, 4);
    ihdr[8] = 8;  // bit depth
    ihdr[9] = 2;  // truecolour
    writeFileSync(path, Buffer.concat([
        Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
        chunk('IHDR', ihdr),
        chunk('IDAT', deflateSync(raw, { level: 9 })),
        chunk('IEND', Buffer.alloc(0))
    ]));
};

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

const CONFIGS = {
    'reference (no palette)': { palette: 0 },
    'palette 64, no denoise': { smoothRadius: 0 },
    'palette 64, denoise 1': { smoothRadius: 1 },
    'palette 64, denoise 2 (default)': { smoothRadius: 2 },
    'palette 64, denoise 3': { smoothRadius: 3 },
    'palette 64, denoise 2 + coherent 1': { smoothRadius: 2, coherentRadius: 1 },
    'palette 16, denoise 2': { palette: 16, smoothRadius: 2 },
    'average mode, denoise 2': { mode: 'average', smoothRadius: 2 }
};

const outDir = process.env.OUT ?? new URL('.color-noise-out/', import.meta.url).pathname;
mkdirSync(outDir, { recursive: true });

const scene = { seed: Number(process.env.SEED ?? 1234), noiseScale: Number(process.env.NOISE ?? 1) };
const only = process.argv.slice(2);
const selected = Object.entries(CONFIGS).filter(([k]) => !only.length || only.some(o => k.startsWith(o)));

console.error(`scene seed ${scene.seed}, noise x${scene.noiseScale}; writing PNGs to ${outDir}`);

const rows = [];
for (const [name, opts] of selected) {
    const t0 = Date.now();
    const run = runPipeline({ scene, ...opts });
    const map = buildTopMap(run);
    renderMap(`${outDir}/${name.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}.png`, map);
    rows.push({ name, m: metrics(map) });
    console.error(`  ${name} (${Date.now() - t0} ms)`);
}

const cols = [
    ['config', r => r.name, 36],
    ['flip%', r => (r.m.flipRate * 100).toFixed(1), 8],
    ['iso%', r => (r.m.isolated * 100).toFixed(2), 8],
    ['cols/mat', r => r.m.perMatEff.toFixed(1), 10],
    ['matErr', r => r.m.matErr.toFixed(4), 9],
    ['edgeErr', r => r.m.edgeErr.toFixed(4), 9],
    ['used', r => r.m.entriesUsed, 6],
    ['redΔ', r => r.m.accents.redAccent.toFixed(3), 7],
    ['blueΔ', r => r.m.accents.blueAccent.toFixed(3), 7]
];

console.log();
console.log(cols.map(([h, , w]) => h.padEnd(w)).join(''));
console.log(cols.map(([, , w]) => '-'.repeat(w - 1).padEnd(w)).join(''));
for (const r of rows) console.log(cols.map(([, f, w]) => String(f(r)).padEnd(w)).join(''));
console.log();

if (process.env.DETAIL) {
    for (const r of rows) {
        console.log(`== ${r.name}`);
        for (const mr of r.m.matRows) {
            console.log(`   ${mr.mat.padEnd(12)} faces=${String(mr.count).padStart(6)}  eff=${String(mr.eff).padStart(3)}  all=${String(mr.colors).padStart(3)}  matErr=${mr.matErr.toFixed(4)}`);
        }
    }
}
