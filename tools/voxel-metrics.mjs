#!/usr/bin/env node
// Voxel grid metrics for the emitted sparse octree.
//
// Decodes a .voxel.json/.voxel.bin pair back into an occupied-voxel set and
// reports the numbers the README's cleanup table quotes, so both rows can be
// re-measured from real runs rather than trusted from history.
//
// Usage:
//   node --import tsx tools/voxel-metrics.mjs <path.voxel.json> [more.voxel.json ...]
//
// Columns:
//   occupied   occupied voxels
//   islands    6-connected components over the occupied set
//   largest    share of occupied voxels in the biggest component
//   roughness  mean |h(x,z) - mean(h of the up-to-8 neighbouring columns)| in
//              voxels, over columns with at least one occupied neighbour
//              column, where h is the highest occupied voxel in the column. A
//              coherent surface reads near 0; a scatter reads high. This is the
//              tool's own definition -- compare rows measured with this tool,
//              not against numbers from elsewhere.
//   scatter    share of occupied voxels with <= 2 of 6 face neighbours. Matches
//              scatterFraction in src/lib/writers/write-voxel.ts, so it
//              cross-checks the decoder against the CLI's own log line.
//   faces/vox  mean occupied face neighbours per occupied voxel.

import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';

const SOLID_LEAF_MARKER = 0xFF000000 >>> 0;

const popcount = (n) => {
    n >>>= 0;
    n -= ((n >>> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
    return (((n + (n >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
};

/**
 * Decode Laine-Karras octree arrays into a set of packed voxel keys.
 *
 * Keys are x + y * nx + z * nx * ny. Voxels outside nx/ny/nz are dropped: the
 * root cube is padded up to a power of two, so a solid region at a high level
 * can cover space the grid does not have.
 */
const decodeOctree = ({ nodes, leafData, treeDepth, nx, ny, nz }) => {
    const voxels = new Set();
    const key = (x, y, z) => x + y * nx + z * nx * ny;

    if (nodes.length === 0) return { nx, ny, nz, voxels, key };

    const addSolidRegion = (li, bx, by, bz) => {
        const span = 4 << li;
        const x0 = bx * span;
        const y0 = by * span;
        const z0 = bz * span;
        const x1 = Math.min(x0 + span, nx);
        const y1 = Math.min(y0 + span, ny);
        const z1 = Math.min(z0 + span, nz);
        for (let z = z0; z < z1; z++) {
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) voxels.add(key(x, y, z));
            }
        }
    };

    const addMixedLeaf = (leafDataIndex, bx, by, bz) => {
        const lo = leafData[leafDataIndex * 2] >>> 0;
        const hi = leafData[leafDataIndex * 2 + 1] >>> 0;
        for (let bit = 0; bit < 64; bit++) {
            const set = bit < 32 ? (lo >>> bit) & 1 : (hi >>> (bit - 32)) & 1;
            if (!set) continue;
            const x = bx * 4 + (bit & 3);
            const y = by * 4 + ((bit >> 2) & 3);
            const z = bz * 4 + ((bit >> 4) & 3);
            if (x >= nx || y >= ny || z >= nz) continue;
            voxels.add(key(x, y, z));
        }
    };

    // Explicit stack of cells: node index plus the cell's level and coordinates.
    const stack = [{ node: 0, li: treeDepth, bx: 0, by: 0, bz: 0 }];
    while (stack.length > 0) {
        const { node, li, bx, by, bz } = stack.pop();
        const word = nodes[node] >>> 0;

        if (word === SOLID_LEAF_MARKER) {
            addSolidRegion(li, bx, by, bz);
            continue;
        }

        const childMask = word >>> 24;
        if (childMask === 0) {
            // Mixed 4x4x4 leaf. Interior nodes always have at least one child,
            // so a zero mask is unambiguous.
            addMixedLeaf(word & 0x00FFFFFF, bx, by, bz);
            continue;
        }

        const baseOffset = word & 0x00FFFFFF;
        for (let oct = 0; oct < 8; oct++) {
            if ((childMask & (1 << oct)) === 0) continue;
            const rank = popcount(childMask & ((1 << oct) - 1));
            stack.push({
                node: baseOffset + rank,
                li: li - 1,
                bx: bx * 2 + (oct & 1),
                by: by * 2 + ((oct >> 1) & 1),
                bz: bz * 2 + ((oct >> 2) & 1)
            });
        }
    }

    return { nx, ny, nz, voxels, key };
};

/**
 * Decode a written .voxel.json / .voxel.bin pair.
 */
const decodeVoxelFiles = async (jsonPath) => {
    const meta = JSON.parse(await readFile(jsonPath, 'utf8'));
    const binPath = jsonPath.replace('.voxel.json', '.voxel.bin');
    const bin = await readFile(binPath);
    const words = new Uint32Array(bin.buffer, bin.byteOffset, bin.byteLength >> 2);

    // subarray clamps silently, so a stale or truncated .bin beside a newer
    // .json would decode fewer voxels and report a plausible but wrong row.
    const expectedWords = meta.nodeCount + meta.leafDataCount;
    if (words.length !== expectedWords) {
        throw new Error(
            `${binPath} holds ${words.length} u32 words, but ${basename(jsonPath)} describes ` +
            `${expectedWords} (nodeCount ${meta.nodeCount} + leafDataCount ${meta.leafDataCount}). ` +
            'The pair is mismatched -- re-export the scene.'
        );
    }

    const nodes = words.subarray(0, meta.nodeCount);
    const leafData = words.subarray(meta.nodeCount, meta.nodeCount + meta.leafDataCount);

    const vr = meta.voxelResolution;
    const nx = Math.round((meta.gridBounds.max[0] - meta.gridBounds.min[0]) / vr);
    const ny = Math.round((meta.gridBounds.max[1] - meta.gridBounds.min[1]) / vr);
    const nz = Math.round((meta.gridBounds.max[2] - meta.gridBounds.min[2]) / vr);

    return decodeOctree({ nodes, leafData, treeDepth: meta.treeDepth, nx, ny, nz });
};

/**
 * Occupancy, connectivity and surface metrics over a decoded voxel set.
 */
const gridMetrics = ({ nx, ny, nz, voxels }) => {
    const occupied = voxels.size;
    if (occupied === 0) {
        return { occupied: 0, islands: 0, largestShare: 0, roughness: 0, scatter: 0, facesPerVoxel: 0 };
    }

    const zStride = nx * ny;
    const unpack = (k) => {
        const z = Math.floor(k / zStride);
        const rem = k - z * zStride;
        const y = Math.floor(rem / nx);
        return [rem - y * nx, y, z];
    };
    const neighborKeys = (x, y, z) => {
        const out = [];
        if (x > 0) out.push(x - 1 + y * nx + z * zStride);
        if (x + 1 < nx) out.push(x + 1 + y * nx + z * zStride);
        if (y > 0) out.push(x + (y - 1) * nx + z * zStride);
        if (y + 1 < ny) out.push(x + (y + 1) * nx + z * zStride);
        if (z > 0) out.push(x + y * nx + (z - 1) * zStride);
        if (z + 1 < nz) out.push(x + y * nx + (z + 1) * zStride);
        return out;
    };

    // Face-neighbour histogram: scatter and faces/voxel in one sweep.
    let faceTotal = 0;
    let scattered = 0;
    for (const k of voxels) {
        const [x, y, z] = unpack(k);
        let faces = 0;
        for (const nk of neighborKeys(x, y, z)) {
            if (voxels.has(nk)) faces++;
        }
        faceTotal += faces;
        if (faces <= 2) scattered++;
    }

    // 6-connected components by BFS over the occupied set.
    const seen = new Set();
    let islands = 0;
    let largest = 0;
    for (const start of voxels) {
        if (seen.has(start)) continue;
        islands++;
        let size = 0;
        const queue = [start];
        seen.add(start);
        while (queue.length > 0) {
            const k = queue.pop();
            size++;
            const [x, y, z] = unpack(k);
            for (const nk of neighborKeys(x, y, z)) {
                if (voxels.has(nk) && !seen.has(nk)) {
                    seen.add(nk);
                    queue.push(nk);
                }
            }
        }
        if (size > largest) largest = size;
    }

    // Top-surface roughness: how far each column's highest voxel sits from the
    // mean of its neighbouring columns' highest voxels.
    const tops = new Map();
    for (const k of voxels) {
        const [x, y, z] = unpack(k);
        const col = x + z * nx;
        const cur = tops.get(col);
        if (cur === undefined || y > cur) tops.set(col, y);
    }
    let devTotal = 0;
    let devCount = 0;
    for (const [col, h] of tops) {
        const x = col % nx;
        const z = (col - x) / nx;
        let sum = 0;
        let n = 0;
        for (let dz = -1; dz <= 1; dz++) {
            for (let dx = -1; dx <= 1; dx++) {
                if (dx === 0 && dz === 0) continue;
                const ax = x + dx;
                const az = z + dz;
                if (ax < 0 || az < 0 || ax >= nx || az >= nz) continue;
                const ah = tops.get(ax + az * nx);
                if (ah === undefined) continue;
                sum += ah;
                n++;
            }
        }
        if (n === 0) continue;
        devTotal += Math.abs(h - sum / n);
        devCount++;
    }

    return {
        occupied,
        islands,
        largestShare: largest / occupied,
        roughness: devCount === 0 ? 0 : devTotal / devCount,
        scatter: scattered / occupied,
        facesPerVoxel: faceTotal / occupied
    };
};

const main = async () => {
    const paths = process.argv.slice(2);
    if (paths.length === 0) {
        console.error('usage: node --import tsx tools/voxel-metrics.mjs <path.voxel.json> ...');
        process.exit(1);
    }
    const pct = (v) => `${(v * 100).toFixed(1)}%`;
    console.log(['file', 'occupied', 'islands', 'largest', 'roughness', 'scatter', 'faces/vox'].join('\t'));
    for (const p of paths) {
        const grid = await decodeVoxelFiles(p);
        const m = gridMetrics(grid);
        console.log([
            basename(p),
            m.occupied,
            m.islands,
            pct(m.largestShare),
            m.roughness.toFixed(2),
            pct(m.scatter),
            m.facesPerVoxel.toFixed(2)
        ].join('\t'));
    }
};

if (process.argv[1] && process.argv[1].endsWith('voxel-metrics.mjs')) {
    await main();
}

export { decodeOctree, decodeVoxelFiles, gridMetrics };
