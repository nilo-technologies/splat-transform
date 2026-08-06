# Voxel Yaw Auto-Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in automatic yaw alignment to the voxel writer so the voxel grid lines up with the scene's dominant surfaces, cutting staircase voxels and `.vox` size, while recording the rotation so `.voxel.json` and `.collision.glb` consumers can undo it.

**Architecture:** A CPU-only estimator (`src/lib/voxel/align-yaw.ts`) votes on the best yaw from per-gaussian surface normals and returns an angle plus a confidence guard. `writeVoxel` composes that yaw into the write transform it already builds, so a single voxelization pass produces every output in the rotated frame; the inverse rotation is then recorded in the `.voxel.json` metadata and as a glTF node rotation. `.vox` is emitted aligned with no metadata, which is the point of the feature.

**Tech Stack:** TypeScript (ES2022, ESM), `playcanvas` (`Quat`, `Vec3`) as a peer dependency, Node's built-in test runner (`node:test` + `node:assert`), Rollup, ESLint (`@playcanvas/eslint-config`), tsx for running TypeScript from tests.

**Design spec:** `specs/2026-08-07-voxel-auto-align-design.md`

## Global Constraints

- `src/lib/` must stay platform-agnostic: no `node:*` imports, no Node-only APIs. Only `src/lib/workers/` may use guarded dynamic `node:` imports, and this feature adds nothing there.
- Naming: classes PascalCase, functions camelCase, types PascalCase, constants UPPER_SNAKE_CASE. Arrow-function `const`s are the prevailing style in `src/lib/`.
- Import order: Node built-ins, then external packages (`playcanvas`), then internal relative paths.
- Use `const`/`let`, never `var`.
- All new public API needs JSDoc with `@param`/`@returns` descriptions (no types in JSDoc — TypeScript provides them) for Typedoc.
- Every output must stay byte-identical to today when the feature is off: no `rotation` key, `version` stays `'1.1'`, GLB node stays `{ mesh: 0 }`.
- Quaternion column convention: `rot_0` is `w`, `rot_1..3` are `x,y,z`. Playcanvas `Quat.set` takes `(x, y, z, w)`. See `src/lib/data-table/gaussian-aabb.ts:88`.
- Raw column conventions: scales are log-space (`linear = Math.exp(v)`), opacity is logit (`alpha = 1/(1+Math.exp(-v))`). See `src/lib/process.ts:242-250`.
- Recorded rotation convention: the quaternion written to both `.voxel.json` and the GLB node is `R_up(-theta)`, mapping voxel frame to source frame, in `[x, y, z, w]` order.
- **Yaw sign convention (verified empirically, do not re-derive):** take the horizontal pair in right-handed cyclic order — `up: 'x'` -> `(y, z)`, `up: 'y'` -> `(z, x)`, `up: 'z'` -> `(x, y)` — and `phi = atan2(b, a)`. Then applying `R_up(+theta)` maps `phi -> phi + theta` for every axis. Checked with `new Quat().setFromEulerAngles(0, 30, 0)`, which maps `(1,0,0)` to `(0.866, 0, -0.5)`: that is `atan2(z, x)` by `-30` but the cyclic pair `(z, x)` by `+30`. Using the naive pair `(x, z)` for `'y'` inverts the yaw and doubles the misalignment.
- `Transform.mul(other)` **mutates the receiver** (`src/lib/utils/math.ts:135-137`: `return this.mul2(this, other)`). Anything read off a Transform must be read before calling `.mul` on it.
- Test files are `test/*.test.mjs`. Full suite: `npm test` (`node --import tsx --test test/*.test.mjs`). Single file: `node --import tsx --test test/<name>.test.mjs`.
- Run `npm run lint` (`eslint src`) and the relevant tests before every commit. Only fix lint issues in code you touched.
- Commit messages follow conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`).

## File Structure

| File | Responsibility |
| --- | --- |
| `src/lib/voxel/align-yaw.ts` | **new** — `estimateAlignYaw` (normals -> weighted histogram -> cyclic correlation -> guarded yaw) and the pure `applyAlignYaw` transform/seed helper. No GPU, no I/O. |
| `test/align-yaw.test.mjs` | **new** — estimator and helper unit tests, no GPU. |
| `src/lib/voxel/index.ts` | export the new module's public surface. |
| `src/lib/index.ts` | re-export for library consumers. |
| `src/lib/writers/collision-glb.ts` | `encodeGlb` and `buildCollisionMesh` gain an optional node rotation. |
| `src/lib/writers/write-voxel.ts` | `autoRotate` option, estimator call, transform composition, `navSeed` rotation, `rotation` metadata field, version bump. |
| `test/write-voxel.test.mjs` | extend: metadata rotation/version cases, `autoRotate` validation. |
| `test/collision-glb-node.test.mjs` | **new** — GLB node rotation cases. |
| `src/lib/types.ts` | `LibOptions.autoRotate`, the bridge the CLI travels through. |
| `src/lib/write.ts` | forward `autoRotate` in the `'voxel'` case. |
| `src/cli/index.ts` | `--auto-rotate[=deg]` option, parsing, warning, usage text. |
| `test/cli.test.mjs` | extend: `--auto-rotate` argument parsing. |
| `specs/2026-08-07-voxel-auto-align-design.md` | record the validation experiment results table. |

**Dependency order:** Tasks 1-4 build the estimator module. Tasks 5 and 6 are independent format changes. Task 7 wires the writer and needs 1, 4, 5, 6. Task 8 exports. Task 9 needs 7. Task 10 needs 9.

---

### Task 1: Estimator core — normals, weights, and the yaw sweep

**Files:**
- Create: `src/lib/voxel/align-yaw.ts`
- Create: `test/align-yaw.test.mjs`

**Interfaces:**
- Consumes: `DataTable`, `Column` from `src/lib/index.js`; `Quat`, `Vec3` from `playcanvas`.
- Produces:
  ```ts
  type UpAxis = 'x' | 'y' | 'z';
  type AlignYawOptions = {
      up?: UpAxis;              // default 'y'
      stepDegrees?: number;     // default 0.125
      minImprovement?: number;  // default 0.02
      opacityCutoff?: number;   // default 0.1
  };
  type AlignYawResult = {
      yawDegrees: number;
      improvement: number;
      cost0: number;
      costBest: number;
      votedCount: number;
      totalWeight: number;
      curve: Float64Array;
      reason?: string;
  };
  const estimateAlignYaw: (dataTable: DataTable, options?: AlignYawOptions) => AlignYawResult;
  ```

- [ ] **Step 1: Write the failing test**

Create `test/align-yaw.test.mjs`. The geometric assertion checks the *invariant* — that applying the returned yaw axis-aligns the normals — using the real playcanvas API, so an inverted sign convention cannot pass:

```javascript
/**
 * Tests for voxel yaw auto-alignment estimation.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { Quat, Vec3 } from 'playcanvas';

import { Column, DataTable } from '../src/lib/index.js';
import { estimateAlignYaw } from '../src/lib/voxel/align-yaw.js';

const COLUMN_NAMES = [
    'x', 'y', 'z',
    'scale_0', 'scale_1', 'scale_2',
    'opacity',
    'rot_0', 'rot_1', 'rot_2', 'rot_3'
];

/**
 * Build a DataTable of flat gaussians standing as vertical walls, one per given
 * yaw. Local X is the flat axis, so a yaw about Y aims the wall's normal.
 *
 * @param {number[]} yawsDeg - Wall orientation per splat, in degrees.
 * @param {object} [opts] - Overrides.
 * @param {number} [opts.thin=0.01] - Linear size of the flat axis.
 * @param {number} [opts.wide=1] - Linear size of the two in-plane axes.
 * @param {number} [opts.alpha=0.9] - Linear opacity.
 * @returns {DataTable} Table with the standard voxel columns.
 */
function makeWalls(yawsDeg, opts = {}) {
    const { thin = 0.01, wide = 1, alpha = 0.9 } = opts;
    const count = yawsDeg.length;
    const data = {};
    for (const name of COLUMN_NAMES) {
        data[name] = new Float32Array(count);
    }

    const q = new Quat();
    for (let i = 0; i < count; i++) {
        q.setFromEulerAngles(0, yawsDeg[i], 0);
        data.rot_0[i] = q.w;
        data.rot_1[i] = q.x;
        data.rot_2[i] = q.y;
        data.rot_3[i] = q.z;
        data.scale_0[i] = Math.log(thin);
        data.scale_1[i] = Math.log(wide);
        data.scale_2[i] = Math.log(wide);
        data.opacity[i] = Math.log(alpha / (1 - alpha));
    }

    return new DataTable(COLUMN_NAMES.map(name => new Column(name, data[name])));
}

/**
 * Rotate a splat's flat-axis normal by a yaw and report how axis-aligned it is.
 * Returns (|x| + |z|) / hypot(x, z): 1 on a grid axis, sqrt(2) at 45 degrees.
 *
 * @param {DataTable} table - Table built by makeWalls.
 * @param {number} row - Row index.
 * @param {number} yawDeg - Yaw to apply about Y, in degrees.
 * @returns {number} Alignment ratio in [1, sqrt(2)].
 */
function alignmentRatio(table, row, yawDeg) {
    const get = name => table.getColumnByName(name).data[row];
    const q = new Quat(get('rot_1'), get('rot_2'), get('rot_3'), get('rot_0')).normalize();
    const n = new Vec3(1, 0, 0);
    q.transformVector(n, n);
    new Quat().setFromEulerAngles(0, yawDeg, 0).transformVector(n, n);
    return (Math.abs(n.x) + Math.abs(n.z)) / Math.hypot(n.x, n.z);
}

describe('estimateAlignYaw', function () {
    it('finds the yaw that axis-aligns a set of tilted walls', function () {
        // Four walls 90 degrees apart, the whole set tilted by 17 degrees.
        const table = makeWalls([17, 107, 197, 287]);

        const result = estimateAlignYaw(table);

        assert.strictEqual(result.reason, undefined, `unexpected guard: ${result.reason}`);
        assert.ok(result.improvement > 0.2, `expected a large improvement, got ${result.improvement}`);
        for (let row = 0; row < 4; row++) {
            assert.ok(alignmentRatio(table, row, 0) > 1.2,
                `row ${row}: fixture should start misaligned`);
            const after = alignmentRatio(table, row, result.yawDegrees);
            assert.ok(after < 1.001,
                `row ${row}: expected axis-aligned after yaw, ratio ${after}`);
        }
    });

    it('returns the smallest equivalent rotation, inside [-45, 45]', function () {
        const table = makeWalls([62, 152, 242, 332]);

        const result = estimateAlignYaw(table);

        assert.ok(result.yawDegrees >= -45 && result.yawDegrees <= 45,
            `expected a normalized yaw, got ${result.yawDegrees}`);
        for (let row = 0; row < 4; row++) {
            const after = alignmentRatio(table, row, result.yawDegrees);
            assert.ok(after < 1.001,
                `row ${row}: expected axis-aligned after yaw, ratio ${after}`);
        }
    });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --import tsx --test test/align-yaw.test.mjs`

Expected: FAIL — cannot resolve `../src/lib/voxel/align-yaw.js`.

- [ ] **Step 3: Write the implementation**

Create `src/lib/voxel/align-yaw.ts`:

```ts
import { Quat, Vec3 } from 'playcanvas';

import type { DataTable } from '../data-table';

/** Axis the alignment yaw rotates about. */
type UpAxis = 'x' | 'y' | 'z';

/**
 * Options for {@link estimateAlignYaw}.
 */
type AlignYawOptions = {
    /** Axis to rotate about. Default: `'y'` */
    up?: UpAxis;
    /** Search resolution in degrees. Default: 0.125 */
    stepDegrees?: number;
    /** Minimum predicted improvement required to report a non-zero yaw. Default: 0.02 */
    minImprovement?: number;
    /** Linear opacity below which a Gaussian does not vote. Default: 0.1 */
    opacityCutoff?: number;
};

/**
 * Result of {@link estimateAlignYaw}.
 */
type AlignYawResult = {
    /** Yaw to apply, in degrees, normalized to [-45, 45]. Zero when no rotation is recommended. */
    yawDegrees: number;
    /** Predicted fraction of surface voxels saved, in [0, 1). */
    improvement: number;
    /** Cost at zero yaw. */
    cost0: number;
    /** Cost at the best yaw. */
    costBest: number;
    /** Number of Gaussians that contributed a vote. */
    votedCount: number;
    /** Sum of vote weights. */
    totalWeight: number;
    /** Cost sampled across [0, 90) degrees. */
    curve: Float64Array;
    /** Set when no rotation is recommended, explaining why. */
    reason?: string;
};

const QUARTER_TURN = Math.PI / 2;

const REQUIRED_COLUMNS = [
    'rot_0', 'rot_1', 'rot_2', 'rot_3',
    'scale_0', 'scale_1', 'scale_2',
    'opacity'
];

// Horizontal component pair in right-handed cyclic order, which makes
// R_up(+theta) map atan2(b, a) to atan2(b, a) + theta for every axis.
const PAIRS: Record<UpAxis, [number, number]> = {
    x: [1, 2], // (y, z)
    y: [2, 0], // (z, x)
    z: [0, 1]  // (x, y)
};

// Voxels touched per unit area by a plane whose normal sits at angle `a` to the
// grid: 1 on an axis, sqrt(2) at 45 degrees.
const gridCost = (a: number): number => Math.abs(Math.cos(a)) + Math.abs(Math.sin(a));

/**
 * Estimates the yaw that best aligns a splat scene's dominant surfaces with the
 * voxel grid axes.
 *
 * Each Gaussian votes for the orientation of its flattest axis - its surface
 * normal - weighted by how flat, how large and how opaque it is. A candidate
 * yaw's cost is how many voxels the voted surfaces would occupy at that
 * orientation, so the minimum is the best-aligned yaw. Rotating a scene by the
 * returned angle before voxelizing removes staircase voxels.
 *
 * Normals are taken into the DataTable's output space using
 * `dataTable.transform.rotation`, so the yaw can be applied directly on top of
 * the table's own transform.
 *
 * @param dataTable - Splat data with rotation, scale and opacity columns.
 * @param options - Search and weighting options.
 * @returns The recommended yaw with the cost curve and diagnostics. When no
 * orientation is meaningfully better, `yawDegrees` is 0 and `reason` explains why.
 * @throws If a required column is missing or an option is out of range.
 *
 * @example
 * ```ts
 * const { yawDegrees, improvement, reason } = estimateAlignYaw(dataTable);
 * if (!reason) {
 *     console.log(`yaw ${yawDegrees.toFixed(2)} saves ${(improvement * 100).toFixed(0)}%`);
 * }
 * ```
 */
const estimateAlignYaw = (dataTable: DataTable, options: AlignYawOptions = {}): AlignYawResult => {
    const {
        up = 'y',
        stepDegrees = 0.125,
        minImprovement = 0.02,
        opacityCutoff = 0.1
    } = options;

    if (!PAIRS[up]) {
        throw new Error(`estimateAlignYaw: invalid up axis '${up}', expected 'x', 'y' or 'z'`);
    }
    if (!(stepDegrees > 0) || stepDegrees > 90) {
        throw new Error(`estimateAlignYaw: stepDegrees must be in (0, 90], got ${stepDegrees}`);
    }
    for (const name of REQUIRED_COLUMNS) {
        if (!dataTable.hasColumn(name)) {
            throw new Error(`estimateAlignYaw: missing required column '${name}'`);
        }
    }

    const rotW = dataTable.getColumnByName('rot_0')!.data;
    const rotX = dataTable.getColumnByName('rot_1')!.data;
    const rotY = dataTable.getColumnByName('rot_2')!.data;
    const rotZ = dataTable.getColumnByName('rot_3')!.data;
    const scale0 = dataTable.getColumnByName('scale_0')!.data;
    const scale1 = dataTable.getColumnByName('scale_1')!.data;
    const scale2 = dataTable.getColumnByName('scale_2')!.data;
    const opacity = dataTable.getColumnByName('opacity')!.data;

    const numBins = Math.max(1, Math.round(90 / stepDegrees));
    const weights = new Float64Array(numBins);
    const binsPerRadian = numBins / QUARTER_TURN;

    const [aAxis, bAxis] = PAIRS[up];
    const preRotation = dataTable.transform.rotation;
    const q = new Quat();
    const n = new Vec3();
    const components = [0, 0, 0];

    let votedCount = 0;
    let totalWeight = 0;

    for (let i = 0; i < dataTable.numRows; i++) {
        const s0 = Math.exp(scale0[i]);
        const s1 = Math.exp(scale1[i]);
        const s2 = Math.exp(scale2[i]);
        if (!Number.isFinite(s0) || !Number.isFinite(s1) || !Number.isFinite(s2)) continue;

        // Logit opacity: +Infinity means fully opaque; NaN fails the comparison.
        const alpha = 1 / (1 + Math.exp(-opacity[i]));
        if (!(alpha >= opacityCutoff)) continue;

        // The flattest local axis is the surface normal.
        let flat = 0;
        let sMin = s0;
        if (s1 < sMin) { sMin = s1; flat = 1; }
        if (s2 < sMin) { sMin = s2; flat = 2; }
        const otherA = flat === 0 ? s1 : s0;
        const otherB = flat === 2 ? s1 : s2;
        const sMid = Math.min(otherA, otherB);
        const sMax = Math.max(otherA, otherB);
        if (!(sMid > 0)) continue;

        // Flat discs vote, round blobs do not; larger patches vote louder.
        const weight = (1 - sMin / sMid) * (sMid * sMax) * alpha;
        if (!(weight > 0)) continue;

        q.set(rotX[i], rotY[i], rotZ[i], rotW[i]);
        if (!(Math.hypot(q.x, q.y, q.z, q.w) > 1e-8)) continue;
        q.normalize();

        n.set(flat === 0 ? 1 : 0, flat === 1 ? 1 : 0, flat === 2 ? 1 : 0);
        q.transformVector(n, n);
        preRotation.transformVector(n, n);

        components[0] = n.x;
        components[1] = n.y;
        components[2] = n.z;
        const a = components[aAxis];
        const b = components[bAxis];
        const h = Math.hypot(a, b);
        // A normal parallel to the up axis (floor, ceiling) carries no yaw signal.
        if (!(h > 1e-6)) continue;

        let phi = Math.atan2(b, a) % QUARTER_TURN;
        if (phi < 0) phi += QUARTER_TURN;
        const bin = Math.min(numBins - 1, (phi * binsPerRadian) | 0);

        const vote = weight * h;
        weights[bin] += vote;
        totalWeight += vote;
        votedCount++;
    }

    const curve = new Float64Array(numBins);

    if (votedCount === 0) {
        return {
            yawDegrees: 0,
            improvement: 0,
            cost0: 0,
            costBest: 0,
            votedCount,
            totalWeight,
            curve,
            reason: 'no eligible gaussians: none were flat, opaque and non-degenerate enough to vote'
        };
    }

    // Both phi and theta land on bin multiples, so the sweep is a cyclic
    // correlation against a precomputed table - no trigonometry in the loop.
    const costTable = new Float64Array(numBins);
    for (let m = 0; m < numBins; m++) {
        costTable[m] = gridCost((m + 0.5) * QUARTER_TURN / numBins);
    }
    for (let j = 0; j < numBins; j++) {
        let sum = 0;
        for (let b = 0; b < numBins; b++) {
            sum += weights[b] * costTable[(b + j) % numBins];
        }
        curve[j] = sum;
    }

    let best = 0;
    for (let j = 1; j < numBins; j++) {
        if (curve[j] < curve[best]) best = j;
    }

    // Parabolic fit across the cyclic neighbours for sub-bin precision.
    const prev = curve[(best - 1 + numBins) % numBins];
    const mid = curve[best];
    const next = curve[(best + 1) % numBins];
    const denom = prev - 2 * mid + next;
    const offset = denom > 0 ? 0.5 * (prev - next) / denom : 0;

    const cost0 = curve[0];
    const costBest = mid;
    const improvement = cost0 > 0 ? 1 - costBest / cost0 : 0;

    let yawDegrees = (best + offset) * (90 / numBins);
    if (yawDegrees > 45) yawDegrees -= 90;

    if (improvement < minImprovement) {
        return {
            yawDegrees: 0,
            improvement,
            cost0,
            costBest,
            votedCount,
            totalWeight,
            curve,
            reason: `no dominant alignment: best yaw saves ${(improvement * 100).toFixed(1)}%, below the ${(minImprovement * 100).toFixed(1)}% threshold`
        };
    }

    return { yawDegrees, improvement, cost0, costBest, votedCount, totalWeight, curve };
};

export { estimateAlignYaw, type AlignYawOptions, type AlignYawResult, type UpAxis };
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --import tsx --test test/align-yaw.test.mjs`

Expected: PASS, both tests.

If `after` comes back near 1.414 while `before` was near 1.0, the cyclic pair for `'y'` is inverted — re-read the yaw sign convention in Global Constraints. Do **not** patch it by negating `yawDegrees` at the end; that would break `up: 'x'` and `up: 'z'`.

- [ ] **Step 5: Lint and commit**

```bash
npm run lint
git add src/lib/voxel/align-yaw.ts test/align-yaw.test.mjs
git commit -m "feat: add voxel yaw alignment estimator"
```

---

### Task 2: Eligibility filters and the confidence guard

**Files:**
- Modify: `test/align-yaw.test.mjs`
- Modify: `src/lib/voxel/align-yaw.ts` (only if a case fails)

**Interfaces:**
- Consumes: `estimateAlignYaw`, `makeWalls`, `COLUMN_NAMES` from Task 1.
- Produces: nothing new. This task proves the Task 1 filters behave.

Each case guards a distinct silent-failure mode. Several may pass on the first run — that is fine here, because the implementation was written in Task 1; what matters is that a future change cannot break them unnoticed.

- [ ] **Step 1: Write the tests**

Append inside `describe('estimateAlignYaw', ...)` in `test/align-yaw.test.mjs`:

```javascript
    it('ignores isotropic blobs, which have no surface orientation', function () {
        const table = makeWalls([17, 107, 197, 287], { thin: 0.5, wide: 0.5 });

        const result = estimateAlignYaw(table);

        assert.strictEqual(result.yawDegrees, 0);
        assert.strictEqual(result.votedCount, 0);
        assert.match(result.reason, /no eligible gaussians/);
    });

    it('reports no dominant alignment when normals are spread evenly', function () {
        const yaws = [];
        for (let i = 0; i < 360; i++) yaws.push(i * 0.25);

        const result = estimateAlignYaw(makeWalls(yaws));

        assert.strictEqual(result.yawDegrees, 0);
        assert.ok(result.votedCount > 0, 'the splats should still have voted');
        assert.match(result.reason, /no dominant alignment/);
    });

    it('does not let faint gaussians outvote opaque ones', function () {
        // 200 faint walls at 30 degrees against 4 opaque walls on-axis. The
        // faint set is under the cutoff, so the opaque set must decide.
        const faint = makeWalls(new Array(200).fill(30), { alpha: 0.02 });
        const opaque = makeWalls([0, 90, 180, 270], { alpha: 0.95 });
        const merged = new DataTable(COLUMN_NAMES.map((name) => {
            const head = faint.getColumnByName(name).data;
            const tail = opaque.getColumnByName(name).data;
            const data = new Float32Array(head.length + tail.length);
            data.set(head, 0);
            data.set(tail, head.length);
            return new Column(name, data);
        }));

        const result = estimateAlignYaw(merged);

        assert.ok(Math.abs(result.yawDegrees) < 1,
            `expected the opaque on-axis walls to win, got ${result.yawDegrees}`);
    });

    it('counts fully opaque splats and skips NaN rows', function () {
        const table = makeWalls([17, 107, 197, 287]);
        const opacityData = table.getColumnByName('opacity').data;
        opacityData[0] = Infinity;   // alpha === 1, votes
        opacityData[1] = NaN;        // skipped

        const result = estimateAlignYaw(table);

        assert.strictEqual(result.votedCount, 3);
        assert.ok(Number.isFinite(result.costBest));
        assert.ok(result.improvement > 0.2);
    });
```

- [ ] **Step 2: Run the tests**

Run: `node --import tsx --test test/align-yaw.test.mjs`

Expected: PASS, all six cases.

If the even-spread case returns a yaw instead of a reason, do **not** raise `minImprovement` to silence it — a uniform distribution must produce a nearly flat curve, so a large improvement means the histogram folding is wrong.

- [ ] **Step 3: Commit**

```bash
npm run lint
git add test/align-yaw.test.mjs src/lib/voxel/align-yaw.ts
git commit -m "test: cover align-yaw eligibility filters and guard"
```

---

### Task 3: Transform-space correctness and the up-axis option

**Files:**
- Modify: `test/align-yaw.test.mjs`
- Modify: `src/lib/voxel/align-yaw.ts` (only if a case fails)

**Interfaces:**
- Consumes: `estimateAlignYaw`, `makeWalls`, `COLUMN_NAMES` from Task 1; `Transform` from `../src/lib/utils/index.js` (exported at `src/lib/utils/index.ts:5`).
- Produces: nothing new.

This is the highest-value test in the plan. `writeVoxel` calls the estimator on a table whose `transform` is normally `Transform.PLY`, and the yaw must come back in output space. Getting it wrong yields a plausible angle that makes alignment worse.

- [ ] **Step 1: Write the tests**

Add `import { Transform } from '../src/lib/utils/index.js';` to the imports, then append inside `describe('estimateAlignYaw', ...)`:

```javascript
    it('returns a yaw in output space when the table carries a transform', function () {
        // Transform.PLY is a 180 degree roll about Z, which negates x and y.
        // Same walls, so the applied yaw must agree modulo the grid's 90 degree
        // symmetry.
        const identity = makeWalls([17, 107, 197, 287]);
        const transformed = new DataTable(
            COLUMN_NAMES.map(name => new Column(name, identity.getColumnByName(name).data.slice())),
            Transform.PLY.clone()
        );

        const plain = estimateAlignYaw(identity);
        const withTransform = estimateAlignYaw(transformed);

        const delta = Math.abs(plain.yawDegrees - withTransform.yawDegrees) % 90;
        assert.ok(Math.min(delta, 90 - delta) < 0.5,
            `expected matching yaws, got ${plain.yawDegrees} and ${withTransform.yawDegrees}`);
        assert.strictEqual(withTransform.reason, undefined);
    });

    it('supports rotating about another up axis', function () {
        const table = makeWalls([17, 107, 197, 287]);

        const aboutY = estimateAlignYaw(table, { up: 'y' });
        const aboutZ = estimateAlignYaw(table, { up: 'z' });

        // These normals lie in the XZ plane, so a Y-up search sees them fully
        // while a Z-up search sees them partly edge-on.
        assert.ok(aboutY.improvement > 0.2);
        assert.ok(aboutZ.improvement <= aboutY.improvement + 1e-9);
    });

    it('rejects an invalid up axis and a nonsense step', function () {
        const table = makeWalls([0, 90]);

        assert.throws(() => estimateAlignYaw(table, { up: 'w' }), /invalid up axis/);
        assert.throws(() => estimateAlignYaw(table, { stepDegrees: 0 }), /stepDegrees/);
    });

    it('throws when a required column is missing', function () {
        const table = makeWalls([0, 90]);
        table.removeColumn('opacity');

        assert.throws(() => estimateAlignYaw(table), /missing required column 'opacity'/);
    });
```

- [ ] **Step 2: Run the tests**

Run: `node --import tsx --test test/align-yaw.test.mjs`

Expected: PASS, all ten cases.

If the transform case fails, the bug is the order in which `preRotation` is applied to `n`: it must be applied *after* the gaussian's own quaternion, matching `_q.set(...).mul2(r, _q)` at `src/lib/data-table/transform.ts:110`.

- [ ] **Step 3: Commit**

```bash
npm run lint
git add test/align-yaw.test.mjs src/lib/voxel/align-yaw.ts
git commit -m "test: cover align-yaw transform space and up-axis option"
```

---

### Task 4: `applyAlignYaw` — compose the transform and rotate the seed

**Files:**
- Modify: `src/lib/voxel/align-yaw.ts`
- Modify: `test/align-yaw.test.mjs`

**Interfaces:**
- Consumes: `Transform` from `../utils`; `Vec3`, `Quat` from `playcanvas`.
- Produces:
  ```ts
  type AlignYawApplied = {
      delta: Transform;
      navSeed?: { x: number; y: number; z: number };
      recordedRotation: [number, number, number, number] | null;
  };
  const applyAlignYaw: (
      delta: Transform,
      navSeed: { x: number; y: number; z: number } | undefined,
      yawDegrees: number,
      up?: UpAxis
  ) => AlignYawApplied;
  ```

This helper exists so the `navSeed` rotation — one line whose failure is silent and only reachable with a GPU — is unit-testable.

- [ ] **Step 1: Write the failing test**

Add a second `describe` block to `test/align-yaw.test.mjs`, and add `applyAlignYaw` to the `align-yaw.js` import:

```javascript
describe('applyAlignYaw', function () {
    it('is a no-op at zero yaw and records nothing', function () {
        const delta = new Transform();

        const result = applyAlignYaw(delta, { x: 1, y: 2, z: 3 }, 0);

        assert.strictEqual(result.recordedRotation, null);
        assert.deepStrictEqual(result.navSeed, { x: 1, y: 2, z: 3 });
        assert.ok(result.delta.isIdentity());
    });

    it('rotates the nav seed into the voxel frame', function () {
        // R_y(90) maps (1, 0, 0) to (0, 0, -1).
        const result = applyAlignYaw(new Transform(), { x: 1, y: 0, z: 0 }, 90);

        assert.ok(Math.abs(result.navSeed.x) < 1e-6, `x was ${result.navSeed.x}`);
        assert.ok(Math.abs(result.navSeed.y) < 1e-6, `y was ${result.navSeed.y}`);
        assert.ok(Math.abs(result.navSeed.z + 1) < 1e-6, `z was ${result.navSeed.z}`);
    });

    it('records the rotation that maps the voxel frame back to source', function () {
        const yaw = 30;

        const result = applyAlignYaw(new Transform(), undefined, yaw);

        const [x, y, z, w] = result.recordedRotation;
        const point = new Vec3(1, 0, 0);
        new Quat().setFromEulerAngles(0, yaw, 0).transformVector(point, point);
        new Quat(x, y, z, w).transformVector(point, point);
        assert.ok(Math.abs(point.x - 1) < 1e-6 && Math.abs(point.z) < 1e-6,
            `round trip failed: ${point.x}, ${point.y}, ${point.z}`);
        assert.strictEqual(result.navSeed, undefined);
    });

    it('composes the yaw on top of an existing delta', function () {
        const result = applyAlignYaw(new Transform().fromEulers(0, 10, 0), undefined, 20);

        const point = new Vec3(1, 0, 0);
        result.delta.transformPoint(point, point);
        const expected = new Vec3(1, 0, 0);
        new Quat().setFromEulerAngles(0, 30, 0).transformVector(expected, expected);
        assert.ok(Math.abs(point.x - expected.x) < 1e-5 && Math.abs(point.z - expected.z) < 1e-5,
            `expected the yaws to add: got ${point.x}, ${point.z}`);
    });

    it('rejects a non-finite yaw', function () {
        assert.throws(() => applyAlignYaw(new Transform(), undefined, NaN), /finite/);
    });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --import tsx --test test/align-yaw.test.mjs`

Expected: FAIL — `applyAlignYaw is not a function`.

- [ ] **Step 3: Write the implementation**

Add `import { Transform } from '../utils';` to `src/lib/voxel/align-yaw.ts`, then append:

```ts
/**
 * Result of {@link applyAlignYaw}.
 */
type AlignYawApplied = {
    /** Write transform with the alignment yaw composed in. */
    delta: Transform;
    /** Seed position rotated into the aligned voxel frame, when one was given. */
    navSeed?: { x: number; y: number; z: number };
    /** Rotation mapping the voxel frame back to source space as `[x, y, z, w]`, or null at zero yaw. */
    recordedRotation: [number, number, number, number] | null;
};

const EULERS: Record<UpAxis, (deg: number) => [number, number, number]> = {
    x: deg => [deg, 0, 0],
    y: deg => [0, deg, 0],
    z: deg => [0, 0, deg]
};

/**
 * Composes an alignment yaw into a write transform, rotating any seed position
 * with it and producing the rotation that maps the aligned frame back to source
 * space for the output metadata.
 *
 * @param delta - Write transform the yaw is applied on top of.
 * @param navSeed - Seed position in source space, if any.
 * @param yawDegrees - Yaw to apply about `up`, in degrees.
 * @param up - Axis the yaw rotates about. Default: `'y'`
 * @returns The composed transform, the rotated seed and the rotation to record.
 * At zero yaw the inputs pass through untouched and `recordedRotation` is null.
 * @throws If `yawDegrees` is not finite or `up` is not an axis.
 */
const applyAlignYaw = (
    delta: Transform,
    navSeed: { x: number; y: number; z: number } | undefined,
    yawDegrees: number,
    up: UpAxis = 'y'
): AlignYawApplied => {
    if (!Number.isFinite(yawDegrees)) {
        throw new Error(`applyAlignYaw: yawDegrees must be finite, got ${yawDegrees}`);
    }
    if (!EULERS[up]) {
        throw new Error(`applyAlignYaw: invalid up axis '${up}', expected 'x', 'y' or 'z'`);
    }
    if (yawDegrees === 0) {
        return { delta, navSeed, recordedRotation: null };
    }

    const [ex, ey, ez] = EULERS[up](yawDegrees);
    const yaw = new Transform().fromEulers(ex, ey, ez);

    // Read everything off `yaw` before `.mul`, which mutates the receiver
    // (src/lib/utils/math.ts:135-137).
    let rotatedSeed = navSeed;
    if (navSeed) {
        const p = new Vec3(navSeed.x, navSeed.y, navSeed.z);
        yaw.rotation.transformVector(p, p);
        rotatedSeed = { x: p.x, y: p.y, z: p.z };
    }
    const inverse = yaw.rotation.clone().invert();

    return {
        // Same ordering as the rotate process action (src/lib/process.ts:388):
        // the yaw applies after delta.
        delta: yaw.mul(delta),
        navSeed: rotatedSeed,
        recordedRotation: [inverse.x, inverse.y, inverse.z, inverse.w]
    };
};
```

Extend the module's export statement to:

```ts
export {
    estimateAlignYaw,
    applyAlignYaw,
    type AlignYawOptions,
    type AlignYawResult,
    type AlignYawApplied,
    type UpAxis
};
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --import tsx --test test/align-yaw.test.mjs`

Expected: PASS, all fifteen cases.

- [ ] **Step 5: Commit**

```bash
npm run lint
git add src/lib/voxel/align-yaw.ts test/align-yaw.test.mjs
git commit -m "feat: add applyAlignYaw transform and seed helper"
```

---

### Task 5: Record the rotation in `.voxel.json`

**Files:**
- Modify: `src/lib/writers/write-voxel.ts` (`VoxelMetadata` at :118-154, `writeOctreeFiles` at :295-337)
- Modify: `test/write-voxel.test.mjs` (`describe('writeOctreeFiles', ...)` at :14)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `writeOctreeFiles(fs, jsonFilename, octree, rotation?: [number, number, number, number] | null)`; `VoxelMetadata.rotation?: [number, number, number, number]`.

- [ ] **Step 1: Write the tests**

In `test/write-voxel.test.mjs`, hoist the octree literal out of the existing test (it currently sits inline at :17-33) into a helper placed above `describe('writeOctreeFiles', ...)`, and point the existing test at it so all three cases share one fixture:

```javascript
function makeOctree() {
    return {
        gridBounds: { min: new Vec3(0, 1, 2), max: new Vec3(3, 4, 5) },
        sceneBounds: { min: new Vec3(-1, -2, -3), max: new Vec3(6, 7, 8) },
        voxelResolution: 0.25,
        leafSize: 4,
        treeDepth: 2,
        numInteriorNodes: 1,
        numMixedLeaves: 1,
        nodes: new Uint32Array([0x11223344, 0xAABBCCDD]),
        leafData: new Uint32Array([0x01020304, 0xFFEEDDCC])
    };
}
```

Then add two cases inside the same `describe`:

```javascript
    it('omits the rotation field and stays at version 1.1 by default', async function () {
        const fs = new MemoryFileSystem();

        await writeOctreeFiles(fs, 'scene.voxel.json', makeOctree());

        const metadata = JSON.parse(new TextDecoder().decode(fs.results.get('scene.voxel.json')));
        assert.strictEqual(metadata.version, '1.1');
        assert.ok(!('rotation' in metadata), 'rotation must not appear when unaligned');
    });

    it('writes the rotation and bumps to version 1.2 when aligned', async function () {
        const fs = new MemoryFileSystem();
        const rotation = [0, -0.2588190451, 0, 0.9659258263];

        await writeOctreeFiles(fs, 'scene.voxel.json', makeOctree(), rotation);

        const metadata = JSON.parse(new TextDecoder().decode(fs.results.get('scene.voxel.json')));
        assert.strictEqual(metadata.version, '1.2');
        assert.deepStrictEqual(metadata.rotation, rotation);
    });
```

- [ ] **Step 2: Run the tests to verify the new one fails**

Run: `node --import tsx --test test/write-voxel.test.mjs`

Expected: the version-1.2 case FAILS (`version` is `'1.1'`, `rotation` undefined). The default case passes already — it is the regression guard.

- [ ] **Step 3: Write the implementation**

In `src/lib/writers/write-voxel.ts`, add to `VoxelMetadata` after `sceneBounds` (:132):

```ts
    /** Rotation mapping voxel space back to source space as `[x, y, z, w]`. Present only in v1.2+, when auto-alignment was applied. */
    rotation?: [number, number, number, number];
```

Change `writeOctreeFiles` (:295) to take the rotation, set the version from it, and attach it:

```ts
const writeOctreeFiles = async (
    fs: FileSystem,
    jsonFilename: string,
    octree: SparseOctree,
    rotation: [number, number, number, number] | null = null
): Promise<void> => {
    // Build metadata object
    const metadata: VoxelMetadata = {
        version: rotation ? '1.2' : '1.1',
        ...
    };

    if (rotation) {
        metadata.rotation = rotation;
    }
```

Add an `@param rotation` line to its JSDoc.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --import tsx --test test/write-voxel.test.mjs`

Expected: PASS, including the pre-existing byte-level `nodes`/`leafData` assertions.

- [ ] **Step 5: Commit**

```bash
npm run lint
git add src/lib/writers/write-voxel.ts test/write-voxel.test.mjs
git commit -m "feat: record voxel alignment rotation in .voxel.json metadata"
```

---

### Task 6: Node rotation in the collision GLB

**Files:**
- Modify: `src/lib/writers/collision-glb.ts` (`encodeGlb` at :21, `nodes` literal at :104, `buildCollisionMesh` at :280-294, `encodeGlb` call at :533)
- Create: `test/collision-glb-node.test.mjs`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `encodeGlb(positions, indices, colors?, nodeRotation?)`; `buildCollisionMesh(grid, gridBounds, voxelResolution, shape?, colorSource?, options?: { nodeRotation?: [number, number, number, number] | null })`.

- [ ] **Step 1: Write the failing test**

Create `test/collision-glb-node.test.mjs`. It parses the GLB container directly, so it needs no grid and no GPU:

```javascript
/**
 * Tests for the collision GLB node transform.
 */

import assert from 'node:assert';
import { describe, it } from 'node:test';

import { encodeGlb } from '../src/lib/writers/collision-glb.js';

/**
 * Extract and parse the JSON chunk of a GLB container.
 *
 * @param {Uint8Array} glb - Encoded GLB bytes.
 * @returns {object} Parsed glTF JSON.
 */
function readGltfJson(glb) {
    const view = new DataView(glb.buffer, glb.byteOffset, glb.byteLength);
    const jsonLength = view.getUint32(12, true);
    const jsonBytes = glb.subarray(20, 20 + jsonLength);
    return JSON.parse(new TextDecoder().decode(jsonBytes));
}

describe('encodeGlb node transform', function () {
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
    const indices = new Uint32Array([0, 1, 2]);

    it('emits a bare mesh node when no rotation is given', function () {
        const gltf = readGltfJson(encodeGlb(positions, indices));

        assert.deepStrictEqual(gltf.nodes, [{ mesh: 0 }]);
    });

    it('emits the node rotation when one is given', function () {
        const rotation = [0, -0.2588190451, 0, 0.9659258263];

        const gltf = readGltfJson(encodeGlb(positions, indices, undefined, rotation));

        assert.deepStrictEqual(gltf.nodes, [{ mesh: 0, rotation }]);
        assert.deepStrictEqual(gltf.scenes, [{ nodes: [0] }]);
    });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --import tsx --test test/collision-glb-node.test.mjs`

Expected: FAIL — `encodeGlb` is not exported from `collision-glb.ts` yet, so the import is undefined.

- [ ] **Step 3: Write the implementation**

In `src/lib/writers/collision-glb.ts`:

1. Widen `encodeGlb` (:21):

```ts
function encodeGlb(
    positions: Float32Array,
    indices: Uint32Array,
    colors?: Float32Array,
    nodeRotation?: [number, number, number, number] | null
): Uint8Array {
```

2. Replace the `nodes` entry in the `gltf` literal (:104):

```ts
        nodes: [nodeRotation ? { mesh: 0, rotation: nodeRotation } : { mesh: 0 }],
```

3. Add an options bag as the sixth parameter of `buildCollisionMesh`, after `colorSource` (:293):

```ts
    options: { nodeRotation?: [number, number, number, number] | null } = {}
```

4. Forward it at the `encodeGlb` call (:533):

```ts
    const glb = encodeGlb(finalMesh.positions, finalMesh.indices, colors, options.nodeRotation);
```

5. Add `encodeGlb` to the module's exports, and document the new parameters in both JSDoc blocks.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `node --import tsx --test test/collision-glb-node.test.mjs test/write-glb.test.mjs test/collision-color.test.mjs`

Expected: PASS. The existing GLB and collision-colour tests must be unaffected, since the new parameter defaults to absent.

- [ ] **Step 5: Commit**

```bash
npm run lint
git add src/lib/writers/collision-glb.ts test/collision-glb-node.test.mjs
git commit -m "feat: support a node rotation in the collision GLB"
```

---

### Task 7: Wire `autoRotate` into `writeVoxel`

**Files:**
- Modify: `src/lib/writers/write-voxel.ts` (options type :57-113, destructure :373-391, validation near :411, transform :459-461, `navSeed` uses :558 and :581, `buildCollisionMesh` call :638-641, `writeOctreeFiles` call :666)
- Modify: `test/write-voxel.test.mjs`

**Interfaces:**
- Consumes: `estimateAlignYaw`, `applyAlignYaw` from Tasks 1 and 4; `writeOctreeFiles` rotation param from Task 5; `buildCollisionMesh` options bag from Task 6.
- Produces: `WriteVoxelOptions.autoRotate?: boolean | number`.

- [ ] **Step 1: Write the failing test**

Add a new `describe` to `test/write-voxel.test.mjs`, following the dummy-device pattern already used at :57-68 (validation runs before the device is created, so it is never invoked):

```javascript
describe('writeVoxel autoRotate validation', function () {
    const dummyCreateDevice = async () => ({});

    /**
     * Minimal table carrying every column writeVoxel requires.
     *
     * @returns {DataTable} One-row table.
     */
    function makeTable() {
        const names = [
            'x', 'y', 'z',
            'rot_0', 'rot_1', 'rot_2', 'rot_3',
            'scale_0', 'scale_1', 'scale_2',
            'opacity'
        ];
        return new DataTable(names.map(name => new Column(name, new Float32Array(1))));
    }

    it('rejects a non-finite autoRotate angle', async function () {
        await assert.rejects(
            () => writeVoxel({
                fs: new MemoryFileSystem(),
                filename: 'scene.voxel.json',
                dataTable: makeTable(),
                createDevice: dummyCreateDevice,
                autoRotate: NaN
            }),
            /autoRotate/
        );
    });
});
```

Check how the existing `writeVoxel` validation tests pass the file system — at :65 they use `createDevice: dummyCreateDevice` and the signature is `writeVoxel(options, fs)`. Match whichever form those tests use rather than the `fs:` property above if they differ.

- [ ] **Step 2: Run the test to verify it fails**

Run: `node --import tsx --test test/write-voxel.test.mjs`

Expected: FAIL — no `autoRotate` validation exists, so the call gets past validation and rejects with an unrelated message (or resolves).

- [ ] **Step 3: Write the implementation**

In `src/lib/writers/write-voxel.ts`:

1. Add the import beside the other `../voxel` imports:

```ts
import { applyAlignYaw, estimateAlignYaw } from '../voxel/align-yaw';
```

2. Add to `WriteVoxelOptions` after `collisionVoxelsSize` (:112):

```ts
    /** Rotate the voxel grid to line up with the scene's dominant surfaces, cutting staircase voxels. `true` estimates the best yaw about Y; a number applies that yaw in degrees verbatim. The rotation is recorded in the `.voxel.json` metadata and as a `.collision.glb` node rotation, so those outputs still land on the unrotated splat; the `.vox` is written in the aligned frame. Default: false */
    autoRotate?: boolean | number;
```

3. Add `autoRotate = false` to the destructure (:373-391).

4. Add validation beside the `collisionVoxelsSize` check (:411):

```ts
    if (typeof autoRotate === 'number' && !Number.isFinite(autoRotate)) {
        throw new Error(`autoRotate must be true, false or a finite angle in degrees, got ${autoRotate}`);
    }
```

5. Replace the transform block at :459-461. `estimateAlignYaw` reads only raw columns plus `dataTable.transform.rotation`, so it runs before any column is materialized:

```ts
    const writeDelta = computeWriteTransform(dataTable.transform, Transform.IDENTITY);

    let alignYaw = 0;
    if (autoRotate !== false) {
        if (typeof autoRotate === 'number') {
            alignYaw = autoRotate;
            if (alignYaw !== 0) {
                logger.info(`auto-rotate: yaw ${alignYaw.toFixed(2)}deg (explicit)`);
            }
        } else {
            const estimate = estimateAlignYaw(dataTable, { opacityCutoff });
            alignYaw = estimate.yawDegrees;
            if (estimate.reason) {
                logger.info(`auto-rotate: no rotation applied - ${estimate.reason}`);
            } else {
                logger.info(`auto-rotate: yaw ${alignYaw.toFixed(2)}deg (est. ${(estimate.improvement * 100).toFixed(0)}% fewer surface voxels, ${fmtCount(estimate.votedCount)} of ${fmtCount(dataTable.numRows)} splats voted)`);
            }
            const bins = estimate.curve.length;
            for (let deg = 0; deg < 90 && bins > 0; deg += 5) {
                const idx = Math.min(bins - 1, Math.round(deg / 90 * bins));
                logger.debug(`auto-rotate cost at ${deg}deg: ${estimate.curve[idx].toFixed(3)}`);
            }
        }
    }

    const aligned = applyAlignYaw(writeDelta ?? new Transform(), navSeed, alignYaw);
    const alignedSeed = aligned.navSeed;
    const recordedRotation = aligned.recordedRotation;
    const delta = aligned.delta;

    let cols: ReturnType<typeof transformColumns> | null = transformColumns(dataTable, voxelColumns, delta);
```

`computeWriteTransform` returns `Transform | null` (`src/lib/data-table/transform.ts:21-24`), hence the `?? new Transform()`. This is equivalent for the unaligned path because `transformColumns` short-circuits on `!delta || delta.isIdentity()` (`transform.ts:48`).

6. Replace `navSeed!` with `alignedSeed!` at the `fillExterior` call (:558) and the `carve` call (:581). Leave the `hasNav`/`hasFillExterior` predicates at :438-442 reading `navSeed` — presence does not change under rotation.

7. Pass the rotation to both writers:

```ts
        const glbBytes = collisionMeshShape ?
            buildCollisionMesh(grid, gridBounds, voxelResolution, collisionMeshShape,
                coloredCollisionMesh ? { ...splatColors!, flatShade: collisionColorFlat } : null,
                { nodeRotation: recordedRotation }) :
            null;
```

```ts
        await writeOctreeFiles(fs, filename, octree, recordedRotation);
```

Leave the `.vox` path (:620-649) untouched: it is written in the aligned frame deliberately.

8. `fmtCount` is already imported in this file (used at :662). Confirm before relying on it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `node --import tsx --test test/write-voxel.test.mjs`

Expected: PASS, including the pre-existing `collisionMesh` validation cases.

- [ ] **Step 5: Verify no unrotated seed remains**

Run: `rg -n 'navSeed!' src/lib/writers/write-voxel.ts`

Expected: no matches. Any remaining `navSeed!` inside the `try` block is the silent-carve bug this feature is most likely to ship.

- [ ] **Step 6: Run the full suite and build**

Run: `npm test && npm run build`

Expected: all tests pass; all four Rollup targets build.

- [ ] **Step 7: Commit**

```bash
npm run lint
git add src/lib/writers/write-voxel.ts test/write-voxel.test.mjs
git commit -m "feat: add autoRotate to the voxel writer"
```

---

### Task 8: Export the estimator as library API

**Files:**
- Modify: `src/lib/voxel/index.ts`
- Modify: `src/lib/index.ts` (voxel value exports near :94, type exports near :55)

**Interfaces:**
- Consumes: `estimateAlignYaw`, `applyAlignYaw` and their types from Tasks 1 and 4.
- Produces: the same names on the package root.

- [ ] **Step 1: Add the exports**

In `src/lib/voxel/index.ts`, matching the existing style:

```ts
export { estimateAlignYaw, applyAlignYaw } from './align-yaw';
export type { AlignYawOptions, AlignYawResult, AlignYawApplied, UpAxis } from './align-yaw';
```

In `src/lib/index.ts`, add `estimateAlignYaw` and `applyAlignYaw` to the voxel export block that already contains `alignGridBounds` and `filterCluster` (:94), and add the four types to the type export block near :55.

- [ ] **Step 2: Verify the public surface resolves**

```bash
cat > ./export-check.tmp.mjs <<'EOF'
import { estimateAlignYaw, applyAlignYaw } from './src/lib/index.js';
console.log(typeof estimateAlignYaw, typeof applyAlignYaw);
EOF
npx tsx ./export-check.tmp.mjs
rm -f ./export-check.tmp.mjs
```

Expected: `function function`

- [ ] **Step 3: Verify docs, build and tests**

Run: `npm run docs && npm run build && npm test`

Expected: no new Typedoc warnings about undocumented or unresolved symbols; build and tests pass.

- [ ] **Step 4: Commit**

```bash
npm run lint
git add src/lib/voxel/index.ts src/lib/index.ts
git commit -m "feat: export the yaw alignment API from the library"
```

---

### Task 9: `--auto-rotate` on the CLI

**Files:**
- Modify: `src/lib/types.ts` (`LibOptions`, after `collisionVoxelsSize` at :99)
- Modify: `src/lib/write.ts` (`case 'voxel'` at :173-193)
- Modify: `src/cli/index.ts` (options :142-170, optional-value map :202-213, parsing near :430-441, options object :595-615, warning after :1136, usage text near :910)
- Modify: `test/cli.test.mjs`

**Interfaces:**
- Consumes: `WriteVoxelOptions.autoRotate` from Task 7.
- Produces: CLI flag `--auto-rotate[=<degrees>]`.

The option travels `CliOptions` -> `LibOptions` -> `write.ts` -> `writeVoxel`. Miss the middle two and the flag parses but never reaches the writer.

- [ ] **Step 1: Write the failing tests**

`test/cli.test.mjs` spawns the CLI through a `runCli(args)` helper (defined at :23-55) that returns `{ code, stdout, stderr }`, and its cases use `--gpu cpu`, the fixture `test/fixtures/splat/minimal.splat`, and `null` as the output. Follow that exactly. Add inside `describe('CLI parsing', ...)`:

```javascript
    it('accepts a bare --auto-rotate without swallowing the output argument', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--auto-rotate',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('accepts an explicit --auto-rotate angle', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--auto-rotate',
            '12.5',
            'null'
        ]);

        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('rejects a non-numeric --auto-rotate value', async () => {
        const result = await runCli([
            '--gpu',
            'cpu',
            'test/fixtures/splat/minimal.splat',
            '--auto-rotate=banana',
            'null'
        ]);

        assert.notStrictEqual(result.code, 0, 'CLI should reject a non-numeric angle');
    });
```

The first two cases are the ones that matter: they prove the `optionalValueOptions` registration works, so a bare flag does not consume the output path and an explicit angle does not become a positional input.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `node --import tsx --test test/cli.test.mjs`

Expected: FAIL — `parseArgs` runs in strict mode, so `--auto-rotate` is an unknown option and the first two cases exit non-zero.

- [ ] **Step 3: Add the option to the library types and dispatcher**

In `src/lib/types.ts`, after `collisionVoxelsSize` (:99):

```ts
    /** Rotate the voxel grid to line up with the scene's dominant surfaces, cutting staircase voxels. `true` estimates the yaw; a number applies that yaw in degrees. Recorded in the `.voxel.json` metadata and the `.collision.glb` node so both still match the unrotated splat; the `.vox` is written aligned. */
    autoRotate?: boolean | number;
```

In `src/lib/write.ts`, inside `case 'voxel'` after `collisionVoxelsSize` (:191):

```ts
                autoRotate: options.autoRotate,
```

- [ ] **Step 4: Add the CLI option**

In `src/cli/index.ts`:

1. In `cliOptionsConfig`'s global block, beside `'collision-voxels-size'` (:154):

```ts
    'auto-rotate': { type: 'string' },
```

2. In `optionalValueOptions` (:202-213), so a bare flag works:

```ts
    ['--auto-rotate', isNumericValue],
```

3. Parse it where `collisionVoxelsSize` is parsed (:435-441):

```ts
    let autoRotate: boolean | number = false;
    if (v['auto-rotate'] !== undefined) {
        autoRotate = v['auto-rotate'] === '' ? true : parseNumber(v['auto-rotate']);
    }
```

4. Add `autoRotate,` to the returned options object beside `collisionVoxelsSize` (:607).

5. Warn when it cannot apply. `outputFormat` is only known in `main`, so put this immediately after the assignment at :1136:

```ts
        if (options.autoRotate !== false && outputFormat !== 'voxel') {
            logger.warn('--auto-rotate has no effect without a .voxel.json output.');
        }
```

6. Add to the usage text beside `--collision-voxels-size` (:910), matching its column alignment exactly:

```
        --auto-rotate      [degrees]        Rotate the voxel grid to line up with the scene's dominant surfaces, cutting
                                            staircase voxels. Bare flag estimates the yaw; a number applies it verbatim.
                                            The .voxel.json and .collision.glb record the rotation so they still match the
                                            unrotated splat; the .vox is written aligned. Default: off
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `node --import tsx --test test/cli.test.mjs`

Expected: PASS, all three new cases plus the pre-existing ones.

- [ ] **Step 6: Verify end to end on a real scene**

```bash
npm run build
node bin/cli.mjs ./scenes/house.ply --auto-rotate --collision-mesh -w ./scenes/tmp-house.voxel.json
```

Expected: an `auto-rotate:` line reporting either a yaw with an estimated improvement or a `no rotation applied` reason. Then confirm the recording matches what was logged:

```bash
node -e "const m=require('./scenes/tmp-house.voxel.json'); console.log(m.version, m.rotation)"
```

Expected: `1.2` plus four numbers when a yaw was applied; `1.1 undefined` when the guard fired. Also confirm the warning path:

```bash
node bin/cli.mjs ./scenes/house.ply --auto-rotate -w ./scenes/tmp-house.sog 2>&1 | rg auto-rotate
```

Expected: `--auto-rotate has no effect without a .voxel.json output.`

Then `rm -f scenes/tmp-house.*`.

- [ ] **Step 7: Commit**

```bash
npm run lint
git add src/lib/types.ts src/lib/write.ts src/cli/index.ts test/cli.test.mjs
git commit -m "feat: add --auto-rotate to the CLI"
```

---

### Task 10: Validation experiment and spec results

**Files:**
- Modify: `specs/2026-08-07-voxel-auto-align-design.md`
- Modify: `src/lib/voxel/align-yaw.ts` (only if the data demands a different `minImprovement`)

**Interfaces:**
- Consumes: the CLI flag from Task 9.
- Produces: a results table in the spec, and either a confirmed 0.02 guard default or a justified change.

The metric is a proxy for occupied voxel count. This task is the evidence that it works. Do not skip it, and do not touch `minImprovement` without measurements in hand.

- [ ] **Step 1: Measure each scene with and without alignment**

For each of `scenes/house.ply`, `scenes/industrial.ply`, `scenes/dungeons-3.ply` and `scenes/landscape.spz`, run the pair below and record: the `auto-rotate` line, grid dimensions, occupied voxel count, `.vox` byte size, `.vox` model count, and wall-clock time.

```bash
node bin/cli.mjs ./scenes/house.ply --collision-voxels ./scenes/tmp-base.vox \
    --voxel-params 0.05,0.1 -w ./scenes/tmp-base.voxel.json

node bin/cli.mjs ./scenes/house.ply --auto-rotate --collision-voxels ./scenes/tmp-aligned.vox \
    --voxel-params 0.05,0.1 -w ./scenes/tmp-aligned.voxel.json
```

`scenes/` is gitignored, so nothing here can be committed by accident.

- [ ] **Step 2: Check the estimate against reality**

On whichever scene showed the largest predicted improvement, sweep explicit angles and confirm the estimator's pick sits at or near the true minimum:

```bash
for deg in -30 -20 -10 0 10 20 30; do
  echo "== $deg"
  node bin/cli.mjs ./scenes/house.ply --auto-rotate=$deg --voxel-params 0.05,0.1 \
      -w ./scenes/tmp-sweep.voxel.json 2>&1 | rg "vox:|voxels|octree depth"
done
```

Expected: occupied voxel count bottoms out near the estimator's yaw. If the true minimum is off by more than a few degrees, **stop and report** — the metric or the sign convention is wrong, and no threshold tuning fixes that.

- [ ] **Step 3: Record the results in the spec**

Add a "Validation results" subsection under "Validation experiment" in `specs/2026-08-07-voxel-auto-align-design.md`: a table of the Step 1 measurements, the Step 2 sweep, and one or two sentences on the guard. State plainly whether `landscape.spz` tripped the guard as predicted.

- [ ] **Step 4: Adjust the guard only if the data says so**

If a scene with obvious structure was rejected, or an organic scene was given a yaw, change the `minImprovement` default in `src/lib/voxel/align-yaw.ts` and say why in the spec. Otherwise change nothing and record that the default held.

- [ ] **Step 5: Final gates and commit**

```bash
rm -f scenes/tmp-*
npm run lint && npm test && npm run build && npm run publint
git add specs/2026-08-07-voxel-auto-align-design.md src/lib/voxel/align-yaw.ts
git commit -m "docs: record voxel auto-align validation results"
```

---

## Notes for the implementer

- The bug this feature is most likely to ship is an unrotated `navSeed`. It only bites when `--seed-pos` meets `--voxel-carve`/`--voxel-external-fill` with a non-zero yaw, and it fails silently by carving the wrong region. Task 4's tests plus Task 7 steps 6 and 5 are the whole defence.
- Do not add auto-alignment to `processDataTable` or any splat writer. Splat outputs must stay byte-identical; the feature lives in the voxel writer.
- If a test passes before you write its implementation, stop and find out why. Usually the assertion is too weak to detect what it claims to check.
- The `.vox` is deliberately the one output with no rotation recorded. If you find yourself adding a rotation to it, re-read the design spec's non-goals.
