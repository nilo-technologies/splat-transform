# Voxel Cleanup Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the cleanup primitives into `writeVoxel` and the CLI behind one `--voxel-cleanup <metres>` dial, off by default, so a scatter-like voxel grid can be turned into a coherent, flat surface without fabricating structure.

**Architecture:** A new `cleanupGrid` orchestrator owns the dial mapping and stage sequencing, keeping `writeVoxel` free of cleanup arithmetic. The candidate mask comes from a second `voxelizeToBuffer` pass at a much lower opacity cutoff, issued before the GPU voxelizer is destroyed; every additive stage intersects with it, which is the anti-fabrication guarantee. The dial is one number that derives the closing/growing radius from `voxelResolution`; the remaining stage constants are fixed because they were measured, not tuned per scene.

**Tech Stack:** TypeScript (ES2022, ESM), Node's built-in test runner (`node:test` + `node:assert`), tsx, ESLint (`@playcanvas/eslint-config`), `parseArgs` from `node:util` for the CLI.

**Design spec:** `specs/2026-08-10-voxel-cleanup-design.md`
**Prerequisite plans:** `specs/2026-08-10-voxel-cleanup-prereq-fixes-plan.md` (landed), `specs/2026-08-10-voxel-cleanup-primitives-plan.md` (must land first)

## Global Constraints

- `src/lib/` must stay platform-agnostic: no `node:*` imports, no Node-only APIs.
- `src/cli/` is the only place Node APIs and CLI argument handling belong.
- Naming: functions camelCase, types PascalCase, constants UPPER_SNAKE_CASE. Arrow-function `const`s are the prevailing style in `src/lib/`.
- Import order: Node built-ins, then external packages (`playcanvas`), then internal relative paths.
- Use `const`/`let`, never `var`.
- All new exported API needs JSDoc with `@param`/`@returns` descriptions and **no type annotations in the JSDoc** — TypeScript provides them. `@throws` where it throws.
- **Off by default is a hard requirement.** With `voxelCleanup` absent or 0, every byte of every output must be identical to before this plan. A test pins this.
- A new option must be threaded through **four** places, all of which forward explicitly with no spread: `Options` (`src/lib/types.ts`), the CLI object literal (`src/cli/index.ts:583-633`), `writeFile`'s dispatch (`src/lib/write.ts:173-195`), and `writeVoxel`'s destructure-with-defaults (`src/lib/writers/write-voxel.ts:386-406`).
- CLI options usable as a bare flag must also be registered in `optionalValueOptions` (`src/cli/index.ts:203-215`) or `normalizeArgv` will swallow the next argv token — typically the output filename.
- Log style: group names are Title-case imperative noun phrases (`'Cleanup'`); `info`/`debug` messages are lowercase `key: value`; counts go through `fmtCount`; groups are `const sub = logger.group(name); … sub.end();` with no `try/finally`.
- Run `npm run lint` and `npm test` before each commit. No new lint errors, no regressions.
- Commit messages follow conventional commits.

## Measured targets

From the design spec, on `scenes/urban.spz` at `--voxel-params 0.1,0.1`. These are the acceptance
criteria for Task 6, not aspirations:

| | occupied | fabricated | faces/vox | components | largest | roughness |
| --- | --- | --- | --- | --- | --- | --- |
| today | 288,184 | 0 | 2.61 | 13,602 | 60.4% | 9.80 |
| `--voxel-cleanup 0.2` | ~202,947 | **0** | ~1.21 | **~64** | ~77.6% | **~3.33** |

## File Structure

| File | Responsibility |
| --- | --- |
| `src/lib/voxel/cleanup.ts` | **New.** `CANDIDATE_CUTOFF`, the dial mapping, and `cleanupGrid` — sequences grow → majority → despeckle and returns per-stage counts. The only place that knows the dial's ratios. |
| `src/lib/voxel/index.ts` | Re-export `cleanupGrid`, its types and `CANDIDATE_CUTOFF`. |
| `src/lib/index.ts` | Add the same to the public API's explicit named list at lines 93-98. |
| `src/lib/types.ts` | Add `voxelCleanup` and `voxelCleanupFill` to `Options`. |
| `src/lib/write.ts` | Forward both to `writeVoxel`. |
| `src/lib/writers/write-voxel.ts` | Add both to `WriteVoxelOptions`, validate them, run the candidate pass, call `cleanupGrid`, log the scatter metric. |
| `src/cli/index.ts` | Register, parse, validate and forward the two flags; help text. |
| `README.md` | Document the dial, the guarantee, and when to reach for it. |
| `test/voxel-cleanup.test.mjs` | **New.** Orchestrator and dial-mapping tests. |
| `test/write-voxel.test.mjs` | Extend: off-by-default byte-identity, and cleanup end-to-end. |
| `test/cli.test.mjs` | Extend: flag parsing and rejection cases. |

`cleanupGrid` is its own module rather than inline in `writeVoxel` because the dial mapping is
policy worth testing directly, and because the `close` plan adds a fourth stage to the same
sequence without touching the writer.

---

## Task 1: `cleanupGrid` orchestrator and dial mapping

Owns `CANDIDATE_CUTOFF`, the dial-to-stage-parameter mapping, and the stage sequence. Takes the
already-built solid and candidate grids; knows nothing about voxelization or the GPU.

**Files:**
- Create: `src/lib/voxel/cleanup.ts`
- Modify: `src/lib/voxel/index.ts`, `src/lib/index.ts`
- Test: `test/voxel-cleanup.test.mjs`

**Interfaces:**
- Consumes: `growGrid`/`GrowResult`, `majorityFilterGrid`, `despeckleGrid` from the primitives plan; `SparseVoxelGrid`.
- Produces:
  - `CANDIDATE_CUTOFF = 0.002`.
  - `type CleanupFillMode = 'none' | 'grow' | 'close' | 'both'`.
  - `type CleanupOptions = { strength: number; voxelResolution: number; fill?: CleanupFillMode }`.
  - `type CleanupStats = { radius: number; grown: number; majorityAdded: number; majorityRemoved: number; despeckled: number; components: number; componentsRemoved: number }`.
  - `type CleanupResult = { grid: SparseVoxelGrid; stats: CleanupStats }`.
  - `cleanupGrid(grid, candidate, options): CleanupResult`. Consumes `grid`, reads `candidate`.
  - `cleanupRadius(strength: number, voxelResolution: number): number` — exported for the CLI's validation messages and for tests.

`close` and `both` are accepted by the type but throw `Error('cleanup fill mode "close" is not implemented yet')` in this plan; the `close` plan replaces the throw. This keeps the enum stable across plans so the CLI does not change twice.

- [ ] **Step 1: Write the failing tests**

Create `test/voxel-cleanup.test.mjs`:

```javascript
/**
 * Tests for the voxel cleanup orchestrator and its dial mapping.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { CANDIDATE_CUTOFF, cleanupGrid, cleanupRadius } from '../src/lib/voxel/cleanup.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

const allCandidate = (n) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = 0; z < n; z++) {
        for (let y = 0; y < n; y++) {
            for (let x = 0; x < n; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

const countVoxels = (g) => {
    let n = 0;
    g.forEachOccupiedVoxel(() => n++);
    return n;
};

// A one-voxel-thick sheet at y === 8 with the listed (x,z) holes.
const sheet = (n, holes) => {
    const g = new SparseVoxelGrid(n, n, n);
    const isHole = new Set(holes.map(([x, z]) => `${x},${z}`));
    for (let z = 2; z < n - 2; z++) {
        for (let x = 2; x < n - 2; x++) {
            if (!isHole.has(`${x},${z}`)) g.setVoxel(x, 8, z);
        }
    }
    return g;
};

describe('cleanupRadius', function () {
    it('maps strength to whole voxels', function () {
        assert.strictEqual(cleanupRadius(0.2, 0.1), 2);
        assert.strictEqual(cleanupRadius(0.5, 0.1), 5);
        assert.strictEqual(cleanupRadius(0.02, 0.01), 2);
    });

    it('rounds to the nearest voxel', function () {
        assert.strictEqual(cleanupRadius(0.24, 0.1), 2);
        assert.strictEqual(cleanupRadius(0.26, 0.1), 3);
    });

    it('clamps a sub-voxel strength up to 1', function () {
        assert.strictEqual(cleanupRadius(0.04, 0.1), 1);
        assert.strictEqual(cleanupRadius(0.0001, 0.1), 1);
    });
});

describe('CANDIDATE_CUTOFF', function () {
    it('is far below the default opacity cutoff', function () {
        assert.strictEqual(CANDIDATE_CUTOFF, 0.002);
        assert.ok(CANDIDATE_CUTOFF < 0.1, 'must admit voxels the solid pass rejects');
    });
});

describe('cleanupGrid', function () {
    it('fills a 1x1 hole and reports it', function () {
        const res = cleanupGrid(sheet(24, [[10, 10]]), allCandidate(24),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(res.grid.getVoxel(10, 8, 10), 1);
        assert.strictEqual(res.stats.grown, 1);
        assert.strictEqual(res.stats.radius, 2);
    });

    it('removes an isolated speck', function () {
        const g = sheet(24, []);
        g.setVoxel(2, 20, 2);
        const res = cleanupGrid(g, allCandidate(24),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(res.grid.getVoxel(2, 20, 2), 0);
        assert.ok(res.stats.majorityRemoved + res.stats.despeckled >= 1);
    });

    it('never adds a voxel outside the candidate mask', function () {
        const empty = new SparseVoxelGrid(24, 24, 24);
        const res = cleanupGrid(sheet(24, [[10, 10]]), empty,
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(res.grid.getVoxel(10, 8, 10), 0, 'the gate must hold');
        assert.strictEqual(res.stats.grown, 0);
        assert.strictEqual(res.stats.majorityAdded, 0);
    });

    it('reports what the gate blocked', function () {
        const empty = new SparseVoxelGrid(24, 24, 24);
        const res = cleanupGrid(sheet(24, [[10, 10]]), empty,
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.ok(res.stats.gateRejected >= 1,
            'the blocked hole is the audit trail and must be reported');
    });

    it('runs majority and despeckle with fill mode none', function () {
        const g = sheet(24, []);
        g.setVoxel(2, 20, 2);
        const res = cleanupGrid(g, allCandidate(24),
            { strength: 0.2, voxelResolution: 0.1, fill: 'none' });
        assert.strictEqual(res.stats.grown, 0, 'no fill stage ran');
        assert.strictEqual(res.grid.getVoxel(2, 20, 2), 0, 'but the speck still goes');
    });

    it('defaults the fill mode to grow', function () {
        const res = cleanupGrid(sheet(24, [[10, 10]]), allCandidate(24),
            { strength: 0.2, voxelResolution: 0.1 });
        assert.strictEqual(res.stats.grown, 1);
    });

    it('reduces the component count on a speckled sheet', function () {
        const g = sheet(24, []);
        // scatter 20 isolated specks well away from the sheet
        for (let i = 0; i < 20; i++) g.setVoxel(2 + (i % 18), 18, 2 + ((i * 7) % 18));
        const before = countVoxels(g);
        const res = cleanupGrid(g, allCandidate(24),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.ok(countVoxels(res.grid) < before, 'specks should be gone');
        assert.ok(res.stats.componentsRemoved >= 1);
    });

    it('rejects a non-positive strength', function () {
        assert.throws(
            () => cleanupGrid(sheet(24, []), allCandidate(24),
                { strength: 0, voxelResolution: 0.1 }),
            /strength must be > 0/
        );
    });

    it('rejects a non-positive voxel resolution', function () {
        assert.throws(
            () => cleanupGrid(sheet(24, []), allCandidate(24),
                { strength: 0.2, voxelResolution: 0 }),
            /voxelResolution must be > 0/
        );
    });

    it('throws a clear not-implemented error for close', function () {
        assert.throws(
            () => cleanupGrid(sheet(24, []), allCandidate(24),
                { strength: 0.2, voxelResolution: 0.1, fill: 'close' }),
            /not implemented yet/
        );
    });

    it('throws a clear not-implemented error for both', function () {
        assert.throws(
            () => cleanupGrid(sheet(24, []), allCandidate(24),
                { strength: 0.2, voxelResolution: 0.1, fill: 'both' }),
            /not implemented yet/
        );
    });

    it('leaves the candidate grid untouched', function () {
        const cand = allCandidate(24);
        const before = [...cand.types];
        cleanupGrid(sheet(24, [[10, 10]]), cand,
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.deepStrictEqual([...cand.types], before);
    });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/voxel-cleanup.test.mjs
```

Expected: FAIL — cannot resolve `../src/lib/voxel/cleanup.js`.

- [ ] **Step 3: Implement `cleanup.ts`**

Create `src/lib/voxel/cleanup.ts`:

```ts
import { despeckleGrid } from './despeckle';
import { growGrid } from './grow';
import { majorityFilterGrid } from './majority';
import { SparseVoxelGrid } from './sparse-voxel-grid';

/**
 * Opacity threshold for the candidate mask.
 *
 * Far below any sensible solid cutoff, so the candidate set covers everywhere
 * the gaussian field has measurable presence. Every stage that adds voxels
 * intersects with it, which is what stops cleanup from inventing structure in
 * empty space: a real void has no density and is therefore untouchable at any
 * radius.
 */
const CANDIDATE_CUTOFF = 0.002;

/** Occupied voxels required among the 27, self included, for the majority filter. */
const MAJORITY_THRESHOLD = 14;

/** Majority filter passes. */
const MAJORITY_ITERATIONS = 2;

/** Components below this many voxels are removed. 64 is one 4x4x4 block. */
const DESPECKLE_MIN_VOXELS = 64;

/** Face neighbours a candidate voxel needs before `grow` fills it. */
const GROW_MIN_NEIGHBORS = 3;

/**
 * Hole-filling algorithm.
 *
 * - `grow` — fill candidate voxels with enough occupied face neighbours.
 * - `close` — morphological closing intersected with the candidate mask.
 * - `both` — `grow`, then `close` on its result.
 * - `none` — skip hole filling; run only the majority filter and despeckle.
 */
type CleanupFillMode = 'none' | 'grow' | 'close' | 'both';

/**
 * Options for {@link cleanupGrid}.
 */
type CleanupOptions = {
    /** Scale of defect to remove, in world units. Must be > 0. */
    strength: number;
    /** Voxel size in world units, used to convert `strength` to voxels. */
    voxelResolution: number;
    /** Hole-filling algorithm. Default: `'grow'` */
    fill?: CleanupFillMode;
};

/**
 * Per-stage counts from {@link cleanupGrid}.
 */
type CleanupStats = {
    /** Defect scale in voxels, derived from the strength dial. */
    radius: number;
    /** Voxels added by the fill stage. */
    grown: number;
    /**
     * Voxels that an ungated pass would have added but the candidate mask
     * blocked. The audit trail for the anti-fabrication guarantee: a large
     * number here means the gate is doing real work.
     */
    gateRejected: number;
    /** Voxels added by the majority filter. */
    majorityAdded: number;
    /** Voxels removed by the majority filter. */
    majorityRemoved: number;
    /** Voxels removed by despeckling. */
    despeckled: number;
    /** Connected components found while despeckling. */
    components: number;
    /** Components removed for being under the size threshold. */
    componentsRemoved: number;
};

/**
 * Result of {@link cleanupGrid}.
 */
type CleanupResult = {
    /** The cleaned grid. */
    grid: SparseVoxelGrid;
    /** What each stage did. */
    stats: CleanupStats;
};

/**
 * Convert the cleanup strength dial into a defect radius in whole voxels.
 *
 * Always at least 1: a strength below one voxel still means "clean up at the
 * finest scale available" rather than "do nothing".
 *
 * @param strength - Defect scale in world units.
 * @param voxelResolution - Voxel size in world units.
 * @returns Radius in voxels, at least 1.
 */
const cleanupRadius = (strength: number, voxelResolution: number): number => {
    return Math.max(1, Math.round(strength / voxelResolution));
};

/**
 * Clean up a voxel grid: fill sampling holes, regularize the surface, drop
 * floating debris.
 *
 * Every stage that adds voxels intersects with `candidate`, so cleanup can only
 * place a voxel where the source gaussians have measurable density. Stages that
 * remove voxels are ungated, since removal cannot fabricate structure.
 *
 * Runs fill (per `options.fill`), then the majority filter, then despeckling.
 * Despeckling is last because the majority filter both creates and destroys
 * islands.
 *
 * @param grid - Grid to clean. **Consumed**: do not reuse it after the call.
 * @param candidate - Voxels permitted to be added. Not modified.
 * @param options - Strength, voxel size and fill mode.
 * @returns The cleaned grid and per-stage counts.
 * @throws If `strength` or `voxelResolution` is not positive, or if the fill
 * mode is not implemented.
 */
const cleanupGrid = (
    grid: SparseVoxelGrid,
    candidate: SparseVoxelGrid,
    options: CleanupOptions
): CleanupResult => {
    const { strength, voxelResolution, fill = 'grow' } = options;

    if (!(strength > 0)) {
        throw new Error(`cleanupGrid: strength must be > 0, got ${strength}`);
    }
    if (!(voxelResolution > 0)) {
        throw new Error(`cleanupGrid: voxelResolution must be > 0, got ${voxelResolution}`);
    }
    if (fill === 'close' || fill === 'both') {
        throw new Error(
            `cleanup fill mode "${fill}" is not implemented yet; use "grow" or "none"`);
    }

    const radius = cleanupRadius(strength, voxelResolution);

    let current = grid;
    let grown = 0;
    let gateRejected = 0;

    if (fill === 'grow') {
        const res = growGrid(current, candidate, {
            minNeighbors: GROW_MIN_NEIGHBORS,
            // A gap of width w needs about w/2 passes to close from both ends,
            // and the dial's radius is half the widest gap we mean to fill.
            maxIterations: 2 * radius + 2
        });
        current = res.grid;
        grown = res.added;
        gateRejected += res.gateRejected;
    }

    const maj = majorityFilterGrid(current, candidate, {
        threshold: MAJORITY_THRESHOLD,
        iterations: MAJORITY_ITERATIONS
    });
    current = maj.grid;
    gateRejected += maj.gateRejected;

    const desp = despeckleGrid(current, { minVoxels: DESPECKLE_MIN_VOXELS });
    current = desp.grid;

    return {
        grid: current,
        stats: {
            radius,
            grown,
            gateRejected,
            majorityAdded: maj.added,
            majorityRemoved: maj.removed,
            despeckled: desp.removed,
            components: desp.components,
            componentsRemoved: desp.componentsRemoved
        }
    };
};

export {
    CANDIDATE_CUTOFF,
    cleanupGrid,
    cleanupRadius,
    type CleanupFillMode,
    type CleanupOptions,
    type CleanupResult,
    type CleanupStats
};
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
npx tsx --test test/voxel-cleanup.test.mjs
```

Expected: PASS, 16 tests.

- [ ] **Step 5: Export from both barrels**

Add to `src/lib/voxel/index.ts`:

```ts
export { CANDIDATE_CUTOFF, cleanupGrid, cleanupRadius } from './cleanup';
export type { CleanupFillMode, CleanupOptions, CleanupResult, CleanupStats } from './cleanup';
```

Then add `CANDIDATE_CUTOFF`, `cleanupGrid` and `cleanupRadius` to the value list, and the four
types to the type list, of the `./voxel` re-export in `src/lib/index.ts:93-98`. Verify:

```bash
npx tsx -e "import('./src/lib/index.ts').then(m => { if (typeof m.cleanupGrid !== 'function') throw new Error('cleanupGrid not exported'); if (m.CANDIDATE_CUTOFF !== 0.002) throw new Error('CANDIDATE_CUTOFF wrong'); console.log('public barrel OK'); })"
```

Expected: prints `public barrel OK`.

- [ ] **Step 6: Full suite and lint**

```bash
npm test && npm run lint
```

- [ ] **Step 7: Commit**

```bash
git add src/lib/voxel/cleanup.ts src/lib/voxel/index.ts src/lib/index.ts test/voxel-cleanup.test.mjs
git commit -m "feat: add cleanupGrid orchestrator and the cleanup strength dial

Owns CANDIDATE_CUTOFF, the strength-to-radius mapping and the stage
sequence: fill, then majority filter, then despeckle. Despeckle runs last
because the majority filter both creates and destroys islands.

The majority, despeckle and grow constants do not scale with the dial --
the majority neighbourhood is structurally 3x3x3, 64 voxels is one block,
and the scatter these target was measured to be scale-invariant. Only the
fill radius comes off the dial.

The close and both fill modes throw a not-implemented error so the enum is
stable before the GPU erode work lands."
```

---

## Task 2: Thread the options through the four layers

Add `voxelCleanup` and `voxelCleanupFill` to `Options`, `write.ts`, and `WriteVoxelOptions`,
with validation in `writeVoxel`. No behaviour yet — this task only makes the options reachable,
which keeps the diff that changes output behaviour (Task 3) small and reviewable.

**Files:**
- Modify: `src/lib/types.ts`, `src/lib/write.ts`, `src/lib/writers/write-voxel.ts`
- Test: `test/write-voxel.test.mjs`

**Interfaces:**
- Consumes: `CleanupFillMode` from Task 1.
- Produces: `Options.voxelCleanup?: number`, `Options.voxelCleanupFill?: CleanupFillMode`, and the same two on `WriteVoxelOptions`. Task 3 reads them; Task 4 sets them from the CLI.

- [ ] **Step 1: Write the failing tests**

Append to `test/write-voxel.test.mjs`. Read the file's existing helpers first — it already has a
harness for calling `writeVoxel` against a `MemoryFileSystem`; reuse it rather than building a
new one. The two tests are:

```javascript
describe('writeVoxel cleanup option validation', function () {
    it('rejects a negative voxelCleanup', async function () {
        await assert.rejects(
            () => runWriteVoxel({ voxelCleanup: -0.1 }),
            /voxelCleanup must be >= 0/
        );
    });

    it('rejects an unknown voxelCleanupFill', async function () {
        await assert.rejects(
            () => runWriteVoxel({ voxelCleanup: 0.2, voxelCleanupFill: 'sideways' }),
            /Invalid voxelCleanupFill/
        );
    });

    it('rejects voxelCleanupFill without voxelCleanup', async function () {
        await assert.rejects(
            () => runWriteVoxel({ voxelCleanupFill: 'grow' }),
            /voxelCleanupFill requires voxelCleanup/
        );
    });

    it('accepts voxelCleanup 0 as explicitly disabled', async function () {
        await assert.doesNotReject(() => runWriteVoxel({ voxelCleanup: 0 }));
    });
});
```

Replace `runWriteVoxel({...})` with whatever the file's existing harness is called, merging the
given fields into its default option set. If no such harness exists, add one modelled on the
file's existing tests and note it in the report.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/write-voxel.test.mjs
```

Expected: FAIL — the options are ignored, so nothing rejects.

- [ ] **Step 3: Add the fields to `Options`**

In `src/lib/types.ts`, after the `autoRotate` field (line 102), matching the single-line JSDoc
style and the `Default: X` ending used throughout:

```ts
    /** Clean up the voxel grid: fill sampling holes, flatten bumpy surfaces and drop floating debris, at this scale in world units. Every added voxel must have gaussian density behind it, so this cannot invent structure in genuinely empty space. 0 or undefined disables it. Default: off */
    voxelCleanup?: number;

    /** Hole-filling algorithm for `voxelCleanup`. `none` runs only the smoothing and debris passes. Requires `voxelCleanup`. Default: `'grow'` */
    voxelCleanupFill?: CleanupFillMode;
```

Add `CleanupFillMode` to the type imports at the top of `src/lib/types.ts`, importing from
`./voxel` alongside whatever it already pulls from there. If it imports nothing from `./voxel`
yet, add `import type { CleanupFillMode } from './voxel';` in the internal-imports group.

- [ ] **Step 4: Forward through `write.ts`**

In `src/lib/write.ts`, in the `case 'voxel':` block (lines 173-195), add two lines after
`autoRotate: options.autoRotate,`:

```ts
                voxelCleanup: options.voxelCleanup,
                voxelCleanupFill: options.voxelCleanupFill,
```

- [ ] **Step 5: Add to `WriteVoxelOptions` and validate**

In `src/lib/writers/write-voxel.ts`, add to the `WriteVoxelOptions` type (which ends around line
118), mirroring the `Options` JSDoc:

```ts
    /** Clean up the voxel grid: fill sampling holes, flatten bumpy surfaces and drop floating debris, at this scale in world units. Every added voxel must have gaussian density behind it. 0 or undefined disables it. Default: off */
    voxelCleanup?: number;

    /** Hole-filling algorithm for `voxelCleanup`. Requires `voxelCleanup`. Default: `'grow'` */
    voxelCleanupFill?: CleanupFillMode;
```

Add both to the destructure at lines 386-406, after `autoRotate = false`:

```ts
        voxelCleanup,
        voxelCleanupFill
```

Then add validation next to the existing option checks (near lines 426-453):

```ts
    if (voxelCleanup !== undefined && !(voxelCleanup >= 0)) {
        throw new Error(`voxelCleanup must be >= 0, got ${voxelCleanup}`);
    }

    const cleanupEnabled = voxelCleanup !== undefined && voxelCleanup > 0;

    if (voxelCleanupFill !== undefined) {
        if (!cleanupEnabled) {
            throw new Error(
                'voxelCleanupFill requires voxelCleanup to be set and greater than 0');
        }
        if (voxelCleanupFill !== 'none' && voxelCleanupFill !== 'grow' &&
            voxelCleanupFill !== 'close' && voxelCleanupFill !== 'both') {
            throw new Error(
                `Invalid voxelCleanupFill: ${voxelCleanupFill}. Expected none, grow, close or both.`);
        }
    }
```

Import `CleanupFillMode` and `cleanupGrid`, `CANDIDATE_CUTOFF` from `../voxel` — add them to the
existing brace import from that module. `cleanupGrid` and `CANDIDATE_CUTOFF` are unused until
Task 3; if lint objects to unused imports, add them in Task 3 instead and note it.

- [ ] **Step 6: Run the tests to verify they pass**

```bash
npx tsx --test test/write-voxel.test.mjs
```

Expected: PASS, including the 4 new cases.

- [ ] **Step 7: Full suite and lint**

```bash
npm test && npm run lint
```

- [ ] **Step 8: Commit**

```bash
git add src/lib/types.ts src/lib/write.ts src/lib/writers/write-voxel.ts test/write-voxel.test.mjs
git commit -m "feat: thread voxelCleanup options through the library layers

Adds voxelCleanup and voxelCleanupFill to Options and WriteVoxelOptions,
forwards them through writeFile's voxel dispatch, and validates them in
writeVoxel. No behaviour change yet -- this keeps the diff that alters
output small and reviewable."
```

---

## Task 3: Candidate mask and the cleanup phase in `writeVoxel`

The behaviour change. Voxelize a second time at `CANDIDATE_CUTOFF` while the GPU voxelizer is
still alive, convert it to a grid without block cleanup, then run `cleanupGrid` between the grid
load and the fill/carve stages so the octree, `.vox` and collision GLB all inherit the result.

**Files:**
- Modify: `src/lib/writers/write-voxel.ts`
- Test: `test/write-voxel.test.mjs`

**Interfaces:**
- Consumes: `cleanupGrid`, `CANDIDATE_CUTOFF` (Task 1); `voxelCleanup`, `voxelCleanupFill`, `cleanupEnabled` (Task 2).
- Produces: no new exports. Task 4 drives it from the CLI.

- [ ] **Step 1: Write the failing tests**

Append to `test/write-voxel.test.mjs`:

```javascript
describe('writeVoxel cleanup behaviour', function () {
    it('produces byte-identical output when cleanup is absent', async function () {
        const a = await runWriteVoxelBytes({});
        const b = await runWriteVoxelBytes({ voxelCleanup: 0 });
        assert.deepStrictEqual(b, a, 'voxelCleanup 0 must match the absent case');
    });

    it('changes the grid when cleanup is enabled', async function () {
        const plain = await runWriteVoxelMeta({});
        const cleaned = await runWriteVoxelMeta({ voxelCleanup: 0.2 });
        assert.notDeepStrictEqual(cleaned, plain,
            'cleanup should alter the octree metadata');
    });

    it('rejects an unimplemented fill mode at the writer level', async function () {
        await assert.rejects(
            () => runWriteVoxel({ voxelCleanup: 0.2, voxelCleanupFill: 'close' }),
            /not implemented yet/
        );
    });
});
```

`runWriteVoxelBytes` must return the emitted `.voxel.bin` bytes from the `MemoryFileSystem`, and
`runWriteVoxelMeta` the parsed `.voxel.json`. Build both on the file's existing harness; if it
does not already expose the written files, extend it and say so in the report.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/write-voxel.test.mjs
```

Expected: the byte-identity test passes trivially (cleanup does nothing yet) and the other two
FAIL. That is the correct starting state.

- [ ] **Step 3: Voxelize the candidate mask**

In `src/lib/writers/write-voxel.ts`, the solid voxelization currently reads (lines 573-584):

```ts
        const buffer = await voxelizeToBuffer(
            bvh, gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
        );
        if (!needsSplatColors) {
            bvh = null;
            pcDataTable = null;
        }
        extentsResult = null;
        cols = null;

        gpuVoxelization.destroy();
        gpuVoxelization = null;
```

Insert the candidate pass before the teardown, so it reuses the uploaded gaussians:

```ts
        const buffer = await voxelizeToBuffer(
            bvh, gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
        );

        // Candidate mask: the same field at a much lower cutoff, marking
        // everywhere the gaussians have measurable presence. Cleanup may only
        // add voxels inside it, so it cannot invent structure in empty space.
        // A second pass rather than a second mask out of one dispatch: a full
        // pass measures ~250ms on a 24x21x38m scene at 0.1m, which is not worth
        // reworking the voxelization shader for.
        let candidateBuffer: BlockMaskBuffer | null = null;
        if (cleanupEnabled) {
            const candSub = logger.group('Candidate mask');
            candidateBuffer = await voxelizeToBuffer(
                bvh, gpuVoxelization, gridBounds, voxelResolution, CANDIDATE_CUTOFF
            );
            candSub.end();
        }

        if (!needsSplatColors) {
            bvh = null;
            pcDataTable = null;
        }
        extentsResult = null;
        cols = null;

        gpuVoxelization.destroy();
        gpuVoxelization = null;
```

`BlockMaskBuffer` is already imported in this file; confirm and add it to the brace import from
`../voxel` if not.

- [ ] **Step 4: Run cleanup after the grid load**

The grid load currently ends at line 607 with `loadSub.end();`, followed at 609-613 by the
`needsGpuDilation` block. Insert the cleanup phase between them:

```ts
        loadSub.end();

        if (cleanupEnabled && candidateBuffer) {
            const cleanSub = logger.group('Cleanup');
            const candidateGrid = SparseVoxelGrid.fromBuffer(
                candidateBuffer, nxInit, nyInit, nzInit
            );
            candidateBuffer.clear();
            candidateBuffer = null;

            const cleaned = cleanupGrid(grid, candidateGrid, {
                strength: voxelCleanup!,
                voxelResolution,
                fill: voxelCleanupFill
            });
            grid = cleaned.grid;
            candidateGrid.releaseStorage();

            const s = cleaned.stats;
            logger.info(
                `cleanup: radius ${s.radius} voxels, +${fmtCount(s.grown)} grown, ` +
                `+${fmtCount(s.majorityAdded)}/-${fmtCount(s.majorityRemoved)} smoothed, ` +
                `-${fmtCount(s.despeckled)} despeckled ` +
                `(${fmtCount(s.componentsRemoved)} of ${fmtCount(s.components)} islands)`);
            logger.info(
                `cleanup gate: ${fmtCount(s.gateRejected)} voxels blocked for having no ` +
                'gaussian density behind them');
            cleanSub.end();
        }

        // Reuse the same device for GPU dilation across exterior, floor, carve.
        const needsGpuDilation = hasFillExterior || hasNav || (hasFloorFill && floorFillDilation > 0);
```

The candidate grid is deliberately built **without** `filterAndFillBlocks`: the candidate set is
evidence, not geometry, and eroding it would narrow the gate.

- [ ] **Step 5: Add the scatter metric**

Still in `write-voxel.ts`, right after the grid load and before the cleanup block, add an
`info` line that measures how scatter-like the grid is and points at the dial when it is bad.
Put the helper next to the other module-level helpers near the top of the file:

```ts
/**
 * Fraction of occupied voxels with at most two of six face neighbours.
 *
 * A coherent surface sits near zero; a sampling scatter runs above 0.3. This is
 * the signal that a scene needs `voxelCleanup`, so it is reported even when
 * cleanup is off.
 *
 * @param grid - Grid to measure.
 * @returns The fraction, or 0 for an empty grid.
 */
const scatterFraction = (grid: SparseVoxelGrid): number => {
    const { nx, ny, nz } = grid;
    let occupied = 0;
    let sparse = 0;
    grid.forEachOccupiedVoxel((x, y, z) => {
        occupied++;
        let n = 0;
        // Every bound is guarded in both directions. getVoxel does no bounds
        // checking and its block index aliases across rows -- on an 8^3 grid
        // getVoxel(8, 0, 0) returns the voxel at (0, 4, 0) -- so an unguarded
        // upper-bound read would silently count a wrapped neighbour.
        if (x + 1 < nx && grid.getVoxel(x + 1, y, z)) n++;
        if (x > 0 && grid.getVoxel(x - 1, y, z)) n++;
        if (y + 1 < ny && grid.getVoxel(x, y + 1, z)) n++;
        if (y > 0 && grid.getVoxel(x, y - 1, z)) n++;
        if (z + 1 < nz && grid.getVoxel(x, y, z + 1)) n++;
        if (z > 0 && grid.getVoxel(x, y, z - 1)) n++;
        if (n <= 2) sparse++;
    });
    return occupied === 0 ? 0 : sparse / occupied;
};
```

and the call site, immediately after `loadSub.end();`:

```ts
        const scatter = scatterFraction(grid);
        logger.info(`surface coherence: ${(scatter * 100).toFixed(0)}% of voxels have <= 2 of 6 neighbours`);
        if (scatter > 0.2 && !cleanupEnabled) {
            logger.warn(
                `this grid is mostly scattered voxels rather than surfaces; ` +
                `--voxel-cleanup ${(voxelResolution * 2).toFixed(3)} would fill the sampling ` +
                'holes and flatten it');
        }
```

**Do not drop the upper-bound guards.** `getVoxel` performs no bounds check: it computes
`(ix >> 2) + (iy >> 2) * nbx + (iz >> 2) * bStride` and reads that block, so an index one past
the end aliases into a legitimate block in the next row. Verified on an 8x8x8 grid with a voxel
set at `(0, 4, 0)`: `getVoxel(8, 0, 0)` returns 1. Guard all six directions.

- [ ] **Step 6: Promote the block-cleanup removal count**

`filterAndFillBlocks` deletes voxels on every run and reports it only at `debug`
(`src/lib/voxel/block-cleanup.ts:189`), so a scene losing 10% of its voxels to it looks silent at
default verbosity. It already computes the counts; return them so the writer can report a large
loss.

Change its return to `{ buffer, voxelsRemoved, voxelsFilled }` and update its single call site in
`write-voxel.ts` (line 590). Keep the existing `logger.debug` line. Then in the writer, after the
call:

```ts
        const removedFraction = filteredBuffer.count > 0 ?
            cleanupCounts.voxelsRemoved / (filteredBuffer.count * 64) :
            0;
        if (removedFraction > 0.05) {
            logger.info(
                `block cleanup removed ${fmtCount(cleanupCounts.voxelsRemoved)} isolated voxels ` +
                `(${(removedFraction * 100).toFixed(0)}% of the grid)`);
        }
```

`filterAndFillBlocks` is exported publicly (`src/lib/index.ts:95`), so this is a breaking change to
a shipped signature. It has one in-repo caller and the change is additive in information, so
returning an object is acceptable — but note it in the commit message so a consumer can see it.

- [ ] **Step 7: Run the tests to verify they pass**

```bash
npx tsx --test test/write-voxel.test.mjs
```

Expected: PASS, all three new cases plus everything already there. The byte-identity test is the
important one — it must still pass.

- [ ] **Step 8: Full suite and lint**

```bash
npm test && npm run lint
```

- [ ] **Step 9: Commit**

```bash
git add src/lib/writers/write-voxel.ts src/lib/voxel/block-cleanup.ts test/write-voxel.test.mjs
git commit -m "feat: run voxel cleanup in writeVoxel behind the candidate mask

Voxelizes a second time at CANDIDATE_CUTOFF while the GPU voxelizer is still
alive, then runs cleanupGrid between the grid load and the fill/carve stages
so the octree, .vox and collision GLB all inherit the cleaned grid. The
candidate grid deliberately skips filterAndFillBlocks: it is evidence, not
geometry, and eroding it would narrow the gate.

Also reports surface coherence at info level and, when a grid is mostly
scatter and cleanup is off, warns with a concrete --voxel-cleanup value; and
reports how many voxels the candidate gate blocked, which is the audit trail
for the anti-fabrication guarantee.

filterAndFillBlocks now returns its removal counts alongside the buffer so a
large silent loss can be reported at info rather than debug. That changes a
publicly exported signature, with one in-repo caller."
```

---

## Task 4: CLI flags

**Files:**
- Modify: `src/cli/index.ts`
- Test: `test/cli.test.mjs`

**Interfaces:**
- Consumes: `Options.voxelCleanup`, `Options.voxelCleanupFill` (Task 2).
- Produces: `--voxel-cleanup <metres>` and `--voxel-cleanup-fill <mode>`.

- [ ] **Step 1: Write the failing tests**

Append to `test/cli.test.mjs`, following the existing `runCli` pattern. Every passing case ends
in `null` as the output sink, as the existing tests do:

```javascript
    it('accepts --voxel-cleanup with a value', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--voxel-cleanup', '0.2',
            'null'
        ]);
        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('accepts a bare --voxel-cleanup without swallowing the output argument', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--voxel-cleanup',
            'null'
        ]);
        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('accepts --voxel-cleanup-fill none', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--voxel-cleanup', '0.2',
            '--voxel-cleanup-fill', 'none',
            'null'
        ]);
        assert.strictEqual(result.code, 0, `CLI failed:\n${result.stderr}\n${result.stdout}`);
    });

    it('rejects an unknown --voxel-cleanup-fill value', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--voxel-cleanup', '0.2',
            '--voxel-cleanup-fill', 'sideways',
            'null'
        ]);
        assert.notStrictEqual(result.code, 0);
        assert.match(result.stderr + result.stdout, /Invalid voxel cleanup fill mode/);
    });

    it('rejects a negative --voxel-cleanup', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--voxel-cleanup', '-1',
            'null'
        ]);
        assert.notStrictEqual(result.code, 0);
    });

    it('errors on --voxel-cleanup-fill without --voxel-cleanup', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--voxel-cleanup-fill', 'grow',
            'null'
        ]);
        assert.notStrictEqual(result.code, 0);
        assert.match(result.stderr + result.stdout, /requires --voxel-cleanup/);
    });

    it('warns that --voxel-cleanup needs a voxel output', async () => {
        const result = await runCli([
            '--gpu', 'cpu',
            'test/fixtures/splat/minimal.splat',
            '--voxel-cleanup', '0.2',
            'out.ply'
        ]);
        assert.match(result.stderr + result.stdout, /--voxel-cleanup has no effect/);
    });
```

The last test writes `out.ply`; check how the existing suite handles output files it does not want
to keep — if it relies on `null`, use a temporary path and delete it, or drop that assertion to a
warning check on a `null` output if the CLI still reaches the warn.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/cli.test.mjs
```

Expected: FAIL — `parseArgs` rejects the unknown options in strict mode.

- [ ] **Step 3: Register the options**

In `src/cli/index.ts`, add to `cliOptionsConfig` after `'voxel-carve'` (line 145):

```ts
    'voxel-cleanup': { type: 'string' },
    'voxel-cleanup-fill': { type: 'string' },
```

`--voxel-cleanup` is usable bare, so add it to `optionalValueOptions` (lines 203-215) with the
numeric predicate. `--voxel-cleanup-fill` always takes a keyword, so it needs its own predicate:

```ts
const isCleanupFillMode = (s: string) => /^(?:none|grow|close|both)$/i.test(s);
```

declared beside `isCollisionMeshShape` (line 197), then both entries:

```ts
    ['--voxel-cleanup', isNumericValue],
    ['--voxel-cleanup-fill', isCleanupFillMode],
```

- [ ] **Step 4: Parse and validate**

In `parseArguments`, after the `--voxel-carve` block (ends line 421), add:

```ts
    const cleanupStr = v['voxel-cleanup'];
    let voxelCleanup: number | undefined;
    if (cleanupStr !== undefined) {
        // Bare flag: two voxels is the scale that closes single-voxel sampling
        // holes without bridging real gaps.
        voxelCleanup = cleanupStr ? parseNumber(cleanupStr, 0) : voxelResolution * 2;
    }

    const cleanupFillStr = v['voxel-cleanup-fill'];
    let voxelCleanupFill: CleanupFillMode | undefined;
    if (cleanupFillStr !== undefined) {
        if (voxelCleanup === undefined || voxelCleanup === 0) {
            throw new Error(
                '--voxel-cleanup-fill requires --voxel-cleanup with a value greater than 0.');
        }
        const normalized = cleanupFillStr.toLowerCase();
        if (normalized !== 'none' && normalized !== 'grow' &&
            normalized !== 'close' && normalized !== 'both') {
            throw new Error(
                `Invalid voxel cleanup fill mode: ${cleanupFillStr}. Expected none, grow, close or both.`);
        }
        voxelCleanupFill = normalized;
    }
```

This must sit after the `--voxel-params` block, because the bare-flag default reads
`voxelResolution`. Import `type CleanupFillMode` from `../lib` alongside the other library types
at the top of the file.

Add both to the `options` object literal (lines 583-633), after `autoRotate,`:

```ts
        voxelCleanup,
        voxelCleanupFill,
```

- [ ] **Step 5: Warn when there is no voxel output**

Beside the existing `--auto-rotate` check at lines 1179-1181:

```ts
    if (options.voxelCleanup !== undefined && outputFormat !== 'voxel') {
        logger.warn('--voxel-cleanup has no effect without a .voxel.json output.');
    }
```

- [ ] **Step 6: Add the help text**

In the `VOXEL OUTPUT (.voxel.json)` block (lines 905-922), after the `--voxel-carve` line,
matching the block's `[value]` / `<value>` and `Default: X` conventions:

```
        --voxel-cleanup    [size]           Fill sampling holes, flatten bumpy surfaces and drop floating
                                            debris at this scale. Only voxels with gaussian density behind
                                            them are ever added, so real gaps and openings survive.
                                            Bare flag uses 2x the voxel size. Default: off
        --voxel-cleanup-fill [none|grow|close|both]   Hole-filling algorithm for --voxel-cleanup. none runs
                                            only the smoothing and debris passes. Default: grow
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
npx tsx --test test/cli.test.mjs
```

Expected: PASS, including the 7 new cases.

- [ ] **Step 8: Full suite and lint**

```bash
npm test && npm run lint
```

- [ ] **Step 9: Commit**

```bash
git add src/cli/index.ts test/cli.test.mjs
git commit -m "feat: add --voxel-cleanup and --voxel-cleanup-fill

--voxel-cleanup takes a scale in world units, defaulting to 2x the voxel
size as a bare flag. --voxel-cleanup-fill selects the hole-filling
algorithm and requires --voxel-cleanup. Both are registered in
optionalValueOptions so a bare flag cannot swallow the output argument,
and the CLI warns when they are passed without a .voxel.json output."
```

---

## Task 5: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:** none.

- [ ] **Step 1: Add the flags to the README option table**

In `README.md`, after the `--voxel-external-fill` line (line 207) in the voxel option block, add
entries matching the surrounding format.

- [ ] **Step 2: Add a prose section**

After the `--auto-rotate` prose (around line 271), add a section covering: what the two symptoms
look like, the one-line cause, the guarantee, and what to pass. Include the measured before/after
so the numbers live somewhere a user will find them:

```markdown
### Cleaning up a scattered voxel grid

Voxelizing thresholds a continuous gaussian density field, and nothing afterwards reconstructs a
surface. When a scene's splats are small relative to the voxel size — common on large outdoor
captures at 5-10 cm — the result is not a surface at all but a scatter of near-isolated voxels:
full of holes, and violently bumpy. `--voxel-cleanup` fixes both, because they are the same
problem.

It runs three passes: fill voxels that look like holes in an existing surface, regularize the
surface with a 3x3x3 majority filter, then drop islands smaller than one 4x4x4 block.

Crucially, **every voxel it adds must have gaussian density behind it.** The cleanup samples the
same field a second time at a far lower opacity threshold and uses that as a mask, so a real
window opening, a real gap between a railing and a deck, or a real void inside a building has no
density and is untouchable at any scale. Morphological closing on its own would fabricate: on the
scene below, an ungated close at the same radius placed ~140,000 voxels in effective vacuum, 46%
of everything it added. The gated pipeline places none.

On a 24x21x38 m city rooftop capture at 10 cm:

| | occupied voxels | disconnected islands | in the largest | surface roughness |
| --- | --- | --- | --- | --- |
| without | 288,184 | 13,602 | 60.4% | 9.8 voxels |
| `--voxel-cleanup 0.2` | 202,947 | 64 | 77.6% | 3.3 voxels |

Note the voxel count goes *down*: it is removing noise, not adding bulk.

```bash
splat-transform city.spz city.voxel.json --voxel-params 0.1,0.1 --voxel-cleanup 0.2
```

Pass roughly twice the voxel size to start. A bare `--voxel-cleanup` does exactly that.
`--voxel-cleanup-fill none` skips hole filling and runs only the smoothing and debris passes, for
scenes whose coverage is already good.

The log reports surface coherence before cleanup, and suggests the flag when a grid is mostly
scatter, so you can tell whether a scene needs it.
```

- [ ] **Step 3: Verify the markdown renders and links are intact**

```bash
npx markdownlint README.md 2>/dev/null || echo "markdownlint not configured; check the diff by eye"
git diff README.md
```

Read the diff and confirm the table renders and the option block alignment matches its neighbours.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document --voxel-cleanup and the anti-fabrication guarantee

Covers the shared cause of holes and bumpiness, the three passes, and the
measured before/after on a city rooftop capture. States plainly that an
ungated close would have fabricated 140,000 voxels in vacuum on that scene
and the gated pipeline fabricates none, since that is the property a user
needs to trust the flag."
```

---

## Task 6: Acceptance run on a real scene

Verification, not implementation. Confirms the shipped feature reproduces the design spec's
measurements on `scenes/urban.spz`.

**Files:** none modified. Findings go in the task report.

**Interfaces:** none.

- [ ] **Step 1: Confirm the scene is available**

```bash
ls -la scenes/urban.spz
```

If absent, stop and report — this task cannot run without it, and the unit tests already cover
correctness. Do not substitute a different scene silently.

- [ ] **Step 2: Baseline, without cleanup**

```bash
npx tsx src/cli/index.ts ./scenes/urban.spz \
  --filter-box -20,-20,-20,20,20,20 \
  --voxel-params 0.1,0.1 \
  --auto-rotate -w \
  /tmp/urban-baseline.voxel.json
```

Record the `surface coherence` line and the octree counts from the output.

- [ ] **Step 3: With cleanup**

```bash
npx tsx src/cli/index.ts ./scenes/urban.spz \
  --filter-box -20,-20,-20,20,20,20 \
  --voxel-params 0.1,0.1 \
  --voxel-cleanup 0.2 \
  --auto-rotate -w \
  /tmp/urban-cleaned.voxel.json
```

Record the `cleanup:` line and the octree counts.

- [ ] **Step 4: Measure both grids**

```bash
npx tsx -e "
(async () => {
const { readFile } = await import('node:fs/promises');
for (const name of ['baseline', 'cleaned']) {
  const meta = JSON.parse(await readFile(\`/tmp/urban-\${name}.voxel.json\`, 'utf8'));
  console.log(name, 'mixedLeaves', meta.numMixedLeaves, 'interior', meta.numInteriorNodes,
    'gridBounds', JSON.stringify(meta.gridBounds.min.map(v => +v.toFixed(2))));
}
})();
"
```

- [ ] **Step 5: Check the acceptance criteria**

The cleaned run must show, relative to baseline:

- the `cleanup:` line reporting a non-zero `grown` and a `componentsRemoved` in the thousands
- `surface coherence` on the baseline above 20%, and the suggestion warning present
- fewer mixed leaves in the cleaned octree than the baseline

The design spec's targets are ~64 components and roughness ~3.33. If the run lands materially
outside those — say more than 200 components, or an *increase* in mixed leaves — do not adjust
the numbers to match. Report the discrepancy with the actual figures; it means a stage is
misbehaving and the primitives' tests missed it.

- [ ] **Step 6: Record the results and clean up**

Write the full output of both runs, the measurements, and a pass/fail against each criterion into
the task report. Then:

```bash
rm -f /tmp/urban-baseline.voxel.* /tmp/urban-cleaned.voxel.*
```

No commit — this task produces evidence, not code.

---

## Remaining work

One plan follows: **`close` mode** (`specs/2026-08-10-voxel-cleanup-close-plan.md`) — the GPU
erode pass, `closeGrid`, `cleanupPad` on the grid bounds, and replacing `cleanupGrid`'s
not-implemented throw for the `close` and `both` modes.
