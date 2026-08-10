# Voxel Cleanup Prerequisite Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two latent correctness bugs in the voxel grid path that the voxel-cleanup feature would otherwise be built on top of: a block-type write that ORs instead of assigns, and unguarded grid-size limits that corrupt surface data silently on large scenes.

**Architecture:** Both fixes are contained. The first is a one-character correction in `applyChunkToDst` plus a comment recording why assignment is required. The second introduces a new module `src/lib/voxel/grid-limits.ts` holding the limit constant and a pure `assertGridFits` guard, called from `writeVoxel` right after the grid bounds are aligned — extracting the check into a pure function is what makes it unit-testable without materialising a multi-gigabyte grid.

**Tech Stack:** TypeScript (ES2022, ESM), Node's built-in test runner (`node:test` + `node:assert`), tsx for running TypeScript from tests, ESLint (`@playcanvas/eslint-config`).

**Design spec:** `specs/2026-08-10-voxel-cleanup-design.md` (section "Prerequisite bug fixes")

## Global Constraints

- `src/lib/` must stay platform-agnostic: no `node:*` imports, no Node-only APIs. Only `src/lib/workers/` may use guarded dynamic `node:` imports, and this work adds nothing there.
- Naming: classes PascalCase, functions camelCase, types PascalCase, constants UPPER_SNAKE_CASE. Arrow-function `const`s are the prevailing style in `src/lib/`.
- Import order: Node built-ins, then external packages (`playcanvas`), then internal relative paths.
- Use `const`/`let`, never `var`.
- All new public API needs JSDoc with `@param`/`@returns` descriptions (no types in JSDoc — TypeScript provides them) for Typedoc.
- Test files live in `test/` with a `.test.mjs` extension and import source modules directly with a `.js` extension (e.g. `../src/lib/voxel/grid-limits.js`), which is the established pattern — see `test/dilation-grid-ops.test.mjs`.
- Run `npm run lint` before each commit. Only fix lint issues in code you are actively modifying.
- Run `npm test` before each commit and confirm no regressions.
- Commit messages follow conventional commits: `fix:`, `feat:`, `test:`, `docs:`.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/lib/voxel/dilation.ts` | Modify line 223 only. No new responsibility. |
| `src/lib/voxel/grid-limits.ts` | **New.** Owns the block-count ceiling for the voxel grid path and the pure guard that enforces it. Single responsibility, no dependencies beyond nothing at all — it is arithmetic and an error message. |
| `src/lib/voxel/index.ts` | Re-export `assertGridFits` and `MAX_GRID_BLOCKS` alongside the other voxel exports. |
| `src/lib/writers/write-voxel.ts` | Call the guard after `alignGridBounds`, before `voxelizeToBuffer`. |
| `test/grid-limits.test.mjs` | **New.** Unit tests for the guard. |

`grid-limits.ts` is deliberately its own module rather than a helper inside `write-voxel.ts`: the limit is a property of `SparseVoxelGrid`/`BlockMaskMap` indexing, not of the writer, and `filter-cluster.ts` may want to share it later.

---

## Task 1: Correct the block-type write in `applyChunkToDst`

`dilation.ts:223` writes a block's 2-bit type with `|=`, which can only ever set bits. That is correct for accumulating *different* blocks into a shared `types` word — 16 blocks pack into each word — but it cannot overwrite a field that already holds a value. Chunks are disjoint today (`innerStep` is a multiple of 4 and grid dims are multiples of 4), so every field is written at most once into freshly zeroed memory and the defect is inert. If two chunks ever wrote the same block, `SOLID (1) | MIXED (2) === 3` — an invalid type. `getVoxel` would then fall through to the mask lookup (`src/lib/voxel/sparse-voxel-grid.ts:151-155`), find no entry, and report the entire block as **empty**.

**No test accompanies this task, deliberately.** The faulty branch is unreachable through any public entry point: `gpuDilate3` is the only caller, its chunks are provably disjoint, and the planned `gpuErode3` reuses the same disjoint chunking. Reaching it would require exporting a module-private helper solely to construct a state the code cannot produce, which is a worse trade than swapping in the existing canonical setter and commenting why. `npm test` must still pass unchanged, which is the regression signal that the change is behaviour-neutral today.

**Files:**
- Modify: `src/lib/voxel/dilation.ts:223` (plus the `./sparse-voxel-grid` import at the top)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. No signature changes.

- [ ] **Step 1: Confirm the current behaviour is unchanged by the fix**

Run the existing dilation suite and record that it passes, so the post-fix run is a real comparison:

```bash
npm test -- --test-name-pattern="gpuDilate3"
```

Expected: PASS. If it fails, stop — the baseline is broken and this plan's assumption is wrong.

- [ ] **Step 2: Apply the fix**

In `src/lib/voxel/dilation.ts`, replace:

```ts
                const globalBlockIdx = baseGlobalIdx + bx;
                const w = globalBlockIdx >>> 4;
                const shift = (globalBlockIdx & 15) << 1;
                dstTypes[w] |= bt << shift;
```

with a call to the canonical setter, which `src/lib/voxel/sparse-voxel-grid.ts:89-93` already
exports and which does the clear-then-set correctly:

```ts
                const globalBlockIdx = baseGlobalIdx + bx;
                // writeBlockType clears the block's own 2-bit field before
                // setting it. OR-ing is equivalent only while chunks are
                // disjoint and the destination starts zeroed; if that stopped
                // holding, SOLID (1) | MIXED (2) would give type 3, which
                // getVoxel resolves to an absent mask and so reports the whole
                // block as empty.
                writeBlockType(dstTypes, globalBlockIdx, bt);
```

Add `writeBlockType` to the existing import from `./sparse-voxel-grid` at the top of
`dilation.ts` (the file already imports `SparseVoxelGrid` from there).

**Do not** replace the line with a bare `dstTypes[w] = bt << shift`. A `types` word packs 16
blocks at 2 bits each, so a bare assignment would wipe the other 15 blocks in the word. The
`|=` was accumulating *different* blocks into a shared word, which is correct; the defect is
only that it cannot overwrite a field that already holds a value.

- [ ] **Step 3: Verify no behaviour change**

```bash
npm test
```

Expected: PASS, with the same results as Step 1. The fix is behaviour-neutral on all currently reachable inputs.

- [ ] **Step 4: Lint**

```bash
npm run lint
```

Expected: no new errors.

- [ ] **Step 5: Commit**

```bash
git add src/lib/voxel/dilation.ts
git commit -m "fix: use writeBlockType in applyChunkToDst instead of OR-ing types

A types word packs 16 blocks at 2 bits each, so a block's field must be
cleared before it is set. OR-ing accumulated different blocks into the
shared word correctly, but could not overwrite an occupied field: if
chunks ever stopped being disjoint, SOLID | MIXED would give type 3,
which getVoxel resolves to an absent mask and so reports the entire
block as empty. writeBlockType already does the clear-then-set."
```

---

## Task 2: Guard the voxel grid against silent index overflow

Voxelizing an unfiltered large scene currently throws `RangeError: Invalid typed array length: -2147483648` from `IntKeyMap` (`src/lib/voxel/block-cleanup.ts:62` sizes it from the occupied block count; `src/lib/utils/int-key-map.ts:38` computes `1 << (32 - Math.clz32(...))`, which is negative past 2^30). Reproduce with `scenes/landscape.spz` unfiltered at 0.1 m: its 3-sigma scene bounds span 593.6 x 370.7 x 882.9 m, giving 3,037,474,944 blocks.

That throw is the benign failure. The same grid also exceeds 2^31 blocks, which trips the `BlockMaskMap` `Int32Array` key overflow (`src/lib/voxel/block-mask-map.ts:12,24,45`): keys past 2^31 store as negative, lookups never match, and every MIXED block's mask reads back as zero — losing precisely the surface blocks while SOLID interiors survive. Past 2^32 the `types` word index itself aliases (`src/lib/voxel/sparse-voxel-grid.ts:75,90,122`).

The ceiling is therefore set by the tightest of the three. `IntKeyMap` needs its capacity under 2^30, and capacity is `ceil(blocks / 0.7)`, so blocks must stay under `0.7 * 2^30` ≈ 751.6e6. `MAX_GRID_BLOCKS = 2^29` (536,870,912) is the clean power of two below that, and it also sits well under the 2^31 `BlockMaskMap` limit and the 2^32 `types` limit. For calibration: the reported `urban.spz` grid is 1.55e6 blocks and a filtered `landscape.spz` at 0.01 m is 90.4e6 blocks, both far inside; a 50 m cube at 0.01 m would be 1.95e9 blocks, which is correctly rejected because its mask storage alone would run to tens of gigabytes.

**Files:**
- Create: `src/lib/voxel/grid-limits.ts`
- Modify: `src/lib/voxel/index.ts`
- Modify: `src/lib/writers/write-voxel.ts` (insert after line 559, before the `voxelizeToBuffer` call at 561)
- Test: `test/grid-limits.test.mjs`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `MAX_GRID_BLOCKS: number` — the constant `536870912`.
  - `assertGridFits(nbx: number, nby: number, nbz: number, voxelResolution: number): void` — throws `Error` when `nbx * nby * nbz > MAX_GRID_BLOCKS`, returns `undefined` otherwise. Later plans call this from the cleanup path too.

- [ ] **Step 1: Write the failing test**

Create `test/grid-limits.test.mjs`:

```javascript
/**
 * Tests for the voxel grid block-count ceiling.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { MAX_GRID_BLOCKS, assertGridFits } from '../src/lib/voxel/grid-limits.js';

describe('assertGridFits', function () {
    it('accepts a grid at the limit', function () {
        // 512^3 = 134217728 blocks, comfortably inside
        assert.doesNotThrow(() => assertGridFits(512, 512, 512, 0.01));
    });

    it('accepts a realistic large scene', function () {
        // filtered landscape.spz at 0.01m: 939 x 285 x 338 = 90.4e6 blocks
        assert.doesNotThrow(() => assertGridFits(939, 285, 338, 0.01));
    });

    it('rejects a grid past the limit', function () {
        // 1484 x 927 x 2208 = 3.04e9 blocks, the unfiltered landscape.spz case
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            /too large/
        );
    });

    it('names the block count and the limit in the error', function () {
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            (err) => {
                assert.match(err.message, /3037474944/);
                assert.match(err.message, new RegExp(String(MAX_GRID_BLOCKS)));
                return true;
            }
        );
    });

    it('suggests both remedies in the error', function () {
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            (err) => {
                assert.match(err.message, /--voxel-params/);
                assert.match(err.message, /--filter-box/);
                return true;
            }
        );
    });

    it('reports the voxel resolution that produced the grid', function () {
        assert.throws(
            () => assertGridFits(1484, 927, 2208, 0.1),
            /0\.1/
        );
    });

    it('exposes the limit as a power of two under the IntKeyMap ceiling', function () {
        assert.strictEqual(MAX_GRID_BLOCKS, 2 ** 29);
        assert.ok(MAX_GRID_BLOCKS < 0.7 * 2 ** 30);
    });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx tsx --test test/grid-limits.test.mjs
```

Expected: FAIL — cannot resolve `../src/lib/voxel/grid-limits.js`.

- [ ] **Step 3: Write the implementation**

Create `src/lib/voxel/grid-limits.ts`:

```ts
/**
 * Block-count ceiling for the voxel grid path.
 *
 * Three separate 32-bit limits sit above the sparse grid, and this constant is
 * the tightest of them expressed as a power of two:
 *
 * - `IntKeyMap` (`../utils/int-key-map.ts`) sizes itself as
 *   `1 << (32 - Math.clz32(capacity - 1))`, which goes negative once capacity
 *   reaches 2^30. `block-cleanup.ts` builds one at `blocks / 0.7`, so blocks
 *   must stay under `0.7 * 2^30` (about 751.6e6).
 * - `BlockMaskMap` (`./block-mask-map.ts`) stores block indices in an
 *   `Int32Array`, so a key at or past 2^31 stores negative, never matches on
 *   lookup, and silently loses every MIXED block's mask — the surface blocks —
 *   while SOLID interiors survive.
 * - `SparseVoxelGrid` (`./sparse-voxel-grid.ts`) indexes its `types` words with
 *   `blockIdx >>> 4`, which aliases past 2^32.
 *
 * 2^29 clears all three. Only the first fails loudly, which is why the guard
 * exists rather than relying on the throw.
 */
const MAX_GRID_BLOCKS = 2 ** 29;

/**
 * Throws when a voxel grid holds more blocks than the sparse grid can index
 * without silently corrupting surface data.
 *
 * @param nbx - Grid block count along X.
 * @param nby - Grid block count along Y.
 * @param nbz - Grid block count along Z.
 * @param voxelResolution - Voxel size in world units, reported in the error so
 * the caller can see which resolution produced the grid.
 * @throws If the total block count exceeds {@link MAX_GRID_BLOCKS}.
 */
const assertGridFits = (
    nbx: number,
    nby: number,
    nbz: number,
    voxelResolution: number
): void => {
    const blocks = nbx * nby * nbz;
    if (blocks > MAX_GRID_BLOCKS) {
        throw new Error(
            `Voxel grid too large: ${nbx}x${nby}x${nbz} blocks (${blocks}) exceeds the ` +
            `${MAX_GRID_BLOCKS} block limit at voxel resolution ${voxelResolution}. ` +
            'Past this size the sparse grid loses surface blocks without reporting it. ' +
            'Either coarsen the resolution with --voxel-params or shrink the region with ' +
            '--filter-box.'
        );
    }
};

export { MAX_GRID_BLOCKS, assertGridFits };
```

- [ ] **Step 4: Run test to verify it passes**

```bash
npx tsx --test test/grid-limits.test.mjs
```

Expected: PASS, 7 tests.

- [ ] **Step 5: Re-export from the voxel barrel**

`src/lib/voxel/index.ts` uses one `export { ... } from './module'` line per module. Add a line
in the same style, placed after the `block-mask-buffer` line so the grid-level modules stay
together:

```ts
export { filterAndFillBlocks } from './block-cleanup';
export { BlockMaskBuffer } from './block-mask-buffer';
export { MAX_GRID_BLOCKS, assertGridFits } from './grid-limits';
export {
    BLOCK_EMPTY, BLOCK_MIXED, BLOCK_SOLID, SparseVoxelGrid, readBlockType, writeBlockType
} from './sparse-voxel-grid';
```

- [ ] **Step 6: Verify the barrel export resolves**

```bash
npx tsx -e "import('./src/lib/voxel/index.ts').then(m => { if (typeof m.assertGridFits !== 'function') throw new Error('assertGridFits not exported'); if (m.MAX_GRID_BLOCKS !== 2 ** 29) throw new Error('MAX_GRID_BLOCKS wrong'); console.log('barrel export OK'); })"
```

Expected: prints `barrel export OK`.

- [ ] **Step 7: Call the guard from `writeVoxel`**

In `src/lib/writers/write-voxel.ts`, add `assertGridFits` to the existing import from `../voxel` (the file already imports `alignGridBounds`, `carve`, `fillExterior`, `fillFloor` and friends from there — add it to that list, keeping alphabetical order if the list is alphabetical).

Then insert the guard between the `alignGridBounds` call and the `voxelizeToBuffer` call. The current code at lines 555-563 reads:

```ts
        let gridBounds = alignGridBounds(
            bounds.min.x - padXZ, bounds.min.y - padY, bounds.min.z - padXZ,
            bounds.max.x + padXZ, bounds.max.y + padY, bounds.max.z + padXZ,
            voxelResolution
        );

        const buffer = await voxelizeToBuffer(
            bvh, gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
        );
```

Change it to:

```ts
        let gridBounds = alignGridBounds(
            bounds.min.x - padXZ, bounds.min.y - padY, bounds.min.z - padXZ,
            bounds.max.x + padXZ, bounds.max.y + padY, bounds.max.z + padXZ,
            voxelResolution
        );

        // Reject oversized grids before any GPU work: past the block-index
        // ceiling the sparse grid drops surface masks silently, so failing here
        // with an actionable message beats producing a hollowed-out result.
        const blockSize = 4 * voxelResolution;
        assertGridFits(
            Math.round((gridBounds.max.x - gridBounds.min.x) / blockSize),
            Math.round((gridBounds.max.y - gridBounds.min.y) / blockSize),
            Math.round((gridBounds.max.z - gridBounds.min.z) / blockSize),
            voxelResolution
        );

        const buffer = await voxelizeToBuffer(
            bvh, gpuVoxelization, gridBounds, voxelResolution, opacityCutoff
        );
```

- [ ] **Step 8: Verify the writer still works on a normal scene**

```bash
npm test -- --test-name-pattern="writeVoxel"
```

Expected: PASS. The guard must not fire on any fixture.

- [ ] **Step 9: Verify the guard fires on the real oversized scene**

This requires `scenes/landscape.spz`, which is not in the repo fixtures. If it is absent, skip this step and note it — the unit tests in Step 1 already cover the guard's logic. If present:

```bash
npx tsx -e "
(async () => {
const { readFile, getInputFormat } = await import('./src/lib/read.ts');
const { NodeReadFileSystem } = await import('./src/cli/node-file-system.ts');
const { DataTable, Column, computeGaussianExtents } = await import('./src/lib/data-table/index.ts');
const { assertGridFits } = await import('./src/lib/voxel/index.ts');
const fsys = new NodeReadFileSystem();
const t = await readFile({ filename: 'scenes/landscape.spz', inputFormat: getInputFormat('scenes/landscape.spz'), options: {}, params: [], fileSystem: fsys });
const cols = ['x','y','z','rot_0','rot_1','rot_2','rot_3','scale_0','scale_1','scale_2','opacity'];
const pc = new DataTable(cols.map(n => new Column(n, t[0].getColumnByName(n).data)));
const b = computeGaussianExtents(pc).sceneBounds;
const res = 0.1, bs = 4 * res;
const dims = [b.max.x-b.min.x, b.max.y-b.min.y, b.max.z-b.min.z].map(v => Math.ceil(v / bs));
try {
  assertGridFits(dims[0], dims[1], dims[2], res);
  throw new Error('guard did NOT fire - expected it to');
} catch (e) {
  if (!/too large/.test(e.message)) throw e;
  console.log('guard fired as expected:', e.message);
}
process.exit(0);
})();
"
```

Expected: prints `guard fired as expected:` followed by the error naming the block count.

- [ ] **Step 10: Full test run and lint**

```bash
npm test && npm run lint
```

Expected: PASS, no new lint errors.

- [ ] **Step 11: Commit**

```bash
git add src/lib/voxel/grid-limits.ts src/lib/voxel/index.ts src/lib/writers/write-voxel.ts test/grid-limits.test.mjs
git commit -m "fix: guard the voxel grid against silent block-index overflow

Past 2^31 blocks, BlockMaskMap stores block indices in an Int32Array as
negative values that never match on lookup, so every MIXED block's mask
reads back as zero. That drops exactly the surface blocks while SOLID
interiors survive, producing a hollow result with no error. IntKeyMap
throws an unactionable RangeError somewhat earlier, and the types word
index aliases somewhat later.

assertGridFits rejects grids over 2^29 blocks -- the clean power of two
below IntKeyMap's 0.7 * 2^30 ceiling -- with a message naming the block
count and pointing at --voxel-params and --filter-box. Reproduced with
an unfiltered landscape.spz at 0.1m: 1484x927x2208 = 3,037,474,944 blocks."
```

---

## Remaining work

This plan covers only the prerequisites. The voxel cleanup feature itself is split into two further plans, written after this one lands:

- **Plan B — cleanup with the `grow` mode.** Candidate-mask plumbing through the GPU voxelizer, `growGrid`, `majorityFilterGrid`, `despeckleGrid`, the `--voxel-cleanup` / `--voxel-cleanup-fill` CLI surface with `grow` as default, observability, and the acceptance run. Ships a complete, working feature.
- **Plan C — the `close` mode.** GPU erode in the dilation pipeline, `closeGrid`, `cleanupPad` on the grid bounds. Additive; `grow` keeps working throughout.

The split follows the spec's delivery order, which deliberately puts `grow` and the CLI ahead of `close` so the feature is measurable before any WGSL is written.
