# Voxel Cleanup Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the four library primitives the density-gated voxel cleanup needs — `clearVoxel`, `growGrid`, `majorityFilterGrid` and `despeckleGrid` — as exported, unit-tested pure functions over `SparseVoxelGrid`.

**Architecture:** Every additive primitive takes a `candidate` grid and may only set voxels that are set in it, which is the feature's anti-fabrication guarantee. `growGrid` works block-wise on 64-bit masks, bit-slicing a 6-neighbour count with carry-save adders; the six directional mask computations move into a shared `block-neighbors.ts` parameterised by a block-reader. `majorityFilterGrid` needs a 3x3x3 neighbourhood, which is 26 cross-block cases in sparse form, so it runs over chunked dense buffers with separable counting instead. `despeckleGrid` labels 6-connected components of the occupied set in one pass with a member buffer capped at the keep threshold.

**Tech Stack:** TypeScript (ES2022, ESM), Node's built-in test runner (`node:test` + `node:assert`), tsx for running TypeScript from tests, ESLint (`@playcanvas/eslint-config`).

**Design spec:** `specs/2026-08-10-voxel-cleanup-design.md`
**Prerequisite plan (landed):** `specs/2026-08-10-voxel-cleanup-prereq-fixes-plan.md`

## Global Constraints

- `src/lib/` must stay platform-agnostic: no `node:*` imports, no Node-only APIs. Only `src/lib/workers/` may use guarded dynamic `node:` imports, and this work adds nothing there.
- Naming: classes PascalCase, functions camelCase, types PascalCase, constants UPPER_SNAKE_CASE. Arrow-function `const`s are the prevailing style in `src/lib/`, except for methods on a class and for the `function` declarations already used in `block-cleanup.ts` and `flood-fill.ts`.
- Import order: Node built-ins, then external packages (`playcanvas`), then internal relative paths.
- Use `const`/`let`, never `var`.
- All new exported API needs JSDoc with `@param` and `@returns` descriptions and no type annotations in the JSDoc — TypeScript provides the types. `@throws` where it throws. This feeds Typedoc.
- Test files live in `test/` with a `.test.mjs` extension and import source modules with a `.js` extension (e.g. `../src/lib/voxel/grow.js`). Use `describe`/`it` from `node:test` and `assert` from `node:assert`.
- Every primitive is **non-destructive on `candidate`** and **consumes `grid`**: it returns a new grid and the caller must not reuse the input. Document this on each function.
- Run `npm run lint` and `npm test` before each commit; no new lint errors, no test regressions. Baseline at the start of this plan is 732 passing.
- Commit messages follow conventional commits (`feat:`, `test:`, `fix:`, `docs:`).

## Reference: existing primitives you will build on

From `src/lib/voxel/sparse-voxel-grid.ts` (all already exported):

```ts
const SOLID_LO = 0xFFFFFFFF >>> 0;   // line 4
const SOLID_HI = 0xFFFFFFFF >>> 0;   // line 5
const BLOCK_EMPTY = 0;               // line 7
const BLOCK_SOLID = 1;               // line 8
const BLOCK_MIXED = 2;               // line 9
```

`SparseVoxelGrid` stores 4x4x4 blocks: `types` packs a 2-bit type per block (16 per word), and
`masks` is a `BlockMaskMap` holding a `[lo, hi]` 64-bit voxel mask for `MIXED` blocks only. Bit
order within a block is `bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4)`, so `lo` holds
`iz` 0-1 and `hi` holds `iz` 2-3.

Relevant members: `nx/ny/nz`, `nbx/nby/nbz`, `bStride` (= `nbx * nby`), `types`, `masks`,
`getBlockType(blockIdx)`, `setBlockType(blockIdx, value)`, `getVoxel(ix,iy,iz)`,
`setVoxel(ix,iy,iz)`, `orBlock(blockIdx, lo, hi)`, `clear()`, `clone()`,
`forEachOccupiedVoxel(cb)`. `masks.slot(key)` returns a slot index (an empty slot if the key is
absent, whose `lo`/`hi` read 0), `masks.set(key, lo, hi)`, `masks.removeAt(slot)`.

The face-mask constants for in-block bit shifts already exist privately in
`src/lib/voxel/block-cleanup.ts:14-24`. Task 2 moves a copy into a shared module; do not edit
`block-cleanup.ts`.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/lib/voxel/sparse-voxel-grid.ts` | Add one method, `clearVoxel`. No other change. |
| `src/lib/voxel/block-neighbors.ts` | **New.** Face-mask constants and `sixNeighborMasks`, which computes the six directional occupancy masks for one block given a block-reader callback. Pure bit arithmetic, no grid dependency. |
| `src/lib/voxel/grow.ts` | **New.** `growGrid` — density-gated neighbour-count region growing. |
| `src/lib/voxel/majority.ts` | **New.** `majorityFilterGrid` — chunked dense 3x3x3 majority filter with separable counting. |
| `src/lib/voxel/despeckle.ts` | **New.** `despeckleGrid` — removes 6-connected components below a voxel-count threshold. |
| `src/lib/voxel/index.ts` | Re-export the three new functions and their option/result types. |
| `src/lib/index.ts` | Add the three functions and their types to the public API. `src/lib/index.ts:93-98` re-exports from `./voxel` with an **explicit named list**, so adding to the voxel barrel alone does not surface anything publicly — both files need the entry. |
| `test/sparse-voxel-grid.test.mjs` | Extend with `clearVoxel` cases. |
| `test/voxel-grow.test.mjs` | **New.** |
| `test/voxel-majority.test.mjs` | **New.** |
| `test/voxel-despeckle.test.mjs` | **New.** |

Each primitive is its own module because each has one responsibility and its own test file, and
because the integration plan composes them in a separate orchestrator. `block-neighbors.ts` is
separate so `growGrid` and the later `close` stage share one copy of the directional-mask
arithmetic.

Nothing in this plan is reachable from the CLI. The integration plan wires it up.

---

## Task 1: `SparseVoxelGrid.clearVoxel`

The class can set a voxel but not unset one. `despeckleGrid` needs to remove voxels, and the
demotion rules are non-obvious: clearing a bit in a `SOLID` block must materialise a `MIXED`
mask of all-ones-minus-one-bit, and clearing the last bit of a `MIXED` block must release the
mask slot and demote the block to `EMPTY`. Getting either wrong corrupts the grid silently,
which is why this is its own task with its own tests.

**Files:**
- Modify: `src/lib/voxel/sparse-voxel-grid.ts` (add a method directly after `setVoxel`, which ends at line 178)
- Test: `test/sparse-voxel-grid.test.mjs`

**Interfaces:**
- Consumes: nothing.
- Produces: `SparseVoxelGrid.prototype.clearVoxel(ix: number, iy: number, iz: number): void`. Task 4 (`despeckleGrid`) calls it.

- [ ] **Step 1: Write the failing tests**

Append to `test/sparse-voxel-grid.test.mjs`. Check the existing imports at the top of that file
first — it already imports `SparseVoxelGrid`; add `BLOCK_EMPTY`, `BLOCK_MIXED` and `BLOCK_SOLID`
to that same import from `../src/lib/voxel/sparse-voxel-grid.js` if they are not present.

```javascript
describe('SparseVoxelGrid.clearVoxel', function () {
    it('clears a voxel from a mixed block', function () {
        const g = new SparseVoxelGrid(8, 8, 8);
        g.setVoxel(1, 1, 1);
        g.setVoxel(2, 1, 1);
        g.clearVoxel(1, 1, 1);
        assert.strictEqual(g.getVoxel(1, 1, 1), 0);
        assert.strictEqual(g.getVoxel(2, 1, 1), 1);
    });

    it('demotes a mixed block to empty when its last voxel goes', function () {
        const g = new SparseVoxelGrid(8, 8, 8);
        g.setVoxel(1, 1, 1);
        assert.strictEqual(g.getBlockType(0), BLOCK_MIXED);
        g.clearVoxel(1, 1, 1);
        assert.strictEqual(g.getBlockType(0), BLOCK_EMPTY);
        assert.strictEqual(g.getVoxel(1, 1, 1), 0);
        assert.strictEqual(g.masks.size, 0, 'mask slot must be released');
    });

    it('demotes a solid block to mixed, keeping the other 63 voxels', function () {
        const g = new SparseVoxelGrid(4, 4, 4);
        for (let z = 0; z < 4; z++) {
            for (let y = 0; y < 4; y++) {
                for (let x = 0; x < 4; x++) g.setVoxel(x, y, z);
            }
        }
        assert.strictEqual(g.getBlockType(0), BLOCK_SOLID);
        g.clearVoxel(2, 3, 3);
        assert.strictEqual(g.getBlockType(0), BLOCK_MIXED);
        assert.strictEqual(g.getVoxel(2, 3, 3), 0);
        let count = 0;
        g.forEachOccupiedVoxel(() => count++);
        assert.strictEqual(count, 63);
    });

    it('clears a hi-word voxel of a solid block', function () {
        // bitIdx >= 32 exercises the hi half; iz 2 and 3 live in hi
        const g = new SparseVoxelGrid(4, 4, 4);
        for (let z = 0; z < 4; z++) {
            for (let y = 0; y < 4; y++) {
                for (let x = 0; x < 4; x++) g.setVoxel(x, y, z);
            }
        }
        g.clearVoxel(0, 0, 2);
        assert.strictEqual(g.getVoxel(0, 0, 2), 0);
        assert.strictEqual(g.getVoxel(0, 0, 1), 1);
        assert.strictEqual(g.getVoxel(0, 0, 3), 1);
    });

    it('is a no-op on an already-empty block', function () {
        const g = new SparseVoxelGrid(8, 8, 8);
        g.clearVoxel(5, 5, 5);
        assert.strictEqual(g.getBlockType(0), BLOCK_EMPTY);
        assert.strictEqual(g.masks.size, 0);
    });

    it('is a no-op on an already-clear voxel of a mixed block', function () {
        const g = new SparseVoxelGrid(8, 8, 8);
        g.setVoxel(1, 1, 1);
        g.clearVoxel(2, 2, 2);
        assert.strictEqual(g.getVoxel(1, 1, 1), 1);
        assert.strictEqual(g.getVoxel(2, 2, 2), 0);
        assert.strictEqual(g.getBlockType(0), BLOCK_MIXED);
    });

    it('round-trips set then clear back to the original state', function () {
        const g = new SparseVoxelGrid(8, 8, 8);
        g.setVoxel(0, 0, 0);
        g.setVoxel(7, 7, 7);
        const before = [...g.types];
        g.setVoxel(3, 3, 3);
        g.clearVoxel(3, 3, 3);
        assert.deepStrictEqual([...g.types], before);
        let count = 0;
        g.forEachOccupiedVoxel(() => count++);
        assert.strictEqual(count, 2);
    });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/sparse-voxel-grid.test.mjs
```

Expected: FAIL — `g.clearVoxel is not a function`.

- [ ] **Step 3: Implement `clearVoxel`**

In `src/lib/voxel/sparse-voxel-grid.ts`, insert this method immediately after the `setVoxel`
method (which closes at line 178) and before `orBlock`:

```ts
    /**
     * Unset a single voxel, demoting the containing block as needed.
     *
     * A `SOLID` block becomes `MIXED` with a full mask minus this one bit; a
     * `MIXED` block whose mask empties releases its mask slot and becomes
     * `EMPTY`. Clearing an already-clear voxel is a no-op.
     *
     * @param ix - Voxel X index.
     * @param iy - Voxel Y index.
     * @param iz - Voxel Z index.
     */
    clearVoxel(ix: number, iy: number, iz: number): void {
        const blockIdx = (ix >> 2) + (iy >> 2) * this.nbx + (iz >> 2) * this.bStride;
        const bt = this.getBlockType(blockIdx);
        if (bt === BLOCK_EMPTY) return;

        const bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4);

        if (bt === BLOCK_SOLID) {
            this.setBlockType(blockIdx, BLOCK_MIXED);
            this.masks.set(blockIdx,
                bitIdx < 32 ? (SOLID_LO & ~(1 << bitIdx)) >>> 0 : SOLID_LO,
                bitIdx >= 32 ? (SOLID_HI & ~(1 << (bitIdx - 32))) >>> 0 : SOLID_HI
            );
            return;
        }

        const s = this.masks.slot(blockIdx);
        if (bitIdx < 32) {
            this.masks.lo[s] = (this.masks.lo[s] & ~(1 << bitIdx)) >>> 0;
        } else {
            this.masks.hi[s] = (this.masks.hi[s] & ~(1 << (bitIdx - 32))) >>> 0;
        }
        if (this.masks.lo[s] === 0 && this.masks.hi[s] === 0) {
            this.masks.removeAt(s);
            this.setBlockType(blockIdx, BLOCK_EMPTY);
        }
    }
```

This mirrors `setVoxel` (lines 158-178) exactly, inverted. Note the guard order: the `EMPTY`
early return must come before `masks.slot`, because `slot` on an absent key returns a free slot
whose `lo`/`hi` are zero and writing through it would corrupt the table.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
npx tsx --test test/sparse-voxel-grid.test.mjs
```

Expected: PASS, including the 7 new cases.

- [ ] **Step 5: Full suite and lint**

```bash
npm test && npm run lint
```

Expected: PASS (739 tests: the 732 baseline plus 7), no new lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/lib/voxel/sparse-voxel-grid.ts test/sparse-voxel-grid.test.mjs
git commit -m "feat: add SparseVoxelGrid.clearVoxel

The class could set a voxel but not unset one. Clearing has two non-obvious
demotions: a SOLID block must materialise a MIXED mask of all-ones minus the
cleared bit, and a MIXED block whose mask empties must release its mask slot
and become EMPTY. Needed by the voxel-cleanup despeckle stage."
```

---

## Task 2: `block-neighbors.ts` — the six directional masks

`growGrid` needs, for one 4x4x4 block, six 64-bit masks saying "for each voxel position, is the
neighbour in direction D occupied?". Within the block that is a bit shift; across a block face it
is a shift of the adjacent block's opposite face. `block-cleanup.ts:87-153` already does this,
but it reads from a `BlockMaskBuffer` through `IntKeyMap` side tables. This task extracts the
arithmetic into a module parameterised by a block-reader so `growGrid` — and later the `close`
stage — share one copy.

`block-cleanup.ts` keeps its own copy; migrating it is a follow-up, not this task. Do not edit it.

**Files:**
- Create: `src/lib/voxel/block-neighbors.ts`
- Test: `test/voxel-grow.test.mjs` (the neighbour helper is tested through its own describe block in the grow test file, since it exists only to serve `growGrid`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `type BlockReader = (bx: number, by: number, bz: number, out: Uint32Array) => void` — writes `[lo, hi]` into `out[0]`, `out[1]` for the block at those block coordinates, or `[0, 0]` when out of bounds.
  - `sixNeighborMasks(read: BlockReader, lo: number, hi: number, bx: number, by: number, bz: number, out: Uint32Array): void` — writes 12 values into `out`: `[+xLo, +xHi, -xLo, -xHi, +yLo, +yHi, -yLo, -yHi, +zLo, +zHi, -zLo, -zHi]`. `out` must have length >= 12. `lo`/`hi` are the subject block's own mask.
  - `NEIGHBOR_SCRATCH_LEN = 12`.

- [ ] **Step 1: Write the failing test**

Create `test/voxel-grow.test.mjs` with just the neighbour-helper block for now (Task 3 appends
to this file):

```javascript
/**
 * Tests for density-gated region growing and its block-neighbour helper.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    NEIGHBOR_SCRATCH_LEN,
    sixNeighborMasks
} from '../src/lib/voxel/block-neighbors.js';
import {
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from '../src/lib/voxel/sparse-voxel-grid.js';

// bitIdx = (ix & 3) + ((iy & 3) << 2) + ((iz & 3) << 4)
const bit = (x, y, z) => {
    const i = (x & 3) + ((y & 3) << 2) + ((z & 3) << 4);
    return i < 32 ? [(1 << i) >>> 0, 0] : [0, (1 << (i - 32)) >>> 0];
};

const emptyReader = (bx, by, bz, out) => {
    out[0] = 0;
    out[1] = 0;
};

describe('sixNeighborMasks', function () {
    it('reports the in-block +X neighbour of a single voxel', function () {
        // voxel at (1,0,0) occupied; position (0,0,0) sees it in +X
        const [lo, hi] = bit(1, 0, 0);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [pLo] = bit(0, 0, 0);
        assert.strictEqual(out[0], pLo, '+X mask should mark position (0,0,0)');
        assert.strictEqual(out[2], 0, '-X mask should be empty');
    });

    it('reports the in-block -X neighbour of a single voxel', function () {
        const [lo, hi] = bit(1, 0, 0);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [mLo] = bit(2, 0, 0);
        assert.strictEqual(out[2], mLo, '-X mask should mark position (2,0,0)');
    });

    it('does not wrap across the lx=3 to lx=0 boundary', function () {
        // voxel at (0,0,0); position (3,0,0) must NOT see it in +X
        const [lo, hi] = bit(0, 0, 0);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [wrapLo] = bit(3, 0, 0);
        assert.strictEqual(out[0] & wrapLo, 0, '+X must not wrap into lx=3');
    });

    it('crosses the Z boundary between lo and hi words', function () {
        // voxel at (0,0,2) is in hi; position (0,0,1) is in lo and sees it in +Z
        const [lo, hi] = bit(0, 0, 2);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(emptyReader, lo, hi, 0, 0, 0, out);
        const [seenLo] = bit(0, 0, 1);
        assert.strictEqual(out[8] & seenLo, seenLo, '+Z must cross lo/hi');
    });

    it('pulls the adjacent block face across +X', function () {
        // subject block empty; neighbour block at bx+1 is fully solid.
        // every position with lx=3 must see an occupied +X neighbour.
        const solidReader = (bx, by, bz, out) => {
            if (bx === 1 && by === 0 && bz === 0) {
                out[0] = SOLID_LO;
                out[1] = SOLID_HI;
            } else {
                out[0] = 0;
                out[1] = 0;
            }
        };
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        sixNeighborMasks(solidReader, 0, 0, 0, 0, 0, out);
        for (let z = 0; z < 4; z++) {
            for (let y = 0; y < 4; y++) {
                const [eLo, eHi] = bit(3, y, z);
                if (eLo) assert.strictEqual(out[0] & eLo, eLo, `+X face at y=${y} z=${z}`);
                if (eHi) assert.strictEqual(out[1] & eHi, eHi, `+X face at y=${y} z=${z}`);
            }
        }
    });

    it('agrees with a brute-force per-voxel neighbour check', function () {
        // Build a small grid, then for one block compare sixNeighborMasks
        // against getVoxel on every position and direction.
        const g = new SparseVoxelGrid(12, 12, 12);
        let seed = 12345;
        const rnd = () => {
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
            return seed / 0x7FFFFFFF;
        };
        for (let z = 0; z < 12; z++) {
            for (let y = 0; y < 12; y++) {
                for (let x = 0; x < 12; x++) {
                    if (rnd() < 0.3) g.setVoxel(x, y, z);
                }
            }
        }
        const read = (bx, by, bz, out) => {
            if (bx < 0 || by < 0 || bz < 0 || bx >= g.nbx || by >= g.nby || bz >= g.nbz) {
                out[0] = 0;
                out[1] = 0;
                return;
            }
            const bi = bx + by * g.nbx + bz * g.bStride;
            const bt = g.getBlockType(bi);
            if (bt === 0) {
                out[0] = 0;
                out[1] = 0;
            } else if (bt === 1) {
                out[0] = SOLID_LO;
                out[1] = SOLID_HI;
            } else {
                const s = g.masks.slot(bi);
                out[0] = g.masks.lo[s];
                out[1] = g.masks.hi[s];
            }
        };
        const own = new Uint32Array(2);
        const out = new Uint32Array(NEIGHBOR_SCRATCH_LEN);
        const dirs = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
        // block (1,1,1) is interior, so all six neighbour blocks exist
        read(1, 1, 1, own);
        sixNeighborMasks(read, own[0], own[1], 1, 1, 1, out);
        for (let lz = 0; lz < 4; lz++) {
            for (let ly = 0; ly < 4; ly++) {
                for (let lx = 0; lx < 4; lx++) {
                    const [pLo, pHi] = bit(lx, ly, lz);
                    for (let d = 0; d < 6; d++) {
                        const [dx, dy, dz] = dirs[d];
                        const expected = g.getVoxel(4 + lx + dx, 4 + ly + dy, 4 + lz + dz);
                        const got = pLo ? ((out[d * 2] & pLo) !== 0) : ((out[d * 2 + 1] & pHi) !== 0);
                        assert.strictEqual(
                            got, expected === 1,
                            `dir ${d} at local (${lx},${ly},${lz})`
                        );
                    }
                }
            }
        }
    });
});
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
npx tsx --test test/voxel-grow.test.mjs
```

Expected: FAIL — cannot resolve `../src/lib/voxel/block-neighbors.js`.

- [ ] **Step 3: Implement `block-neighbors.ts`**

Create `src/lib/voxel/block-neighbors.ts`:

```ts
/**
 * Directional neighbour-occupancy masks for a 4x4x4 voxel block.
 *
 * Bit layout matches `SparseVoxelGrid`: `bitIdx = lx + ly*4 + lz*16`, with `lo`
 * holding `lz` 0-1 and `hi` holding `lz` 2-3.
 *
 * `block-cleanup.ts` carries an equivalent private copy of this arithmetic
 * because it reads from a `BlockMaskBuffer` through `IntKeyMap` side tables
 * rather than from a grid. Migrating it onto this module is a follow-up.
 */

/** `lx == 0` positions in each 32-bit word. */
const FACE_X0 = 0x11111111;
/** `lx == 3` positions in each 32-bit word. */
const FACE_X3 = 0x88888888;
/** `ly == 0` positions in each 32-bit word. */
const FACE_Y0 = 0x000F000F;
/** `ly == 3` positions in each 32-bit word. */
const FACE_Y3 = 0xF000F000;
/** `lz == 0` positions: `lo` bits 0-15. */
const FACE_Z0_LO = 0x0000FFFF;
/** `lz == 3` positions: `hi` bits 16-31. */
const FACE_Z3_HI = 0xFFFF0000 >>> 0;

/** Number of `Uint32Array` entries {@link sixNeighborMasks} writes. */
const NEIGHBOR_SCRATCH_LEN = 12;

/**
 * Reads one block's `[lo, hi]` voxel mask into `out[0]`, `out[1]`.
 *
 * Implementations must write `[0, 0]` for block coordinates outside the grid,
 * which is what makes out-of-grid read as empty.
 */
type BlockReader = (bx: number, by: number, bz: number, out: Uint32Array) => void;

const adj = new Uint32Array(2);

/**
 * Compute the six directional neighbour-occupancy masks for one block.
 *
 * For direction `D`, the returned mask has a bit set at voxel position `p` when
 * the voxel at `p + D` is occupied — whether that neighbour lies inside this
 * block or in the adjacent one.
 *
 * @param read - Callback supplying any block's `[lo, hi]` mask.
 * @param lo - The subject block's own low mask word.
 * @param hi - The subject block's own high mask word.
 * @param bx - Subject block X coordinate.
 * @param by - Subject block Y coordinate.
 * @param bz - Subject block Z coordinate.
 * @param out - Destination of length >= {@link NEIGHBOR_SCRATCH_LEN}, filled as
 * `[+xLo, +xHi, -xLo, -xHi, +yLo, +yHi, -yLo, -yHi, +zLo, +zHi, -zLo, -zHi]`.
 */
const sixNeighborMasks = (
    read: BlockReader,
    lo: number,
    hi: number,
    bx: number,
    by: number,
    bz: number,
    out: Uint32Array
): void => {
    // In-block shifts. Each masks off the face that would wrap around.
    // +X: position p sees p+1, valid while lx < 3
    out[0] = (lo >>> 1) & ~FACE_X3;
    out[1] = (hi >>> 1) & ~FACE_X3;
    // -X: position p sees p-1, valid while lx > 0
    out[2] = (lo << 1) & ~FACE_X0;
    out[3] = (hi << 1) & ~FACE_X0;
    // +Y: p sees p+4
    out[4] = (lo >>> 4) & ~FACE_Y3;
    out[5] = (hi >>> 4) & ~FACE_Y3;
    // -Y: p sees p-4
    out[6] = (lo << 4) & ~FACE_Y0;
    out[7] = (hi << 4) & ~FACE_Y0;
    // +Z: p sees p+16, which crosses lo->hi at lz 1->2
    out[8] = (lo >>> 16) | (hi << 16);
    out[9] = hi >>> 16;
    // -Z: p sees p-16, crossing hi->lo at lz 2->1
    out[10] = lo << 16;
    out[11] = (hi << 16) | (lo >>> 16);

    // Cross-block faces. Our lx=3 column sees the neighbour's lx=0 column,
    // shifted up by 3 lanes; and symmetrically for the other five directions.
    read(bx + 1, by, bz, adj);
    out[0] |= (adj[0] & FACE_X0) << 3;
    out[1] |= (adj[1] & FACE_X0) << 3;

    read(bx - 1, by, bz, adj);
    out[2] |= (adj[0] & FACE_X3) >>> 3;
    out[3] |= (adj[1] & FACE_X3) >>> 3;

    read(bx, by + 1, bz, adj);
    out[4] |= (adj[0] & FACE_Y0) << 12;
    out[5] |= (adj[1] & FACE_Y0) << 12;

    read(bx, by - 1, bz, adj);
    out[6] |= (adj[0] & FACE_Y3) >>> 12;
    out[7] |= (adj[1] & FACE_Y3) >>> 12;

    read(bx, by, bz + 1, adj);
    out[9] |= (adj[0] & FACE_Z0_LO) << 16;

    read(bx, by, bz - 1, adj);
    out[10] |= (adj[1] & FACE_Z3_HI) >>> 16;

    // Normalize: the shifts above can leave results as signed int32.
    for (let i = 0; i < NEIGHBOR_SCRATCH_LEN; i++) out[i] = out[i] >>> 0;
};

export {
    NEIGHBOR_SCRATCH_LEN,
    sixNeighborMasks,
    type BlockReader
};
```

Note the module-level `adj` scratch array: `sixNeighborMasks` is called once per block in a hot
loop, so it must not allocate. This makes the function non-reentrant, which is fine — nothing in
`src/lib/` is concurrent — and is worth the comment it carries.

- [ ] **Step 4: Run the test to verify it passes**

```bash
npx tsx --test test/voxel-grow.test.mjs
```

Expected: PASS, 6 tests. The brute-force test is the important one: it checks all 6 directions at
all 64 positions of an interior block against `getVoxel`, so any shift or face-mask error fails it.

- [ ] **Step 5: Full suite and lint**

```bash
npm test && npm run lint
```

Expected: PASS, no new lint errors.

- [ ] **Step 6: Commit**

```bash
git add src/lib/voxel/block-neighbors.ts test/voxel-grow.test.mjs
git commit -m "feat: add sixNeighborMasks for block-wise neighbour occupancy

Computes the six directional neighbour-occupancy masks for one 4x4x4 block
given a block-reader callback, so callers reading from a SparseVoxelGrid do
not need block-cleanup's IntKeyMap side tables. Verified against a
brute-force getVoxel check over all 64 positions and 6 directions."
```

---

## Task 3: `growGrid`

Iteratively fill any candidate voxel with at least `minNeighbors` of its 6 face neighbours
occupied. This is the default hole-filling stage. It can only add voxels that are set in
`candidate`, which is the anti-fabrication guarantee.

Behaviour to preserve, from the design spec's measurements: on a 1-voxel-thick sheet a voxel has
at most 4 in-plane neighbours, so at `minNeighbors = 3` a 1x1 hole fills in one iteration, a 1x3
slit takes two (ends first, then the middle), and a 3x3 square hole **never** fills — its corners
see 2, its edge middles 1, its centre 0. That conservatism is intentional and is pinned by a test.

**Files:**
- Create: `src/lib/voxel/grow.ts`
- Modify: `src/lib/voxel/index.ts`
- Test: `test/voxel-grow.test.mjs` (append a second describe block)

**Interfaces:**
- Consumes: `sixNeighborMasks`, `NEIGHBOR_SCRATCH_LEN`, `type BlockReader` from `./block-neighbors` (Task 2).
- Produces:
  - `type GrowOptions = { minNeighbors?: number; maxIterations?: number }` — defaults 3 and 4.
  - `type GrowResult = { grid: SparseVoxelGrid; added: number; iterations: number }`.
  - `growGrid(grid: SparseVoxelGrid, candidate: SparseVoxelGrid, options?: GrowOptions): GrowResult`. Consumes `grid`; does not modify `candidate`. The integration plan calls this.

- [ ] **Step 1: Write the failing tests**

Append to `test/voxel-grow.test.mjs`. Add `growGrid` to the imports:

```javascript
import { growGrid } from '../src/lib/voxel/grow.js';

// A grid where every voxel is a candidate, for tests not exercising the gate.
const allCandidate = (nx, ny, nz) => {
    const g = new SparseVoxelGrid(nx, ny, nz);
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

// A one-voxel-thick sheet at y === yPlane, with the listed (x,z) holes.
const sheet = (n, yPlane, holes) => {
    const g = new SparseVoxelGrid(n, n, n);
    const isHole = new Set(holes.map(([x, z]) => `${x},${z}`));
    for (let z = 0; z < n; z++) {
        for (let x = 0; x < n; x++) {
            if (!isHole.has(`${x},${z}`)) g.setVoxel(x, yPlane, z);
        }
    }
    return g;
};

describe('growGrid', function () {
    it('fills a 1x1 hole in a sheet in one iteration', function () {
        const g = sheet(8, 4, [[4, 4]]);
        const res = growGrid(g, allCandidate(8, 8, 8), { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
        assert.strictEqual(res.added, 1);
    });

    it('fills a 1x3 slit in exactly two iterations, not one', function () {
        const holes = [[3, 4], [4, 4], [5, 4]];
        const one = growGrid(sheet(12, 4, holes), allCandidate(12, 12, 12),
            { minNeighbors: 3, maxIterations: 1 });
        assert.strictEqual(one.grid.getVoxel(4, 4, 4), 0, 'middle needs a second pass');
        assert.strictEqual(one.grid.getVoxel(3, 4, 4), 1, 'ends fill first');

        const two = growGrid(sheet(12, 4, holes), allCandidate(12, 12, 12),
            { minNeighbors: 3, maxIterations: 2 });
        assert.strictEqual(two.grid.getVoxel(4, 4, 4), 1);
        assert.strictEqual(two.added, 3);
    });

    it('never fills a 3x3 square hole, at any iteration count', function () {
        const holes = [];
        for (let x = 3; x <= 5; x++) {
            for (let z = 3; z <= 5; z++) holes.push([x, z]);
        }
        const res = growGrid(sheet(12, 4, holes), allCandidate(12, 12, 12),
            { minNeighbors: 3, maxIterations: 32 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 0, 'centre must stay open');
        assert.strictEqual(res.added, 0, 'nothing in a 3x3 hole reaches 3 neighbours');
    });

    it('does not grow outward from a flat sheet face', function () {
        // A complete sheet: every voxel just above it has exactly 1 occupied
        // neighbour, so nothing should be added anywhere.
        const res = growGrid(sheet(8, 4, []), allCandidate(8, 8, 8),
            { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.added, 0);
    });

    it('respects the candidate gate: an empty candidate adds nothing', function () {
        const empty = new SparseVoxelGrid(8, 8, 8);
        const res = growGrid(sheet(8, 4, [[4, 4]]), empty,
            { minNeighbors: 3, maxIterations: 8 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 0, 'gate must block the fill');
        assert.strictEqual(res.added, 0);
    });

    it('reports gate-rejected voxels', function () {
        const empty = new SparseVoxelGrid(8, 8, 8);
        const res = growGrid(sheet(8, 4, [[4, 4]]), empty,
            { minNeighbors: 3, maxIterations: 8 });
        assert.strictEqual(res.gateRejected, 1,
            'the hole was eligible but blocked, and must be counted once');
    });

    it('reports zero gate-rejected when everything is a candidate', function () {
        const res = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.gateRejected, 0);
    });

    it('respects the candidate gate per voxel', function () {
        // Two 1x1 holes; only one is a candidate.
        const cand = new SparseVoxelGrid(12, 12, 12);
        cand.setVoxel(4, 4, 4);
        const res = growGrid(sheet(12, 4, [[4, 4], [8, 8]]), cand,
            { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
        assert.strictEqual(res.grid.getVoxel(8, 4, 8), 0);
        assert.strictEqual(res.added, 1);
    });

    it('terminates early when an iteration adds nothing', function () {
        const res = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 3, maxIterations: 16 });
        assert.strictEqual(res.iterations, 2,
            'one productive pass plus one that adds nothing');
    });

    it('honours a higher minNeighbors', function () {
        // A 1x1 hole in a sheet has 4 neighbours, so k=4 fills it but k=5 does not.
        const four = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 4, maxIterations: 4 });
        assert.strictEqual(four.grid.getVoxel(4, 4, 4), 1);

        const five = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8),
            { minNeighbors: 5, maxIterations: 4 });
        assert.strictEqual(five.grid.getVoxel(4, 4, 4), 0);
    });

    it('fills an interior void with 6 neighbours at k=6', function () {
        // A 3x3x3 solid cube with its centre missing: the centre has all 6.
        const g = new SparseVoxelGrid(8, 8, 8);
        for (let z = 2; z <= 4; z++) {
            for (let y = 2; y <= 4; y++) {
                for (let x = 2; x <= 4; x++) {
                    if (!(x === 3 && y === 3 && z === 3)) g.setVoxel(x, y, z);
                }
            }
        }
        const res = growGrid(g, allCandidate(8, 8, 8), { minNeighbors: 6, maxIterations: 2 });
        assert.strictEqual(res.grid.getVoxel(3, 3, 3), 1);
        assert.strictEqual(res.added, 1);
    });

    it('crosses block boundaries', function () {
        // Hole at (4,4,4) sits at a block corner (blocks are 4^3), so its
        // neighbours live in four different blocks.
        const g = sheet(12, 4, [[4, 4]]);
        const res = growGrid(g, allCandidate(12, 12, 12), { minNeighbors: 3, maxIterations: 4 });
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
    });

    it('leaves the candidate grid untouched', function () {
        const cand = allCandidate(8, 8, 8);
        const before = [...cand.types];
        growGrid(sheet(8, 4, [[4, 4]]), cand, { minNeighbors: 3, maxIterations: 4 });
        assert.deepStrictEqual([...cand.types], before);
    });

    it('defaults to minNeighbors 3 and maxIterations 4', function () {
        const res = growGrid(sheet(8, 4, [[4, 4]]), allCandidate(8, 8, 8));
        assert.strictEqual(res.grid.getVoxel(4, 4, 4), 1);
    });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/voxel-grow.test.mjs
```

Expected: FAIL — cannot resolve `../src/lib/voxel/grow.js`.

- [ ] **Step 3: Implement `grow.ts`**

Create `src/lib/voxel/grow.ts`:

```ts
import {
    NEIGHBOR_SCRATCH_LEN,
    sixNeighborMasks,
    type BlockReader
} from './block-neighbors';
import { popcount } from './morton';
import {
    BLOCK_EMPTY,
    BLOCK_SOLID,
    SOLID_HI,
    SOLID_LO,
    SparseVoxelGrid
} from './sparse-voxel-grid';

/**
 * Options for {@link growGrid}.
 */
type GrowOptions = {
    /** Minimum occupied face neighbours a candidate voxel needs. Default: 3 */
    minNeighbors?: number;
    /** Cap on passes; growing stops early once a pass adds nothing. Default: 4 */
    maxIterations?: number;
};

/**
 * Result of {@link growGrid}.
 */
type GrowResult = {
    /** The grown grid. */
    grid: SparseVoxelGrid;
    /** Voxels added across all passes. */
    added: number;
    /**
     * Voxels that met the neighbour threshold but were blocked by the candidate
     * mask. The audit trail for the anti-fabrication guarantee: these are the
     * voxels an ungated fill would have invented.
     */
    gateRejected: number;
    /** Passes actually run, including the final unproductive one. */
    iterations: number;
};

/**
 * Bit-sliced "at least `k` of the six masks set at this position".
 *
 * The three planes hold the neighbour count in binary: `c2 c1 c0`, maximum 6.
 *
 * @param k - Threshold, 1 to 6.
 * @param c0 - Count bit 0.
 * @param c1 - Count bit 1.
 * @param c2 - Count bit 2.
 * @returns Mask of positions whose count is at least `k`.
 */
const atLeast = (k: number, c0: number, c1: number, c2: number): number => {
    switch (k) {
        case 1: return c2 | c1 | c0;
        case 2: return c2 | c1;
        case 3: return c2 | (c1 & c0);
        case 4: return c2;
        case 5: return c2 & (c1 | c0);
        case 6: return c2 & c1;
        default: return 0;
    }
};

/** Reads a grid block's mask, or zeroes when out of bounds. */
const makeReader = (g: SparseVoxelGrid): BlockReader => {
    const { nbx, nby, nbz, bStride, masks } = g;
    return (bx: number, by: number, bz: number, out: Uint32Array): void => {
        if (bx < 0 || by < 0 || bz < 0 || bx >= nbx || by >= nby || bz >= nbz) {
            out[0] = 0;
            out[1] = 0;
            return;
        }
        const bi = bx + by * nbx + bz * bStride;
        const bt = g.getBlockType(bi);
        if (bt === BLOCK_EMPTY) {
            out[0] = 0;
            out[1] = 0;
        } else if (bt === BLOCK_SOLID) {
            out[0] = SOLID_LO;
            out[1] = SOLID_HI;
        } else {
            const s = masks.slot(bi);
            out[0] = masks.lo[s];
            out[1] = masks.hi[s];
        }
    };
};

/**
 * Fill voxels that look like holes in an existing surface, restricted to a
 * candidate mask.
 *
 * A voxel is filled when it is empty, set in `candidate`, and has at least
 * `minNeighbors` of its six face neighbours occupied. Passes repeat until one
 * adds nothing or `maxIterations` is reached; each pass reads a snapshot, so the
 * result does not depend on block iteration order.
 *
 * Because every addition must be in `candidate`, this can never place a voxel
 * where the source data has no support — see the design spec's anti-fabrication
 * guarantee.
 *
 * @param grid - Grid to grow. **Consumed**: do not reuse it after the call.
 * @param candidate - Voxels permitted to be filled. Not modified.
 * @param options - Threshold and iteration cap.
 * @returns The grown grid with the number of voxels added and passes run.
 */
const growGrid = (
    grid: SparseVoxelGrid,
    candidate: SparseVoxelGrid,
    options: GrowOptions = {}
): GrowResult => {
    const { minNeighbors = 3, maxIterations = 4 } = options;

    const { nbx, nby, nbz, bStride } = grid;
    const readCandidate = makeReader(candidate);
    const own = new Uint32Array(2);
    const cand = new Uint32Array(2);
    const nb = new Uint32Array(NEIGHBOR_SCRATCH_LEN);

    let current = grid;
    let added = 0;
    let gateRejected = 0;
    let iterations = 0;

    for (let iter = 0; iter < maxIterations; iter++) {
        iterations++;
        const readCurrent = makeReader(current);
        const next = current.clone();
        let addedThisPass = 0;

        // Only blocks with candidate voxels can gain anything, and the
        // candidate grid is far smaller than the full block space.
        for (let bz = 0; bz < nbz; bz++) {
            for (let by = 0; by < nby; by++) {
                for (let bx = 0; bx < nbx; bx++) {
                    readCandidate(bx, by, bz, cand);
                    if (cand[0] === 0 && cand[1] === 0) continue;

                    readCurrent(bx, by, bz, own);
                    // A fully solid block has nothing to gain.
                    if (own[0] === SOLID_LO && own[1] === SOLID_HI) continue;

                    sixNeighborMasks(readCurrent, own[0], own[1], bx, by, bz, nb);

                    // Carry-save accumulate the six masks into a 3-bit count.
                    let lo0 = 0;
                    let lo1 = 0;
                    let lo2 = 0;
                    let hi0 = 0;
                    let hi1 = 0;
                    let hi2 = 0;
                    for (let d = 0; d < 6; d++) {
                        const mLo = nb[d * 2];
                        const cLo = lo0 & mLo;
                        lo0 ^= mLo;
                        const c1Lo = lo1 & cLo;
                        lo1 ^= cLo;
                        lo2 |= c1Lo;

                        const mHi = nb[d * 2 + 1];
                        const cHi = hi0 & mHi;
                        hi0 ^= mHi;
                        const c1Hi = hi1 & cHi;
                        hi1 ^= cHi;
                        hi2 |= c1Hi;
                    }

                    const eligibleLo = (~own[0] & atLeast(minNeighbors, lo0, lo1, lo2)) >>> 0;
                    const eligibleHi = (~own[1] & atLeast(minNeighbors, hi0, hi1, hi2)) >>> 0;
                    const newLo = (eligibleLo & cand[0]) >>> 0;
                    const newHi = (eligibleHi & cand[1]) >>> 0;
                    // Counted only on the first pass: a voxel the gate blocks
                    // stays eligible every pass, so summing would multiply it.
                    if (iter === 0) {
                        gateRejected += popcount((eligibleLo & ~cand[0]) >>> 0) +
                            popcount((eligibleHi & ~cand[1]) >>> 0);
                    }
                    if (newLo === 0 && newHi === 0) continue;

                    addedThisPass += popcount(newLo) + popcount(newHi);
                    next.orBlock(bx + by * nbx + bz * bStride, newLo, newHi);
                }
            }
        }

        if (addedThisPass === 0) {
            next.releaseStorage();
            break;
        }
        added += addedThisPass;
        current.releaseStorage();
        current = next;
    }

    return { grid: current, added, gateRejected, iterations };
};

export { growGrid, type GrowOptions, type GrowResult };
```

Two details that matter. First, each pass writes into a `clone()` so the neighbour reads all see
the previous pass's state — growing in place would let a voxel added earlier in the scan seed
another later in the same pass, making the result depend on iteration order. Second, the
`SOLID_LO`/`SOLID_HI` early-out and the candidate-empty early-out are what keep the block scan
cheap on a sparse grid.

Check `popcount`'s export before relying on it: it is imported by `block-cleanup.ts:2` from
`./morton`. If its signature differs from `(n: number) => number`, adapt the two call sites.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
npx tsx --test test/voxel-grow.test.mjs
```

Expected: PASS, 20 tests (6 from Task 2 plus 14 here).

- [ ] **Step 5: Export from both barrels**

`src/lib/voxel/index.ts` uses one `export { ... } from './module'` line per module. Add:

```ts
export { growGrid } from './grow';
export type { GrowOptions, GrowResult } from './grow';
```

Then surface it publicly. `src/lib/index.ts:93-98` re-exports from `./voxel` through an explicit
named list, so the voxel barrel alone is not enough. Add `growGrid` to the value list (keeping
its alphabetical order within the list) and `GrowOptions`, `GrowResult` to the type list:

```ts
// Voxel
export {
    alignGridBounds, applyAlignYaw, carve, estimateAlignYaw, fillExterior, fillFloor,
    filterAndFillBlocks, filterCluster, filterFloaters, findClusterVoxelFlood, growGrid,
    voxelizeToBuffer,
    BlockMaskBuffer, SparseVoxelGrid, BLOCK_EMPTY, BLOCK_MIXED, BLOCK_SOLID
} from './voxel';
export type { AlignYawApplied, AlignYawOptions, AlignYawResult, GrowOptions, GrowResult, NavSeed, NavSimplifyResult, UpAxis } from './voxel';
```

Verify both resolve:

```bash
npx tsx -e "Promise.all([import('./src/lib/voxel/index.ts'), import('./src/lib/index.ts')]).then(([v, p]) => { if (typeof v.growGrid !== 'function') throw new Error('voxel barrel'); if (typeof p.growGrid !== 'function') throw new Error('public barrel'); console.log('both barrels OK'); })"
```

Expected: prints `both barrels OK`.

- [ ] **Step 6: Full suite and lint**

```bash
npm test && npm run lint
```

Expected: PASS, no new lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/lib/voxel/grow.ts src/lib/voxel/index.ts src/lib/index.ts test/voxel-grow.test.mjs
git commit -m "feat: add growGrid, density-gated neighbour-count region growing

Fills empty voxels that are set in the candidate mask and have at least
minNeighbors of six face neighbours occupied, iterating until a pass adds
nothing. Counts are bit-sliced with carry-save adders over the six
directional masks, so a block's 64 voxels are evaluated in ~24 bit ops per
word.

Every addition is gated by the candidate mask, so the stage cannot place a
voxel where the gaussian field has no support. At minNeighbors 3 the
conservatism is deliberate and pinned by tests: a 1x1 hole fills in one
pass, a 1x3 slit in two, and a 3x3 square hole never fills."
```

---

## Task 4: `majorityFilterGrid`

A 3x3x3 majority filter: a voxel is on when at least `threshold` of the 27 voxels in its
neighbourhood (itself included) are occupied. This is the surface-regularization stage — on
`urban.spz` it takes top-surface roughness from 9.80 voxels to about 3.3 and cuts the component
count from thousands to tens.

26 cross-block neighbours in sparse form is 6 faces plus 12 edges plus 8 corners, so this runs
over chunked dense buffers instead, with **separable** counting: an X pass yielding 0-3, a Z pass
yielding 0-9, a Y pass yielding 0-27. Three linear passes rather than 27 taps per voxel.

Additions are gated by `candidate`; removals are not, since removing cannot fabricate.

**Files:**
- Create: `src/lib/voxel/majority.ts`
- Modify: `src/lib/voxel/index.ts`
- Test: `test/voxel-majority.test.mjs`

**Interfaces:**
- Consumes: nothing from Tasks 2-3.
- Produces:
  - `type MajorityOptions = { threshold?: number; iterations?: number; chunkInner?: number }` — defaults 14, 2 and 256. `chunkInner` exists so tests can force many small chunks.
  - `type MajorityResult = { grid: SparseVoxelGrid; added: number; removed: number }`.
  - `majorityFilterGrid(grid, candidate, options?): MajorityResult`. Consumes `grid`; does not modify `candidate`.

- [ ] **Step 1: Write the failing tests**

Create `test/voxel-majority.test.mjs`:

```javascript
/**
 * Tests for the 3x3x3 majority filter.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { majorityFilterGrid } from '../src/lib/voxel/majority.js';
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

// A solid slab spanning y in [y0, y1] across the whole grid interior.
const slab = (n, y0, y1) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = 0; z < n; z++) {
        for (let y = y0; y <= y1; y++) {
            for (let x = 0; x < n; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};

describe('majorityFilterGrid', function () {
    it('removes an isolated voxel', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0);
        assert.strictEqual(res.removed, 1);
    });

    it('removes a 1-voxel bump on a slab', function () {
        const g = slab(16, 6, 9);
        g.setVoxel(8, 10, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 10, 8), 0, 'bump should go');
        assert.strictEqual(res.grid.getVoxel(8, 9, 8), 1, 'slab top should stay');
    });

    it('fills a 1-voxel dent in a slab', function () {
        const g = slab(16, 6, 9);
        g.clearVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 1, 'interior dent should fill');
    });

    it('preserves the interior of a thick slab', function () {
        // The slab spans the full x/z extent, so its edge columns border
        // out-of-grid (counted empty) and see only 12 of 27 -- those DO erode.
        // Interior voxels, well away from the boundary, must survive untouched.
        const res = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        for (let y = 5; y <= 10; y++) {
            assert.strictEqual(res.grid.getVoxel(8, y, 8), 1, `interior y=${y}`);
        }
        assert.strictEqual(res.grid.getVoxel(8, 11, 8), 1, 'top surface centre');
        assert.strictEqual(res.grid.getVoxel(8, 4, 8), 1, 'bottom surface centre');
        assert.strictEqual(res.added, 0, 'a solid slab needs no additions');
    });

    it('erodes a slab edge that borders out-of-grid', function () {
        // Documents the boundary convention that the previous test works around.
        const res = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(0, 8, 8), 0,
            'x=0 column sees 2 of 3 x-slices, 12 of 27');
        assert.ok(res.removed > 0);
    });

    it('gates additions by the candidate mask', function () {
        const g = slab(16, 6, 9);
        g.clearVoxel(8, 8, 8);
        const empty = new SparseVoxelGrid(16, 16, 16);
        const res = majorityFilterGrid(g, empty, { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0, 'gate must block the fill');
        assert.strictEqual(res.added, 0);
        assert.ok(res.gateRejected >= 1, 'the blocked dent must be counted');
    });

    it('does not gate removals by the candidate mask', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const empty = new SparseVoxelGrid(16, 16, 16);
        const res = majorityFilterGrid(g, empty, { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0, 'removal needs no candidate');
        assert.strictEqual(res.removed, 1);
    });

    it('produces the same result whatever the chunk size', function () {
        const build = () => {
            const g = new SparseVoxelGrid(32, 32, 32);
            let seed = 999;
            const rnd = () => {
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
                return seed / 0x7FFFFFFF;
            };
            for (let z = 0; z < 32; z++) {
                for (let y = 0; y < 32; y++) {
                    for (let x = 0; x < 32; x++) {
                        if (rnd() < 0.4) g.setVoxel(x, y, z);
                    }
                }
            }
            return g;
        };
        const big = majorityFilterGrid(build(), allCandidate(32),
            { threshold: 14, iterations: 2, chunkInner: 256 });
        const small = majorityFilterGrid(build(), allCandidate(32),
            { threshold: 14, iterations: 2, chunkInner: 8 });

        const dump = (g) => {
            const out = [];
            g.forEachOccupiedVoxel((x, y, z) => out.push(`${x},${y},${z}`));
            return out.sort();
        };
        assert.deepStrictEqual(dump(small.grid), dump(big.grid),
            'chunked result must equal single-chunk result');
        assert.strictEqual(small.added, big.added);
        assert.strictEqual(small.removed, big.removed);
    });

    it('iterates: two passes differ from one on a noisy sheet', function () {
        const build = () => {
            const g = new SparseVoxelGrid(24, 24, 24);
            for (let z = 2; z < 22; z++) {
                for (let x = 2; x < 22; x++) {
                    if ((x + z) % 3 !== 0) g.setVoxel(x, 12, z);
                }
            }
            return g;
        };
        const one = majorityFilterGrid(build(), allCandidate(24),
            { threshold: 10, iterations: 1 });
        const two = majorityFilterGrid(build(), allCandidate(24),
            { threshold: 10, iterations: 2 });
        assert.notStrictEqual(countVoxels(one.grid), countVoxels(two.grid));
    });

    it('leaves the candidate grid untouched', function () {
        const cand = allCandidate(16);
        const before = [...cand.types];
        majorityFilterGrid(slab(16, 6, 9), cand, { threshold: 14, iterations: 1 });
        assert.deepStrictEqual([...cand.types], before);
    });

    it('treats out-of-grid as empty', function () {
        // A slab touching y=0: the bottom layer's below-neighbours are outside
        // the grid and must count as empty, so a thin slab at the edge erodes.
        const g = new SparseVoxelGrid(16, 16, 16);
        for (let z = 0; z < 16; z++) {
            for (let x = 0; x < 16; x++) g.setVoxel(x, 0, z);
        }
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 0, 8), 0,
            'a 1-thick sheet at the grid floor has at most 9 of 27');
    });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/voxel-majority.test.mjs
```

Expected: FAIL — cannot resolve `../src/lib/voxel/majority.js`.

- [ ] **Step 3: Implement `majority.ts`**

Create `src/lib/voxel/majority.ts`:

```ts
import { SparseVoxelGrid } from './sparse-voxel-grid';

/**
 * Options for {@link majorityFilterGrid}.
 */
type MajorityOptions = {
    /** Occupied voxels required among the 27, self included, to be on. Default: 14 */
    threshold?: number;
    /** Passes to run. Default: 2 */
    iterations?: number;
    /**
     * Inner chunk edge in voxels. Lower it to bound peak memory or to force many
     * chunks in tests; the result is identical either way. Default: 256
     */
    chunkInner?: number;
};

/**
 * Result of {@link majorityFilterGrid}.
 */
type MajorityResult = {
    /** The filtered grid. */
    grid: SparseVoxelGrid;
    /** Voxels turned on. */
    added: number;
    /** Voxels turned off. */
    removed: number;
    /**
     * Voxels that reached the threshold but were blocked by the candidate mask.
     * The audit trail for the anti-fabrication guarantee.
     */
    gateRejected: number;
};

/**
 * Regularize a voxel surface with a 3x3x3 majority filter.
 *
 * A voxel ends up occupied when at least `threshold` of the 27 voxels in its
 * neighbourhood — itself included — are occupied. Turning a voxel **on** also
 * requires it to be set in `candidate`; turning one **off** does not, since
 * removal cannot fabricate structure.
 *
 * Counting is separable, so each pass is three linear sweeps (X then Z then Y)
 * over a dense chunk rather than 27 taps per voxel. Chunks carry a halo equal to
 * the iteration count, which makes every chunk's interior exact and the result
 * independent of `chunkInner`.
 *
 * @param grid - Grid to filter. **Consumed**: do not reuse it after the call.
 * @param candidate - Voxels permitted to be turned on. Not modified.
 * @param options - Threshold, pass count and chunk size.
 * @returns The filtered grid with counts of voxels added and removed.
 */
const majorityFilterGrid = (
    grid: SparseVoxelGrid,
    candidate: SparseVoxelGrid,
    options: MajorityOptions = {}
): MajorityResult => {
    const { threshold = 14, iterations = 2, chunkInner = 256 } = options;
    const { nx, ny, nz } = grid;

    let current = grid;
    let added = 0;
    let removed = 0;
    let gateRejected = 0;

    // Each pass reads the whole previous state, so passes are sequential and
    // every chunk of a pass sees the same input grid.
    for (let iter = 0; iter < iterations; iter++) {
        const next = new SparseVoxelGrid(nx, ny, nz);

        // One voxel of reach per pass; a single pass is applied per chunk here,
        // so a halo of 1 makes each chunk's interior exact.
        const halo = 1;
        const step = Math.max(4, chunkInner);

        for (let cz = 0; cz < nz; cz += step) {
            for (let cy = 0; cy < ny; cy += step) {
                for (let cx = 0; cx < nx; cx += step) {
                    const innerX = Math.min(step, nx - cx);
                    const innerY = Math.min(step, ny - cy);
                    const innerZ = Math.min(step, nz - cz);

                    // Outer region includes the halo, clamped to the grid.
                    const ox = Math.max(0, cx - halo);
                    const oy = Math.max(0, cy - halo);
                    const oz = Math.max(0, cz - halo);
                    const ex = Math.min(nx, cx + innerX + halo);
                    const ey = Math.min(ny, cy + innerY + halo);
                    const ez = Math.min(nz, cz + innerZ + halo);
                    const ow = ex - ox;
                    const oh = ey - oy;
                    const od = ez - oz;

                    const src = new Uint8Array(ow * oh * od);
                    let anyOccupied = false;
                    for (let z = 0; z < od; z++) {
                        for (let y = 0; y < oh; y++) {
                            const rowBase = y * ow + z * ow * oh;
                            for (let x = 0; x < ow; x++) {
                                if (current.getVoxel(ox + x, oy + y, oz + z)) {
                                    src[rowBase + x] = 1;
                                    anyOccupied = true;
                                }
                            }
                        }
                    }
                    if (!anyOccupied) continue;

                    // Separable counting. sumX[i] counts the 3-window along X,
                    // sumZ adds the Z window on top, sumY the Y window: 27 total.
                    const sumX = new Uint8Array(src.length);
                    for (let z = 0; z < od; z++) {
                        for (let y = 0; y < oh; y++) {
                            const base = y * ow + z * ow * oh;
                            for (let x = 0; x < ow; x++) {
                                let s = src[base + x];
                                if (x > 0) s += src[base + x - 1];
                                if (x + 1 < ow) s += src[base + x + 1];
                                sumX[base + x] = s;
                            }
                        }
                    }
                    const sumZ = new Uint8Array(src.length);
                    const zStride = ow * oh;
                    for (let z = 0; z < od; z++) {
                        for (let y = 0; y < oh; y++) {
                            const base = y * ow + z * zStride;
                            for (let x = 0; x < ow; x++) {
                                let s = sumX[base + x];
                                if (z > 0) s += sumX[base + x - zStride];
                                if (z + 1 < od) s += sumX[base + x + zStride];
                                sumZ[base + x] = s;
                            }
                        }
                    }

                    // Y pass folded into the decision, so no third buffer.
                    for (let z = 0; z < od; z++) {
                        const gz = oz + z;
                        if (gz < cz || gz >= cz + innerZ) continue;
                        for (let y = 0; y < oh; y++) {
                            const gy = oy + y;
                            if (gy < cy || gy >= cy + innerY) continue;
                            const base = y * ow + z * zStride;
                            for (let x = 0; x < ow; x++) {
                                const gx = ox + x;
                                if (gx < cx || gx >= cx + innerX) continue;

                                let count = sumZ[base + x];
                                if (y > 0) count += sumZ[base + x - ow];
                                if (y + 1 < oh) count += sumZ[base + x + ow];

                                const was = src[base + x] === 1;
                                const on = count >= threshold;
                                if (on) {
                                    if (was) {
                                        next.setVoxel(gx, gy, gz);
                                    } else if (candidate.getVoxel(gx, gy, gz)) {
                                        next.setVoxel(gx, gy, gz);
                                        added++;
                                    } else {
                                        gateRejected++;
                                    }
                                } else if (was) {
                                    removed++;
                                }
                            }
                        }
                    }
                }
            }
        }

        current.releaseStorage();
        current = next;
    }

    return { grid: current, added, removed, gateRejected };
};

export { majorityFilterGrid, type MajorityOptions, type MajorityResult };
```

The halo is 1, not `iterations`, because a single majority pass is applied per chunk visit and
the loop re-chunks for every pass. That keeps each chunk's interior exact with the smallest
possible halo, at the cost of re-extracting the dense buffer once per pass — a fair trade given
`getVoxel` is O(1) and the alternative is an `iterations`-wide halo plus in-chunk iteration.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
npx tsx --test test/voxel-majority.test.mjs
```

Expected: PASS, 12 tests. The chunk-equivalence test is the load-bearing one — it compares
`chunkInner: 8` against `chunkInner: 256` on a 32³ random grid over two iterations, so any halo
error changes the result and fails it.

- [ ] **Step 5: Export from both barrels**

Add to `src/lib/voxel/index.ts`:

```ts
export { majorityFilterGrid } from './majority';
export type { MajorityOptions, MajorityResult } from './majority';
```

Then add `majorityFilterGrid` to the value list and `MajorityOptions`, `MajorityResult` to the
type list of the `./voxel` re-export in `src/lib/index.ts:93-98`, as Task 3 did for `growGrid`.
Verify:

```bash
npx tsx -e "import('./src/lib/index.ts').then(m => { if (typeof m.majorityFilterGrid !== 'function') throw new Error('not exported'); console.log('public barrel OK'); })"
```

Expected: prints `public barrel OK`.

- [ ] **Step 6: Full suite and lint**

```bash
npm test && npm run lint
```

Expected: PASS, no new lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/lib/voxel/majority.ts src/lib/voxel/index.ts src/lib/index.ts test/voxel-majority.test.mjs
git commit -m "feat: add majorityFilterGrid for voxel surface regularization

A voxel ends up on when at least threshold of its 27-voxel neighbourhood is
occupied. Additions are gated by the candidate mask; removals are not, since
removing cannot fabricate structure.

Runs over chunked dense buffers with separable counting -- X then Z then a
Y fold into the decision -- so a pass is three linear sweeps instead of 27
taps per voxel. Chunks carry a 1-voxel halo, which a test pins by comparing
chunkInner 8 against 256 over two iterations on a random grid."
```

---

## Task 5: `despeckleGrid`

Remove 6-connected components of occupied voxels below `minVoxels`. On `urban.spz` the raw grid
holds 13,602 components with 21% of all voxels in islands under 64 voxels; this is what takes the
final count to tens.

One pass: flood from each unvisited occupied voxel, collecting members into a reusable buffer
capped at `minVoxels` entries. Under the cap the whole component is in hand and gets cleared;
over it, the component is a keeper and the buffer is discarded. Every occupied voxel is visited
once, so the pass is O(occupied) with O(minVoxels) scratch.

**Files:**
- Create: `src/lib/voxel/despeckle.ts`
- Modify: `src/lib/voxel/index.ts`
- Test: `test/voxel-despeckle.test.mjs`

**Interfaces:**
- Consumes: `SparseVoxelGrid.clearVoxel` (Task 1).
- Produces:
  - `type DespeckleOptions = { minVoxels?: number }` — default 64.
  - `type DespeckleResult = { grid: SparseVoxelGrid; removed: number; components: number; componentsRemoved: number }`.
  - `despeckleGrid(grid, options?): DespeckleResult`. Consumes and mutates `grid` in place, returning the same instance — unlike the other two primitives, because clearing is done through `clearVoxel` and no copy is needed. Document that clearly.

- [ ] **Step 1: Write the failing tests**

Create `test/voxel-despeckle.test.mjs`:

```javascript
/**
 * Tests for connected-component despeckling.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { despeckleGrid } from '../src/lib/voxel/despeckle.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';

const countVoxels = (g) => {
    let n = 0;
    g.forEachOccupiedVoxel(() => n++);
    return n;
};

// A solid axis-aligned box, inclusive bounds.
const box = (g, x0, y0, z0, x1, y1, z1) => {
    for (let z = z0; z <= z1; z++) {
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) g.setVoxel(x, y, z);
        }
    }
};

describe('despeckleGrid', function () {
    it('removes a single isolated voxel', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 0);
        assert.strictEqual(res.removed, 1);
        assert.strictEqual(res.componentsRemoved, 1);
    });

    it('removes a 27-voxel island at minVoxels 64', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        box(g, 4, 4, 4, 6, 6, 6);           // 3x3x3 = 27
        assert.strictEqual(countVoxels(g), 27);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(countVoxels(res.grid), 0);
        assert.strictEqual(res.removed, 27);
    });

    it('keeps a 125-voxel island at minVoxels 64', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        box(g, 4, 4, 4, 8, 8, 8);           // 5x5x5 = 125
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(countVoxels(res.grid), 125);
        assert.strictEqual(res.removed, 0);
        assert.strictEqual(res.componentsRemoved, 0);
    });

    it('keeps exactly at the threshold and removes one below', function () {
        const atLimit = new SparseVoxelGrid(16, 16, 16);
        box(atLimit, 0, 0, 0, 3, 3, 3);     // 4x4x4 = 64
        const keep = despeckleGrid(atLimit, { minVoxels: 64 });
        assert.strictEqual(countVoxels(keep.grid), 64, '64 is not below 64');

        const below = new SparseVoxelGrid(16, 16, 16);
        box(below, 0, 0, 0, 3, 3, 3);
        below.clearVoxel(3, 3, 3);          // 63
        const drop = despeckleGrid(below, { minVoxels: 64 });
        assert.strictEqual(countVoxels(drop.grid), 0);
    });

    it('removes small islands and keeps large ones together', function () {
        const g = new SparseVoxelGrid(32, 32, 32);
        box(g, 2, 2, 2, 8, 8, 8);           // 343, keep
        g.setVoxel(20, 20, 20);             // 1, drop
        box(g, 24, 24, 24, 25, 25, 25);     // 8, drop
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(countVoxels(res.grid), 343);
        assert.strictEqual(res.removed, 9);
        assert.strictEqual(res.components, 3);
        assert.strictEqual(res.componentsRemoved, 2);
    });

    it('counts a component spanning many blocks as one', function () {
        // A 1-voxel-wide bar 20 long crosses five 4-wide blocks.
        const g = new SparseVoxelGrid(32, 32, 32);
        for (let x = 4; x < 24; x++) g.setVoxel(x, 8, 8);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.components, 1);
        assert.strictEqual(res.removed, 20, 'a 20-voxel bar is below 64');
    });

    it('treats diagonal contact as disconnected', function () {
        // Two voxels touching only at a corner are two components.
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(4, 4, 4);
        g.setVoxel(5, 5, 5);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.components, 2);
        assert.strictEqual(res.removed, 2);
    });

    it('is a no-op at minVoxels 0', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = despeckleGrid(g, { minVoxels: 0 });
        assert.strictEqual(countVoxels(res.grid), 1);
        assert.strictEqual(res.removed, 0);
    });

    it('handles a fully solid grid as one component', function () {
        const g = new SparseVoxelGrid(8, 8, 8);
        box(g, 0, 0, 0, 7, 7, 7);
        const res = despeckleGrid(g, { minVoxels: 64 });
        assert.strictEqual(res.components, 1);
        assert.strictEqual(countVoxels(res.grid), 512);
    });

    it('defaults minVoxels to 64', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        box(g, 4, 4, 4, 6, 6, 6);           // 27
        const res = despeckleGrid(g);
        assert.strictEqual(countVoxels(res.grid), 0);
    });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
npx tsx --test test/voxel-despeckle.test.mjs
```

Expected: FAIL — cannot resolve `../src/lib/voxel/despeckle.js`.

- [ ] **Step 3: Implement `despeckle.ts`**

Create `src/lib/voxel/despeckle.ts`:

```ts
import { SparseVoxelGrid } from './sparse-voxel-grid';

/**
 * Options for {@link despeckleGrid}.
 */
type DespeckleOptions = {
    /**
     * Components with fewer than this many voxels are removed. 64 is one 4x4x4
     * block, the grid's natural island unit. 0 disables removal. Default: 64
     */
    minVoxels?: number;
};

/**
 * Result of {@link despeckleGrid}.
 */
type DespeckleResult = {
    /** The despeckled grid — the same instance that was passed in. */
    grid: SparseVoxelGrid;
    /** Voxels removed. */
    removed: number;
    /** Connected components found. */
    components: number;
    /** Components removed for being under the threshold. */
    componentsRemoved: number;
};

/**
 * Remove small disconnected islands of voxels.
 *
 * Labels 6-connected components of the occupied set and clears every component
 * holding fewer than `minVoxels` voxels. One pass: each component is flooded
 * once, with members collected into a buffer capped at `minVoxels`, so a
 * component that exceeds the cap is recognised as a keeper without the buffer
 * ever growing. Every occupied voxel is visited exactly once.
 *
 * Unlike the other cleanup stages this mutates `grid` in place and returns that
 * same instance, because clearing needs no copy.
 *
 * @param grid - Grid to despeckle. **Mutated in place.**
 * @param options - Minimum component size to keep.
 * @returns The grid with counts of voxels and components removed.
 */
const despeckleGrid = (
    grid: SparseVoxelGrid,
    options: DespeckleOptions = {}
): DespeckleResult => {
    const { minVoxels = 64 } = options;
    if (minVoxels <= 0) {
        return { grid, removed: 0, components: 0, componentsRemoved: 0 };
    }

    const { nx, ny, nz } = grid;
    const visited = new SparseVoxelGrid(nx, ny, nz);

    // BFS ring buffer of packed voxel coordinates, grown geometrically. Packing
    // into one number keeps the queue a flat typed array.
    let queue = new Int32Array(1 << 12);
    // Members of the component in progress, capped: past the cap the component
    // is known to be a keeper and the list is no longer needed.
    const members = new Int32Array(minVoxels);

    const pack = (x: number, y: number, z: number): number => x + y * nx + z * nx * ny;

    let removed = 0;
    let components = 0;
    let componentsRemoved = 0;

    // Voxels of every sub-threshold component, cleared after the walk. Deferring
    // the clears keeps `grid` unmodified while `forEachOccupiedVoxel` iterates
    // its `types`/`masks`, and avoids materialising a seed list of every
    // occupied voxel -- which on a real scene is millions of entries.
    const toClear: number[] = [];

    grid.forEachOccupiedVoxel((sx, sy, sz) => {
        if (visited.getVoxel(sx, sy, sz)) return;

        components++;
        visited.setVoxel(sx, sy, sz);
        queue[0] = pack(sx, sy, sz);
        let head = 0;
        let tail = 1;
        let size = 0;

        while (head < tail) {
            const v = queue[head++];
            const x = v % nx;
            const y = ((v / nx) | 0) % ny;
            const z = (v / (nx * ny)) | 0;
            if (size < minVoxels) members[size] = v;
            size++;

            for (let d = 0; d < 6; d++) {
                const dx = d === 0 ? 1 : d === 1 ? -1 : 0;
                const dy = d === 2 ? 1 : d === 3 ? -1 : 0;
                const dz = d === 4 ? 1 : d === 5 ? -1 : 0;
                const ax = x + dx;
                const ay = y + dy;
                const az = z + dz;
                if (ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz) continue;
                if (!grid.getVoxel(ax, ay, az)) continue;
                if (visited.getVoxel(ax, ay, az)) continue;
                visited.setVoxel(ax, ay, az);
                if (tail === queue.length) {
                    const bigger = new Int32Array(queue.length * 2);
                    bigger.set(queue);
                    queue = bigger;
                }
                queue[tail++] = pack(ax, ay, az);
            }
        }

        if (size < minVoxels) {
            for (let i = 0; i < size; i++) toClear.push(members[i]);
            removed += size;
            componentsRemoved++;
        }
    });

    for (let i = 0; i < toClear.length; i++) {
        const v = toClear[i];
        grid.clearVoxel(v % nx, ((v / nx) | 0) % ny, (v / (nx * ny)) | 0);
    }

    visited.releaseStorage();
    return { grid, removed, components, componentsRemoved };
};

export { despeckleGrid, type DespeckleOptions, type DespeckleResult };
```

Two notes. Clears are deferred to after the walk for two reasons: `clearVoxel` would otherwise
mutate the `types`/`masks` arrays that `forEachOccupiedVoxel` is walking, and driving the outer
loop straight off `forEachOccupiedVoxel` avoids materialising a seed list of every occupied voxel
— millions of entries on a real scene. `toClear` is bounded by the voxels in sub-threshold
components, about 60k on `urban.spz`. And the queue holds the whole component, so peak memory is
proportional to the largest component — for a scene whose
shell is one big component that is the shell's voxel count as `Int32Array`, about 4 bytes per
voxel. Acceptable, and bounded by the grid-size guard from the prerequisite plan.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
npx tsx --test test/voxel-despeckle.test.mjs
```

Expected: PASS, 10 tests.

- [ ] **Step 5: Export from both barrels**

Add to `src/lib/voxel/index.ts`:

```ts
export { despeckleGrid } from './despeckle';
export type { DespeckleOptions, DespeckleResult } from './despeckle';
```

Then add `despeckleGrid` to the value list and `DespeckleOptions`, `DespeckleResult` to the type
list of the `./voxel` re-export in `src/lib/index.ts:93-98`, as Tasks 3 and 4 did. Verify:

```bash
npx tsx -e "import('./src/lib/index.ts').then(m => { if (typeof m.despeckleGrid !== 'function') throw new Error('not exported'); console.log('public barrel OK'); })"
```

Expected: prints `public barrel OK`.

- [ ] **Step 6: Full suite and lint**

```bash
npm test && npm run lint
```

Expected: PASS, no new lint errors.

- [ ] **Step 7: Commit**

```bash
git add src/lib/voxel/despeckle.ts src/lib/voxel/index.ts src/lib/index.ts test/voxel-despeckle.test.mjs
git commit -m "feat: add despeckleGrid to remove small voxel islands

Labels 6-connected components of the occupied set and clears any holding
fewer than minVoxels voxels, defaulting to 64 -- one 4x4x4 block. Single
pass: members go into a buffer capped at minVoxels, so a component that
exceeds the cap is recognised as a keeper without the buffer growing, and
every occupied voxel is visited exactly once.

Mutates in place via clearVoxel, unlike the other cleanup stages, since
removal needs no copy."
```

---

## Remaining work

This plan delivers the primitives as exported, tested library functions. Two plans follow, to be
written after this one lands:

- **Integration plan.** The `cleanupGrid` orchestrator and the dial mapping
  (`r = max(1, round(voxelCleanup / voxelResolution))`), the candidate mask's second
  `voxelizeToBuffer` pass at `CANDIDATE_CUTOFF = 0.002`, `writeVoxel` wiring, the
  `--voxel-cleanup` / `--voxel-cleanup-fill` CLI surface with `grow` as default, the `info`-level
  scatter metric, README and JSDoc, and the acceptance run against `urban.spz` targeting
  components <= 100, roughness <= 4.0 and 0 fabricated voxels.
- **`close` mode plan.** The GPU erode mode in the dilation pipeline, `closeGrid`, and
  `cleanupPad` on the grid bounds. Purely additive; `grow` keeps working throughout.
