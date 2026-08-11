# Sheet-Aware Majority Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the 3x3x3 majority filter from deleting 1-voxel-thick surfaces and rounding solid
edges, by gating removals on occupied face-neighbour count as well as neighbourhood density.

**Architecture:** One rule change inside the existing separable majority pass in
`src/lib/voxel/majority.ts`: a voxel under `threshold` is removed only if it also has fewer than
`keepFaceNeighbors` (default 3) occupied face-adjacent neighbours, read as six boundary-checked
taps into the dense chunk buffer the pass already builds. Additions are untouched, so the
candidate-mask anti-fabrication guarantee is unaffected. The new `kept` counter is threaded to
`CleanupStats.majorityKept` and the cleanup log line. A new `tools/voxel-metrics.mjs` decodes the
emitted octree so the README's measurement table can be re-derived from a real run.

**Tech Stack:** TypeScript (ES2022, ESM), Node's built-in test runner via `tsx`, ESLint with
`@playcanvas/eslint-config`, WebGPU through the `webgpu` (Dawn) dev dependency for CLI runs.

**Spec:** `specs/2026-08-10-voxel-cleanup-sheet-aware-design.md`, which amends Stage 2 of
`specs/2026-08-10-voxel-cleanup-design.md`.

## Global Constraints

- `src/lib/` must stay platform-agnostic: no `node:*` imports, no Node-only APIs. Only
  `src/lib/workers/` is exempt. `tools/` and `test/` are Node-only and may import freely.
- Run the whole suite with `npm test` (`node --import tsx --test test/*.test.mjs`). A single file:
  `node --import tsx --test test/voxel-majority.test.mjs`.
- Run `npm run lint` (`eslint src`) before each commit. It does not cover `test/` or `tools/`;
  match surrounding style there by hand.
- Public API needs JSDoc per `AGENTS.md`: description, `@param` with description, `@returns`.
  `MajorityOptions`, `MajorityResult`, `CleanupStats` are all exported from `src/lib/index.ts:99`.
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`.
- Do not commit `dist/` or `docs/` build output.
- 4-space indent, single quotes, semicolons, arrow-function consts — match each file's neighbours.
- **Never adjust an expected test value to make a test pass.** Every number in this plan was
  measured against a dense reference implementation of the rule. A mismatch means the
  implementation is wrong, or the reference was; report the discrepancy with actual figures
  instead of editing the assertion.

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `src/lib/voxel/majority.ts` | modify | The rule: `keepFaceNeighbors` option, six face taps, `kept` counter |
| `src/lib/voxel/cleanup.ts` | modify | Pass the constant, surface `majorityKept` in `CleanupStats` |
| `src/lib/writers/write-voxel.ts` | modify (line 726) | Report kept voxels in the cleanup log line |
| `test/voxel-majority.test.mjs` | modify | Invert 2 pins, fix 1 comment, add 7 cases, 1 equivalence line |
| `test/voxel-cleanup.test.mjs` | modify | Fix the stale helper comment, add 3 end-to-end cases |
| `tools/voxel-metrics.mjs` | create | Decode `.voxel.json`/`.bin`, report occupied/islands/largest/roughness/scatter |
| `test/voxel-metrics.test.mjs` | create | Roundtrip pin: grid -> octree -> decode -> identical voxel set |
| `README.md` | modify (~299, 311-313) | Correct the majority description and re-measured table |

Task order: the rule and its pins first (Task 1), the remaining majority pins (Task 2), the
orchestrator wiring and end-to-end pins (Task 3), the log line (Task 4), the measurement tool
(Task 5), then the real-scene measurement and docs (Task 6). Tasks 1-4 are the shippable change;
5-6 produce the evidence and the docs.

---

## Task 1: The sheet-aware removal rule

**Files:**
- Modify: `src/lib/voxel/majority.ts`
- Test: `test/voxel-majority.test.mjs:62-88`, `:180-190`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `majorityFilterGrid(grid, candidate, options)` where `options` gains
  `keepFaceNeighbors?: number` (default `3`) and the returned `MajorityResult` gains
  `kept: number`. Full result shape after this task:
  `{ grid: SparseVoxelGrid, added: number, removed: number, kept: number, gateRejected: number }`.

- [ ] **Step 1: Invert the two pins that assert the current erasure behaviour**

In `test/voxel-majority.test.mjs`, replace the whole `it('erodes a slab edge that borders
out-of-grid', ...)` block at lines 76-88 with:

```javascript
    it('keeps a slab edge that borders out-of-grid', function () {
        // Inverted deliberately: this used to assert 'erodes a slab edge that
        // borders out-of-grid'. The density count at the grid corner (0,8,0) is
        // still 12 of 27 -- (2/3)*(2/3)*1*27, since X and Z each lose a third to
        // out-of-grid while Y stays interior to the slab -- but the voxel has 4
        // occupied face neighbours (+x, +z, and both Y), so the sheet-aware gate
        // spares it. No voxel of a solid slab has fewer than 3: the minimum sits
        // at the bottom grid corner (0,4,0), with +x, +z and +y.
        const res = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(0, 8, 0), 1, 'the edge must survive');
        assert.strictEqual(res.removed, 0, 'a solid slab has nothing removable');
        assert.ok(res.kept >= 1, 'the spared under-threshold voxels must be audited');
    });
```

Then replace the whole `it('treats out-of-grid as empty', ...)` block at lines 180-190 with:

```javascript
    it('keeps a 1-thick sheet at the grid floor', function () {
        // Inverted deliberately: this used to assert 'treats out-of-grid as
        // empty' by requiring the sheet be erased. The convention is unchanged
        // -- the below-neighbours are outside the grid and count as empty, so
        // the density count peaks at 9 of 27 -- but every interior voxel has 4
        // in-plane face neighbours and survives. Only the sheet's own 4 convex
        // corners, at 2 face neighbours, go. The convention itself is now pinned
        // through the addition path by 'treats out-of-grid as empty when adding'.
        const g = new SparseVoxelGrid(16, 16, 16);
        for (let z = 0; z < 16; z++) {
            for (let x = 0; x < 16; x++) g.setVoxel(x, 0, z);
        }
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(8, 0, 8), 1, 'the surface must survive');
        assert.strictEqual(countVoxels(res.grid), 252, '256 less the 4 corners');
        assert.strictEqual(res.removed, 4);
        for (const [x, z] of [[0, 0], [15, 0], [0, 15], [15, 15]]) {
            assert.strictEqual(res.grid.getVoxel(x, 0, z), 0, `corner ${x},${z} has 2 faces`);
        }
    });
```

- [ ] **Step 2: Run the two tests and verify they fail**

Run: `node --import tsx --test test/voxel-majority.test.mjs`

Expected: both new tests FAIL against the current implementation — `keeps a slab edge...` fails on
`getVoxel(0, 8, 0)` being `0`, and `keeps a 1-thick sheet at the grid floor` fails on
`getVoxel(8, 0, 8)` being `0`. Everything else in the file passes.

- [ ] **Step 3: Add the option and the counter to the types**

In `src/lib/voxel/majority.ts`, add to `MajorityOptions` after the `chunkInner` member (line 15):

```typescript
    /**
     * Occupied face-adjacent neighbours that spare an under-threshold voxel from
     * removal. A voxel on a 1-voxel-thick sheet has at most 4, all in-plane, so
     * 3 keeps sheet interiors and straight edges while still shaving bumps (1),
     * scatter (0-2) and stick tips (2). Out-of-grid neighbours count as empty.
     * Set 0 to remove on the density test alone. Default: 3
     */
    keepFaceNeighbors?: number;
```

Add to `MajorityResult` after `removed` (line 27):

```typescript
    /**
     * Voxels under `threshold` that the face-neighbour gate spared: exactly the
     * set a density-only filter would have deleted. Accumulates over
     * `iterations`, as `added` and `removed` do.
     */
    kept: number;
```

- [ ] **Step 4: Correct the function's own JSDoc**

In `src/lib/voxel/majority.ts`, replace the first paragraph of the `majorityFilterGrid` doc
comment (lines 38-41) with:

```typescript
 * A voxel stays occupied when at least `threshold` of the 27 voxels in its
 * neighbourhood — itself included — are occupied, **or** when it has at least
 * `keepFaceNeighbors` occupied face-adjacent neighbours. The second clause is
 * what keeps thin surfaces: a 1-voxel-thick sheet can never reach a threshold
 * above 9, so a density-only rule deletes sheets rather than smoothing them,
 * and rounds the convex edges off solid volumes for the same reason.
 *
 * Turning a voxel **on** requires it to be set in `candidate`; turning one
 * **off** does not, since removal cannot fabricate structure.
```

- [ ] **Step 5: Implement the gate**

In `src/lib/voxel/majority.ts`, extend the destructure at line 59:

```typescript
    const { threshold = 14, iterations = 2, chunkInner = 256, keepFaceNeighbors = 3 } = options;
```

Add the counter beside `removed` (line 64):

```typescript
    let kept = 0;
```

Replace the `} else if (was) {` branch in the decision loop (lines 165-167) with:

```typescript
                                } else if (was) {
                                    // Sheet-aware removal. `src` is the pre-pass
                                    // snapshot the density count came from, so
                                    // both tests see identical state, and halo=1
                                    // puts all six neighbours of every inner
                                    // voxel inside this buffer. A neighbour
                                    // outside it is out-of-grid — the outer
                                    // region is only clamped at the grid
                                    // boundary — and counts as empty, matching
                                    // the density count's convention.
                                    let faceCount = 0;
                                    if (x > 0 && src[base + x - 1]) faceCount++;
                                    if (x + 1 < ow && src[base + x + 1]) faceCount++;
                                    if (y > 0 && src[base + x - ow]) faceCount++;
                                    if (y + 1 < oh && src[base + x + ow]) faceCount++;
                                    if (z > 0 && src[base + x - zStride]) faceCount++;
                                    if (z + 1 < od && src[base + x + zStride]) faceCount++;
                                    if (faceCount >= keepFaceNeighbors) {
                                        // Must write: `next` starts empty, so a
                                        // removal is expressed by not writing.
                                        // Counting alone would drop the voxel.
                                        next.setVoxel(gx, gy, gz);
                                        kept++;
                                    } else {
                                        removed++;
                                    }
                                }
```

Update the return statement (line 179):

```typescript
    return { grid: current, added, removed, kept, gateRejected };
```

- [ ] **Step 6: Run the file and verify everything passes**

Run: `node --import tsx --test test/voxel-majority.test.mjs`

Expected: PASS, all tests. The pre-existing cases that must still pass unchanged are `removes an
isolated voxel`, `removes a 1-voxel bump on a slab`, `fills a 1-voxel dent in a slab`, `preserves
the interior of a thick slab`, both candidate-gate tests, `produces the same result whatever the
chunk size`, `iterates: two passes differ from one on a noisy volume`, and `leaves the candidate
grid untouched`.

- [ ] **Step 7: Fix the now-wrong comment on the thick-slab test**

The test at lines 62-74 still passes, but its comment explains that full-extent edge columns "DO
erode", which is no longer true. Replace the comment lines inside
`it('preserves the interior of a thick slab', ...)` — the three lines beginning
`// The slab spans the full x/z extent` — with:

```javascript
        // The slab spans the full x/z extent, so its edge columns border
        // out-of-grid (counted empty) and see only 12 of 27. They survive
        // anyway: the sheet-aware gate spares any voxel with >= 3 occupied face
        // neighbours, which 'keeps a slab edge that borders out-of-grid' pins
        // directly. Interior voxels are untouched under either rule.
```

Leave every assertion in that test alone.

- [ ] **Step 8: Run the full suite and lint**

Run: `npm test`
Expected: PASS. `test/voxel-cleanup.test.mjs` exercises `cleanupGrid`, which calls
`majorityFilterGrid`, so watch it specifically — if `removes an isolated speck` or `reduces to a
single component on a speckled slab` fails, stop and report: single-voxel specks have 0 face
neighbours and must still be removed.

Run: `npm run lint`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/lib/voxel/majority.ts test/voxel-majority.test.mjs
git commit -m "fix: keep thin surfaces in the voxel majority filter

The removal rule was a thickness test, not a roughness test: a
1-voxel-thick sheet cannot reach 14 of 27, so the filter deleted thin
surfaces instead of smoothing them, and rounded the convex edges off
solid volumes for the same reason.

Removals are now gated on connectivity too -- a voxel goes only if it is
under threshold AND has fewer than keepFaceNeighbors (default 3) occupied
face neighbours. Additions are untouched, so the candidate-mask
guarantee is unaffected: this only removes less. MajorityResult.kept
audits exactly what a density-only pass would have deleted.

The two tests that pinned the old erasure are inverted, and the
out-of-grid-as-empty convention they carried is unchanged -- it is now
pinned through the addition path instead."
```

---

## Task 2: Retention, tradeoff and audit pins

**Files:**
- Modify: `test/voxel-majority.test.mjs`

**Interfaces:**
- Consumes: `majorityFilterGrid` with `keepFaceNeighbors` and `kept` from Task 1.
- Produces: no source changes; these are characterization pins for later tasks to rely on.

These tests assert behaviour Task 1 already implements, so they pass on first run. That is the
point: they pin the measured cost of the change so nobody later "fixes" the tradeoffs silently.
If any expected number disagrees, Task 1 is wrong — report it, do not edit the number.

- [ ] **Step 1: Add a sheet helper beside the existing `slab` helper**

In `test/voxel-majority.test.mjs`, after the `slab` helper (line 36), add:

```javascript
// A 1-voxel-thick sheet at height y, spanning [x0,x1] x [z0,z1].
const sheet = (n, y, x0, x1, z0, z1) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = z0; z <= z1; z++) {
        for (let x = x0; x <= x1; x++) g.setVoxel(x, y, z);
    }
    return g;
};

// A solid axis-aligned box, inclusive bounds.
const box = (n, x0, x1, y0, y1, z0, z1) => {
    const g = new SparseVoxelGrid(n, n, n);
    for (let z = z0; z <= z1; z++) {
        for (let y = y0; y <= y1; y++) {
            for (let x = x0; x <= x1; x++) g.setVoxel(x, y, z);
        }
    }
    return g;
};
```

- [ ] **Step 2: Add the retention, tradeoff and audit cases**

Insert these before the final `it('leaves the candidate grid untouched', ...)` test:

```javascript
    it('keeps an interior 1-thick sheet, bevelling only its corners', function () {
        // A 10x10 sheet well inside the grid: 100 voxels, of which the 4 convex
        // corners have 2 face neighbours and go. The bevel then advances one
        // diagonal per pass, because each removal exposes two new 2-neighbour
        // voxels, so the loss per corner is 1 after one pass and 3 after two.
        const one = majorityFilterGrid(sheet(16, 8, 3, 12, 3, 12), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(countVoxels(one.grid), 96, '100 less 1 voxel per corner');
        for (const [x, z] of [[3, 3], [12, 3], [3, 12], [12, 12]]) {
            assert.strictEqual(one.grid.getVoxel(x, 8, z), 0, `corner ${x},${z}`);
        }
        assert.strictEqual(one.grid.getVoxel(3, 8, 8), 1, 'straight edge stays');
        assert.strictEqual(one.grid.getVoxel(8, 8, 8), 1, 'interior stays');

        const two = majorityFilterGrid(sheet(16, 8, 3, 12, 3, 12), allCandidate(16),
            { threshold: 14, iterations: 2 });
        assert.strictEqual(countVoxels(two.grid), 88, '3 voxels per corner after two passes');
        assert.strictEqual(two.grid.getVoxel(3, 8, 8), 1, 'straight edges never erode');
    });

    it('preserves a solid interior cube exactly', function () {
        // Today's rule rounds this to 136 over two passes: a convex edge sees 12
        // of 27 and a corner 8, both under threshold. With the gate, an edge has
        // 4 face neighbours and a corner 3, so the cube is untouched.
        const res = majorityFilterGrid(box(16, 5, 10, 5, 10, 5, 10), allCandidate(16),
            { threshold: 14, iterations: 2 });
        assert.strictEqual(countVoxels(res.grid), 216);
        assert.strictEqual(res.removed, 0);
        assert.strictEqual(res.grid.getVoxel(5, 5, 5), 1, 'corner');
        assert.strictEqual(res.grid.getVoxel(5, 8, 5), 1, 'edge');
    });

    it('audits the voxels it spared', function () {
        // `kept` accumulates over passes, like added and removed: the cube
        // presents the same 8 corners and 48 edge voxels to every pass.
        const one = majorityFilterGrid(box(16, 5, 10, 5, 10, 5, 10), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(one.kept, 56, '8 corners + 48 edge voxels');
        const two = majorityFilterGrid(box(16, 5, 10, 5, 10, 5, 10), allCandidate(16),
            { threshold: 14, iterations: 2 });
        assert.strictEqual(two.kept, 112, 'per-pass counter, summed over 2 passes');

        const slabRes = majorityFilterGrid(slab(16, 4, 11), allCandidate(16),
            { threshold: 14, iterations: 1 });
        assert.strictEqual(slabRes.kept, 144, 'the full-extent slab edge columns');
    });

    it('reports nothing kept when there is nothing to spare', function () {
        const g = new SparseVoxelGrid(16, 16, 16);
        g.setVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.kept, 0, '0 face neighbours is not a surface');
        assert.strictEqual(res.removed, 1);
    });

    it('keeps a 2x2 bump patch on a slab: the documented tradeoff', function () {
        // Each of the four voxels has 2 in-plane neighbours plus the slab below
        // = 3, so the patch survives where a single-voxel bump does not. This is
        // the price of keeping 1-thick sheets and it is intended: do not "fix"
        // it without re-reading the design spec's tradeoffs section.
        const g = slab(16, 6, 9);
        for (const [x, z] of [[6, 6], [7, 6], [6, 7], [7, 7]]) g.setVoxel(x, 10, z);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 2 });
        for (const [x, z] of [[6, 6], [7, 6], [6, 7], [7, 7]]) {
            assert.strictEqual(res.grid.getVoxel(x, 10, z), 1, `patch voxel ${x},${z}`);
        }
    });

    it('still shaves a stick off a slab', function () {
        // A 1x1x3 stick: the tip has 1 face neighbour and the shaft 2, so the
        // whole thing goes in one pass. Poles and railings are not protected.
        const g = slab(16, 6, 9);
        for (let y = 10; y <= 12; y++) g.setVoxel(6, y, 6);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        for (let y = 10; y <= 12; y++) {
            assert.strictEqual(res.grid.getVoxel(6, y, 6), 0, `stick voxel y=${y}`);
        }
        assert.strictEqual(res.grid.getVoxel(6, 9, 6), 1, 'the slab top stays');
    });

    it('treats out-of-grid as empty when adding', function () {
        // The convention the old erosion pin carried, moved to the addition
        // path, which this change does not touch. A dent at the grid corner sees
        // 11 of 27 -- two axes lose a third to out-of-grid and the dent itself
        // is empty -- so it stays open, while an interior dent at 26 fills.
        const g = slab(16, 4, 11);
        g.clearVoxel(0, 8, 0);
        g.clearVoxel(8, 8, 8);
        const res = majorityFilterGrid(g, allCandidate(16), { threshold: 14, iterations: 1 });
        assert.strictEqual(res.grid.getVoxel(0, 8, 0), 0, 'grid-corner dent stays open');
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 1, 'interior dent fills');
        assert.strictEqual(res.added, 1, 'exactly the interior dent');
    });
```

- [ ] **Step 3: Pin `kept` in the chunk-equivalence test**

`kept` is per-voxel derived state, so a halo mistake in the face taps would show up as a chunk-size
dependence that `added` and `removed` could both miss. In
`it('produces the same result whatever the chunk size', ...)`, after
`assert.strictEqual(small.removed, big.removed);` (line 139) add:

```javascript
        assert.strictEqual(small.kept, big.kept,
            'the face taps must read the halo, not the chunk edge');
```

- [ ] **Step 4: Run the file**

Run: `node --import tsx --test test/voxel-majority.test.mjs`
Expected: PASS, including all seven new tests.

- [ ] **Step 5: Commit**

```bash
git add test/voxel-majority.test.mjs
git commit -m "test: pin sheet retention, solid preservation and the kept counter

Characterization pins for the sheet-aware removal rule: a 10x10 interior
sheet keeps 96 of 100 after one pass and 88 after two with only its
corners bevelled, a solid 6^3 cube is preserved exactly where it used to
erode to 136, and kept reports 56 per pass on that cube (112 over two,
since it accumulates) and 0 for a lone voxel.

Also pins the two tradeoffs as intended behaviour so they are not
silently 'fixed': 2x2 bump patches survive, sticks do not. The
chunk-equivalence test now compares kept, which is the assertion that
would catch a face tap reading the chunk edge instead of the halo."
```

---

## Task 3: Thread `majorityKept` through `cleanupGrid`

**Files:**
- Modify: `src/lib/voxel/cleanup.ts`
- Test: `test/voxel-cleanup.test.mjs`

**Interfaces:**
- Consumes: `majorityFilterGrid(...)` returning `kept`, from Task 1.
- Produces: `CleanupStats` gains `majorityKept: number`; `cleanupGrid` passes
  `keepFaceNeighbors: MAJORITY_KEEP_FACE_NEIGHBORS` (3). Task 4 reads `stats.majorityKept`.

- [ ] **Step 1: Write the failing test**

In `test/voxel-cleanup.test.mjs`, add inside `describe('cleanupGrid', ...)`, before
`it('rejects a non-positive strength', ...)`:

```javascript
    it('reports the thin-surface voxels it spared', function () {
        // The slab's full-extent edge columns are under the majority threshold
        // and spared by the face-neighbour gate, so the counter must be live and
        // non-zero on ordinary input.
        const n = 24;
        const res = cleanupGrid(slab(n, 4, 11, []), allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.ok(res.stats.majorityKept > 0,
            'kept must be surfaced in the stats, not swallowed');
    });
```

- [ ] **Step 2: Run it and verify it fails**

Run: `node --import tsx --test test/voxel-cleanup.test.mjs`
Expected: FAIL — `res.stats.majorityKept` is `undefined`, so the `> 0` assertion fails.

- [ ] **Step 3: Add the constant, the stat and the JSDoc**

In `src/lib/voxel/cleanup.ts`, after `MAJORITY_ITERATIONS` (line 21) add:

```typescript
/**
 * Occupied face neighbours that spare an under-threshold voxel from the majority
 * filter. A 1-voxel-thick sheet tops out at 4, so 3 keeps sheets and solid
 * convex edges whole while still shaving bumps, scatter and stick tips.
 */
const MAJORITY_KEEP_FACE_NEIGHBORS = 3;
```

Add to `CleanupStats` after `majorityRemoved` (line 68):

```typescript
    /**
     * Voxels the majority filter spared for being part of a thin surface: what a
     * density-only filter would have deleted. Large on scenes with genuine
     * 1-voxel-thick structure such as roof decks and fences.
     */
    majorityKept: number;
```

Pass the option in the `majorityFilterGrid` call (lines 156-159):

```typescript
    const maj = majorityFilterGrid(current, candidate, {
        threshold: MAJORITY_THRESHOLD,
        iterations: MAJORITY_ITERATIONS,
        keepFaceNeighbors: MAJORITY_KEEP_FACE_NEIGHBORS
    });
```

And report it in the returned stats, after `majorityRemoved: maj.removed,` (line 173):

```typescript
            majorityKept: maj.kept,
```

Export the constant by adding `MAJORITY_KEEP_FACE_NEIGHBORS,` to the export block at the bottom of
the file, immediately after `CANDIDATE_CUTOFF,`. Then add it to the value export list in
`src/lib/voxel/index.ts:22`, which currently reads
`export { CANDIDATE_CUTOFF, cleanupGrid, cleanupRadius } from './cleanup';`, so that it becomes:

```typescript
export { CANDIDATE_CUTOFF, MAJORITY_KEEP_FACE_NEIGHBORS, cleanupGrid, cleanupRadius } from './cleanup';
```

Finally add `MAJORITY_KEEP_FACE_NEIGHBORS` to the alphabetical value export list in
`src/lib/index.ts` — it sits in the `from './voxel'` group that begins
`alignGridBounds, applyAlignYaw, carve, cleanupGrid, cleanupRadius, despeckleGrid,` at line 94.
Insert it in alphabetical position within that list.

- [ ] **Step 4: Run the test and verify it passes**

Run: `node --import tsx --test test/voxel-cleanup.test.mjs`
Expected: PASS, all tests in the file.

- [ ] **Step 5: Fix the stale rationale on the `slab` helper**

The helper's comment at `test/voxel-cleanup.test.mjs:27-33` justifies avoiding a thin sheet with
"a thin sheet gets erased wholesale by majority regardless of what grow does", which is exactly the
behaviour being changed. Replace the comment block above `const slab = (n, y0, y1, holes) => {`
with:

```javascript
// A thick slab (y in [y0,y1]) spanning the grid's XZ interior, with full-depth
// column holes at the listed (x,z) positions. Thickness is what makes the
// majority filter's density test reachable here: a 1-voxel-thick sheet caps out
// at 9 of 27 and only survives via the face-neighbour gate, which
// 'keeps a 1-thick sheet with holes end to end' below covers separately.
```

- [ ] **Step 6: Add the end-to-end sheet case**

The regression that matters for real scenes: a thin surface must survive the whole pipeline, not
just the majority stage. Add after the test from Step 1:

```javascript
    it('keeps a 1-thick sheet with holes end to end', function () {
        // 20x20 sheet at y=8 with three 1-voxel holes. grow fills them (each has
        // 4 occupied face neighbours), majority keeps the surface and bevels its
        // 4 corners by 3 voxels each over 2 passes, despeckle keeps it as one
        // 388-voxel component. Under the old rule the sheet was erased outright.
        const n = 24;
        const g = new SparseVoxelGrid(n, n, n);
        for (let z = 2; z <= 21; z++) {
            for (let x = 2; x <= 21; x++) g.setVoxel(x, 8, z);
        }
        for (const [x, z] of [[8, 8], [12, 15], [15, 9]]) g.clearVoxel(x, 8, z);
        const res = cleanupGrid(g, allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(res.grid.getVoxel(8, 8, 8), 1, 'hole filled');
        assert.strictEqual(res.grid.getVoxel(11, 8, 11), 1, 'sheet interior alive');
        assert.strictEqual(res.grid.getVoxel(2, 8, 11), 1, 'straight edge alive');
        assert.strictEqual(res.stats.components, 1, 'one surface, not debris');
        assert.strictEqual(countVoxels(res.grid), 388, '400 less 3 voxels per corner');
        assert.ok(res.stats.majorityKept > 0);
    });
```

- [ ] **Step 7: Add the despeckle-interaction cases**

`majority` no longer pre-shrinks chunky islands, so the size threshold is now their only backstop.
Pin both ends of that. Add after the previous test:

```javascript
    it('still drops a small floating blob', function () {
        // A solid 3x3x3 blob survives majority now -- its corners have 3 face
        // neighbours -- so despeckle is what removes it, at 27 < 64 voxels.
        const n = 24;
        const g = new SparseVoxelGrid(n, n, n);
        for (let z = 10; z <= 12; z++) {
            for (let y = 10; y <= 12; y++) {
                for (let x = 10; x <= 12; x++) g.setVoxel(x, y, z);
            }
        }
        const res = cleanupGrid(g, allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(countVoxels(res.grid), 0, 'the blob must go');
        assert.strictEqual(res.stats.componentsRemoved, 1);
    });

    it('keeps a blob at the despeckle threshold, documenting the interaction', function () {
        // A solid 4x4x4 blob is exactly DESPECKLE_MIN_VOXELS, so it survives.
        // Under the old rule majority eroded it to 8 voxels first and despeckle
        // then dropped it. The size threshold now carries this alone: if real
        // scenes show surviving blobs, DESPECKLE_MIN_VOXELS is the dial, not
        // keepFaceNeighbors.
        const n = 24;
        const g = new SparseVoxelGrid(n, n, n);
        for (let z = 10; z <= 13; z++) {
            for (let y = 10; y <= 13; y++) {
                for (let x = 10; x <= 13; x++) g.setVoxel(x, y, z);
            }
        }
        const res = cleanupGrid(g, allCandidate(n),
            { strength: 0.2, voxelResolution: 0.1, fill: 'grow' });
        assert.strictEqual(countVoxels(res.grid), 64);
        assert.strictEqual(res.stats.componentsRemoved, 0);
    });
```

- [ ] **Step 8: Run the suite and lint**

Run: `npm test`
Expected: PASS.

Run: `npm run lint`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add src/lib/voxel/cleanup.ts src/lib/voxel/index.ts src/lib/index.ts test/voxel-cleanup.test.mjs
git commit -m "feat: report majorityKept from cleanupGrid

cleanupGrid passes keepFaceNeighbors explicitly, next to the threshold
and iteration constants it already spells out, and surfaces the spared
count as CleanupStats.majorityKept.

End-to-end pins: a 1-thick sheet with holes now survives the whole
pipeline (grow fills, majority keeps, despeckle keeps one component)
where it used to be erased. Also pins the despeckle interaction -- a 3^3
blob is still dropped on size, a 4^3 blob at the threshold survives
where majority used to erode it first -- and rewrites the slab helper's
comment, which justified itself with the erasure being removed."
```

---

## Task 4: Report kept voxels in the cleanup log

**Files:**
- Modify: `src/lib/writers/write-voxel.ts:726`

**Interfaces:**
- Consumes: `CleanupStats.majorityKept` from Task 3.
- Produces: no API surface; log text only.

- [ ] **Step 1: Extend the log line**

In `src/lib/writers/write-voxel.ts`, the cleanup log at lines 725-729 currently reads:

```typescript
            logger.info(
                `cleanup: radius ${s.radius} voxels, +${fmtCount(s.grown)} grown, ` +
                `+${fmtCount(s.majorityAdded)}/-${fmtCount(s.majorityRemoved)} smoothed, ` +
                `-${fmtCount(s.despeckled)} despeckled ` +
                `(${fmtCount(s.componentsRemoved)} of ${fmtCount(s.components)} islands)`);
```

Replace the `smoothed` line with two lines so the kept count reads as part of smoothing:

```typescript
                `+${fmtCount(s.majorityAdded)}/-${fmtCount(s.majorityRemoved)} smoothed ` +
                `(${fmtCount(s.majorityKept)} thin-surface voxels kept), ` +
```

- [ ] **Step 2: Verify on a real scene**

Run:

```bash
node bin/cli.mjs ./scenes/cabin.spz \
  --voxel-params 0.1,0.1 \
  --voxel-cleanup 0.2 \
  -w /tmp/cabin-cleanup.voxel.json 2>&1 | grep -A1 "cleanup:"
```

Expected: a `cleanup:` line that includes `thin-surface voxels kept` with a non-zero count, followed
by the `cleanup gate:` line. If the run fails for lack of a GPU adapter, report that — the
`webgpu` dev dependency provides Dawn, and this same command shape is what the integration plan
used for its acceptance run.

- [ ] **Step 3: Clean up and lint**

```bash
rm -f /tmp/cabin-cleanup.voxel.*
npm run lint
```

Expected: no lint errors.

- [ ] **Step 4: Commit**

```bash
git add src/lib/writers/write-voxel.ts
git commit -m "feat: log the thin-surface voxels cleanup kept

The kept count is the audit trail for sheet-aware removal, in the same
way gateRejected is for the anti-fabrication guarantee: it is exactly
what a density-only majority filter would have deleted."
```

---

## Task 5: `tools/voxel-metrics.mjs`

**Files:**
- Create: `tools/voxel-metrics.mjs`
- Create: `test/voxel-metrics.test.mjs`

**Interfaces:**
- Consumes: nothing from earlier tasks. Reads the octree format written by
  `writeOctreeFiles` (`src/lib/writers/write-voxel.ts:346`) and built by `buildSparseOctree`
  (exported from `src/lib/index.ts:68`).
- Produces: `decodeVoxelFiles(jsonPath)` -> `{ nx, ny, nz, voxels: Set<number>, key(x,y,z) }` and
  `gridMetrics({ nx, ny, nz, voxels })` ->
  `{ occupied, islands, largestShare, roughness, scatter, facesPerVoxel }`. Task 6 runs the CLI
  entry point.

The format, read off `sparse-octree.ts`: `.voxel.bin` is `nodeCount` little-endian `Uint32`
node words followed by `leafDataCount` `Uint32` leaf mask words. Node word encoding is
`((childMask & 0xFF) << 24) | baseOffset`, with three cases — `0xFF000000` exactly
(`SOLID_LEAF_MARKER`, a wholly solid region at this level), a word whose top byte is 0 (a mixed
4x4x4 leaf, low 24 bits index `leafData` in pairs), and anything else (interior node, children
appended in increasing octant order at `baseOffset + popcount(childMask & ((1 << oct) - 1))`).
Octant bit 0 is X, bit 1 is Y, bit 2 is Z (`sparse-octree.ts:527-529`). Node 0 is the root, at
level `treeDepth`, where a level-`li` cell spans `4 << li` voxels per axis. Within a 4x4x4 block
the bit index is `x + (y << 2) + (z << 4)` (`sparse-voxel-grid.ts:154`), low word for bits 0-31.

- [ ] **Step 1: Write the roundtrip test first**

The README's numbers will come out of this decoder, so it needs a pin that does not depend on any
scene or GPU. Create `test/voxel-metrics.test.mjs`:

```javascript
/**
 * Tests for the voxel metrics tool's octree decoder and metric definitions.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';
import { Vec3 } from 'playcanvas';

import { buildSparseOctree } from '../src/lib/writers/sparse-octree.js';
import { SparseVoxelGrid } from '../src/lib/voxel/sparse-voxel-grid.js';
import { decodeOctree, gridMetrics } from '../tools/voxel-metrics.mjs';

// buildSparseOctree needs world bounds; one voxel per unit keeps the mapping
// from grid indices to world coordinates trivial.
const boundsFor = (nx, ny, nz) => ({
    min: new Vec3(0, 0, 0),
    max: new Vec3(nx, ny, nz)
});

const roundtrip = (grid, nx, ny, nz, dense) => {
    // buildSparseOctree has two emitters -- Morton streams, and a dense mip
    // build chosen by shouldUseDenseMipBuild or forced with options.dense
    // (sparse-octree.ts:669). Both write the same node format, so the decoder
    // must handle either; every case below runs through both.
    const octree = buildSparseOctree(grid, boundsFor(nx, ny, nz), boundsFor(nx, ny, nz), 1,
        { dense });
    return decodeOctree({
        nodes: octree.nodes,
        leafData: octree.leafData,
        treeDepth: octree.treeDepth,
        nx,
        ny,
        nz
    });
};

const dump = (voxels) => [...voxels].sort((a, b) => a - b);

// Runs a case through both emitters. `build` must return a fresh grid each
// call: buildSparseOctree may consume the one it is given.
const bothPaths = (name, build, nx, ny, nz, check) => {
    for (const dense of [false, true]) {
        check(roundtrip(build(), nx, ny, nz, dense), `${name} (dense=${dense})`);
    }
};

describe('decodeOctree', function () {
    it('recovers a mixed leaf exactly', function () {
        const expected = new Set();
        const build = () => {
            const g = new SparseVoxelGrid(16, 16, 16);
            expected.clear();
            for (const [x, y, z] of [[0, 0, 0], [1, 2, 3], [15, 15, 15], [4, 0, 9], [7, 7, 7]]) {
                g.setVoxel(x, y, z);
                expected.add(x + y * 16 + z * 16 * 16);
            }
            return g;
        };
        bothPaths('mixed leaf', build, 16, 16, 16, (got, label) => {
            assert.strictEqual(got.nx, 16, label);
            assert.deepStrictEqual(dump(got.voxels), dump(expected), label);
        });
    });

    it('recovers solid leaves and whole solid subtrees', function () {
        // A solid 8x8x8 corner: whole blocks are SOLID and their parents
        // aggregate, so this exercises SOLID_LEAF_MARKER above the leaf level.
        const n = 16;
        const expected = new Set();
        const build = () => {
            const g = new SparseVoxelGrid(n, n, n);
            expected.clear();
            for (let z = 0; z < 8; z++) {
                for (let y = 0; y < 8; y++) {
                    for (let x = 0; x < 8; x++) {
                        g.setVoxel(x, y, z);
                        expected.add(x + y * n + z * n * n);
                    }
                }
            }
            return g;
        };
        bothPaths('solid subtree', build, n, n, n, (got, label) => {
            assert.strictEqual(got.voxels.size, 512, label);
            assert.deepStrictEqual(dump(got.voxels), dump(expected), label);
        });
    });

    it('recovers a sparse shell across many blocks', function () {
        const n = 32;
        const expected = new Set();
        const build = () => {
            const g = new SparseVoxelGrid(n, n, n);
            expected.clear();
            let seed = 7;
            const rnd = () => {
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF;
                return seed / 0x7FFFFFFF;
            };
            for (let z = 0; z < n; z++) {
                for (let y = 0; y < n; y++) {
                    for (let x = 0; x < n; x++) {
                        if (rnd() < 0.1) {
                            g.setVoxel(x, y, z);
                            expected.add(x + y * n + z * n * n);
                        }
                    }
                }
            }
            return g;
        };
        bothPaths('sparse shell', build, n, n, n, (got, label) => {
            assert.deepStrictEqual(dump(got.voxels), dump(expected), label);
        });
    });

    it('recovers an empty grid as no voxels', function () {
        bothPaths('empty', () => new SparseVoxelGrid(16, 16, 16), 16, 16, 16, (got, label) => {
            assert.strictEqual(got.voxels.size, 0, label);
        });
    });
});

describe('gridMetrics', function () {
    it('counts components and the largest share', function () {
        const n = 16;
        const voxels = new Set();
        const key = (x, y, z) => x + y * n + z * n * n;
        // Component A: a 3x3 plate (9 voxels). Component B: 2 voxels, apart.
        for (let z = 2; z <= 4; z++) {
            for (let x = 2; x <= 4; x++) voxels.add(key(x, 8, z));
        }
        voxels.add(key(12, 8, 12));
        voxels.add(key(12, 8, 13));
        const m = gridMetrics({ nx: n, ny: n, nz: n, voxels });
        assert.strictEqual(m.occupied, 11);
        assert.strictEqual(m.islands, 2);
        assert.strictEqual(m.largestShare, 9 / 11);
    });

    it('reports zero roughness for a flat surface and more for a jagged one', function () {
        const n = 16;
        const key = (x, y, z) => x + y * n + z * n * n;
        const flat = new Set();
        for (let z = 2; z <= 12; z++) {
            for (let x = 2; x <= 12; x++) flat.add(key(x, 8, z));
        }
        assert.strictEqual(gridMetrics({ nx: n, ny: n, nz: n, voxels: flat }).roughness, 0);

        const jagged = new Set();
        for (let z = 2; z <= 12; z++) {
            for (let x = 2; x <= 12; x++) jagged.add(key(x, 8 + ((x + z) % 2) * 4, z));
        }
        assert.ok(gridMetrics({ nx: n, ny: n, nz: n, voxels: jagged }).roughness > 3,
            'alternating 4-voxel steps must read as rough');
    });

    it('matches the scatter definition the writer logs', function () {
        // scatterFraction (write-voxel.ts:72) is the share of occupied voxels
        // with at most 2 of 6 face neighbours. A lone voxel has 0.
        const n = 16;
        const voxels = new Set([8 + 8 * n + 8 * n * n]);
        assert.strictEqual(gridMetrics({ nx: n, ny: n, nz: n, voxels }).scatter, 1);
    });
});
```

- [ ] **Step 2: Run it and verify it fails**

Run: `node --import tsx --test test/voxel-metrics.test.mjs`
Expected: FAIL — cannot resolve `../tools/voxel-metrics.mjs`.

- [ ] **Step 3: Write the tool**

Create `tools/voxel-metrics.mjs`:

```javascript
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
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `node --import tsx --test test/voxel-metrics.test.mjs`
Expected: PASS, all seven tests.

If the roundtrip tests fail, the decoder is misreading the format — do not adjust the expected
voxel sets, which come from the grid itself. Re-read `appendMixedLeaf` and the octant loop in
`src/lib/writers/sparse-octree.ts:460-545`, and check whether `buildSparseOctree` took the
streaming path (`sparse-octree.ts:890+`) rather than the dense one for the grid size used.

- [ ] **Step 5: Cross-check the decoder against a real artifact**

The repo ships `scenes/rooftops.voxel.json` and its `.bin`. Run:

```bash
node --import tsx tools/voxel-metrics.mjs scenes/rooftops.voxel.json
```

Expected: a row with a plausible occupied count in the hundreds of thousands to low millions
(`numMixedLeaves` is 360,011, so occupied must be at least that and at most 64x it), `islands`
well above 1, and `largest` under 100%. A decoder bug typically shows as an absurd occupied count
or a near-100% scatter. Record the row in the task report.

- [ ] **Step 6: Run the suite and commit**

Run: `npm test`
Expected: PASS. (`npm run lint` covers `src` only, so it will not see these files; keep the style
consistent with `tools/color-noise-bench.mjs` by hand.)

```bash
git add tools/voxel-metrics.mjs test/voxel-metrics.test.mjs
git commit -m "test: add a voxel metrics tool with a decoder roundtrip pin

The README's cleanup table quotes occupied voxels, islands, largest-
component share and surface roughness, and nothing in the repo could
re-derive them -- the numbers were carried from an out-of-tree harness.
This decodes the emitted octree back into a voxel set and computes all
four, so both rows can be re-measured from real runs and the grow-vs-
close comparison later has a defined yardstick.

Roughness is the tool's own definition (mean deviation of each column's
top voxel from its neighbouring columns' mean), documented in the header
so rows are compared against each other, not against history. Scatter
duplicates scatterFraction from write-voxel deliberately: it must match
the CLI's own log line, which cross-checks the decoder on real data.

Pinned by a grid -> octree -> decode roundtrip over mixed leaves, solid
subtrees, a sparse shell and an empty grid, so the numbers the README
quotes rest on a tested decoder."
```

---

## Task 6: Measure the real scene and update the docs

**Files:**
- Modify: `README.md` (the prose at ~299 and the table at 311-313)

**Interfaces:**
- Consumes: the CLI from Tasks 1-4 and `tools/voxel-metrics.mjs` from Task 5.
- Produces: evidence and documentation. No source changes.

This task produces the honest numbers. The README's table was measured on `urban.spz` cropped to a
40 m box at 10 cm — the same command the integration plan used for its acceptance run — and all
four columns move with this change. Both rows are re-measured with the same tool so they stay
comparable to each other.

- [ ] **Step 0: Build, so the CLI actually runs**

`src/cli/index.ts` only *exports* `main` — nothing invokes it (`src/cli/index.ts:1366`,
`bin/cli.mjs`). Running that module directly, as `npx tsx src/cli/index.ts <args>`, imports it and
exits 0 having done nothing, which looks like a pass and produces no output. Task 4 hit exactly
that. Build first and drive the real entry point:

```bash
npm run build
node bin/cli.mjs --help | head -3
```

Expected: the build exits 0, and `--help` prints the version banner with the current commit hash.
Every CLI invocation below uses `node bin/cli.mjs`. If you change source after building, rebuild
before re-measuring.

- [ ] **Step 1: Measure the baseline, without cleanup**

```bash
node bin/cli.mjs ./scenes/urban.spz \
  --filter-box -20,-20,-20,20,20,20 \
  --voxel-params 0.1,0.1 \
  --auto-rotate -w \
  /tmp/urban-baseline.voxel.json 2>&1 | tee /tmp/urban-baseline.log
```

Record the `surface coherence` line.

- [ ] **Step 2: Measure with cleanup**

```bash
node bin/cli.mjs ./scenes/urban.spz \
  --filter-box -20,-20,-20,20,20,20 \
  --voxel-params 0.1,0.1 \
  --voxel-cleanup 0.2 \
  --auto-rotate -w \
  /tmp/urban-cleaned.voxel.json 2>&1 | tee /tmp/urban-cleaned.log
```

Record the `cleanup:` line, including the new kept count, and the `cleanup gate:` line.

- [ ] **Step 3: Compute the metrics for both**

```bash
node --import tsx tools/voxel-metrics.mjs \
  /tmp/urban-baseline.voxel.json /tmp/urban-cleaned.voxel.json
```

- [ ] **Step 4: Cross-check the decoder against the CLI's own log**

The tool's `scatter` column for the baseline must match the `surface coherence` percentage the
baseline run logged, to within rounding. If it does not, the decoder is wrong and the numbers
cannot be published — stop and report both figures.

- [ ] **Step 5: Check the result against expectations**

Expected direction, from the design spec:

- `occupied` for the cleaned run should now be **higher** than the pre-change 202,947, because thin
  surfaces are retained rather than deleted. It should still be at or below the baseline: cleanup
  removes noise rather than adding bulk, and additions remain candidate-gated.
- `islands` should stay in the same order as before (tens, against 13,602 baseline).
- `roughness` may be slightly worse than the pre-change figure — the 2x2-bump tradeoff — and must
  still be far better than baseline.
- `kept` in the `cleanup:` line must be substantial. It is the thin structure this change exists to
  save.

If `occupied` came out *lower* than 202,947, or `islands` jumped into the thousands, something is
wrong: report the figures rather than publishing them.

- [ ] **Step 6: Confirm the silhouette visually on the rooftops scene**

The measurement above says nothing about whether roof planes look right. Run the pipeline on
`scenes/rooftops.spz` with the flags you normally use for it — if you have no standing command,
use:

```bash
node bin/cli.mjs ./scenes/rooftops.spz \
  --voxel-params 0.1,0.1 \
  --voxel-cleanup 0.2 \
  --collision-voxels /tmp/rooftops-cleaned.vox \
  --auto-rotate -w \
  /tmp/rooftops-cleaned.voxel.json
```

Open the `.vox` output and confirm roof planes and walls are present rather than perforated or
missing. Record the command used and the verdict. The convex-corner bevel is the one cost not yet
checked against real geometry — synthetic sheets have 4 clean corners, a rooftop capture has many
— so look specifically at whether corners read as chamfered.

- [ ] **Step 7: Update the README prose**

At `README.md:298-299` the pipeline description reads:

```
It runs three passes: fill voxels that look like holes in an existing surface, regularize the
surface with a 3x3x3 majority filter, then drop islands smaller than one 4x4x4 block.
```

Replace with:

```
It runs three passes: fill voxels that look like holes in an existing surface, regularize the
surface with a 3x3x3 majority filter, then drop islands smaller than one 4x4x4 block. The
smoothing pass keeps any voxel with at least three occupied face neighbours, so a 1-voxel-thick
roof deck or wall survives it — a pure density test would delete thin surfaces rather than
smooth them, since they can never reach the threshold — while single-voxel bumps, scatter and
thin poles are still shaved off.
```

- [ ] **Step 8: Update the README table**

Replace both data rows at `README.md:312-313` with the measured figures from Step 3, keeping the
column order (occupied voxels, disconnected islands, in the largest, surface roughness). Add this
line immediately after the table's existing "Note the voxel count goes *down*" sentence at line
315:

```
Measured with `tools/voxel-metrics.mjs`, which defines roughness as the mean deviation of each
column's topmost voxel from its neighbours' — compare the rows against each other rather than
against figures from elsewhere.
```

If the cleaned run's occupied count no longer sits below the baseline, the "Note the voxel count
goes *down*" sentence is false and must be corrected to match the measurement, not the other way
around.

- [ ] **Step 9: Clean up and commit**

```bash
rm -f /tmp/urban-baseline.voxel.* /tmp/urban-cleaned.voxel.* /tmp/urban-*.log
rm -f /tmp/rooftops-cleaned.voxel.* /tmp/rooftops-cleaned.vox
git add README.md
git commit -m "docs: re-measure the voxel cleanup table after sheet-aware removal

Sheet-aware removal changes every column, so both rows were re-measured
in one session with tools/voxel-metrics.mjs rather than carried over.
The table now says which tool produced it and how roughness is defined,
because the previous figures came from an out-of-tree harness and were
not reproducible.

Also states what the smoothing pass keeps, since 'regularize the surface
with a 3x3x3 majority filter' gave no hint that it used to delete every
1-voxel-thick surface it touched."
```

- [ ] **Step 10: Report the evidence**

Write into the task report: both CLI log excerpts, the metrics table, the scatter cross-check, the
visual verdict on the rooftops scene, and the before/after comparison against the pre-change
figures (202,947 occupied, 64 islands, 77.6% largest, 3.3 roughness). Call out any column that
moved in an unexpected direction.

---

## Remaining work after this plan

`close` mode, unchanged: `specs/2026-08-10-voxel-cleanup-close-plan.md` tasks 1-4 — the erode
shader mode with its row-tail fill, `gpuErode3`, `closeGrid`, `sparseAndGrids`, async
`cleanupGrid`, and `cleanupPad` in `write-voxel`. Its comparative measurement should use
`tools/voxel-metrics.mjs` from Task 5 so grow and close are measured on one definition, and it
must run after this plan: comparing against a filter that erases thin sheets would measure the
wrong baseline.
