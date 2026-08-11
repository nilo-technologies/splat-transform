# Sheet-Aware Majority Removal

Addendum to `specs/2026-08-10-voxel-cleanup-design.md`, amending **Stage 2 — `majority`**. The
rest of that design — the candidate mask, the anti-fabrication guarantee, `grow`, `despeckle`,
the dial mapping, the pipeline placement — is unchanged in code. One consequence reaches
`despeckle`'s *role* without touching its implementation, and is recorded below.

## Problem

The 3x3x3 majority filter regularizes the surface by removing every occupied voxel whose
27-neighbourhood count falls under `MAJORITY_THRESHOLD = 14`. On thin structure that test is
not a roughness test, it is a thickness test: a 1-voxel-thick sheet can never reach 14, because
the two out-of-plane layers are empty by construction. In-plane it tops out at 9 of 27.

So the filter does not smooth thin sheets, it deletes them. Walls, roof decks, fences and any
other single-voxel-thick surface disappear wherever the earlier stages failed to thicken them
to two voxels. This is visible on the rooftops scene as lost silhouette — whole roof planes
gone, not merely rougher.

The same test also rounds off solid geometry, because a convex edge or corner is under threshold
too. Measured on a dense reference implementation at `threshold = 14`:

| input | today, 1 pass | today, 2 passes |
| --- | --- | --- |
| 10x10 1-thick sheet, grid interior | 100 -> **0** | 100 -> 0 |
| solid 6x6x6 cube, grid interior | 216 -> 160 | 216 -> 136 |

The sheet is not thinned, it is annihilated in a single pass. The cube loses all 12 edges and
8 corners.

It is also why the measured pipeline in the parent design lands at 202,947 occupied voxels
against a 288,184 baseline. Part of that reduction is the intended removal of scatter; part of
it is thin structure being erased, and the two are indistinguishable in the current counters.

The current behaviour is deliberately pinned by two tests, so this change must invert them:

- `test/voxel-majority.test.mjs:76-88` — "erodes a slab edge that borders out-of-grid"
- `test/voxel-majority.test.mjs:180-190` — "treats out-of-grid as empty", asserting a 1-thick
  sheet at the grid floor is erased

Worth noting that the parent design's own test plan asked for "Preserves a flat plane exactly".
A pure density threshold cannot deliver that, and the two tests above are where the
implementation quietly settled for the opposite. This change makes the code meet the original
intent rather than inventing a new one.

## Solution

Gate removals on local connectivity instead of on neighbourhood density alone.

New `MajorityOptions.keepFaceNeighbors`, default `3`. An occupied voxel is removed only when
**both** conditions hold:

- its 27-neighbourhood count is below `threshold` (as today), and
- it has fewer than `keepFaceNeighbors` occupied face-adjacent neighbours.

Additions are untouched: still `count >= threshold` **and** set in `candidate`. The
anti-fabrication guarantee is unaffected — this change only removes less, never adds more.

### Why 3

On a 1-voxel-thick sheet a voxel has at most 4 face neighbours, all in-plane. That gives a
clean separation between "part of a surface" and "sticking out of one":

| structure | occupied face neighbours | outcome at `keepFaceNeighbors = 3` |
| --- | --- | --- |
| sheet interior | 4 | kept |
| sheet straight edge | 3 | kept |
| sheet convex corner | 2 | removed |
| isolated voxel | 0 | removed |
| 1-voxel bump on a surface | 1 | removed |
| stick / pole interior | 2 | removed |
| 2x2 bump patch on a surface | 3 | **kept** — see tradeoffs |

A threshold of 4 keeps sheet interiors but shaves a full 1-voxel ring off every free border per
pass: the same 10x10 sheet goes 100 -> 64 -> 36. A threshold of 2 keeps the sheet whole
(100 -> 100) but also keeps sticks — a 1x1x3 stick on a slab loses only its tip. 3 is the value
that keeps sheets and solids whole while still shaving protrusions:

| input | `keep=2` | **`keep=3`** | `keep=4` | today |
| --- | --- | --- | --- | --- |
| 10x10 sheet, 1 pass | 100 | **96** | 64 | 0 |
| 10x10 sheet, 2 passes | 100 | **88** | 36 | 0 |
| solid 6^3 cube, 2 passes | 216 | **216** | — | 136 |
| 1x1x3 stick on a slab | 2 of 3 kept | **0 of 3 kept** | 0 of 3 | 0 of 3 |

The cube row is a second win beyond thin sheets: at `keep=3` a solid convex edge (4 face
neighbours) and corner (3) both survive, so the filter stops rounding off solid geometry.

### Implementation

Six boundary-checked taps into the dense `src` chunk buffer in `majority.ts`, read in the same
decision loop that already folds the Y pass (`src/lib/voxel/majority.ts:139-170`).

- `src` is the pre-pass snapshot, the same buffer `count` is derived from, so the face test and
  the density test see identical state.
- The retained voxel must be **written**: today's `else if (was) { removed++; }` branch only
  counts, since `next` starts empty and a removal is expressed by not writing. The gate turns
  that branch into `next.setVoxel(gx, gy, gz); kept++`. Without the write, retention silently
  does nothing.
- The existing `halo = 1` already covers all six face neighbours of every inner voxel, so
  chunk-size independence is preserved and the existing equivalence test keeps its meaning.
- A neighbour outside the outer buffer is out-of-grid — with `halo = 1` the outer region is
  clamped only at the grid boundary — and counts as **empty**, matching the convention the
  density count already uses.

Only reached for voxels that are occupied and under threshold, so the cost is six byte loads on
the small minority of voxels that are removal candidates today.

`MajorityResult` gains `kept`: voxels that were under threshold but survived the face-neighbour
gate. This is the audit counter for the change — it is exactly the set of voxels the old filter
would have deleted. Like `added` and `removed` it **accumulates over passes**, so a two-pass run
on a shape that keeps re-presenting the same candidates reports double the single-pass figure
(an interior solid 6^3 cube: 56 at `iterations: 1`, 112 at 2). Single-pass figures for reference:
144 for a full-extent slab on a 16^3 grid, 56 for that cube (its 12 edges and 8 corners), 252 for
a 1-thick 16x16 floor sheet.

### Interaction with `despeckle`

Retaining chunky structure means `majority` no longer pre-shrinks small floating islands, so
`despeckle` becomes their only backstop. A free-floating solid 4x4x4 blob is eroded 64 -> 8 by
two of today's passes and then dropped by `despeckle` at `DESPECKLE_MIN_VOXELS = 64`; under the
new rule it stays at 64 (its corners have 3 face neighbours) and, being exactly at the threshold
rather than below it, survives the pipeline.

This is consistent with the intent — the same property that keeps a real 4^3 detail is the one
that keeps a 4^3 blob — but it means the size threshold now does this work alone, where it
previously inherited a partly-eroded input. `DESPECKLE_MIN_VOXELS` stays at 64; a 3^3 blob (27
voxels) is still removed. If real scenes show surviving small blobs, the dial to turn is
`DESPECKLE_MIN_VOXELS`, not `keepFaceNeighbors`.

### Wiring

- `src/lib/voxel/cleanup.ts` — new `MAJORITY_KEEP_FACE_NEIGHBORS = 3` constant passed through;
  `CleanupStats` gains `majorityKept`.
- `src/lib/writers/write-voxel.ts:726` — the cleanup log line reports the kept count alongside
  `+added/-removed`.
- JSDoc, per AGENTS.md, since `MajorityOptions` and `MajorityResult` are both public
  (`src/lib/index.ts:99`): the new `keepFaceNeighbors` and `kept` members get doc comments, and
  `majorityFilterGrid`'s own description — which currently states the removal rule as pure
  density (`majority.ts:38-41`) — is corrected, as is `CleanupStats.majorityKept`.

No structural change to the pipeline: stage order, the `grow` / `close` / `both` / `none` modes
and the dial mapping are all as the parent design specifies.

## Tradeoffs

Stated plainly. The numbers below are measured on a dense reference implementation of the rule
(16^3 grids, `threshold = 14`, `keepFaceNeighbors = 3`), not estimated.

- **Convex corners chamfer, and the chamfer advances one diagonal per pass.** A corner voxel has
  2 face neighbours and goes; that exposes its two in-plane neighbours at 2, which go on the next
  pass. On a 10x10 1-thick sheet the loss per convex corner is 1 voxel after 1 pass, 3 after 2,
  6 after 3 — triangular growth, a 45-degree bevel. At the production `iterations = 2` that is
  3 voxels per corner, against the whole sheet today. Free straight edges do not erode at all
  (3 face neighbours), so the effect is strictly cornerwise and bounded. It does mean raising
  `iterations` is disproportionately expensive at corners; if that ever matters, the fix is to
  freeze voxels the gate kept in an earlier pass, which costs another grid of state and is not
  worth it until measurement says so.
- **2x2 bump clumps on a surface survive**, at 1 and at 2 passes. Each of the four voxels has 2
  in-plane neighbours plus 1 into the surface below = 3. Shell roughness on `urban.spz` may
  therefore measure slightly *worse* than the parent design's 3.33 voxels. Retaining a real
  1-thick roof is worth more than removing a 2x2 pimple, but the number should be reported
  honestly, not hidden.
- **Poles and sticks are still shaved**, exactly as today: a 1x1x3 stick on a slab is removed
  entirely in one pass. Thin vertical members — railings, antennae, lamp posts — have 2 face
  neighbours and are not protected by this change. If they matter, that is separate work.

## Test plan

Four existing sites change — three in `test/voxel-majority.test.mjs`, one in
`test/voxel-cleanup.test.mjs` — and six cases are added.

Inverted pins. All expected values below are measured against a dense reference implementation
of the rule, so the tests encode observed behaviour rather than predicted behaviour:

- `:76-88` "erodes a slab edge that borders out-of-grid" becomes a retention pin: the grid
  corner `(0,8,0)` of `slab(16,4,11)` survives, because it has 4 occupied face neighbours (+x,
  +z, ±y) even though its density count is 12 of 27. Its `res.removed > 0` assertion inverts to
  `res.removed === 0`: the minimum face-neighbour count anywhere in that slab is 3, at the
  bottom grid corner `(0,4,0)`.
- `:180-190` "treats out-of-grid as empty" becomes a retention pin: the 1-thick sheet at the
  grid floor keeps its centre `(8,0,8)` and 252 of its 256 voxels. The 4 removals are the
  sheet's own convex corners at 2 face neighbours, so the test asserts `removed === 4` and names
  them — retention of the surface, not of every voxel.
- `:62-74` "preserves the interior of a thick slab" keeps every assertion but its comment is now
  wrong — it explains that full-extent edge columns "DO erode". Correct the comment; do not
  weaken the test.
- `test/voxel-cleanup.test.mjs:27-33` — the `slab` helper's comment justifies avoiding a thin
  sheet because "a thin sheet gets erased wholesale by majority regardless of what grow does",
  which is precisely the behaviour being inverted. The helper stays (its column holes are still
  what those tests need) but the rationale must be rewritten, and the new end-to-end sheet case
  below lives in this same file.

The out-of-grid-as-empty convention still needs a pin, and moves to the addition path, which
this change does not touch: a dent cleared at the grid corner of a slab is **not** filled (its
count is 11 of 27, under threshold) while an interior dent is (26 of 27). That pins the
convention through behaviour that remains gated by density alone.

New cases:

- A 10x10 1-voxel-thick sheet in the grid interior keeps 96 of 100 voxels after one pass and 88
  after two, with the losses asserted to be exactly the corner bevels. Pins both the fix and its
  measured cost.
- The same sheet with scattered 1-voxel holes survives end-to-end through `cleanupGrid`
  (`test/voxel-cleanup.test.mjs`), which is the regression that matters for the rooftops scene.
- A 2x2 bump patch on a slab survives one pass and two. Documents the tradeoff as intended
  behaviour so a future reader does not "fix" it silently.
- A 1x1x3 stick attached to a slab is removed entirely in one pass.
- An interior solid 6x6x6 cube is preserved exactly (216 voxels, `removed === 0`) through two
  passes, where today it erodes to 136. Pins the stop-rounding-solids effect.
- `kept` counts the retained removal candidates, asserted on a **single** pass since the counter
  accumulates: 56 for that cube at `iterations: 1` (112 at 2), and 0 for a lone isolated voxel,
  which reports `removed === 1`. Exercises the audit counter in both directions.
- A free-floating solid 3x3x3 blob is still removed end-to-end by `cleanupGrid`, while a 4x4x4
  one now survives. Pins the `despeckle` interaction above, including that the size threshold is
  the thing keeping the guarantee.

Unchanged and expected to still pass: the isolated-voxel and 1-voxel-bump removals, the dent
fill, the candidate-gate tests, and the untouched-candidate test. The two chunk-equivalence tests
keep their assertions and gain one line each: `kept` must match across chunk sizes too, since it
is per-voxel derived state and would expose a halo mistake that `added` and `removed` could miss.

The risk flagged in review — "iterates: two passes differ from one on a noisy volume"
(`:142-171`), where the 0.6-density block has a mean face-neighbour count of 3.6 and most voxels
are now retained — has been checked and holds: 3028 voxels after one pass against 3119 after two.
Additions still differ between passes. No new discriminator is needed and the test is left alone.

## Landing order

1. This change: sheet-aware majority removal, its wiring and its tests.
2. Manual confirmation on the rooftops scene with the user's existing command — visually
   confirm silhouette retention before going further.
3. The `close` plan as already written, `specs/2026-08-10-voxel-cleanup-close-plan.md` tasks
   1-4, unchanged: erode shader mode with row-tail fill, `gpuErode3`, `closeGrid`,
   `sparseAndGrids`, async `cleanupGrid`, `cleanupPad` in `write-voxel`.
4. Re-measure `grow` against `close` with sheet-aware removal in place, and update the README.
   `close` is expected to seal the wide 3x3-and-larger perforations `grow` provably cannot; the
   parent design's measurement table should be extended, not replaced, so the effect of this
   change is legible.

Step 1 lands first because it is a small change to already-shipped code that alters what every
later measurement means. Measuring `close` against a filter that erases thin sheets would
compare the wrong baseline.

## Out of scope

- Protecting sticks and poles. Would need a different descriptor than face-neighbour count —
  a local thickness or curvature estimate — and risks keeping genuine scatter.
- Making `threshold` adaptive to local thickness. The face-neighbour gate is the cheap
  approximation of that idea; a real one belongs with the RANSAC planarity work the parent
  design already lists as out of scope.
- Freezing gate-kept voxels across passes to stop the corner bevel advancing. Needs another
  grid of state, and would also protect scatter that a later pass could legitimately remove.
  Revisit only if corner loss shows up in the rooftops measurement.
- Any change to `grow`, the candidate mask, or the dial mapping. `despeckle` keeps
  `DESPECKLE_MIN_VOXELS = 64` as well; the interaction noted above is a consequence to observe on
  real scenes, not a reason to retune the threshold pre-emptively.
