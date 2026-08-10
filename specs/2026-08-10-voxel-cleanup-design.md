# Voxel Cleanup: Density-Gated Hole Filling and Surface Smoothing

**Date:** 2026-08-10
**Scope:** `src/lib/voxel/`, `src/lib/gpu/gpu-voxelization.ts`, `src/lib/gpu/shaders/dilation.ts`, `src/lib/gpu/gpu-dilation.ts`, `src/lib/writers/write-voxel.ts`, `src/lib/types.ts`, `src/lib/index.ts`, `src/cli/index.ts`

## Problem

Voxelizing `scenes/urban.spz` at `--voxel-params 0.1,0.1` produces a structure riddled with
holes and violently bumpy surfaces. Measured on that scene, grid 464x428x500, after the
current pipeline:

| metric | value |
| --- | --- |
| occupied voxels | 288,184 |
| disconnected 6-connected components | 13,602 |
| share of voxels in the largest component | 60.4% |
| components under 64 voxels ("specks") | 13,438, holding 59,847 voxels (21% of all) |
| voxels with <= 2 of 6 face neighbours | 32.8% |
| exposed faces per voxel | 2.61 |
| top-surface height-field roughness | 9.80 voxels (98 cm of jitter between adjacent columns) |
| solid runs per vertical column | 1.61 |

### Root cause

Both symptoms are one cause. The voxelizer thresholds a continuous Gaussian density field
and nothing afterwards reconstructs or regularizes a surface. At 0.1 m the scene's splats
are sub-voxel, so the occupancy set is not a surface at all — it is a scatter of
near-isolated voxels around wherever splat centres happen to land.

| | `urban.spz` @ 0.1 m | `landscape.spz` @ 0.01 m |
| --- | --- | --- |
| median 3-sigma extent, max axis | 0.062 m = **0.62 voxels** | 0.180 m = **18 voxels** |
| splats whose whole 3-sigma extent is under one voxel | **56.9%** | 4.8% |
| splats in region | 191,874 | 500,000 |

Coverage is governed by `splat 3-sigma extent / voxel size`. Above 1 the splats overlap into
a continuous shell; far below 1 they cannot. `landscape.spz` looks correct only because it is
run at a resolution where that ratio is 18, giving a genuinely thick 12.6 M-voxel shell.

Holes are the shell never having been a connected sheet. Bumpiness is the same scatter — the
isolated and dangling voxels *are* the bumps, and they produce the 98 cm height jitter.

Two aggravating factors, neither the primary cause:

- `filterAndFillBlocks` (`src/lib/voxel/block-cleanup.ts:157`) runs unconditionally and only
  ever removes voxels lacking a 6-connected neighbour. On this scene it deletes 10.4% of
  voxels, worsening the holes, and reports it at `debug` only.
- `--filter-cluster` with defaults drops 9.1% of splats (191,874 to 174,406).

### Rejected alternatives, with the measurements that rejected them

- **Morphological opening** to shave bumps: erases the 1-2-voxel-thin sheets that constitute
  the shell. Largest component collapses from 82.2% to 27.5%.
- **Supersample then max-pool** (voxelize at 0.05 or 0.025, pool down): the scatter is
  approximately scale-invariant — voxels with <= 2 neighbours are 32.8% at 0.1 m and 29.9% at
  0.05 m. Finer sampling does not yield a coherent sheet, and costs 8x memory and time per
  halving.
- **Lowering the opacity cutoff**: 0.1 to 0.02 adds 64% more voxels but only moves the
  <= 2-neighbour share from 32.8% to 28.5%. It adds fog, not structure.
- **Ungated morphological closing**: fabricates structure. `close r=2` adds 305,608 voxels,
  and 45.7% of them sit where the Gaussian field contributes under 0.02% opacity — effective
  vacuum. Even `r=1` fabricates 32% of what it adds. This is the reason the design below is
  gated.

## Solution

A post-voxelization cleanup stage over the `SparseVoxelGrid`, off by default, driven by one
strength dial plus a mode selector. Every stage that *adds* voxels is intersected with a
**candidate mask** derived from the same Gaussian density field at a much lower opacity
threshold.

### The anti-fabrication guarantee

The cleanup may only place a voxel where the Gaussian field reaches at least
`CANDIDATE_CUTOFF` opacity. A genuine void — a window opening, the gap between a railing and
a deck, the interior of a sealed building — has no Gaussian density and is therefore
untouchable at any radius or iteration count. Only the space *between* Gaussians that are
already present can be filled.

Removals need no gate; removing a voxel cannot fabricate structure.

Measured effect of the gate on `urban.spz`, where "fabricated" counts voxels placed where the
field contributes under 0.02% opacity:

| variant | occupied | fabricated | faces/vox | components | largest | roughness |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (today) | 288,184 | 0 | 2.61 | 13,602 | 60.4% | 9.80 |
| close r=2, ungated | 593,792 | **139,610** | 1.09 | 2,462 | 96.8% | 5.21 |
| close r=2 ungated + majority + despeckle | 492,472 | **139,610** | 0.85 | 47 | 89.4% | 3.18 |
| close r=2, gated | 416,703 | 0 | 2.01 | 8,837 | 83.1% | 7.26 |
| grow k>=4, gated | 302,931 | 0 | 2.36 | 12,819 | 67.3% | 9.43 |
| grow k>=3, gated | 370,539 | 0 | 1.88 | 10,246 | 73.3% | 8.65 |
| **grow k>=3 + majority + despeckle** | **202,947** | **0** | **1.21** | **64** | 77.6% | **3.33** |
| close r=2 gated + majority + despeckle | 205,000 | 0 | 1.27 | 64 | 85.6% | 3.57 |

Comparing the last two rows against the ungated third row: the guarantee costs 64 components
against 47, and roughness 3.33 against 3.18 — that is, almost nothing. The gated pipeline also
ends *below* the baseline voxel count, so it removes noise rather than adding bulk; no
thickening is involved.

### User-facing surface

```
--voxel-cleanup <metres>       strength; absent or 0 = disabled
--voxel-cleanup-fill <mode>    none | grow | close | both      (default: grow)
```

Modes:

- `grow` — hole filling by neighbour-count region growing (stage 1a).
- `close` — hole filling by density-gated morphological closing (stage 1b).
- `both` — `grow` first, then `close` on its result.
- `none` — skip hole filling entirely and run only the majority filter and despeckle. This is
  the smoothing-and-denoising-only mode, for scenes whose coverage is already adequate.

`--voxel-cleanup-fill` without `--voxel-cleanup`, or alongside `--voxel-cleanup 0`, is an
error: there is no strength to apply. `--voxel-cleanup` without `--voxel-cleanup-fill` uses
`grow`.

`WriteVoxelOptions` and `Options` gain:

```ts
/** Cleanup strength in world units. 0 or undefined disables cleanup entirely. */
voxelCleanup?: number;
/** Hole-filling algorithm. Default: 'grow'. */
voxelCleanupFill?: 'none' | 'grow' | 'close' | 'both';
```

The four stage functions are exported individually from `src/lib/index.ts` with explicit
parameters, so library consumers can compose their own pipeline without the dial:

```ts
growGrid(grid, candidate, { minNeighbors, maxIterations }): SparseVoxelGrid
closeGrid(grid, candidate, gpuDilation, radius): SparseVoxelGrid
majorityFilterGrid(grid, candidate, { threshold, iterations }): SparseVoxelGrid
despeckleGrid(grid, { minVoxels }): SparseVoxelGrid
```

Ownership contract, matching how `writeVoxel` already hands grids down its pipeline: each
stage function **consumes** its input grid and returns a new one. The caller must not reuse the
input afterwards. This keeps peak memory at two grids rather than five, which matters at the
grid sizes these scenes reach. The `candidate` grid is *not* consumed — every stage reads it.

### Dial mapping

```
r = max(1, round(voxelCleanup / voxelResolution))     // defect scale in voxels

grow      : minNeighbors = 3, maxIterations = 2r + 2, early exit when an iteration adds nothing
close     : radius = r
majority  : threshold = 14 of 27, iterations = 2
despeckle : minVoxels = 64
candidate : CANDIDATE_CUTOFF = 0.002 opacity
```

The majority, despeckle and candidate constants do not scale with the dial. Justification:
the majority filter's neighbourhood is structurally 3x3x3; 64 voxels is one 4x4x4 block, the
codebase's natural island unit, and the scatter these thresholds target was measured to be
scale-invariant. All are overridable through the stage functions.

Stage order: fill (`grow`, then `close` when mode is `both`), then majority, then despeckle.
Despeckle must run last because the majority filter both creates and destroys islands.

### Pipeline placement

Cleanup runs in `writeVoxel` immediately after `SparseVoxelGrid.fromBuffer`
(`src/lib/writers/write-voxel.ts:589`) and before `fillExterior`, `fillFloor`, `carve` and the
crop, so the octree, `.vox` and collision GLB all inherit the cleaned grid.

`needsGpuDilation` (`write-voxel.ts:597`) extends to include the `close` and `both` modes.

### Candidate mask

`src/lib/gpu/gpu-voxelization.ts:191-202` already holds `totalSigma` and compares it once.
It gains a second comparison against `CANDIDATE_CUTOFF` and a second `atomicOr` into a
doubled results region. The expensive Gaussian evaluation is unchanged, so the candidate mask
costs no additional voxelization pass. `voxelizeToBuffer` returns both buffers and builds the
candidate one only when cleanup is enabled, so the default path allocates nothing extra.

`filterAndFillBlocks` continues to run on the solid buffer only, unchanged. Every measurement
in this document was taken with it enabled, so removing or gating it would invalidate the
figures above.

Memory: at a 0.002 gate the candidate set runs roughly 3-4x the solid set (793,764 against
288,184 on `urban.spz`). On `landscape.spz` at 0.01 m that is on the order of an extra 80 MB
of block masks. `CANDIDATE_CUTOFF` is the lever if that becomes a problem.

### Grid padding

Only the `close` mode needs it — its dilate step would otherwise clip at the grid edge.
`cleanupPad = (r + 1) * voxelResolution` folds into the `padXZ` / `padY` computation at
`write-voxel.ts:547-559`, and unlike `floorPad` it applies on **all three axes** because
closing is isotropic. `cropToOccupied` trims the pad back off, so output bounds are
unaffected.

`grow` needs no padding: it can never leave the candidate mask, which already lies inside the
3-sigma-derived scene bounds.

### Stage 1a — `grow`

New `src/lib/voxel/grow.ts`, a direct generalization of `block-cleanup.ts`. It reuses that
file's six per-direction neighbour masks unchanged — in-block bit shifts
(`block-cleanup.ts:87-111`) and cross-block faces via `IntKeyMap`
(`block-cleanup.ts:116-153`).

Where `block-cleanup` ORs and ANDs the six masks, `grow` bit-slices a **count**: three bit
planes accumulated with carry-save adders across the six masks (a maximum of 6 needs 3 bits,
about 24 bit operations per 32-bit word). New voxels are
`~occupied & candidate & (count >= minNeighbors)`.

Each iteration reads a snapshot, so the result is independent of block iteration order. The
iteration runs over the **union** of the solid and candidate block sets, because a hole's
block may hold no solid voxels yet.

What `minNeighbors = 3` can and cannot fill, on a 1-voxel-thick sheet where a voxel has at
most 4 in-plane neighbours:

| defect | behaviour |
| --- | --- |
| 1x1 hole | 4 occupied neighbours, filled in 1 iteration |
| 1x2 slit | 3 each, filled in 1 iteration |
| 1x3 slit | ends have 3 and fill first, the middle then has 4, filled in 2 iterations |
| 1xN slit | fills inward from both ends, about N/2 iterations |
| 3x3 square hole | corners have 2, edge middles 1, centre 0 — **never filled** |

That last row is the intended conservatism, and it holds independently of the candidate gate:
`grow` fills gaps *between* neighbouring surface voxels, not open areas. Thicker structures
fill more readily because out-of-plane neighbours also count. This is why `close` exists as an
alternative mode — it is the one that can bridge 2D-wide gaps, and it is why it needs the
candidate gate more urgently.

### Stage 1b — `close`

`A | (erode(dilate(A, r), r) & candidate)`.

- `src/lib/gpu/shaders/dilation.ts`: parameterize `dilateXWgsl` and `dilateYZWgsl` by
  operation (`OR` for dilate, `AND` for erode) and by out-of-grid value (0 for dilate, 1 for
  erode). The shaders are otherwise identical, so this is parameterization rather than
  duplication.
- `src/lib/gpu/gpu-dilation.ts`: pipelines for both variants; `dispatchX` and `dispatchYZ`
  take a mode.
- `src/lib/voxel/dilation.ts`: `gpuErode3` alongside `gpuDilate3`.
- `src/lib/voxel/close.ts`: the union via `sparseOrGrids` (`src/lib/voxel/grid-ops.ts:127`)
  and a new sparse AND.

The erode out-of-grid convention matters: treating outside as empty would shave the outer `r`
voxels off anything touching the grid edge. With `cleanupPad` in place the practical effect
is nil, but the convention is explicit in the shader and pinned by a test so that removing
the pad later cannot introduce a silent boundary bug.

### Stage 2 — `majority`

New `src/lib/voxel/majority.ts`. A 3x3x3 neighbourhood in sparse block form means 6 faces
plus 12 edges plus 8 corners; a chunked dense pass is both simpler and faster.

- Chunk as `src/lib/voxel/dilation.ts` does, with a halo of `iterations` voxels block-aligned
  through `blockAlignedExtent`. That makes each chunk's inner region exact and the result
  independent of chunk order.
- `CHUNK_INNER = 256` rather than dilation's 512: two occupancy buffers plus one count buffer
  at 256^3 is roughly 50 MB, against roughly 400 MB at 512^3.
- Counting is **separable**: an X pass yielding 0-3, a Z pass yielding 0-9, a Y pass yielding
  0-27, all into `Uint8Array`. Three linear passes rather than 27 taps per voxel.
- `out = count >= threshold ? (occupied || candidate) : 0`. Additions gated, removals not.
- Chunks with no occupied blocks are skipped, mirroring `chunkIsEmpty`
  (`src/lib/voxel/dilation.ts:315`).

### Stage 3 — `despeckle`

New `src/lib/voxel/despeckle.ts`. 6-connected component labelling reusing the two-level BFS
in `src/lib/voxel/flood-fill.ts` — a block queue for wholly-solid blocks and a voxel queue for
mixed blocks — whose queues grow geometrically and throw rather than truncate
(`flood-fill.ts:80`).

Two passes bound memory: pass 1 records each component's seed voxel and size; pass 2 re-floods
only the components below `minVoxels` and clears them. Sub-threshold components are by
definition small, so pass 2 is cheap.

## Delivery order

Each group is independently shippable and testable.

1. The two prerequisite bug fixes, in the order listed below.
2. Candidate mask plumbing through the GPU voxelizer and `voxelizeToBuffer`.
3. Stage 1a `grow`, plus its `growGrid` export.
4. Stage 2 `majority` and stage 3 `despeckle`, plus their exports.
5. CLI and `Options` wiring for `--voxel-cleanup` / `--voxel-cleanup-fill`, default `grow`.
   At this point the default mode is fully functional and the acceptance run can be performed.
6. Stage 1b `close`, including the GPU erode mode and `cleanupPad`. This is where the
   `dilation.ts:223` fix from group 1 stops being merely defensive.
7. Observability and documentation.

`close` lands after the CLI wiring deliberately: `grow` is the default and needs no new GPU
code, so the feature is usable and measurable before the shader work begins.

## Prerequisite bug fixes

These land first, each independently testable. A third item, originally reported as a CLI
parsing bug, was retracted on verification and is recorded below for the record.

### 1. `dilation.ts:223` cannot overwrite an occupied block-type field

```
dstTypes[w] |= bt << shift;
```

`|=` can only set bits. That is correct for accumulating *different* blocks into a shared
`types` word — 16 blocks pack into each word at 2 bits apiece — but it cannot overwrite a field
that already holds a value. Chunks are disjoint today, so every field is written at most once
into freshly zeroed memory and the defect is inert. If two chunks ever wrote the same block,
`SOLID (1) | MIXED (2) = 3`, an invalid type. `getVoxel` then falls through to the mask lookup
(`src/lib/voxel/sparse-voxel-grid.ts:151-155`), finds no entry, and reports the whole block as
**empty**. Stage 1b adds a second consumer of this path, so the latent trap is fixed before it
is built upon.

Fix: call the canonical `writeBlockType` (`src/lib/voxel/sparse-voxel-grid.ts:89-93`, already
exported), which clears the field before setting it. **Not** a bare `=`: that would wipe the
other 15 blocks sharing the word.

### 2. Unguarded grid-size limits

Voxelizing unfiltered `landscape.spz` at 0.1 m throws
`RangeError: Invalid typed array length: -2147483648` from `IntKeyMap`
(`src/lib/voxel/block-cleanup.ts:62`, `src/lib/utils/int-key-map.ts:38`). Its 3-sigma scene
bounds span 593.6 x 370.7 x 882.9 m because of outlier splats, giving 3,037,474,944 blocks.

That throw is the *lucky* failure. The same grid also exceeds 2^31 blocks, tripping the silent
`BlockMaskMap` `Int32Array` key overflow (`src/lib/voxel/block-mask-map.ts:12,24,45`), which
loses every MIXED block's mask — that is, every surface block — while SOLID interiors survive.
At 0.01 m the same scene reaches roughly 3.04e12 blocks, past 2^32 into silent `types` word aliasing
(`src/lib/voxel/sparse-voxel-grid.ts:75,90,122`).

Fix: an explicit grid-size guard on the voxel write path, mirroring the one
`src/lib/voxel/filter-cluster.ts:195-199` already applies, raising an actionable error that
names the block count and suggests a filter or a coarser resolution. A better message for the
`IntKeyMap` limit alone would not address the silent variants.

The ceiling is set by the tightest of the three limits. `IntKeyMap` needs its capacity under
2^30 and sizes itself at `blocks / 0.7`, so blocks must stay under `0.7 * 2^30` (about 751.6e6).
`MAX_GRID_BLOCKS = 2^29` (536,870,912) is the clean power of two below that and clears the 2^31
and 2^32 limits as well. For calibration: `urban.spz` is 1.55e6 blocks and a filtered
`landscape.spz` at 0.01 m is 90,453,870 blocks, both far inside.

This never affected the reported `urban.spz` output, whose grid is 1.55e6 blocks.

### 3. Retracted: `--filter-cluster <path>` is not a bug

An earlier draft of this spec claimed that `--filter-cluster ./scenes/scene.voxel.json`
silently swallowed the path and misdirected the output. That claim was wrong and the proposed
warning would have been a regression.

`--filter-cluster` is an optional-value option that consumes its next token only when the token
is numeric (`src/cli/index.ts:196-215`). A path is not numeric, so it falls through to the
positionals and becomes the output file. That is the deliberate, tested idiom for every
optional-value option — `test/cli.test.mjs:457`, "accepts a bare `--auto-rotate` without
swallowing the output argument", exercises exactly this shape with `--auto-rotate null`. A
warning on the pattern would fire on every correct invocation.

No filename form can be eaten by accident either: `./scenes/scene.voxel.json`, `null`,
`out.ply`, `1.ply`, `12.ply` and `2024-scene.ply` all fail `isNumericValue`. Only a digits-only
filename such as `123` would be consumed, and the run then fails on a missing output rather
than misbehaving quietly.

The reported command supplied no other output positional, so `scenes/scene.voxel.json` is
where it asked the output to go. No change.

## Observability

The pipeline is currently silent about exactly what is wrong. Added at `info`:

- After voxelization: the scatter metric — share of occupied voxels with <= 2 of 6 neighbours,
  plus the component count. When the share exceeds 20%, suggest `--voxel-cleanup`.
- `filterAndFillBlocks`' removal count promoted from `debug` (`block-cleanup.ts:189`) when it
  exceeds 5% of voxels.
- Per cleanup stage: voxels added, voxels removed, and **voxels rejected by the candidate
  gate**. That last figure is the audit trail for the fabrication question.

## Test plan

`test/voxel-cleanup.test.mjs`, synthetic grids throughout.

Anti-fabrication, the load-bearing test:

- A 1-voxel hole in a plane with an empty candidate mask stays open, for `grow` and for
  `close`, at every radius and iteration count.

`grow`:

- Fills a 1x1 hole in a sheet at `minNeighbors = 3`, in one iteration.
- Fills a 1x3 slit in exactly two iterations, and not in one.
- Does **not** fill a 3x3 square hole at any iteration count, pinning the documented
  conservatism.
- Does not grow outward from a flat wall face (such a voxel has 1 occupied neighbour).
- Terminates early when an iteration adds nothing.

`close`:

- Seals a 1-voxel hole at `r = 1`; a 3-voxel hole only at `r = 2`.
- A solid slab touching the grid edge is not chipped, pinning the erode out-of-grid
  convention.
- Two chunks writing the same block never yield type 3 (regression for bug fix 1).

`majority`:

- Removes a 1-voxel bump on a plane; fills a 1-voxel dent.
- Preserves a flat plane exactly.
- Removes an isolated voxel.
- Chunk equivalence: `CHUNK_INNER = 8` matches a single chunk covering the whole grid.

`despeckle`:

- A 27-voxel island is removed at `minVoxels = 64`; a 100-voxel island survives.
- A component straddling several blocks is sized correctly.

Wiring:

- Off by default: with `voxelCleanup` absent, emitted `.voxel.bin` and `.vox` bytes match a
  fixture captured before the change. This is the regression guard for existing consumers.
- Dial mapping: `voxelCleanup: 0.2` at `voxelResolution: 0.1` yields `r = 2`; a sub-voxel
  value clamps to `r = 1`.
- `--voxel-cleanup-fill` without `--voxel-cleanup` errors.
- A grid exceeding `MAX_GRID_BLOCKS` raises the guard error naming the block count, the limit
  and the voxel resolution, and pointing at both `--voxel-params` and `--filter-box`
  (regression for bug fix 2).

Acceptance on the real scene, run manually and recorded in the implementation plan. The
reported command with the new flag added:

```
splat-transform ./scenes/urban.spz \
  --filter-box -20,-20,-20,20,20,20 \
  --filter-cluster \
  --voxel-params 0.1,0.1 \
  --voxel-cleanup 0.2 \
  --collision-color solid \
  --collision-color-palette 64 \
  --collision-color-smooth 0 \
  --collision-color-coherent 0 \
  --collision-voxels ./scenes/urban.vox \
  --auto-rotate -w \
  ./scenes/urban.voxel.json
```

must reach components <= 100, top-surface roughness <= 4.0 voxels, and 0 fabricated voxels.

## Out of scope

True planarity through RANSAC plane fitting and snapping. The majority filter takes roughness
from 9.80 to about 3.33 voxels, a 2.9x improvement that will look markedly cleaner, but the
result is not planar. Worth revisiting only after the cleanup dial has been evaluated on real
scenes, and it carries real risk of destroying legitimately curved geometry.
