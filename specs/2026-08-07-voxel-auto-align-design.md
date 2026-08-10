# Voxel yaw auto-alignment

- **Date:** 2026-08-07
- **Status:** implemented and released in v2.7.1-nilo.6 (`6c095eb`..`0890251`, 17 commits)
- **Scope:** `writeVoxel` and its outputs only

## Problem

The voxel grid is always world-axis-aligned. When a scene's dominant surfaces sit
at an angle to the world axes, every wall becomes a staircase: the number of
voxels a planar surface occupies scales with `|n_x| + |n_y| + |n_z|` for unit
normal `n`, which is 1 when the normal hits an axis and `sqrt(2)` when it sits at
45 degrees. The result is up to 41% more surface voxels than necessary, a larger
`.vox`, a heavier octree, and a model that misrepresents the original shape when
opened in MagicaVoxel.

A yaw about the up axis is enough to fix this for the scenes that matter
(buildings, interiors, terrain with man-made structure), because the offending
surfaces are near-vertical walls whose normals lie in the horizontal plane.

Today the only lever is `--rotate 0,<deg>,0`, which requires the user to guess
the angle and rotates the exported splat too.

## Goals

- Estimate the best yaw automatically, cheaply, before the single voxelization pass.
- Apply it to the voxel family only. Splat outputs stay untouched.
- Record the rotation so `.voxel.json` and `.collision.glb` consumers can put the
  data back onto the unrotated splat.
- Emit `.vox` in the aligned frame, since that is the point of the exercise.
- Expose the estimator as library API, not just a CLI flag.

## Non-goals

Explicitly out of scope, listed so they are not smuggled in:

- Translation / phase snapping of the grid origin to dominant planes.
- Pitch and roll (full 3-DOF alignment).
- A second voxelization pass to produce an aligned `.vox` alongside an unaligned
  octree and mesh. Rejected: doubles the most expensive stage.
- A CLI dry-run that prints the angle without voxelizing. The estimate is
  sub-millisecond and is logged during the real run; library callers can call the
  estimator directly.
- Representing the yaw inside the `.vox` file. MagicaVoxel's `nTRN` carries an
  integer `_t` and 90-degree-step `_r` only, so an arbitrary yaw cannot be
  expressed.

## Behaviour and output contract

Opt-in. When `autoRotate` is absent or resolves to a zero yaw, every output is
byte-identical to today.

Let `theta` be the applied yaw about engine `+Y`, and frame **V** the rotated
frame in which `p_V = R_y(theta) * p_source`. All voxelization, filling,
cropping, octree building, mesh extraction and `.vox` encoding happen in V, from
one voxelization pass.

| Output | Frame | Change |
| --- | --- | --- |
| `.vox` | V | none; the aligned grid is the deliverable |
| `.collision.glb` | V vertices, node rotation `R_y(-theta)` | `nodes: [{ mesh: 0, rotation: [x,y,z,w] }]` |
| `.voxel.json` / `.voxel.bin` | V | new `rotation` field; `gridBounds`/`sceneBounds` stay in V |
| splat outputs (`.sog`, `.ply`, `.spz`, ...) | source | none |

One convention for both recorded values: the quaternion written to the GLB node
and to `.voxel.json` is the same, `R_y(-theta)`, mapping **V to source**, in
`[x, y, z, w]` order (playcanvas / glTF component order).

`theta` pivots about the engine origin, not the scene centre. This is harmless
for alignment quality (it only shifts where the grid crop lands) and consumers
reconstruct exactly from the recorded quaternion.

### Metadata version

`VoxelMetadata.version` stays `'1.1'` when no `rotation` field is written and
becomes `'1.2'` when one is. A loader that ignores unknown fields would place
collision geometry off by `theta`; the version bump gives it a detectable signal.

### Seed coordinates

`navSeed` (CLI `--seed-pos`) is a source-frame coordinate consumed *inside*
`writeVoxel` by `fillExterior` and `carve` against the rotated grid, so it is
rotated by `R_y(theta)` before use. `navCapsule` height/radius,
`navExteriorRadius` and floor fill are scalars or Y-axis-only and are unaffected.

Process actions that take world coordinates (`--filter-box`, `--filter-cluster`
seed, `--filter-sphere`, `--filter-floaters`) run before the writer and keep
source-frame semantics. `--collision-voxels-size` is unaffected: coarse cells
align to the grid origin either way.

### Explicit angle

`autoRotate` accepts a number as well as `true`. A number skips estimation
entirely and applies that yaw verbatim (no guard, no improvement figure), which
gives an escape hatch when the estimate is wrong and lets a build pin an angle
for reproducibility. `autoRotate: 0` is equivalent to off: no rotation, no
recorded field.

## Estimator

New module `src/lib/voxel/align-yaw.ts`. Pure CPU, no GPU, no `node:` imports, so
it works in browser builds. It allocates nothing per splat: one pass over the raw
columns into a fixed histogram.

```ts
estimateAlignYaw(dataTable: DataTable, options?: AlignYawOptions): AlignYawResult
```

### Per-gaussian normal

Conventions taken from `computeGaussianExtents` (src/lib/data-table/gaussian-aabb.ts:88-89)
and `forwardTransforms` (src/lib/process.ts:242-250):

- `q = Quat(rot_1, rot_2, rot_3, rot_0).normalize()` — `rot_0` is `w`
- linear scales `s_i = exp(scale_i)`
- `alpha = 1 / (1 + exp(-opacity))`
- `k = argmin_i s_i`, the flattest local axis; surface normal `n = q * e_k`
- `n` is then rotated by `dataTable.transform.rotation`, the raw-to-engine
  rotation, so the yaw returned is in engine space. No transformed columns are
  materialized; the estimator reads raw columns and rotates the resulting normal.

A gaussian is skipped when any input is NaN, when scales are degenerate
(non-finite or `s_mid == 0`), when the rotation quaternion is degenerate
(`hypot(x, y, z, w) <= 1e-8`, so it cannot be normalized), or when
`alpha < opacityCutoff` — a splat too faint to produce a solid voxel must not
vote. `opacity = +Infinity` yields `alpha = 1` and votes at full weight (real
files contain these).

### Vote weight

With sorted `s_min <= s_mid <= s_max`:

```
w = flatness * area * alpha
flatness = 1 - s_min / s_mid      // 0 for isotropic blobs, ->1 for a disc
area     = s_mid * s_max          // voxel count scales with patch area
```

`flatness` is what stops organic fuzz from outvoting walls. Both factors are
invariant to (or a global constant factor of) the *uniform* scale in the write
transform, which is why raw scale columns can be read directly.

This invariance does not extend to non-uniform scale. The estimator rotates
normals by `dataTable.transform.rotation` alone, but under a non-uniform scale
(for example `--scale 1,2,1`) true engine-space normals follow the
inverse-transpose of the linear part, so both the normal directions and the
`s_mid * s_max` area term skew. In practice the effect is bounded — a skewed
vote distribution flattens the cost curve and so lowers the measured
`improvement`, tending to trip the 2% guard rather than bake a wrong yaw — and
`autoRotate` accepts an explicit angle to bypass estimation entirely. Handling
non-uniform scale properly is out of scope.

### Cost function

The horizontal component pair is taken in right-handed cyclic order for the
chosen up axis, which makes the rotation sign uniform across all three:

| `up` | pair `(a, b)` |
| --- | --- |
| `'x'` | `(y, z)` |
| `'y'` | `(z, x)` |
| `'z'` | `(x, y)` |

With `h = hypot(a, b)`, `phi = atan2(b, a)` and `g(a) = |cos a| + |sin a|`:

```
cost(theta) = sum over gaussians of  w * h * g(phi + theta)
```

The component along the up axis is invariant under the rotation and is dropped.

Verified empirically rather than derived on paper, because the sign is easy to
invert: `new Quat().setFromEulerAngles(0, 30, 0)` maps `(1,0,0)` to
`(0.866, 0, -0.5)`, i.e. `atan2(z, x)` by `-30` degrees but the cyclic pair
`(z, x)` by `+30`. Using the naive pair `(x, z)` for `up: 'y'` would yield a yaw
of the wrong sign, doubling the misalignment instead of removing it. `g` has period 90 degrees, minima
at multiples of 90 and maximum `sqrt(2)` at 45, so the ceiling on improvement is
`1 - 1/sqrt(2)`, about 29% fewer surface voxels.

### Evaluation

1. One O(N) pass accumulates `W[bin] += w * h` into 720 bins of `phi mod 90deg`
   (0.125 degree resolution).
2. Precompute `G[m] = g((m + 0.5) * 90deg / bins)`. Because both `phi` and
   `theta` land on bin multiples, the cost sweep is a cyclic correlation with no
   trigonometry in the inner loop:
   `curve[j] = sum over bins of W[b] * G[(b + j) mod bins]`.
   720 x 720 multiply-adds, sub-millisecond and independent of splat count.
3. Parabolic fit across the argmin and its two cyclic neighbours for sub-bin
   precision.
4. Normalize into `[-45, 45]` degrees, the smallest equivalent rotation.

### Guard

`improvement = 1 - cost(bin_min) / cost(0)`, where `bin_min` is the argmin bin
rather than the sub-bin `theta*` from the parabolic fit. The parabolic fit refines
the reported angle but not the reported saving. Since the fitted parabola's vertex
value lies at or below `cost(bin_min)` by construction, the reported saving is
conservative with respect to the fit. When `improvement < minImprovement`
(default 0.02), or when no gaussian was eligible, the result is `yawDegrees: 0`
with a `reason` string: nothing is rotated, no metadata is written, `.vox` comes
out exactly as today. This is what keeps an organic scene from acquiring a
meaningless bake.

### Options and result

```ts
type AlignYawOptions = {
    up?: 'x' | 'y' | 'z';      // default 'y'; library-only, CLI is always 'y'
    stepDegrees?: number;      // default 0.125
    minImprovement?: number;   // default 0.02
    opacityCutoff?: number;    // default 0.1, matching the writer's default
};

type AlignYawResult = {
    yawDegrees: number;
    improvement: number;
    cost0: number;
    costBest: number;
    votedCount: number;
    totalWeight: number;
    curve: Float64Array;       // sampled cost over [0, 90)
    reason?: string;           // set when the guard fired
};

applyAlignYaw(
    delta: Transform,
    navSeed: { x: number; y: number; z: number } | undefined,
    yawDegrees: number,
    up?: UpAxis                // default 'y'
): AlignYawApplied

type AlignYawApplied = {
    delta: Transform;                     // write transform with the yaw composed in
    navSeed?: { x: number; y: number; z: number };  // seed rotated into the aligned frame, when given
    recordedRotation: [number, number, number, number] | null;  // the inverse, for metadata; null at zero yaw
};
```

`UpAxis`, `AlignYawOptions`, `AlignYawResult` and `AlignYawApplied` are all public
type exports (`src/lib/index.ts:98`), alongside the `estimateAlignYaw` and
`applyAlignYaw` value exports (`src/lib/index.ts:94`).

For the rotated component pair per up axis, see the cost-function table above —
it is the single source of truth for the sign convention. The recorded quaternion
is about the same axis.

### Logging

Always one info line, e.g.
`auto-rotate: yaw 13.5deg (est. 12% fewer surface voxels, 184K of 235K splats voted)`.
When the guard fires the line reports the `reason` and that no rotation was
applied. At `--verbose`, the cost curve sampled every 5 degrees.

## Code shape

| File | Change |
| --- | --- |
| `src/lib/voxel/align-yaw.ts` | new: `estimateAlignYaw`, `AlignYawOptions`, `AlignYawResult`, `UpAxis`, and the pure `applyAlignYaw(delta, navSeed, yawDegrees, up?)` helper returning `AlignYawApplied` |
| `src/lib/voxel/index.ts` | export the above |
| `src/lib/index.ts` | public export + types, JSDoc with `@example` for typedoc |
| `src/lib/writers/write-voxel.ts` | `WriteVoxelOptions.autoRotate?: boolean \| number`; estimate right after `delta` is computed (as implemented, write-voxel.ts:477-507) and compose `delta = R_y(theta).mul(delta)`; rotate `navSeed`; `VoxelMetadata.rotation?: [x,y,z,w]` and `version: '1.1' \| '1.2'`; `writeOctreeFiles(fs, filename, octree, rotation?)` |
| `src/lib/writers/collision-glb.ts` | `buildCollisionMesh(...)` gains a 6th optional `options?: { nodeRotation?: [x,y,z,w] }` |
| `src/cli/index.ts` | `--auto-rotate[=deg]` global option, registered in `optionalValueOptions` with `isNumericValue`; warn-and-ignore without a voxel output; usage text beside the `--collision-*` block |

`.mul` ordering follows the existing rotate action (src/lib/process.ts:388):
`R_y(theta).mul(delta)` means the yaw is applied after the write transform.

No changes to `processDataTable`, no new process action, no changes to splat
writers. Both `buildCollisionMesh` and `VoxelMetadata` are public API, so all
changes are additive.

`applyAlignYaw` exists as a separate pure function so the `navSeed` rotation —
the one line whose failure is silent and GPU-only — is unit-testable.

Bundle impact: none on the worker. `worker-entry.ts` imports only `./tasks`,
which reaches `quantize-1d-core` and a `TypedArray` type, never
`src/lib/voxel/index.ts`, so the new export cannot drag `DataTable` (and with it
the playcanvas engine) into `dist/worker.mjs`.

## Testing

Estimator unit tests (`test/align-yaw.test.mjs`, no GPU). The geometric tests
assert invariants rather than a literal signed angle, so an inverted sign
convention cannot pass:

1. Flat gaussians with normals 17 degrees off-axis: applying the returned yaw
   yields `|n_x| + |n_z| ~= hypot(n_x, n_z)`, i.e. actually axis-aligned
2. Periodicity: input at 62 degrees returns a value in `[-45, 45]` equivalent mod 90
3. Isotropic blobs: no eligible votes, `yawDegrees: 0` with `reason`
4. Uniformly random normals: contrast under the guard, `yawDegrees: 0` with `reason`
5. Dominant faint set at 30 degrees vs weaker opaque set at 0 returns ~0, proving
   the `alpha < opacityCutoff` gate
6. `+Infinity` opacity votes at full weight; NaN rows are skipped without
   poisoning the histogram
7. Same geometry expressed with a `Transform.PLY` transform vs pre-baked identity
   returns the same engine-space yaw — this catches a wrong pre-rotation, the
   most likely silent bug in the feature
8. `up: 'z'` on a Z-up variant
9. `applyAlignYaw` rotates `navSeed` by `R_y(theta)` and leaves it untouched at
   `theta = 0`

Format tests (no GPU):

10. `writeOctreeFiles` with a rotation writes `version: '1.2'` and the correct
    quaternion; without one it writes `'1.1'` and **no `rotation` key**
11. `buildCollisionMesh` with `nodeRotation` produces `nodes[0].rotation` in the
    GLB JSON chunk; without it the node stays exactly `{ mesh: 0 }`
12. Validation path with the existing dummy device: `autoRotate: 13.5` accepted,
    non-finite rejected

Gates: `npm run lint`, `npm test`, `npm run build` and `npm run docs` (no new
typedoc warnings).

### Validation experiment

The metric is a proxy, so it needs evidence once rather than per run. On
`scenes/house.ply`, `scenes/industrial.ply`, `scenes/dungeons-3.ply` and
`scenes/landscape.spz`, run with and without `--auto-rotate` and record:
predicted improvement, measured occupied voxel count, `.vox` size, grid dims and
`.vox` model count. Also compare the estimator's argmin against a real coarse
voxelization sweep on one scene.

This decides two things: whether the 0.02 guard is calibrated, and whether the
proxy ranks angles correctly. Expectation: `landscape.spz` (organic terrain)
trips the guard and stays at 0 degrees; if it does not, the guard is too loose.
Results table lands in this document.

### Validation results

Run on 2026-08-07, CLI built from `9978b5c` (`--collision-voxels`,
`--voxel-params 0.05,0.1` unless noted). Occupied voxel counts and `.vox` byte
sizes are exact (parsed from the `XYZI` chunk), not the rounded figures the CLI
log prints.

#### Step 1: with vs. without `--auto-rotate`

| Scene | `auto-rotate` result | `.vox` grid dims | occupied voxels | `.vox` size | models | wall time |
| --- | --- | --- | --- | --- | --- | --- |
| `house.ply` | no rotation — best yaw saves 0.0% (unguarded: yaw 0.19deg, improvement 0.0037%), below 2.0% threshold | 20×21×18 (both) | 4382 (both) | 18624 B (both) | 1 | base 0.61s / aligned 0.66s |
| `industrial.ply` | no rotation — best yaw saves 0.0% (unguarded: yaw -0.13deg, improvement 0.0015%), below 2.0% threshold | 22×22×12 (both) | 1704 (both) | 7912 B (both) | 1 | base 0.69s / aligned 0.72s |
| `dungeons-3.ply` | no rotation — best yaw saves 0.2% (unguarded: yaw 1.67deg, improvement 0.183%), below 2.0% threshold | 19×20×12 (both) | 2017 (both) | 9164 B (both) | 1 | base 0.56s / aligned 0.59s |
| `landscape.spz`¹ | no rotation — best yaw saves 0.1% (unguarded: yaw 4.88deg, improvement 0.051%), below 2.0% threshold | 289×426×173 (both) | 1,606,758 (both) | 6,428,558 B (both) | 3 | base 6.08s / aligned 6.31s |

¹ `landscape.spz` crashes with `--voxel-params 0.05,0.1` as specified — see
"Deviation and a bug found" below. Its row uses `--voxel-params 2.0,0.1`
instead; the base-vs-aligned comparison is still apples-to-apples since both
runs used the same params.

For all four scenes the guard fired, so `.vox`, `.voxel.bin` and `.voxel.json`
are **byte-identical** with and without `--auto-rotate` (`cmp` confirms it;
`.voxel.json` stays `version: "1.1"` with no `rotation` key), exactly as the
output contract promises for a zero-yaw result.

None of the four real scenes has a real yaw problem: all four are already
within about 5 degrees of axis-aligned, so `--auto-rotate` correctly declines
to touch any of them. Since none of them exercises the "genuine misalignment"
path, a fifth, synthetic case was added to test that path against real
splat geometry (not the synthetic constructed Gaussians `align-yaw.test.mjs`
already covers): `dungeons-3.ply` rotated by a known `+20deg` yaw
(`--rotate 0,20,0`), which recovers this scene's own tiny inherent tilt too:

| Scene | `auto-rotate` result | `.vox` grid dims | occupied voxels | `.vox` size | wall time |
| --- | --- | --- | --- | --- | --- |
| `dungeons-3.ply` + `--rotate 0,20,0`, no `--auto-rotate` | n/a | 24×23×12 | 2050 | 9296 B | 0.57s |
| same, `--auto-rotate` | yaw -18.33deg, est. 8% fewer surface voxels (193K/262K splats voted) | 20×20×12 | 2021 | 9180 B | 0.57s |

Applying the estimated -18.33deg to a scene rotated by +20deg leaves a residual
of +1.67deg — matching `dungeons-3.ply`'s own unguarded estimate (1.67deg)
almost exactly, and `costBest` for the rotated case (1.080603) matches
`dungeons-3.ply`'s own `costBest` (1.080602) to 5 decimal places. The estimator
is measuring the same underlying geometry either way, as it should. Occupied
voxels dropped 1.4% (2050 → 2021), correctly signed but smaller in magnitude
than the estimator's 8% cost-function figure — expected, since the cost
function is a staircase proxy over surface-facing normals, not a literal count
of the final `.vox` (which also includes interior fill and is affected by
grid cropping to the rotated AABB; see "Risks").

#### Step 2: sweep vs. the estimator's pick

Swept `dungeons-3.ply` (the real scene with the largest, if tiny, predicted
improvement) at the angles the brief specifies, `--voxel-params 0.05,0.1`,
comparing exact `.vox` occupied-voxel counts:

| yaw (deg) | -30 | -20 | -10 | 0 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| occupied voxels | 2055 | 2036 | 2019 | **2017** | 2031 | 2050 | 2049 |

At this 10-degree granularity the minimum sits at 0deg, matching the
unguarded estimate of 1.67deg to well within "a few degrees." Given how flat
this curve is, this sweep alone is a weak test of the sign convention, so the
same sweep was repeated on the synthetic `+20deg`-rotated `dungeons-3.ply`
from Step 1, which has a real, sizeable misalignment to find:

| yaw (deg) | -30 | -20 | -18.33 (estimate) | -10 | 0 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupied voxels | 2019 | **2017** | 2021 | 2031 | 2050 | 2049 | 2075 | 2082 |

At 10-degree granularity this also looks clean: 0deg (2050) is clearly the
worst point, and the minimum sampled (-20deg, 2017) sits 1.67deg from the
estimate.

**A second round of review flagged that `house.ply`/`industrial.ply`'s "tiny
misalignment" claim rested only on the estimator's own cost curve, not an
independent sweep — a partial circularity, since the metric under test would
be validating itself.** Investigating that gap uncovered something more
consequential than a missing data point for two scenes: **the sweep
methodology itself has a real, reproducible noise floor**, found by filling in
finer angle steps (1-2deg) around zero on top of the two coarse sweeps above.

`house.ply` swept the same way, plus 1-2-degree steps out to +/-10:

| yaw (deg) | -30 | -20 | -10 | -7 | -6 | -5 | -4 | -3 | -2 | -1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 | 20 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupied voxels | 4668 | 4627 | 4583 | 4544 | 4579 | 4600 | 4614 | 4570 | 4494 | 4416 | **4382** | 4415 | 4490 | 4542 | 4565 | 4546 | 4527 | 4533 | 4576 | 4630 | 4705 |

This one holds up: 0deg is an unambiguous global minimum across all 21
sampled points (next closest, +/-1deg, is 33-34 voxels / 0.75% higher), even
though the 4-7deg region on both sides wobbles non-monotonically — that wobble
is the same noise found below, just not near enough to zero to threaten the
conclusion. **`house.ply` is independently confirmed**, matching its 0.19deg
unguarded estimate.

Filling in the same 1-2deg steps around zero for real (unrotated)
`dungeons-3.ply` tells a different story:

| yaw (deg) | -10 | -5 | -2 | -1 | 0 | 1 | 2 | 5 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupied voxels | 2019 | 2019 | 2006 | 1999 | 2017 | 2013 | 2030 | **1995** | 2031 |

The finer-grained global minimum among these points is +5deg (1995), not
0deg — lower than the value the original coarse sweep reported as "the"
minimum. The swing across this narrow band (1995-2030, ~1.7%) is comparable
to the scene's own predicted improvement (0.18%). The 10-degree sweep's
"confirmed" verdict above should be read as *consistent with* a small true
misalignment (the region far from zero is not favoured), not as a sharp,
degree-precise confirmation — the noise floor and the signal are the same
size here.

And filling in the same steps around the synthetic `+20deg`-rotated scene's
predicted -18.33deg optimum:

| yaw (deg) | -25 | -22 | -20 | -18.33 | -16 | -14 | -10 | -5 | 0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupied voxels | 2019 | 2006 | 2017 | 2021 | 2024 | **1978** | 2031 | 2011 | 2050 |

Same pattern: the finer-grained minimum (-14deg, 1978) doesn't land on -20deg
or -18.33deg either. What *does* hold up robustly is the big picture: 0deg
(2050) is unambiguously worse than every sampled point from -25deg to -10deg
(1978-2031), a genuine ~3.5% swing in the predicted direction. That's a real,
useful confirmation of *sign and rough scale*, not of *exact-degree
precision* — the same caveat as the real `dungeons-3.ply` sweep, just with a
large-enough signal that the directional conclusion survives the noise.

Open follow-up: this sweep cannot separate quantization aliasing from genuine
estimator bias, because both would displace the minimum away from -18.33deg.
Re-running it at a finer voxel size would distinguish them — aliasing shrinks
with resolution while bias does not. Worth doing before anyone treats the
per-degree numbers above as an estimator accuracy measurement.

Attempting the same investigation on `industrial.ply` (both the brief's
7-point sweep and a finer set) makes the noise floor obvious rather than
subtle:

| yaw (deg) | -30 | -20 | -15 | -10 | -5 | -2 | -1 | -0.5 | -0.1 | 0 | 0.1 | 0.5 | 1 | 2 | 5 | 10 | 15 | 20 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| occupied voxels | 1657 | 1625 | 1623 | 1609 | 1576 | 1624 | 1665 | 1687 | 1702 | **1704** | 1705 | 1707 | 1684 | 1625 | 1578 | 1619 | 1622 | 1628 | 1659 |

0deg is a local *maximum* here, not a minimum — higher than every neighbour
out to +/-10deg — before the curve settles into a broad, roughly symmetric
plateau around 1610-1660 further out. There's no discontinuity right at zero
(0.1deg and 0.5deg sit smoothly between 0deg and 1deg, ruling out a code-path
artifact from `autoRotate: 0` skipping the rotation pipeline entirely — see
below), so this is a real, continuous, reproducible response, not noise from
non-determinism (re-running 0deg and -10deg twice each gave identical counts).
The swing (1576-1707, ~7.6%) dwarfs the scene's own predicted improvement
(0.0015%). **`industrial.ply`'s sweep does not confirm the near-zero
estimate — it's inconclusive**, not contradictory (the response is
essentially symmetric in sign, which is at least consistent with no strongly
preferred nonzero direction). Its "tiny misalignment" claim continues to rest
on the estimator's own smooth cost curve alone.

**Root cause (most likely), not chased further given the answer doesn't
change the outcome:** `--auto-rotate=N` for any `N != 0` (even 0.1deg) always
routes through the full rotate-then-recrop pipeline, re-deriving the grid's
AABB in the rotated frame before voxelizing; `--auto-rotate=0` (or no flag)
skips rotation and alignment entirely, per the documented `autoRotate: 0`
contract (a direct `cmp` of `industrial.ply`'s `--auto-rotate=0` output
against its no-flag baseline confirmed byte-identical output).
Because a voxel grid is quantized, even a fractional-degree rotation shifts
which voxels sub-pixel-scale surface detail falls into, independent of any
real macroscopic alignment change — a form of spatial aliasing. This noise
is on the order of 1.5-8% of occupied-voxel count at this resolution
(`--voxel-params 0.05`), scene-dependent, and is comparable to or larger than
the true signal for every real scene here except `house.ply`. It was
continuous through zero in every case checked (no jump), so it is not the
`autoRotate: 0` code-path branch itself causing a discontinuity — it's an
inherent property of quantized voxelization responding to small rotations,
which the `autoRotate: 0` special case merely sits inside of like any other
sample point.

#### Guard calibration

`minImprovement` (default 0.02) **held; not changed.** Evidence for keeping
it, now weighted by how much independent confirmation each scene actually has:

- **Independently confirmed by a clean sweep:** `house.ply` (0.19deg
  estimate; unambiguous global minimum at 0deg across 21 points, both coarse
  and fine). The synthetic `+20deg` misalignment is confirmed for *direction
  and rough scale* (0deg is unambiguously worst; the -25..-10deg region is
  consistently 0.9-3.5% better) but not for exact-degree precision, since the
  finer-grained minimum landed at -14deg rather than -18.33/-20deg.
- **Consistent with, but not sharply confirmed by, an independent sweep:**
  real `dungeons-3.ply` (1.67deg estimate). The 10-degree sweep's minimum
  landed at 0deg; a finer sweep's minimum landed at +5deg instead. Both are
  within the scene's own noise floor (~1.7%, comparable to its 0.18%
  predicted improvement), so the honest reading is "no evidence of a
  larger hidden misalignment," not "the estimate is confirmed to the
  degree."
- **Not independently confirmed; rests on the estimator's own cost curve
  alone:** `industrial.ply` (-0.13deg estimate) — its sweep is dominated by
  a 7.6% noise floor roughly 5000x its 0.0015% predicted improvement, and
  does not resolve to a clear minimum anywhere. `landscape.spz` was never
  swept in this experiment at all (the guard fired; only its own cost curve
  was inspected, at 5-degree steps via `--verbose`).
- The synthetic +20deg misalignment's 7.94% predicted improvement (from the
  estimator's own cost units, independent of the sweep) is comfortably clear
  of the 2% threshold, and Task 1's synthetic-Gaussian finding (17deg tilt to
  ~19.9% improvement, per that task's brief) points the same direction: a
  guard set at 2% does not swallow a real, sizeable yaw problem.

No scene in this experiment falls near the 2% boundary, so this run cannot
distinguish "0.02 is exactly right" from "0.02 is roughly right"; it only
rules out "0.02 is badly miscalibrated" in either direction. The discovery of
a ~1.5-8% sweep noise floor is, if anything, a point in favour of keeping a
conservative guard rather than lowering it: below roughly that magnitude of
predicted improvement, a single real voxelization pass at typical CLI
resolutions cannot reliably measure whether rotating actually helped, so a
guard that declines to act on smaller predictions is declining to act on
signals the pipeline can't itself verify anyway.

#### Deviation and a bug found

`landscape.spz` with the brief's literal `--voxel-params 0.05,0.1` crashes
during the filtering stage. This was directly confirmed both **without**
`--auto-rotate` and **with** it — not inferred from the non-crashing
`2.0,0.1` run used elsewhere in this report:

```
$ node bin/cli.mjs .../landscape.spz --collision-voxels ./scenes/tmp-landscape-base.vox \
    --voxel-params 0.05,0.1 -w ./scenes/tmp-landscape-base.voxel.json
...
✗ RangeError: Invalid typed array length: -2147483648
    at new Float64Array (<anonymous>)
    at new IntKeyMap (.../dist/cli.mjs:46964:21)
    at filterAndFillBlocks (.../dist/cli.mjs:68677:22)

$ node bin/cli.mjs .../landscape.spz --auto-rotate --collision-voxels ./scenes/tmp-landscape-crash-check.vox \
    --voxel-params 0.05,0.1 -w ./scenes/tmp-landscape-crash-check.voxel.json
...
  · auto-rotate: no rotation applied - no dominant alignment: best yaw saves 0.1%, below the 2.0% threshold
  ▸ Build voxels
    ...
      ✗ RangeError: Invalid typed array length: -2147483648
    at new Float64Array (<anonymous>)
    at new IntKeyMap (.../dist/cli.mjs:46964:21)
    at filterAndFillBlocks (.../dist/cli.mjs:68677:22)
```

Identical crash, same stack, in both cases — `--auto-rotate` logs its (correct,
guard-fired) decision before the crash, since the crash is in the filtering
stage that runs after alignment regardless of which yaw was chosen.

`landscape.spz`'s scene extents are roughly 593 × 370 × 882 units — two to
three orders of magnitude larger than `house.ply`/`industrial.ply`/
`dungeons-3.ply` (~1 unit across). At `0.05` that is well over a billion 4×4×4
blocks, and `IntKeyMap`'s capacity computation (`1 << (32 - Math.clz32(...))`)
overflows a 32-bit signed shift at that scale, producing a negative
`Float64Array` length. `--voxel-params 2.0,0.1` avoids it and was used for
`landscape.spz`'s row above instead; this is an existing bug in
`filterAndFillBlocks`'s block-count handling, not introduced by this feature,
and out of this task's scope to fix (Task 10 only touches this spec and,
conditionally, `minImprovement`). Worth its own follow-up ticket.

## Risks

- **Old `.voxel.json` loaders** ignoring `rotation` misplace collision data by
  `theta`. Mitigated by opt-in, the 1.2 version bump and the log line; the
  consumer is ours, so the residual risk is accepted.
- **Wrong angle on organic scenes.** Bounded: at most a 45-degree yaw, voxel
  outputs only, with the guard and the explicit-angle override as escapes.
- **Grid dims can grow while occupied count drops.** Aligning a diagonal building
  shrinks its AABB, but a terrain's frame-V AABB may grow slightly. Sparse `.vox`
  size follows occupied count while the 256-per-axis tiling follows dims, so
  model count may move either way. The experiment measures it.
- **`.vox` and `.collision.glb` now present the same grid in different frames.**
  A user diffing them side by side needs the usage text to say so.

## Follow-ups

Deliberately deferred, in rough order of expected value:

1. Grid-origin phase snapping, so dominant planes land on voxel boundaries rather
   than straddling them. Complements yaw and is likely the next largest win.
2. Exposing the up-axis choice on the CLI for Z-up scenes.
3. Pitch/roll alignment for scenes with sloped dominant planes.
