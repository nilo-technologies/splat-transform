# Voxel yaw auto-alignment

- **Date:** 2026-08-07
- **Status:** approved, not yet implemented
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
(non-finite or `s_mid == 0`), or when `alpha < opacityCutoff` — a splat too faint
to produce a solid voxel must not vote. `opacity = +Infinity` yields `alpha = 1`
and votes at full weight (real files contain these).

### Vote weight

With sorted `s_min <= s_mid <= s_max`:

```
w = flatness * area * alpha
flatness = 1 - s_min / s_mid      // 0 for isotropic blobs, ->1 for a disc
area     = s_mid * s_max          // voxel count scales with patch area
```

`flatness` is what stops organic fuzz from outvoting walls. Both factors are
invariant to (or a global constant factor of) the uniform scale in the write
transform, which is why raw scale columns can be read directly.

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

`improvement = 1 - cost(theta*) / cost(0)`. When `improvement < minImprovement`
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
```

For `up: 'x'` the rotated component pair is `(y, z)`; for `'y'` it is `(x, z)`;
for `'z'` it is `(x, y)`. The recorded quaternion is about the same axis.

### Logging

Always one info line, e.g.
`auto-rotate: yaw 13.5deg (est. 12% fewer surface voxels, 184K of 235K splats voted)`.
When the guard fires the line reports the `reason` and that no rotation was
applied. At `--verbose`, the cost curve sampled every 5 degrees.

## Code shape

| File | Change |
| --- | --- |
| `src/lib/voxel/align-yaw.ts` | new: `estimateAlignYaw`, `AlignYawOptions`, `AlignYawResult`, and the pure `applyAlignYaw(delta, navSeed, theta)` helper |
| `src/lib/voxel/index.ts` | export the above |
| `src/lib/index.ts` | public export + types, JSDoc with `@example` for typedoc |
| `src/lib/writers/write-voxel.ts` | `WriteVoxelOptions.autoRotate?: boolean \| number`; estimate right after `delta` is computed (write-voxel.ts:459) and compose `delta = R_y(theta).mul(delta)`; rotate `navSeed`; `VoxelMetadata.rotation?: [x,y,z,w]` and `version: '1.1' \| '1.2'`; `writeOctreeFiles(fs, filename, octree, rotation?)` |
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
