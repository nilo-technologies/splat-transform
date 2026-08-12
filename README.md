# SplatTransform - 3D Gaussian Splat Converter

[![NPM Version](https://img.shields.io/npm/v/@playcanvas/splat-transform.svg)](https://www.npmjs.com/package/@playcanvas/splat-transform)
[![NPM Downloads](https://img.shields.io/npm/dw/@playcanvas/splat-transform)](https://npmtrends.com/@playcanvas/splat-transform)
[![License](https://img.shields.io/npm/l/@playcanvas/splat-transform.svg)](https://github.com/playcanvas/splat-transform/blob/main/LICENSE)
[![Discord](https://img.shields.io/badge/Discord-5865F2?style=flat&logo=discord&logoColor=white&color=black)](https://discord.gg/RSaMRzg)
[![Reddit](https://img.shields.io/badge/Reddit-FF4500?style=flat&logo=reddit&logoColor=white&color=black)](https://www.reddit.com/r/PlayCanvas)
[![X](https://img.shields.io/badge/X-000000?style=flat&logo=x&logoColor=white&color=black)](https://x.com/intent/follow?screen_name=playcanvas)

| [User Guide](https://developer.playcanvas.com/user-manual/splat-transform/) | [API Reference](https://api.playcanvas.com/splat-transform/) | [Blog](https://blog.playcanvas.com/) | [Forum](https://forum.playcanvas.com/) |

SplatTransform is an open source library and CLI tool for converting and editing Gaussian splats. It can:

📥 Read PLY, Compressed PLY, SOG, SPZ, SPLAT, KSPLAT, LCC and LCC2 formats  
📤 Write PLY, Compressed PLY, SOG, SPZ, GLB, CSV, HTML Viewer, LOD, Voxel and WebP image formats  
📊 Generate statistical summaries for data analysis  
🔗 Merge multiple splats  
🔄 Apply transformations to input splats  
🎛️ Filter out Gaussians or spherical harmonic bands  
🔀 Reorder splats for improved spatial locality  
⚙️ Procedurally generate splats using JavaScript generators

The library is platform-agnostic and can be used in both Node.js and browser environments.

## Installation

Install or update to the latest version:

```bash
npm install -g @playcanvas/splat-transform
```

For library usage, install as a dependency:

```bash
npm install @playcanvas/splat-transform
```

For running on a backend with Docker (including GPU/Vulkan setup), see the [Docker Backend Guide](https://developer.playcanvas.com/user-manual/splat-transform/docker/).

> [!TIP]
> For one-off conversions without installing anything, try [SuperSplat Convert](https://superspl.at/convert) — a browser-based frontend to splat-transform. See the [Convert page docs](https://developer.playcanvas.com/user-manual/supersplat/convert/) for details.

## Guides

- [Streamed SOG Guide](https://developer.playcanvas.com/user-manual/splat-transform/#generating-lod-format) — build a multi-LOD streamed SOG from a single PLY.
- [LOD Streaming Guide](https://developer.playcanvas.com/user-manual/gaussian-splatting/building/lod-streaming/) — load and render streamed SOG output in a PlayCanvas app.
- [Collision Mesh Guide](https://developer.playcanvas.com/user-manual/splat-transform/collision/) — generate voxel/collision data from a splat scene.
- [Docker Backend Guide](https://developer.playcanvas.com/user-manual/splat-transform/docker/) — run splat-transform on a backend (incl. GPU/Vulkan setup).

## Format Specifications

| Format | Description |
| ------ | ----------- |
| [PLY](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/) | Industry-standard uncompressed format for source, editing and interchange |
| [SOG](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/sog/) | Super-compressed format for web delivery (`meta.json` + WebP textures, bundled or unbundled) |
| [Streamed SOG](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/streamed-sog/) | Multi-LOD chunked SOG for streaming very large scenes (`lod-meta.json`) |
| [Voxel](https://developer.playcanvas.com/user-manual/splat-transform/voxel-format/) | Sparse voxel octree for collision detection (`.voxel.json` / `.voxel.bin`) |

## CLI Usage

```bash
splat-transform [GLOBAL] input [ACTIONS]  ...  output [ACTIONS]
```

**Key points:**
- Input files become the working set; ACTIONS are applied in order
- The last file is the output; actions after it modify the final result
- Use `null` as output to discard file output

## Supported Formats

| Format | Input | Output | Description |
| ------ | ----- | ------ | ----------- |
| `.ply` | ✅ | ✅ | Standard PLY format |
| `.sog` | ✅ | ✅ | Bundled super-compressed format (recommended) |
| `meta.json` | ✅ | ✅ | Unbundled super-compressed format (accompanied by `.webp` textures) |
| `.compressed.ply` | ✅ | ✅ | Compressed PLY format (auto-detected and decompressed on read) |
| `.spz` | ✅ | ✅ | Compressed splat format (Niantic format, v2–4) |
| `.lcc` | ✅ | ❌ | LCC file format (XGRIDS) |
| `.lcc2` | ✅ | ❌ | LCC2 file format (XGRIDS, octree) |
| `.ksplat` | ✅ | ❌ | Compressed splat format (mkkellogg format) |
| `.splat` | ✅ | ❌ | Compressed splat format (antimatter15 format) |
| `.mjs` | ✅ | ❌ | Generate a scene using an mjs script (Beta) |
| `.glb` | ❌ | ✅ | Binary glTF with [KHR_gaussian_splatting](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_gaussian_splatting) extension |
| `.csv` | ❌ | ✅ | Comma-separated values spreadsheet |
| `.html` | ❌ | ✅ | HTML viewer app (single-page or unbundled) based on SOG |
| `.voxel.json` | ❌ | ✅ | Sparse voxel octree for collision detection |
| `lod-meta.json` | ❌ | ✅ | Streamed LOD data stored in SOG chunks |
| `.webp` | ❌ | ✅ | Lossless WebP image rendered from a camera view via GPU rasterizer |
| `null` | ❌ | ✅ | Discard output (useful with `--summary` for analysis-only runs) |

## Actions

Actions execute in the order specified and can be repeated. Any action may appear after any input or output file:

```none
-t, --translate        <x,y,z>          Translate Gaussians by (x, y, z)
-r, --rotate           <x,y,z>          Rotate Gaussians by Euler angles (x, y, z), in degrees
-s, --scale            <factor>         Uniformly scale Gaussians by factor
-H, --filter-harmonics <0|1|2|3>        Remove spherical harmonic bands > n
-N, --filter-nan                        Remove Gaussians with NaN values and most Inf values;
                                          retains +Infinity in opacity and -Infinity in scale_*
-B, --filter-box       <x,y,z,X,Y,Z>    Remove Gaussians outside box (min, max corners)
-S, --filter-sphere    <x,y,z,radius>   Remove Gaussians outside sphere (center, radius)
-V, --filter-value     <name,cmp,value> Keep Gaussians where <name> <cmp> <value>
                                          cmp ∈ {lt,lte,gt,gte,eq,neq}
                                          opacity, scale_*, f_dc_* use transformed values
                                          (linear opacity 0-1, linear scale, linear color 0-1).
                                          Append _raw for raw PLY values (e.g. opacity_raw).
-F, --decimate         <n|n%>           Simplify to n Gaussians via progressive pairwise merging
                                          Use n% to keep a percentage of Gaussians
-G, --filter-floaters  [size,op,min]    Remove Gaussians not contributing to any solid voxel.
                                          Evaluates each Gaussian at occupied voxel centers.
                                          Default: size=0.05, opacity=0.1, min=0.004 (1/255).
                                          Bare flag (no value) uses all defaults.
-D, --filter-cluster   [res,op,min]     Keep only the connected cluster at --seed-pos.
                                          GPU-voxelizes at coarse resolution (res world units/voxel).
                                          Default: res=1.0, opacity=0.999, min=0.1.
                                          Bare flag (no value) uses all defaults.
-p, --params           <key=val,...>    Pass parameters to .mjs generator script
-l, --lod              <n>              Tag the Gaussians with LOD level n (n >= 0, or -1 for environment)
-m, --summary                           Print per-column statistics to stdout
-M, --morton-order                      Reorder Gaussians by Morton code (Z-order curve)
```

## General Options

```none
-h, --help                              Show this help and exit
-v, --version                           Show version and exit
-q, --quiet                             Suppress non-error output
    --verbose                           Show debug-level diagnostics
    --mem                               Show memory usage in progress output
    --tty                               Interactive bar rendering (default on a TTY; --no-tty to disable)
-w, --overwrite                         Overwrite output file if it exists
    --max-workers      <n>              Worker threads for CPU-heavy stages such as SOG encoding.
                                          0 runs everything inline on the calling thread. Peak memory
                                          scales with worker count, since each worker holds its own
                                          WebP WASM heap. Default: min(4, cores - 1)
```

## GPU Options

Used by SOG compression and GPU voxelization (`--filter-cluster`, `--filter-floaters`, `.voxel.json` output).

```none
-L, --list-gpus                         List available GPU adapters and exit
-g, --gpu              <n|cpu>          Device for GPU operations: GPU adapter index | 'cpu'
                                          ('cpu' disables GPU and is incompatible with
                                          GPU-only features like --filter-cluster)
```

## SOG Compression Options

Apply when writing `.sog`, `meta.json`, `lod-meta.json`, or `.html` outputs.

```none
-i, --iterations       <n>              Iterations for SH compression (more=better). Default: 10
```

## SPZ Output Options

Apply when writing `.spz` outputs.

```none
    --spz-version      <3|4>            The SPZ format version to write. Default: 4
```

## HTML Viewer Output Options

Apply when writing `.html` outputs.

```none
-E, --viewer-settings  <settings.json>  HTML viewer settings JSON file
-U, --unbundled                         Generate unbundled HTML viewer with separate files
```

> [!NOTE]
> See the [SuperSplat Viewer Settings Schema](https://github.com/playcanvas/supersplat-viewer?tab=readme-ov-file#settings-schema) for details on how to pass data to the `-E` option.

## LCC / LCC2 Input Options

Apply when reading `.lcc` and `.lcc2` files.

```none
-O, --lod-select       <n,n,...>        Comma-separated LOD levels to read from LCC / LCC2 input
```

## LOD Output Options

Apply when writing `lod-meta.json` (multi-LOD streaming SOG bundle).

```none
-C, --lod-chunk-count  <n>              Approximate number of Gaussians per LOD chunk in K. Default: 512
-X, --lod-chunk-extent <n>              Approximate size of an LOD chunk in world units (m). Default: 16
```

See [Generating Streamed SOG](https://developer.playcanvas.com/user-manual/splat-transform/#generating-lod-format) for an end-to-end walkthrough.

## Voxel Output Options

Apply when writing `.voxel.json` (sparse voxel octree for collision detection). See the [Collision Mesh Guide](https://developer.playcanvas.com/user-manual/splat-transform/collision/) for a deep dive on each step and tuning.

```none
    --voxel-params     [size,opacity]   Voxel size and opacity threshold. Default: 0.05,0.1
    --voxel-external-fill [size]        Seal exterior voxels via boundary flood fill (interior scenes).
                                          [size] (world units) is the dilation distance applied
                                          before the flood fill to bridge small wall gaps.
                                          --seed-pos is used to verify the volume is enclosed at
                                          the seed; the fill is skipped if the seed is reachable
                                          from outside.
                                          Default size: 1.6
    --voxel-floor-fill [size]           Fill each column upward from bottom until hitting solid (exterior scenes).
                                          Optional size (world units): only patch XZ areas surrounded by floor
                                          within 2*size; large empty exterior areas are left alone.
                                          Default size: 1.6
    --voxel-carve      [h,r]            Carve navigable space using capsule flood fill from seed.
                                          Default: height=1.6, radius=0.2
    --voxel-cleanup    [size]           Fill sampling holes, flatten bumpy surfaces and drop floating
                                          debris at this scale. Only voxels with gaussian density behind
                                          them are ever added, so real gaps and openings survive.
                                          Bare flag uses 2x the voxel size. Default: off
    --voxel-cleanup-fill [none|grow|close|both]
                                          Hole-filling algorithm for --voxel-cleanup. none runs only
                                          the smoothing and debris passes. Default: grow
    --seed-pos         <x,y,z>          Seed position for voxel fill/carve and --filter-cluster.
                                          Default: 0,0,0
-K, --collision-mesh   [smooth|faces|voxel|tris]
                                          Generate collision mesh (.collision.glb). voxel and tris additionally
                                          bake splat colors into a COLOR_0 vertex attribute, using an unlit
                                          material so the baked colors are not shaded again.
                                          Default: smooth
    --collision-color  [average|solid]  Vertex color algorithm for voxel/tris meshes. average blends the nearby
                                          splats; solid takes an opacity-weighted median so each surface keeps
                                          one flat color. Ignored for smooth/faces. Default: average
    --collision-color-palette  <n|colors>
                                          Quantize vertex colors to at most n colors, then give each vertex its
                                          nearest palette entry. The palette is chosen in Oklab with lightness
                                          weighted below chroma, and only colors forming spatially coherent
                                          regions can claim a slot — so small strongly-colored features survive
                                          instead of collapsing into the dominant tone, and scattered color
                                          noise cannot consume the budget.
                                          Given a comma-separated list of hex colors instead ('#3243aa,4444ff',
                                          #rgb or #rrggbb, the hash only needed on the first), palette selection
                                          is skipped and every vertex snaps to its nearest listed color, which
                                          the output then matches exactly. Quote the value — an unquoted # starts
                                          a shell comment. Default: off
    --collision-color-flat              Give every face one uniform color instead of interpolating across it.
                                          Combined with --collision-color-palette the face takes its dominant
                                          palette entry, so the output never exceeds the palette. Default: false
    --collision-color-smooth   <r>      Edge-preserving denoise within r voxels (fractional allowed) before the
                                          palette is built: each vertex averages only the neighbours whose color
                                          is perceptually close to its own, so noise inside a material averages
                                          out while boundaries between materials stay crisp. Max 8; 0 disables.
                                          Default: 2 with --collision-color-palette, off otherwise
    --collision-color-coherent <r>      After palette assignment, snap each vertex to the dominant palette color
                                          within r voxels. Removes leftover speckle while leaving palette
                                          entries exact and color boundaries crisp. Max 8; 0 disables.
                                          Default: off
    --collision-voxels  <file.vox>      Also write the collision voxels as a MagicaVoxel .vox model, coloured
                                          per voxel straight from the splats. Needs no collision mesh.
                                          Default: off
    --collision-voxels-size    <size>   Voxel size for the .vox model only, leaving the octree and collision
                                          mesh at the finer --voxel-params size. A region wider than 256 voxels
                                          per axis is tiled into several models automatically, so this is for
                                          keeping the model count and file size sensible rather than for
                                          fitting the axis limit. Must be >= --voxel-params size and is
                                          rounded to a whole multiple of it.
                                          Default: same as --voxel-params size
    --auto-rotate      [degrees]        Rotate the voxel grid to line up with the scene's dominant surfaces
                                          before voxelizing, cutting staircase voxels. Bare flag estimates the
                                          best yaw (about Y) from the splats' own surface normals; a number
                                          applies that yaw verbatim, skipping estimation.
                                          Default: off
```

`--auto-rotate` yaws the whole voxel grid — octree, `.collision.glb`, and `.vox` alike — before voxelizing, which turns a diagonal wall from a staircase of extra voxels into flat, efficient faces. A bare flag estimates the yaw by having each Gaussian vote for the orientation of its flattest axis, weighted by how flat, large, and opaque it is; if nothing wins clearly it skips rotation entirely and says why in the log line, rather than applying a meaningless angle to an organic scene.

The `.voxel.json` metadata and `.collision.glb` node both record the rotation that was applied, so they still line up with the original, unrotated splat when loaded. The `.vox` is written in the aligned frame with no rotation recorded — MagicaVoxel cannot express an arbitrary yaw, and an aligned model is the entire point of the flag.

```bash
# Estimate the best yaw automatically
splat-transform building.ply building.voxel.json --auto-rotate

# Apply a known 15 degree yaw directly, skipping estimation
splat-transform building.ply building.voxel.json --auto-rotate 15
```

### Cleaning up a scattered voxel grid

Voxelizing thresholds a continuous gaussian density field, and nothing afterwards reconstructs a
surface. When a scene's splats are small relative to the voxel size — common on large outdoor
captures at 5-10 cm — the result is not a surface at all but a scatter of near-isolated voxels:
full of holes, and violently bumpy. `--voxel-cleanup` fixes both, because they are the same
problem.

It runs three passes: fill voxels that look like holes in an existing surface, regularize the
surface with a 3x3x3 majority filter, then drop islands smaller than one 4x4x4 block. The
smoothing pass keeps any voxel with at least three occupied face neighbours, so a 1-voxel-thick
roof deck or wall survives it — a pure density test would delete thin surfaces rather than
smooth them, since they can never reach the threshold — while single-voxel bumps, scatter and
thin poles are still shaved off.

Crucially, **every voxel it adds must have gaussian density behind it.** The cleanup samples the
same field a second time at a far lower opacity threshold and uses that as a mask, so a real
window opening, a real gap between a railing and a deck, or a real void inside a building has no
density and is untouchable at any scale. Morphological closing on its own would fabricate: on the
scene below, an ungated close at the same radius placed ~140,000 voxels in effective vacuum, 46%
of everything it added. The gated pipeline places none.

On a city rooftop capture cropped to a 40 m box at 10 cm (52x35x50 m of occupied grid after
auto-rotate):

| | occupied voxels | disconnected islands | in the largest | surface roughness |
| --- | --- | --- | --- | --- |
| without | 324,997 | 11,349 | 67.8% | 2.35 voxels |
| `--voxel-cleanup 0.2` | 328,717 | 112 | 84.0% | 1.43 voxels |

Note the voxel count barely moves — +1.1% here: hole filling and scatter removal roughly cancel, so
cleanup redistributes voxels onto surfaces rather than adding bulk, and every addition is
density-gated (41.9K were blocked on this scene for having none behind them).

Measured with `tools/voxel-metrics.mjs`, which defines roughness as the mean deviation of each
column's topmost voxel from its neighbours' — compare the rows against each other rather than
against figures from elsewhere.

```bash
splat-transform city.spz city.voxel.json --voxel-params 0.1,0.1 --voxel-cleanup 0.2
```

Pass roughly twice the voxel size to start. A bare `--voxel-cleanup` does exactly that.
`--voxel-cleanup-fill none` skips hole filling and runs only the smoothing and debris passes, for
scenes whose coverage is already good.

The log reports surface coherence before cleanup, and suggests the flag when a grid is mostly
scatter, so you can tell whether a scene needs it.

The two spatial options are independent: `--collision-color-smooth` cleans the colors *before* the palette is built, `--collision-color-coherent` cleans the assignment *after*. Radii are in voxels, so they scale with `--voxel-params` size.

Smoothing runs at radius `2` by default whenever a palette is requested, because quantizing amplifies colour noise rather than hiding it. Palette entries are seeded from binned candidate colours, so a material's share of the palette follows how widely its colours *spread*, not how much of the mesh it covers: a noisy material such as reconstructed foliage spreads across many bins, collects a dozen entries, and neighbouring faces then alternate between them. The result is quantized output that reads as noisier than the splats it came from, even at a small palette size — the smooth colour gradient became hard-edged patches. Denoising first removes the spread, so those slots go to genuinely distinct materials instead.

The filter is edge-preserving, so this costs very little detail: a vertex only averages neighbours already within about two just-noticeable differences of its own colour, which excludes anything across a material boundary. On the reference diorama, the default takes same-material colour changes from ~22% of neighbouring face pairs down to ~7% and isolated speckle from ~3.8% to ~0.7%, while *lowering* colour error both overall and at material boundaries. Raise it towards `3`–`4` for very noisy captures, or pass `0` to turn it off and get the previous behaviour. Add `--collision-color-coherent 1` on top to absorb any speckle that survives into the assignment.

`--collision-voxels` writes the collision voxels as a MagicaVoxel model. Colours are sampled once per voxel, at its centre and using the average of its exposed face normals, then run through the same denoise and palette steps as the mesh — so every colour option above applies, and the `.vox` opens looking like the `.glb`. Sampling per voxel rather than per mesh vertex is both the natural granularity for the format (a MagicaVoxel voxel carries one colour) and far cheaper: on a 48 m landscape at 2 cm it is ~186K samples instead of ~18M, seconds instead of minutes. It also means the `.vox` needs no collision mesh at all — `--collision-voxels` works on its own.

Only voxels with at least one exposed face are sampled. A voxel enclosed on all six sides cannot be seen, so a BVH query for it buys nothing — and on a solid volume those are the large majority: a 200x180x204 house at 5 mm holds 3.07M occupied voxels of which only 292K (9.5%) are on the surface. Enclosed voxels then take the colour of the nearest surface voxel by breadth-first flood, which costs no BVH work and keeps the interior sensible if the model is later sliced open. Because the flood copies already-palettised colours it adds no palette entries, and because the palette is chosen from the surface alone it is not skewed by invisible voxels.

`XYZI` stores each coordinate in a single byte, so **one model spans at most 256 voxels per axis** — at 2 cm voxels a 5.12 m cube. A larger region is split into tiled models placed by the `nTRN`/`nGRP`/`nSHP` scene graph, so this is handled rather than rejected. Tiles are aligned to the occupied region, sized tightly around their own contents, and only non-empty tiles become models. A model also holds at most 255 colours, so a larger palette, or none, is reduced to 255 for the `.vox` only.

The one hard limit left is model count, capped at 256. That is a practical ceiling rather than a format one — the format only bounds node ids to int32 — but MagicaVoxel is not usable with thousands of objects. It is checked immediately after the grid is cropped, before any colouring, and the error names the smallest voxel size that fits:

```
The .vox model would be 9608x2726x1972 voxels, which needs more than 256 models of
256x256x256 to represent. The occupied region spans 48.0x13.6x9.9 world units, so
the .vox needs a voxel size of at least 0.01. Pass --collision-voxels-size 0.01 to
coarsen only the .vox and keep the collision grid, or raise --voxel-params to
coarsen everything.
```

The suggested size is a whole multiple of the collision voxel size and is verified to fit, so it works first time. Tiling makes it easy to ask for a model far larger than is useful, so a warning fires past 8M voxels:

```
! the .vox holds 152M voxels (578.4MB); MagicaVoxel is unlikely to open it usefully.
  Pass a larger --collision-voxels-size to coarsen the model without touching the
  collision grid.
```

`--collision-voxels-size` is the usual answer: it decouples the `.vox` from the collision resolution, reducing the grid for the model only (a coarse voxel is solid when any fine voxel inside it is), so the octree and `.collision.glb` keep their detail.

```bash
# 2 cm octree and collision mesh, 20 cm .vox
splat-transform landscape.spz landscape.voxel.json \
    --voxel-params 0.02,0.1 \
    --collision-voxels landscape.vox --collision-voxels-size 0.2
```

Note that collision *mesh* size grows with the square of the inverse voxel size: at 2 cm the landscape above is 36.8M triangles and a 2.06 GB `.glb`, which no engine will load usefully. Pick `--voxel-params` for the mesh you actually want and use `--collision-voxels-size` for the model.

Palette selection deliberately favours hue coverage over per-vertex colour accuracy, so a small strongly-coloured feature is kept rather than averaged away. That balance is set by a handful of tuning constants at the top of [`src/lib/mesh/palette.ts`](src/lib/mesh/palette.ts) — how far lightness is discounted against chroma (`LIGHTNESS_WEIGHT`), candidate colour granularity (`L_BIN_STEP` / `AB_BIN_STEP`), and what separates a real region from scattered noise (`SUPPORT_*`). They were calibrated against one reference scene that is ~92% warm by vertex count, so a scene with a very different colour balance may want different values; that file documents each constant.

To re-measure any of it, run `node --import tsx tools/color-noise-bench.mjs`. It builds a synthetic diorama whose true per-material colours are known, runs the real colouring pipeline over it, and reports spatial noise, colour drift and small-feature survival side by side, writing a top-down PNG per configuration so the numbers can be checked by eye.

## Image Output Options

Apply when writing `.webp` (lossless WebP rendered via GPU rasterizer).

```none
    --projection       <pinhole|equirect>  Camera projection. Default: pinhole.
                                        equirect = 360°×180° panorama from --camera; --fov must be
                                        omitted; --resolution must be 2:1 (default 2048x1024).
    --camera           <x,y,z>          Camera position in world space. Default: 2,1,-2
    --look-at          <x,y,z>          Camera target point. Default: 0,0,0
    --up               <x,y,z>          World up vector. Default: 0,1,0
    --fov              <degrees>        Vertical field of view in degrees. Default: 60. Rejected with --projection equirect.
    --resolution       <WxH>            Output resolution, e.g. 1920x1080. Default: 1280x720 (pinhole) or 2048x1024 (equirect)
    --near             <n>              Near clip distance. Default: 0.2 (matches reference 3DGS)
    --background       <r,g,b[,a]>      Background color in [0,1]. Default: 0,0,0,1
    --f-stop           <N>              Aperture as a photographic f-stop (e.g. 2.8, 5.6, 11). Enables defocus blur;
                                        smaller = more blur. Pinhole only. Default: disabled (no defocus).
    --focus-distance   <n>              Camera-space Z of the focus plane (world units). Default: distance to --look-at.
                                        Pinhole only; only meaningful with --f-stop.
    --sensor-size      <n>              Vertical sensor height in world units. Gives --f-stop a physical meaning.
                                        Default: 0.024 (35mm full-frame, world units = meters). Scale to your world:
                                        world unit = decimeter → 0.24, world unit = millimeter → 24.
    --camera-end       <x,y,z>          End camera position. When set, enables camera motion blur: the renderer
                                        averages sub-frames with the camera interpolated from --camera (shutter open)
                                        to --camera-end (shutter close). Default: disabled (no motion blur).
    --look-at-end      <x,y,z>          End camera target. Default: same as --look-at. Only with --camera-end.
    --up-end           <x,y,z>          End up vector. Default: same as --up. Only with --camera-end.
    --shutter          <0..1>           Fraction of the start→end segment integrated, centered on the midpoint
                                        (1.0 = full motion; 0.5 = 180° shutter). Default: 1. Only with --camera-end.
    --motion-samples   <n>              Sub-frames to accumulate for motion blur. Cost is N× a single render.
                                        Default: 16. Only with --camera-end.
```

## Examples

### Basic Operations

```bash
# Simple format conversion
splat-transform input.ply output.csv

# Convert from .splat format
splat-transform input.splat output.ply

# Convert from .ksplat format
splat-transform input.ksplat output.ply

# Convert to compressed PLY
splat-transform input.ply output.compressed.ply

# Uncompress a compressed PLY back to standard PLY
# (compressed .ply is detected automatically on read)
splat-transform input.compressed.ply output.ply

# Convert to SOG bundled format
splat-transform input.ply output.sog

# Convert to SOG unbundled format
splat-transform input.ply output/meta.json

# Convert from SOG (bundled) back to PLY
splat-transform scene.sog restored.ply

# Convert from SOG (unbundled folder) back to PLY
splat-transform output/meta.json restored.ply

# Convert to standalone HTML viewer (bundled, single file)
splat-transform input.ply output.html

# Convert to unbundled HTML viewer (separate CSS, JS, and SOG files)
splat-transform -U input.ply output.html

# Convert to HTML viewer with custom settings
splat-transform -E settings.json input.ply output.html
```

### Transformations

```bash
# Scale and translate
splat-transform bunny.ply -s 0.5 -t 0,0,10 bunny_scaled.ply

# Rotate by 90 degrees around Y axis
splat-transform input.ply -r 0,90,0 output.ply

# Chain multiple transformations
splat-transform input.ply -s 2 -t 1,0,0 -r 0,0,45 output.ply
```

### Filtering

```bash
# Remove entries containing NaN and Inf
splat-transform input.ply --filter-nan output.ply

# Filter by opacity values (keep only splats with opacity > 0.5)
splat-transform input.ply -V opacity,gt,0.5 output.ply

# Strip spherical harmonic bands higher than 2
splat-transform input.ply --filter-harmonics 2 output.ply

# Simplify to 50000 splats via progressive pairwise merging
splat-transform input.ply --decimate 50000 output.ply

# Simplify to 25% of original splat count
splat-transform input.ply -F 25% output.ply
```

### Advanced Usage

```bash
# Combine multiple files with different transforms
splat-transform -w cloudA.ply -r 0,90,0 cloudB.ply -s 2 merged.compressed.ply

# Apply final transformations to combined result
splat-transform input1.ply input2.ply output.ply -t 0,0,10 -s 0.5
```

### Statistical Summary

Generate per-column statistics for data analysis or test validation:

```bash
# Print summary, then write output
splat-transform input.ply --summary output.ply

# Print summary without writing a file (discard output)
splat-transform input.ply -m null

# Print summary before and after a transform
splat-transform input.ply --summary -s 0.5 --summary output.ply
```

The summary includes min, max, median, mean, stdDev, nanCount and infCount for each column in the data.

### Generators (Beta)

Generator scripts can be used to synthesize gaussian splat data. See [gen-grid.mjs](generators/gen-grid.mjs) for an example.

```bash
splat-transform gen-grid.mjs -p width=10,height=10,scale=10,color=0.1 scenes/grid.ply -w
```

### Voxel Format

The voxel format stores sparse voxel octree data for collision detection. It consists of two files: `.voxel.json` (metadata) and `.voxel.bin` (binary octree data). Pass `-K` to also emit a `.collision.glb` mesh derived from the voxel grid, and `--collision-voxels` to additionally write those voxels as a MagicaVoxel `.vox` model (optionally at a coarser size via `--collision-voxels-size`).

For a step-by-step walkthrough of each option (with illustrations), see the [Collision Mesh Guide](https://developer.playcanvas.com/user-manual/splat-transform/collision/).

#### Recommended pipeline

```bash
splat-transform input.ply \
    --filter-cluster --seed-pos x,y,z \
    [--voxel-external-fill | --voxel-floor-fill] [--voxel-carve] \
    [-K [smooth|faces]] \
    output.voxel.json
```

`--filter-cluster` isolates the central scene and discards stray floaters before voxelization. `--seed-pos` is shared by `--filter-cluster` and the voxel fill/carve passes — set it once to a known-walkable point inside the scene.

#### Interior scenes (rooms, indoor scans)

Use `--voxel-external-fill` to seal the void around the room interior, then `--voxel-carve` to hollow out the navigable space:

```bash
splat-transform room.ply \
    --filter-cluster --seed-pos 0,1,0 \
    --voxel-external-fill --voxel-carve \
    -K room.voxel.json
```

#### Exterior scenes (outdoor objects, terrain)

Use `--voxel-floor-fill` to fill the ground beneath surfaces, optionally followed by `--voxel-carve`:

```bash
splat-transform terrain.ply \
    --filter-cluster --seed-pos 0,0,0 \
    --voxel-floor-fill \
    -K terrain.voxel.json
```

#### Other examples

```bash
# Voxelize with custom resolution and opacity threshold
splat-transform --voxel-params 0.1,0.3 input.ply output.voxel.json

# Custom carve capsule (height, radius)
splat-transform --seed-pos 1,0,0 --voxel-carve 2.0,0.3 input.ply output.voxel.json

# Watertight voxel-face collision mesh
splat-transform -K faces input.ply output.voxel.json

# Colored voxel collision mesh, quantized to a 16-color palette
splat-transform -K voxel --collision-color-palette 16 input.ply output.voxel.json

# Restricted to an explicit palette instead of one chosen from the scene
splat-transform -K voxel --collision-color-palette '#3243aa,4444ff' input.ply output.voxel.json

# Same, with flat per-voxel faces and a wider denoise for a noisy capture
splat-transform -K voxel --collision-color solid --collision-color-flat \
    --collision-color-palette 16 --collision-color-smooth 3 --collision-color-coherent 1 \
    input.ply output.voxel.json

# Also emit a MagicaVoxel model of the same voxels and colors
splat-transform -K voxel --collision-color solid --collision-color-flat \
    --collision-color-palette 16 --collision-color-coherent 1 \
    --collision-voxels output.vox \
    input.ply output.voxel.json

# MagicaVoxel model on its own - no collision mesh needed
splat-transform --collision-color solid --collision-color-palette 24 \
    --collision-voxels output.vox \
    input.ply output.voxel.json

# Fine octree, coarse .vox: the model fits 256 voxels per axis either way
splat-transform --voxel-params 0.02,0.1 --collision-color-palette 24 \
    --collision-voxels output.vox --collision-voxels-size 0.2 \
    input.ply output.voxel.json

# Rotate the voxel grid to line up with the scene before voxelizing
splat-transform --auto-rotate --collision-voxels output.vox input.ply output.voxel.json
```

### Image Rendering

Render a splat scene to a lossless WebP image from a given camera view. Rendering runs on the GPU.

```bash
# Default 1280x720 render
splat-transform input.ply view.webp

# Custom camera and resolution
splat-transform input.ply view.webp \
    --camera 2,1,-2 --look-at 0,0,0 \
    --fov 50 --resolution 1920x1080

# Transparent background
splat-transform input.ply view.webp --background 0,0,0,0

# Defocus blur (focus on look-at, f/2.8 aperture)
splat-transform input.ply view.webp --f-stop 2.8

# Defocus with explicit focus distance and a smaller world scale
splat-transform input.ply view.webp \
    --f-stop 2.8 --focus-distance 3 --sensor-size 0.1

# 360° equirectangular panorama from camera position
splat-transform input.ply pano.webp \
    --projection equirect --camera 0,1,0 --look-at 0,1,1

# Camera motion blur (dolly from start to end pose over the shutter)
splat-transform input.ply view.webp \
    --camera 2,1,-2 --camera-end 3,1,-2 \
    --motion-samples 16 --shutter 1
```

### Device Selection for SOG Compression

When compressing to SOG format, you can control which device (GPU or CPU) performs the compression:

```bash
# List available GPU adapters
splat-transform --list-gpus

# Let WebGPU automatically choose the best GPU (default behavior)
splat-transform input.ply output.sog

# Explicitly select a GPU adapter by index
splat-transform -g 0 input.ply output.sog  # Use first listed adapter
splat-transform -g 1 input.ply output.sog  # Use second listed adapter

# Use CPU for compression instead (much slower but always available)
splat-transform -g cpu input.ply output.sog
```

> [!NOTE]
> When `-g` is not specified, WebGPU automatically selects the best available GPU. Use `-L` to list available adapters with their indices and names. The order and availability of adapters depends on your system and GPU drivers. Use `-g <index>` to select a specific adapter, or `-g cpu` to force CPU computation.

> [!WARNING]
> CPU compression can be significantly slower than GPU compression (often 5-10x slower). Use CPU mode only if GPU drivers are unavailable or problematic.

## Getting Help

```bash
# Show version
splat-transform --version

# Show help
splat-transform --help
```

---

## Library Usage

SplatTransform exposes a programmatic API for reading, processing, and writing Gaussian splat data.

### Basic Import

```typescript
import {
    readFile,
    writeFile,
    getInputFormat,
    getOutputFormat,
    DataTable,
    processDataTable
} from '@playcanvas/splat-transform';
```

### Key Exports

| Export | Description |
| ------ | ----------- |
| `readFile` | Read splat data from various formats |
| `writeFile` | Write splat data to various formats |
| `getInputFormat` | Detect input format from filename |
| `getOutputFormat` | Detect output format from filename |
| `DataTable`, `Column` | Core data structures for splat data |
| `combine` | Merge multiple DataTables into one |
| `convertToSpace` | Convert a DataTable between coordinate spaces |
| `processDataTable` | Apply a sequence of processing actions |
| `computeSummary` | Generate statistical summary of data |
| `sortMortonOrder` | Sort indices by Morton code for spatial locality |
| `sortByVisibility` | Sort indices by visibility score for filtering |
| `writeVoxel` | Write sparse voxel octree files, plus the optional `.collision.glb` and `.vox` |
| `writeImage` | Render a camera view to a lossless WebP image (requires GPU) |
| `renderSplats` | Lower-level renderer returning the raw RGBA byte buffer |
| `SparseVoxelGrid` | The voxel grid every collision/voxel path operates on |
| `buildCollisionVox` | Encode a grid as a MagicaVoxel `.vox`, colours sampled from the splats |
| `buildCollisionMesh` | Extract a collision mesh from a grid and encode it as GLB |
| `buildSparseOctree` | Build the sparse voxel octree `writeVoxel` serialises |
| `estimateAlignYaw`, `applyAlignYaw` | Estimate and apply the yaw `writeVoxel`'s `autoRotate` option uses to align the voxel grid |
| `voxelFaces`, `marchingCubes` | Mesh extraction from a grid |
| `GaussianBVH`, `computeGaussianExtents` | Build the colour source the above need |

### File System Abstractions

The library uses abstract file system interfaces for maximum flexibility:

**Reading:**
- `UrlReadFileSystem` - Read from URLs (browser/Node.js)
- `MemoryReadFileSystem` - Read from in-memory buffers
- `ZipReadFileSystem` - Read from ZIP archives

**Writing:**
- `MemoryFileSystem` - Write to in-memory buffers
- `ZipFileSystem` - Write to ZIP archives

### Example: Reading and Processing

```typescript
import { Vec3 } from 'playcanvas';
import {
    readFile,
    writeFile,
    getInputFormat,
    getOutputFormat,
    processDataTable,
    UrlReadFileSystem,
    MemoryFileSystem
} from '@playcanvas/splat-transform';

// Read a PLY file from URL
const fileSystem = new UrlReadFileSystem();
const inputFormat = getInputFormat('scene.ply');

const dataTables = await readFile({
    filename: 'https://example.com/scene.ply',
    inputFormat,
    options: { iterations: 10 },
    params: [],
    fileSystem
});

// Apply transformations
const processed = processDataTable(dataTables[0], [
    { kind: 'scale', value: 0.5 },
    { kind: 'translate', value: new Vec3(0, 1, 0) },
    { kind: 'filterNaN' }
]);

// Write to in-memory buffer
const memFs = new MemoryFileSystem();
const outputFormat = getOutputFormat('output.ply', {});

await writeFile({
    filename: 'output.ply',
    outputFormat,
    dataTable: processed,
    options: {}
}, memFs);

// Get the output data
const outputBuffer = memFs.files.get('output.ply');
```

### Processing Actions

The `processDataTable` function accepts an array of actions:

```typescript
type ProcessAction =
    | { kind: 'translate'; value: Vec3 }
    | { kind: 'rotate'; value: Vec3 }       // Euler angles in degrees
    | { kind: 'scale'; value: number }
    | { kind: 'filterNaN' }
    | { kind: 'filterByValue'; columnName: string; comparator: 'lt'|'lte'|'gt'|'gte'|'eq'|'neq'; value: number }
    | { kind: 'filterBands'; value: 0|1|2|3 }
    | { kind: 'filterBox'; min: Vec3; max: Vec3 }
    | { kind: 'filterSphere'; center: Vec3; radius: number }
    | { kind: 'filterFloaters'; voxelResolution?: number; opacityCutoff?: number; minContribution?: number } // GPU
    | { kind: 'filterCluster'; voxelResolution?: number; seed?: Vec3; opacityCutoff?: number; minContribution?: number } // GPU
    | { kind: 'decimate'; count: number | null; percent: number | null }
    | { kind: 'param'; name: string; value: string }
    | { kind: 'lod'; value: number }
    | { kind: 'summary' }
    | { kind: 'mortonOrder' };
```

> [!NOTE]
> `filterFloaters` and `filterCluster` require a GPU device — pass `createDevice` via the `ProcessOptions` argument to `processDataTable`.

### Voxel and Collision Generation

`writeVoxel` composes the voxel pipeline and writes files. Every stage it uses is
also exported, so a consumer can drive the same paths directly — in the browser
included — and keep the bytes in memory instead of writing them.

Colouring needs a `GaussianBVH` over the splats plus their colour columns:

```javascript
import {
    buildCollisionVox, buildCollisionMesh, computeGaussianExtents,
    GaussianBVH, SparseVoxelGrid
} from '@playcanvas/splat-transform';
import { Vec3 } from 'playcanvas';

// a grid can come from the voxelizer, or be filled directly
const grid = new SparseVoxelGrid(16, 16, 16);
for (let z = 0; z < 16; z++)
    for (let y = 0; y < 16; y++)
        for (let x = 0; x < 16; x++) grid.setVoxel(x, y, z);

const voxelResolution = 0.5;
const gridBounds = { min: new Vec3(0, 0, 0), max: new Vec3(8, 8, 8) };

const { extents } = computeGaussianExtents(splats, 0.1);
const colorSource = {
    bvh: new GaussianBVH(splats, extents),
    columns: {
        f_dc_0: splats.getColumnByName('f_dc_0').data,
        f_dc_1: splats.getColumnByName('f_dc_1').data,
        f_dc_2: splats.getColumnByName('f_dc_2').data,
        opacity: splats.getColumnByName('opacity').data
    },
    mode: 'solid',      // or 'average'
    palette: 64         // optional; also smoothRadius / coherentRadius
};

// MagicaVoxel model, as raw bytes
const vox = buildCollisionVox(grid, gridBounds, voxelResolution, colorSource);

// collision mesh, as GLB bytes
const glb = buildCollisionMesh(grid, gridBounds, voxelResolution, 'voxel', {
    ...colorSource,
    flatShade: true
});
```

The `.vox` limits can be inspected before committing to any colouring work, which
is the cheap part of the pipeline:

```javascript
import {
    assertVoxFits, countVoxModels, downsampleGrid,
    enumerateOccupied, minVoxFactorForModels, MAX_VOX_MODELS
} from '@playcanvas/splat-transform';

const occupied = enumerateOccupied(grid);          // count + tight bounds
countVoxModels(occupied, 1);                        // models needed at full detail
minVoxFactorForModels(occupied);                    // smallest reduction that fits
assertVoxFits(occupied, voxelResolution, 1);        // throws with an actionable message

// reduce for the model only, leaving `grid` untouched
const plan = downsampleGrid(grid, gridBounds, voxelResolution, 4);
const smaller = buildCollisionVox(plan.grid, plan.gridBounds, plan.voxelResolution, colorSource);
```

`forEachExposedFace` and `SparseVoxelGrid.forEachOccupiedVoxel` iterate the sparse
block structure directly, so custom meshing or analysis costs time proportional to
occupancy rather than to grid volume:

```javascript
import { forEachExposedFace } from '@playcanvas/splat-transform';

let faces = 0;
forEachExposedFace(grid, (x, y, z, bucket) => faces++);  // bucket: 0..5 = -X,+X,-Y,+Y,-Z,+Z

grid.forEachOccupiedVoxel((x, y, z) => { /* ... */ });
```

### Auto-Rotate (Voxel Grid Alignment)

The easiest way to use `--auto-rotate` programmatically is the same option `writeVoxel` exposes to the CLI — pass `autoRotate: true` (estimate the best yaw) or a number (apply that yaw in degrees, skipping estimation):

```typescript
import { writeVoxel, MemoryFileSystem } from '@playcanvas/splat-transform';

const fs = new MemoryFileSystem();
await writeVoxel({
    filename: 'building.voxel.json',
    dataTable: myDataTable,
    autoRotate: true,                      // or e.g. 15 to apply a known yaw directly
    collisionMesh: true,
    createDevice: async () => myGraphicsDevice
}, fs);
```

For finer control — inspecting the cost curve, overriding the up axis, or applying the rotation to a `DataTable` outside `writeVoxel`'s own pipeline — call the estimator and its transform helper directly:

```typescript
import { estimateAlignYaw, applyAlignYaw } from '@playcanvas/splat-transform';
import { Transform } from '@playcanvas/splat-transform';

const result = estimateAlignYaw(myDataTable); // options: { up, stepDegrees, minImprovement, opacityCutoff }

if (result.reason) {
    console.log(`no rotation applied: ${result.reason}`);
} else {
    console.log(`yaw ${result.yawDegrees.toFixed(2)} saves ~${(result.improvement * 100).toFixed(0)}%`);

    // Compose the yaw into a write transform (and rotate a nav seed alongside it, if any):
    const { delta, navSeed, recordedRotation } = applyAlignYaw(new Transform(), mySeed, result.yawDegrees);
    // `recordedRotation` ([x, y, z, w]) maps the aligned frame back to source space -
    // record it wherever your own pipeline needs to undo the rotation later.
}
```

`estimateAlignYaw` only reads the `DataTable`'s rotation, scale, and opacity columns plus its `transform` — no GPU, no file I/O — so it runs anywhere the library does, including the browser.

### Custom Logging

Configure the logger for your environment:

```typescript
import { logger } from '@playcanvas/splat-transform';

logger.setLogger({
    log: console.log,
    warn: console.warn,
    error: console.error,
    debug: console.debug,
    progress: (text) => process.stdout.write(text),
    output: console.log
});

logger.setQuiet(true); // Suppress non-error output
```
