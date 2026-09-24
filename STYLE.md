# Pixel Video Style Specification

These dimensions and timing rules are measurements of the reference-video styles.
Creator links and attribution are in [README.md](README.md). Use this file as a visual
specification, and use `pxv analyze` to measure a new reference rather than guessing.

| Reference mechanism | Style | Native canvas | Upscale | Duration | Audio |
| --- | --- | --- | --- | --- | --- |
| MLP training on MNIST | v1 neon | 320×180 | ×6 | 56 s | NES-style effects |
| Diffusion U-Net | v1 neon | 320×180 | ×6 | 21.5 s loop | NES-style effects |
| V-JEPA 2.1 | v2 phosphor | 480×270 | ×4 | 29 s | Silent |
| π0.5 | v2 phosphor | 480×270 | ×4 | 36 s | Silent |

Both styles produce 1920×1080 at 60 fps. Draw each frame on the native pixel grid and
upscale by the listed integer factor with nearest-neighbor sampling. Snap all positions
to whole canvas pixels. Diagonals should have visible pixel steps; circles use pixel rings.

## v1: neon loop

Use a near-black background (`#03030b`), no bloom, no vignette, and no grid. Apart from
small 3×5 numbers, the screen is largely wordless. Quantize images and UI colors to the
same 32-color palette (`pixelkit.NEON32`) so noise, frames, and interface belong together.

| Element | Color or behavior |
| --- | --- |
| Idle outlines | `#32467b`; fill `#080c1b` |
| Idle wires | `#1b2341`, solid or short dashed |
| Forward calculation | Cyan `#55d3d5` → light blue `#20b1fb` → blue `#1037b4` |
| Backward calculation | Pink `#fd5aac` → magenta `#fc1c76` → dark red `#7b092c` |
| Time or condition | Yellow `#fcc01c` for the active value |
| Result | Brief white outline flash |

For a diffusion loop, keep the U-Net and its skip connections on screen. Show the forward
noise schedule, one sampled time step, a loss calculation, the reverse operation, and a
return to the initial composition. A one-pixel screen nudge can mark a result; a short
leftward hit can mark a loss. Dissolve all imagery back toward the opening arrangement
before the loop restarts.

For an MLP training loop, place input pixels at the left, hidden layers in the center,
and class probabilities at the right. Light layers in forward order, show the prediction
and loss, then propagate the pink backward signal. Keep most weights dim; highlight only
enough connections to explain the update. Pitch can rise on the forward pass and descend
on the backward pass.

## v2: phosphor explainer

The 480×270 canvas uses a `#090b19` background. Add a dot-grid pixel every six canvas
pixels at x≡3, y≡3. For bloom, blur a 2× low-resolution copy of the ink with σ≈2.6×2,
then blend it behind the sharp upscaled pixels at roughly half strength. Keep the ink
itself sharp; a second blur over text makes it hard to read.

Reserve y<18 for the title/status/progress header and y≥243 for captions. The main
content sits around y=20…238. Place the title at (10,6), status at x=240, stage label
near the progress squares, and small progress squares near the upper-right edge.
The dotted caption rule is at y=243; caption lines start at y=249 and y=259.

| Element | Rule |
| --- | --- |
| Title | Bright white `#f0f2f9`, LCD font |
| Status | Centered, muted `#c3c9e8`; report an equation or size |
| Idle module | Dim outline and label, present from the start |
| Active module | Accent color and animated data |
| Completed module | Cooler, less bright state |
| Wires | Dim when idle; move particles while data travels |
| Captions | Two lines, about 120 Latin characters/s; second line starts 0.2 s later |

Draw the whole mechanism as a dim, stable system map from about 0.3 s onward. Each
stage lights its module and wire and changes the data inside the frame. Do not introduce
an entirely new diagram at every stage. Use real or reproducible toy-scale values for
matrices, vectors, token counts, and plots. When an image comes from a paper or project,
cite it in the caption.

## Text and covers

Use the built-in proportional 5×7 LCD font for labels and captions. The 3×5 font is
available for tiny numbers. `pxv check` detects unavailable glyphs and text that runs
into the title or caption bands. Keep status lines compact, e.g. `STEP 6 / 10 · t = 0.5`.
Use the first caption line for the action and the second for quantitative detail or
honest caveats such as `ILLUSTRATIVE` or `THE MODEL WAS NOT RUN`.

A 16:9 cover may reuse a representative video frame. The 4:3 (2400×1800) and 3:4
(1800×2400) covers enlarge the mechanism content with a pixel title and byline. Crop
around the important part of the system, preserve the pixel grid, and leave space
between title, image, and source credit.

## Reference-video analysis

`pxv analyze` provides scale, grid, palette, motion, OCR, and frame-comparison tools.
Check native resolution and integer scaling before copying colors or coordinates.
Measure several frames to distinguish actual motion from a palette change or bloom.
