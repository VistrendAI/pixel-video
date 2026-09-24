---
name: pixel-video
description: >-
  Create an English pixel-art mechanism explainer from a topic, paper, or code repository.
  Draw at 480×270 or 320×180, upscale by an integer factor to 1080p at 60 fps, and
  animate one mechanism step by step. Use real computed values when available; label
  illustrative inputs clearly. Includes project setup, fact tracking, storyboard templates,
  reusable widgets, layout checks, rendering, covers, and reference-video analysis.
  Use for pixel-art, 8-bit, retro LCD, or CRT-style explanations of algorithms and papers.
---

# Pixel Video

Create a short animation that follows **one mechanism** from input to output. Each stage
should transform data visibly. The status line reports the current equation or size; the
first caption line explains what happens, and the second gives numbers, sources, or caveats.
The video-style references and credits are in [README.md](README.md); measured design
details are in [STYLE.md](STYLE.md).

Call `pxv` through its installed full path, because symlinks break asset discovery:

```bash
PXV="$HOME/.claude/skills/pixel-video/pxv"
"$PXV" init MyTopic --paper 2501.00001 --code https://github.com/example/project
"$PXV" stills scene.py --times 0.5,4,8,12
"$PXV" check scene.py
"$PXV" still scene.py --t 8 --full
"$PXV" render scene.py
"$PXV" cover scene.py --t 12 --title "How it works" --byline "Paper and figure source"
```

## 1. Start a project

`pxv init NAME` creates a dated project under the current directory. Use `--root DIR` to
choose a parent directory or `--dir DIR` for an exact path. The project contains `scene.py`,
`prep.py`, `FACTS.md`, and `paper/`, `code/`, `data/`, and `out/` directories.

| Input | Next step |
| --- | --- |
| Topic only | Find the original paper and official code; resolve ambiguous paper versions. |
| Paper URL, arXiv ID, or PDF | Run `pxv init --paper`; use a local PDF if the download fails. |
| Code repository | Run `pxv init --code`; inspect the released configuration before drawing. |

Use the v2 explainer by default. Use v1 for a single wordless loop with 8-bit sound.

## 2. Build `FACTS.md`

Read the paper, appendix, figures, and relevant code/configuration. `paper/paper.txt` is
the extracted text, `paper/pages/` contains page previews, and `paper/figs/` contains
extracted bitmaps. Vector figures may need a high-resolution crop:

```bash
"$PXV" fig paper/paper.pdf --page 3 --box x0,y0,x1,y1
```

Give **every number shown on screen** a source: a page, table, equation, or code location.
Do not invent or casually round figures. When prose and a released configuration differ,
record both and explain which one the animation uses.

## 3. Pick one mechanism and storyboard it

Follow one forward pass, training step, or sampling loop. Keep only the context needed to
understand the new step. Give that step more screen time than routine context.

| | v2 explainer | v1 loop |
| --- | --- | --- |
| Length | About 25–45 s, 5–10 stages | About 15–60 s |
| Per stage | 2.5–6 s; hold the completed action for at least 1 s | Use one explicit event timeline |
| Text | Title, status equation, two caption lines | Numbers only, if needed |
| Audio | Silent by default | NES-style effects |

Useful flows include input → tokenization → encoder → predictor → loss for representation
learning; noise schedule → training prediction → sampling loop for diffusion; and prompt →
prefill → cache → decode for KV Cache. Keep the entire system map visible from about 0.3 s,
dim when idle. At each stage, light up the active module, animate the connection, and fill
its data slots. The [v2 scene template](templates/scene_v2.py) and
[KV Cache example](examples/kv_cache.py) show this structure.

Write short factual captions. Put the action on line one and concrete sizes, counts,
sources, or caveats on line two. Mark toy examples `TOY MODEL`, schematic elements
`ILLUSTRATIVE`, and unrun models `THE MODEL WAS NOT RUN`. The status line should describe
the current operation, such as `224 × 224 → 16 × 16 = 256 TOKENS`.

## 4. Compute data before drawing

Prefer, in order:

1. Run a small input through the official code and capture intermediate tensors.
2. Reproduce the core mathematics at toy scale with seeded NumPy calculations.
3. Use cited table or figure values from the paper.

Put calculations in `prep.py`, save them to `data/data.npz`, and load them in `scene.py`.
Seed every random draw during setup because rendering can evaluate frames out of order.
If a model requires substantial hardware, size the run for the available machine.

## 5. Draw and check

Use a 480×270 canvas for v2 and a 320×180 canvas for v1. Place data flow left to right.
Put stable positions in named integer constants. `pixelkit/widgets.py` contains modules,
wires, token grids, matrices, vectors, plots, tables, robot arms, and other reusable parts;
see [the widget gallery](examples/widgets_gallery.png).

The v2 content area is approximately y=20…238. Keep drawings out of the title and caption
bands. Draw the caption chrome last. A static system map prevents sparse early frames;
animate state and data rather than moving the whole diagram around.

```bash
"$PXV" stills scene.py --times 0.5,4,8,12
"$PXV" check scene.py
"$PXV" still scene.py --t 8 --full
```

`check` catches missing glyphs, text overflow and overlap, title collisions, content in
reserved bands, long motionless stretches, and loop seams. It cannot verify factual
accuracy or every frame between samples. Inspect the contact sheet and several full-size
frames after it passes.

## 6. Render and export

```bash
"$PXV" render scene.py
"$PXV" cover scene.py --t 12 --title "How it works" --byline "Source: paper and figure"
```

`render` writes the MP4 to `out/`. `cover` can create 16:9, 4:3, and 3:4 images. Use
`--crop34` to focus a portrait cover and `--title34` to shorten its title. Choose outputs
based on the actual delivery request.

## Files

| Path | Purpose |
| --- | --- |
| `pxv` | Project creation, checks, rendering, covers, and analysis CLI |
| `pixelkit/` | Canvas, fonts, visual widgets, audio, and rendering |
| `templates/` | v1/v2 scenes, data preparation, and fact sheet |
| `examples/` | KV Cache, diffusion, MLP, and widget gallery |
| `tools/analyze.py` | Reference-video scale, grid, palette, motion, and OCR analysis |

The English LCD font is built into `pixelkit/fonts.py`. If a glyph is unavailable, `pxv
check` reports it; revise the text before rendering.
