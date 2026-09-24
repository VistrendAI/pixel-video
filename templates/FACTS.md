# {NAME} · Facts and sources

Every number in the video or captions needs a source here: paper page, table, equation,
figure, or code file and line. Mark uncertain details as `TO VERIFY`. Do not invent or
casually round numbers. If a released config differs from the paper, record both.

## 1. One-sentence mechanism

What does this method do, and which transformation makes it distinctive?

## 2. The one mechanism to animate

- Scope: one training step, inference loop, tokenization step, or module internals.
- Why this step: where does the paper's new contribution happen?
- Style: v2 explainer or v1 wordless loop.

## 3. Input → stages → output

| Stage | What happens | Shape or value | Source |
| --- | --- | --- | --- |
| Input | | | |
| | | | |

## 4. Training objective and loss

Write the original formula and define each term.

## 5. Inference loop

Record step count, schedule, stopping condition, and active modules.

## 6. Scale and hyperparameters

Record layers, widths, parameters, token counts, resolutions, and frame rates.

## 7. Results to show

Choose one figure or result number and record its exact source. State when original
figure colors are used.

## 8. Values that can be computed

- Official code entry point and smallest useful input:
- Required GPU or model weights, if any:
- NumPy toy-scale reproduction and seed:
- `prep.py` output file, shapes, and displayed values:

## 9. Illustrative elements

List everything schematic or simulated. Put the relevant caveat in the video's
second caption line, such as `ILLUSTRATIVE` or `THE MODEL WAS NOT RUN`.

## 10. Visual assets

- Paper figure, page number, and extraction path:
- Project video or demo URL and frame source:
