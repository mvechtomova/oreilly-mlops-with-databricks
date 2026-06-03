---
name: png-to-excalidraw
description: Convert raw PNG diagrams in visuals_raw/ into Excalidraw JSON files in visuals/ that match the brand palette and styling. Trigger whenever the user asks to turn a PNG (or screenshot/sketch) into an Excalidraw diagram, recreate a diagram in Excalidraw, restyle a PNG diagram with the project palette, or mentions converting files under visuals_raw/.
---

# png-to-excalidraw

Convert a raw PNG diagram into an `.excalidraw.json` file that matches the source layout and content, restyled with the project's brand palette and consistent Excalidraw element styling.

## When to use

Trigger for any of: "turn this PNG into Excalidraw", "recreate this diagram", "convert visuals_raw/X.png", "redo this in Excalidraw", "make an Excalidraw version of...". If the user pastes or points at a PNG inside `visuals_raw/`, assume this skill applies.

## Input

A PNG filename or path, passed as the argument or referenced in the user's message.

Resolution rules:
- Filename only (e.g. `mlops_components.png`) → look in `visuals_raw/<arg>`.
- Path given → use as-is. Must be a `.png`.
- Not found → list PNGs in `visuals_raw/` with `ls visuals_raw/*.png` and stop.
- Not a PNG → stop and tell the user.
- Missing argument → ask which PNG to convert; list available ones.

## Output

Write to `visuals/<basename_without_ext>.excalidraw.json`. If the file already exists, ask before overwriting — the user may have hand-edited it.

## Procedure

### 1. View the source

Use `Read` on the PNG. The image renders directly in the tool result — you can see boxes, labels, icons, arrows, and groupings.

### 2. Inventory

Make a mental (or short scratch) list of:
- Every distinct card / box and its label
- **The internal layout of each card** — is text centered as a single column, or split side-by-side (e.g. label on the left, code or value on the right)? Note this per card before placing anything. See "Inspect each card's internal layout" in step 5.
- Multi-line text inside cards (e.g. tool lists like "GitHub Actions / Jenkins / Azure pipelines", or code examples with several lines)
- Section containers / group backgrounds (if any)
- Icons or accent shapes
- Arrows and connectors with their direction
- Rows and columns — most diagrams here are grids; note the dimensions (e.g. 4×5)

The goal is to **match layout and content**, not redesign. Same boxes, same labels, same relative positions, and the **same intra-card arrangement**.

### 3. Plan the canvas

- Default canvas width: 1100–1400px depending on column count.
- Pick a uniform box size for each row/column and reuse it across the grid — this is what makes the result look intentional rather than ad-hoc. Both example PNGs use a strict grid.
- Leave consistent gutters between cards — **minimum 20px on both axes** (horizontal *and* vertical). This applies to free-form / scattered layouts (e.g. the "ML code surrounded by components" diagram) just as much as grids: when boxes have varying heights and aren't on a shared row baseline, it's easy to leave two of them 0–5px apart or overlapping. After placing each box, compute `next.y - (prev.y + prev.height)` (and the x equivalent) for neighbors and make sure it's ≥ 20.
- If a card title is longer than the card is wide (e.g. "Human Review & Feedback" in a 230px card), its text overflows and collides with neighboring tiles. Prefer **widening the cards** in that row (keep a normal ~30–40px gutter) over spreading the tiles far apart — a huge gutter looks worse than slightly larger boxes. When you resize a card, also update its title and body text elements' `x` and `width` to match, or they render off-center.
- `appState.viewBackgroundColor` is the canvas color (`bg`, `#f6f5f3`).

### 4. Emit the JSON

Top-level shape:

```json
{
  "type": "excalidraw",
  "version": 2,
  "source": "claude",
  "elements": [ ... ],
  "appState": {
    "viewBackgroundColor": "#f6f5f3",
    "gridSize": null
  },
  "files": {}
}
```

Element schema mirrors [visuals/ci_cd_architecture.excalidraw.json](../../../visuals/ci_cd_architecture.excalidraw.json). The critical wiring for text-inside-shape is:

- The shape has `boundElements: [{ "type": "text", "id": "<text_id>" }]`.
- The text element has `containerId: "<shape_id>"`.
- The text's `x`/`y`/`width`/`height` should match the shape's; Excalidraw re-centers it on load using `textAlign` and `verticalAlign`.

Give each element a stable `id` and a unique numeric `seed` / `version` / `versionNonce` (any positive integer; just don't collide).

### 5. Apply the style rules

Every element uses these defaults:

| Field            | Value                                                        |
| ---------------- | ------------------------------------------------------------ |
| `fontFamily`     | `6` (Excalidraw "Normal" — Nunito, current preset). For reference: `5` = Excalifont (Hand-drawn preset), `8` = Comic Shanns (Code preset). Legacy IDs `1` (Virgil), `2` (Helvetica), `3` (Cascadia) fall back to hand-drawn rendering in current Excalidraw — do **not** use them. |
| `roundness`      | `{ "type": 3 }` on rectangles; `null` on text and lines       |
| `roughness`      | `0` (Architect — smoothest). `1` = Artist, `2` = Cartoonist — do **not** use those unless asked. |
| `fillStyle`      | `"solid"`                                                    |
| `textAlign`      | `"center"`                                                   |
| `verticalAlign`  | `"middle"`                                                   |
| `strokeWidth`    | `2` for boxes, `1.5` for arrows                              |
| `fontSize`       | `20` for titles, `16` for body text, `14` for badges/labels  |
| `opacity`        | `100`                                                        |

### 6. Apply the color palette by role

Named palette — use the **role**, not raw hex, when reasoning about which color goes where:

| Name             | Hex       | Role                                                       |
| ---------------- | --------- | ---------------------------------------------------------- |
| `bg`             | `#f6f5f3` | Canvas background (`viewBackgroundColor`)                  |
| `surface`        | `#ffffff` | Primary card / box fill                                    |
| `surface-alt`    | `#f0efed` | Nested card or section-container fill                      |
| `text`           | `#1a1a1a` | Primary labels, card titles                                |
| `text-secondary` | `#6b6b6b` | Sub-labels, captions, tool lists inside cards              |
| `text-tertiary`  | `#9a9a9a` | Metadata, footnotes                                        |
| `border`         | `#e0dfdd` | Neutral stroke — section containers, ungrouped cards       |
| `border-light`   | `#ebeae8` | Subtle dividers, nested borders                            |
| `accent`         | `#f24b37` | **Orange** — card borders of the first/primary section (e.g. MLOps), highlights, icon backgrounds |
| `accent-hover`   | `#e04530` | Secondary accent — use sparingly for variation             |
| `link`           | `#337FF9` | **Blue** — card borders of the second section (e.g. LLMOps), link text (NOT arrows — see arrow rule) |

**Color-code cards by section using orange and blue.** This is the preferred look: cards in the first/primary section get the orange `accent` stroke; cards in the second section get the blue `link` stroke. Card fill stays `surface` (white). Only fall back to the neutral `border` stroke when a diagram has no section grouping. Keep titles, sub-text, and section containers neutral so the orange/blue card borders carry the visual distinction.

Default role-to-element mapping:

- **Canvas background** → `bg`
- **Component card, primary section** → fill `surface`, stroke `accent` (orange `#f24b37`)
- **Component card, secondary section** → fill `surface`, stroke `link` (blue `#337FF9`)
- **Component card, no sections** → fill `surface`, stroke `border`
- **Section / group container behind cards** → stroke `text-tertiary` (`#9a9a9a`) or `border`, transparent or `surface-alt` fill
- **Card title** → `text`
- **Sub-text inside a card** (e.g. tool lists) → `text-secondary`
- **Icon / accent shape / category marker** → fill `accent`; text on top → `surface` if the icon is large enough, otherwise `text`
- **Arrow / connector** → stroke `text` (black `#1a1a1a`) or `text-tertiary` (grey `#9a9a9a`) — see the arrow rule below
- **Tertiary metadata** → `text-tertiary`

### Arrows: always black or grey, straight, matching Figure5.1 / Figure6.2

Arrows are **never** colored (no orange, no blue). Use only black `text` (`#1a1a1a`) or grey `text-tertiary` (`#9a9a9a`) for the stroke. Match the exact arrow shape used in [visuals/Figure5.1.excalidraw.json](../../../visuals/Figure5.1.excalidraw.json) and [visuals/Figure6.2.excalidraw.json](../../../visuals/Figure6.2.excalidraw.json):

- `strokeColor`: `#1a1a1a` (black) or `#9a9a9a` (grey)
- `strokeWidth`: `1.5`
- `roughness`: `0`
- `roundness`: `null` — **straight** lines, not curved
- `points`: exactly two points `[[0, 0], [dx, dy]]` (a single straight segment)
- `startArrowhead`: `null`
- `endArrowhead`: `"triangle"` (the filled head used in both Figure5.1 and Figure6.2) — always use `triangle`, never `arrow`
- `fillStyle`: `"solid"`, `strokeStyle`: `"solid"`

### 7. Write and report

Write the JSON file to `visuals/<basename>.excalidraw.json`. Report:
- The output path
- Element count
- One-line summary of grid dimensions or layout (e.g. "5×4 grid of component cards")

## Element template (rectangle with centered text)

`strokeColor` below is the orange `accent` (`#f24b37`) for a primary-section card; use the blue `link` (`#337FF9`) for second-section cards, or neutral `border` (`#e0dfdd`) when there are no sections.

```json
{
  "id": "card_orchestrator",
  "type": "rectangle",
  "x": 40,
  "y": 40,
  "width": 240,
  "height": 140,
  "strokeColor": "#f24b37",
  "backgroundColor": "#ffffff",
  "fillStyle": "solid",
  "strokeWidth": 2,
  "roughness": 0,
  "opacity": 100,
  "angle": 0,
  "seed": 101,
  "version": 1,
  "versionNonce": 101,
  "isDeleted": false,
  "boundElements": [{ "type": "text", "id": "card_orchestrator_text" }],
  "updated": 1,
  "link": null,
  "locked": false,
  "groupIds": [],
  "roundness": { "type": 3 }
}
```

```json
{
  "id": "card_orchestrator_text",
  "type": "text",
  "x": 40,
  "y": 40,
  "width": 240,
  "height": 140,
  "text": "Orchestrator\nLakeflow jobs",
  "fontSize": 20,
  "fontFamily": 6,
  "textAlign": "center",
  "verticalAlign": "middle",
  "strokeColor": "#1a1a1a",
  "backgroundColor": "transparent",
  "fillStyle": "solid",
  "strokeWidth": 1,
  "roughness": 0,
  "opacity": 100,
  "angle": 0,
  "seed": 102,
  "version": 1,
  "versionNonce": 102,
  "isDeleted": false,
  "boundElements": null,
  "updated": 1,
  "link": null,
  "locked": false,
  "groupIds": [],
  "containerId": "card_orchestrator"
}
```

For a card with a title + secondary sub-text, use two separate text elements stacked inside the card (one bound, or both free-positioned at the card's x/y with explicit offsets) — match what looks closest to the source PNG.

### Inspect each card's internal layout

Before placing text, decide for each card whether its contents are **stacked-centered**, **split inside one card**, or **label-outside-box**. Defaulting to centered title-over-body loses information density and visually drifts away from the source.

- **Stacked-centered**: title on top, sub-text below, both centered horizontally inside one bordered card. Use this only when the source clearly shows centered content with a single visible border.
- **Label-outside-box** (most common for API / cheatsheet diagrams like MLflow `log_*` ↔ `load_*`): the bordered/colored rectangle wraps **only** the code or value on the right; the label (title + sub-text) is free text sitting to the **left** of the rectangle with no border around it. This is the default when the source shows a side-by-side label↔code pairing — do not put a border around the label.

For label-outside-box rows:
- **No rectangle around the label.** Only the code area gets a bordered rectangle (`strokeColor` = section accent: orange for primary, blue for secondary).
- **Three unbound text elements** per row plus one rectangle: title, description, code, code_box.
- Label (left, no border):
  - Title: `textAlign: "left"`, `verticalAlign: "top"`, `fontSize: 15`, color `text`.
  - Description: `textAlign: "left"`, `verticalAlign: "top"`, `fontSize: 11`, color `text-secondary`, just below the title.
  - Vertically center the title+description block against the code box: `label_top = code_box.y + (code_box.h - (title_h + 4 + desc_h)) / 2`.
  - Allot ~180–220px of width for the label column, with a ~50–60px horizontal gutter before the code box.
- Code box (right, the only bordered element on the row):
  - **Size the box to the code, not the row.** Width ~280–320px is enough for typical one-line snippets at `fontSize: 11` (with the Code font, ~7px/char, so ~38 chars fits in 280px including inner padding). Height ~70–80px fits up to 3 wrapped code lines. Avoid stretching the box to fill the canvas — leftover horizontal space belongs in the gutter between label and box, or as outer canvas margin, not inside the box.
  - The code text element needs **inner padding** so the rectangle's stroke doesn't kiss the glyphs, and its `y` must be computed manually for true vertical centering. Standard pattern:
    - `text.x = box.x + 14`
    - `text.width = box.width - 28`
    - `text.height = lines * fontSize * lineHeight` (the natural text height, NOT the box height)
    - `text.y = box.y + (box.height - text.height) / 2`
    - `textAlign: "left"`, `verticalAlign: "top"`
  - **Don't rely on `verticalAlign: "middle"` for unbound text.** Excalidraw anchors unbound (`containerId: null`) text to its `y` regardless of `verticalAlign`. Stretching `text.height` to match `box.height` and setting `verticalAlign: "middle"` does *not* center the text — single-line code sticks to the top of the box. The only reliable centering for unbound text is to (1) size `text.height` to the actual text height and (2) compute `text.y` from the box height minus that.
  - To compute text height: for `fontSize: 11`, `lineHeight: 1.4`, one line ≈ 15.4px, three lines ≈ 46.2px. Add a `\n`-count step in your generator and derive `text.height` programmatically — never eyeball it.
  - Use `fontFamily: 8` (Comic Shanns, the Excalidraw "Code" preset), `fontSize: 11`, `lineHeight: 1.4`.
  - For multi-line code, wrap with explicit `\n` and indentation; keep the box height constant across the grid for visual rhythm rather than enlarging individual boxes.
- **Never bind the code text to the box** (`containerId: null`, box's `boundElements: []`). Binding stretches the line height and re-centers the text in ways that defeat the manual vertical centering above.
- **Quote code from the source PNG verbatim.** Don't append extra method calls or attribute chains "for completeness" if the source shows only `mlflow.get_run()` — the user wants the diagram to match what's in the image, not a fuller reference. When in doubt, read fewer lines from the PNG, not more.

Common mistakes (don't do these):
- Putting a border around the entire row (label + code). The label has no border.
- Making code boxes ~500px+ wide because the row is wide. Boxes hug the code; the label uses the leftover space; extra width goes into the gutter.
- Trusting `verticalAlign: "middle"` to center unbound text inside a taller box. It doesn't — unbound text always anchors to its `y`. Compute `y = box.y + (box.height - text.height) / 2` manually, with `text.height` set to the natural text height (lines × fontSize × lineHeight).
- Setting `text.x = box.x` with no inset. The first glyph sits on the stroke. Inset by ~14px on each side.
- Padding the code snippet with extra information not present in the source (e.g. expanding `mlflow.get_run()` into `run = mlflow.get_run(...); run.data.params; run.info...`).

### Line spacing across cards with different item counts

When card body text is **bound to its container** (`containerId` set, rectangle's `boundElements` includes the text), Excalidraw stretches `lineHeight` to fill the container's height. With a 140px-tall card and `fontSize: 16`:

- 3 lines → `lineHeight ≈ 2.917` (gap of ~47px between lines)
- 4 lines → `lineHeight ≈ 2.188`
- 2 lines → `lineHeight ≈ 4.375` ← **huge gap, looks wrong** next to other cards

For visual consistency across a grid where most cards have 3 items but some have 2, **unbind the 2-item cards** and match the dominant card's `lineHeight`:

1. On the rectangle, set `"boundElements": []`.
2. On the text, set `"containerId": null`.
3. On the text, set `lineHeight` to match the dominant cards (e.g. `2.9166666666666665` if most cards have 3 lines).
4. On the text, set `height` to `lineHeight * fontSize * num_lines` (e.g. 2 lines × 2.917 × 16 ≈ 93).
5. On the text, set `y` to `card_y + (card_height - text_height) / 2` to center the group.

Do this for every card whose line count differs from the dominant one.

## Verification (after writing)

Tell the user to drag the file into `excalidraw.com` and check:
- Warm stone (`#f6f5f3`) background, rounded corners, no leftover default Excalidraw stroke (`#1971c2`) or black.
- Card borders are color-coded by section: orange (`#f24b37`) for the primary section, blue (`#337FF9`) for the second.
- Selecting a shape shows **Architect** sloppiness and **Normal** font in the right panel.
- Text centered horizontally and vertically in every card, with even line spacing across cards (2-line cards unbound — see line-spacing rule above).
- No two boxes overlap or sit closer than ~20px on either axis — check this programmatically before reporting (loop over box bounding boxes), don't eyeball it. Stacked boxes with different heights are the usual offender.
- Layout matches the source PNG's grouping and order.
