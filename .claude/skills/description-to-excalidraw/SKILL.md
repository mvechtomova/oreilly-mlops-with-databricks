---
name: description-to-excalidraw
description: Create an Excalidraw JSON diagram in visuals/ from a written description (no source image), using the project's brand palette and styling. Trigger whenever the user describes a diagram in words and asks to draw/create/make it in Excalidraw, sketch a diagram from a spec, build a new diagram from scratch, or otherwise wants an Excalidraw file without an existing PNG to convert.
---

# description-to-excalidraw

Create an `.excalidraw.json` file from a **written description** of a diagram — a list of boxes, sections, arrows, and layout in words — restyled with the project's brand palette and consistent Excalidraw element styling. This is the sibling of [png-to-excalidraw](../png-to-excalidraw/SKILL.md): same palette, same style rules, same flaws-and-corrections; the only difference is there is no source image, so you build the layout from the description instead of tracing one.

## When to use

Trigger for any of: "draw a diagram of...", "make an Excalidraw showing X → Y → Z", "create a diagram with these boxes...", "sketch the architecture where...", "I need a new figure for the chapter that shows...". If the user describes the *content* of a diagram in words and wants an Excalidraw file, and there is no PNG to convert, this skill applies. If they point at a PNG in `visuals_raw/`, use [png-to-excalidraw](../png-to-excalidraw/SKILL.md) instead.

## Input

A natural-language description of the diagram, passed as the argument or in the user's message. It may specify some or all of: the boxes/cards and their labels, sections or groupings, arrows and their direction, the grid or flow shape, and any code/sub-text inside cards.

When the description is **underspecified**, fill gaps with the sensible defaults below rather than stalling — but ask the user first when a load-bearing choice is genuinely ambiguous:
- **Grid vs. flow vs. free-form**: if the description implies a sequence ("A then B then C"), default to a left-to-right or top-to-bottom flow with arrows. If it implies categories/components, default to a grid. If unsure which, ask.
- **Section grouping / color-coding**: if the user names two or more distinct groups, color-code them (orange primary, dark grey secondary — see palette). If they name none, use neutral borders.
- **Number of boxes and labels**: use exactly what the user lists. Do not invent extra boxes to "round out" the diagram, and do not drop ones they named.

## Output

The user usually wants a specific figure name. Resolve the output path in this order:
1. If the user gives a name (e.g. "call it Figure5.12" or "save as model_lifecycle"), write to `visuals/<that>.excalidraw.json`.
2. Otherwise, derive a short kebab/Figure-style slug from the description and **confirm it with the user** before writing.

If the target file already exists, ask before overwriting — the user may have hand-edited it (see [[excalidraw-open-editor-clobber]]: also avoid writing while they have it open in the VS Code editor, or your write gets clobbered).

## Procedure

### 1. Parse the description into an inventory

From the words, build the same inventory png-to-excalidraw builds from an image:
- Every distinct card / box and its exact label (verbatim from the user).
- **The internal layout of each card** — stacked-centered title+body, or split label↔code side-by-side? Decide per card (see step 5 / "Inspect each card's internal layout").
- Multi-line text or code inside cards (quote any code the user gives **verbatim**; don't pad it with extra calls).
- Section containers / group backgrounds and which cards belong to each.
- Icons or accent shapes.
- Arrows and connectors with direction, and which boxes they join.
- The overall shape: grid (note dimensions, e.g. 4×5), linear flow, or free-form.

The goal is to **realize the description faithfully** — same boxes, same labels, the grouping and flow the user asked for — not to redesign or embellish.

### 2. Plan the canvas

- Default canvas width: 1100–1400px depending on column count.
- Pick a uniform box size per row/column and reuse it across the grid — this is what makes the result look intentional rather than ad-hoc.
- Leave consistent gutters — **minimum 20px on both axes** (horizontal *and* vertical). This applies to free-form / scattered layouts just as much as grids: when boxes have varying heights and aren't on a shared row baseline, it's easy to leave two of them 0–5px apart or overlapping. After placing each box, compute `next.y - (prev.y + prev.height)` (and the x equivalent) for neighbors and make sure it's ≥ 20.
- If a card title is longer than the card is wide (e.g. "Human Review & Feedback" in a 230px card), its text overflows and collides with neighbours. Prefer **widening the cards** in that row (keep a normal ~30–40px gutter) over spreading the tiles far apart — a huge gutter looks worse than slightly larger boxes. When you resize a card, also update its title and body text elements' `x` and `width` to match, or they render off-center.
- For a flow diagram, lay boxes on a shared baseline (same `y` for a horizontal flow, same `x` for a vertical one) and reserve ~60–120px between them for the arrows.
- `appState.viewBackgroundColor` is the canvas color (`bg`, `#f6f5f3`).

### 3. Emit the JSON

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

### 4. Apply the style rules

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

### 5. Apply the color palette by role

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

**Do not use blue.** The palette is orange-first, then dark greys and black. Blue (`#337FF9`) is retired — never use it as a second section color or anywhere else.

**Color-code cards using orange as the primary accent.** Cards get the orange `accent` stroke; card fill stays `surface` (white). When a diagram has two distinct sections and you need a second stroke to tell them apart, use a neutral dark grey (`text-secondary` `#6b6b6b`) for the second section, not blue. When there is no section grouping, use the neutral `border` stroke. Keep titles, sub-text, and section containers neutral so the orange (and, where needed, dark-grey) card borders carry the visual distinction.

Default role-to-element mapping:

- **Canvas background** → `bg`
- **Component card, primary section** → fill `surface`, stroke `accent` (orange `#f24b37`)
- **Component card, secondary section** → fill `surface`, stroke `text-secondary` (dark grey `#6b6b6b`)
- **Component card, no sections** → fill `surface`, stroke `border`
- **Section / group container behind cards** → stroke `text-tertiary` (`#9a9a9a`) or `border`, transparent or `surface-alt` fill
- **Card title** → `text`
- **Sub-text inside a card** (e.g. tool lists) → `text-secondary`
- **Icon / accent shape / category marker** → fill `accent`; text on top → `surface` if the icon is large enough, otherwise `text`
- **Arrow / connector** → stroke `text` (black `#1a1a1a`) or `text-tertiary` (grey `#9a9a9a`) — see the arrow rule below
- **Tertiary metadata** → `text-tertiary`

**No em-dashes or en-dashes in any text content** (see [[no-em-dashes-in-prose]]): in labels, titles, and captions use commas or colons instead. This applies to the words rendered inside the diagram, not just prose.

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

**Many-to-one (converging) arrows must NOT share an endpoint.** When several arrows point at the same target box (e.g. three workspaces → one metastore), giving them all the identical end coordinate stacks the arrowheads so they read as a single merged arrow. Instead, spread the entry points along the target's near edge — at least ~16px apart — so each arrowhead lands on its own spot. Compute distinct endpoints: for `n` arrows entering a box of height `h` at center `cy`, target `cy + (i - (n-1)/2) * gap` for `i` in `0..n-1` (e.g. `gap = 18`). The arrows still fan in toward the box but stay visually separate. Verify after writing that no two arrows in a converging group share the same `(x+dx, y+dy)`.

**Prefer perfectly horizontal (or vertical) arrows, and align the boxes they connect to make that possible.** A connector that runs at a slight slant (e.g. `dy = -12` over `dx = 210`) reads as a mistake. Before drawing a cross arrow, move one of the two boxes so their centers line up on the arrow's axis, then draw the arrow straight: for a horizontal connector set both endpoints to the shared center `y` and `dy = 0`; for a vertical one share the center `x` and `dx = 0`. Nudging a box up or down a few px to hit a clean horizontal is almost always worth it. Only fall back to a slanted single segment when alignment is genuinely impossible (the two boxes must stay at different levels for layout reasons).

**Arrows must keep a small gap from every box — they may not touch or overlap one, not even by a pixel.** Inset both ends by ~3px from the box edges: start the arrow ~3px outside the source box and end its tip ~3px short of the target box (so the arrowhead points *at* the edge without crossing it). An arrow crossing a *grouping/section container* outline to reach a card nested inside it is acceptable (that is how you connect into a container); clipping through any *other* box is not. After writing, validate programmatically: for every arrow, take its segment bounding box inflated by 2px and assert it does not intersect any rectangle except the two it connects (and section containers it must pass through). Fix and re-emit before reporting if any arrow clips a box. Also give vertical pipeline connectors the same ~3px gap at both ends rather than butting them flush against the cards.

### 6. Write and report

Write the JSON file to `visuals/<name>.excalidraw.json`. Report:
- The output path
- Element count
- One-line summary of grid dimensions or layout (e.g. "5×4 grid of component cards" or "left-to-right 4-stage flow")

## Element template (rectangle with centered text)

`strokeColor` below is the orange `accent` (`#f24b37`) for a primary-section card; use dark grey `text-secondary` (`#6b6b6b`) for second-section cards, or neutral `border` (`#e0dfdd`) when there are no sections. Never blue.

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

For a card with a title + secondary sub-text, use two separate text elements stacked inside the card (one bound, or both free-positioned at the card's x/y with explicit offsets) — match what the description calls for.

### Inspect each card's internal layout

Before placing text, decide for each card whether its contents are **stacked-centered**, **split inside one card**, or **label-outside-box**. Defaulting to centered title-over-body loses information density.

- **Stacked-centered**: title on top, sub-text below, both centered horizontally inside one bordered card. Use this when the description shows centered content with a single visible border.
- **Label-outside-box** (most common for API / cheatsheet diagrams like MLflow `log_*` ↔ `load_*`): the bordered/colored rectangle wraps **only** the code or value on the right; the label (title + sub-text) is free text sitting to the **left** of the rectangle with no border around it. This is the default when the description shows a side-by-side label↔code pairing — do not put a border around the label.

For label-outside-box rows:
- **No rectangle around the label.** Only the code area gets a bordered rectangle (`strokeColor` = section accent: orange for primary, dark grey for secondary).
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
- **Quote code from the description verbatim.** Don't append extra method calls or attribute chains "for completeness" if the user wrote only `mlflow.get_run()` — the diagram should match what they asked for, not a fuller reference. When in doubt, write less, not more.

Common mistakes (don't do these):
- Putting a border around the entire row (label + code). The label has no border.
- Making code boxes ~500px+ wide because the row is wide. Boxes hug the code; the label uses the leftover space; extra width goes into the gutter.
- Trusting `verticalAlign: "middle"` to center unbound text inside a taller box. It doesn't — unbound text always anchors to its `y`. Compute `y = box.y + (box.height - text.height) / 2` manually, with `text.height` set to the natural text height (lines × fontSize × lineHeight).
- Setting `text.x = box.x` with no inset. The first glyph sits on the stroke. Inset by ~14px on each side.
- Padding the code snippet with extra information not present in the description.
- Inventing boxes, labels, or arrows the user did not describe to "complete" the picture.

### Bound text height must be the natural text height, not the container height

When you bind a centered label to its box (`containerId` set), do **not** set the text element's `height`/`y` to the box's full height/top. Excalidraw reserves internal padding (~5px each side); if the bound text's `height` is ≥ the container's available height, it stops vertical-centering and **top-aligns** the label, so it renders stuck to the top of the box. Set the bound text's `height` to the natural one-line height (`round(fontSize * lineHeight)`, e.g. ~19 for `fontSize: 16`) and its `y` to `box.y + (box.height - text_height) / 2`. With a correctly sized height, `verticalAlign: "middle"` then centers it. This bit us on the principal boxes in Figure5.6 (`Hotel_booking team`, `dev/acc/prd SPN`).

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
- Card borders are color-coded by section: orange (`#f24b37`) for the primary section, dark grey (`#6b6b6b`) for the second. No blue anywhere.
- Selecting a shape shows **Architect** sloppiness and **Normal** font in the right panel.
- Text centered horizontally and vertically in every card, with even line spacing across cards (2-line cards unbound — see line-spacing rule above).
- **Vertical centering is mandatory and must be checked programmatically before reporting — never eyeball it.** The recurring bug: a bound text element given the container's full `height` (instead of its natural text height) makes Excalidraw top-align the label. Generate every bound label with `text.height = round(numLines * fontSize * lineHeight)` and `text.y = box.y + (box.height - text.height) / 2`, then loop over all text elements with a `containerId` and assert `abs((box.y + box.height/2) - (text.y + text.height/2)) <= 0.6`; for unbound labels assert the same against their intended center. Fail loudly and fix before writing the final file — do not report success with any label off-center.
- No two boxes overlap or sit closer than ~20px on either axis — check this programmatically before reporting (loop over box bounding boxes), don't eyeball it. Stacked boxes with different heights are the usual offender.
- The diagram contains exactly the boxes, labels, sections, and arrows the user described — nothing invented, nothing dropped.
