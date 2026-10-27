"""rig_sheet_import.py - bring in a FINISHED, hand-made pixel sprite sheet (no source photos needed).

The sheet it understands is the "green cells" layout: every body part is drawn inside its own cell (a green box with a red
border), the cells are arranged like the body (head on top, neck, arms beside the chest ...).  A sheet can hold
several such figures side by side (for example the living body and the corpse materials).

What it does:  find the cells -> work out which cell is which of the 17 body parts from where it sits -> cut the pixels
that are NOT cell colour out of each cell -> give every part sensible joints -> hand the parts to the rig as the FRONT
view (side and back are then guessed by the normal 'synth views' logic until you import / cut real ones).

Pure numpy / scipy / PIL - no GUI in this file.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GREEN, RED, ORANGE = (44, 222, 81), (237, 28, 36), (255, 127, 39)

# rows of the figure, top to bottom; inside a row left -> right as you SEE it (the character's RIGHT side is on the left)
BANDS = (("head",), ("neck",), ("uarmR", "chest", "uarmL"), ("farmR", "abdomen", "farmL"),
         ("handR", "pelvis", "handL"), ("thighR", "thighL"), ("shinR", "shinL"), ("footR", "footL"))


# ----------------------------------------------------------------------------------------------- finding cells
def load_rgb(path_or_img):
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    return np.asarray(im.convert("RGBA"))


def find_cells(rgba, border_cols=(RED, ORANGE), fill=GREEN, max_size=60, min_size=5):
    """Every box enclosed by a border colour whose inside is mostly cell-green.
    Returns [dict(box=(y0, x0, y1, x1), mask=bool(h, w) region inside the border)] sorted top-left."""
    rgb = rgba[..., :3].astype(int)
    border = np.zeros(rgb.shape[:2], bool)
    for c in border_cols: border |= (rgb == c).all(-1)
    lab, n = ndi.label(~border)
    G = (rgb == fill).all(-1)
    out = []
    for i, s in enumerate(ndi.find_objects(lab), 1):
        if s is None: continue
        y, x = s; h, w = y.stop - y.start, x.stop - x.start
        if not (min_size <= h <= max_size and min_size <= w <= max_size): continue
        m = lab[s] == i
        if G[s][m].sum() < 0.2 * m.sum(): continue
        out.append(dict(box=(y.start, x.start, y.stop, x.stop), mask=m))
    out.sort(key=lambda c: (c["box"][0], c["box"][1]))
    return out


def cell_pixels(rgba, cell, fill=GREEN):
    """The drawing inside one cell as RGBA: everything that is not cell-green (cropped to its own bounding box,
    None when the cell is empty)."""
    y0, x0, y1, x1 = cell["box"]; sub = rgba[y0:y1, x0:x1].copy()
    keep = cell["mask"] & ~(sub[..., :3].astype(int) == fill).all(-1) & (sub[..., 3] > 0)
    if not keep.any(): return None
    out = np.zeros_like(sub); out[keep] = sub[keep]; out[keep, 3] = 255
    ys, xs = np.nonzero(keep)
    return np.ascontiguousarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def split_panels(cells, gap=15):
    """Group the cells into separate figures (sheets drawn side by side): a gap wider than `gap` px between neighbouring
    columns starts a new figure. Returned left to right."""
    if not cells: return []
    order = sorted(cells, key=lambda c: c["box"][1]); panels = [[order[0]]]; right = order[0]["box"][3]
    for c in order[1:]:
        if c["box"][1] - right > gap: panels.append([])
        panels[-1].append(c); right = max(right, c["box"][3])
    return [p for p in panels if len(p) >= 8]


# ----------------------------------------------------------------------------------------------- which cell is which part
def _centre(c): return ((c["box"][1] + c["box"][3]) / 2.0, (c["box"][0] + c["box"][2]) / 2.0)


def assign(panel):
    """panel: list of cells of ONE figure -> dict part key -> cell.  Uses only the layout:
    1. cells far out to the sides (spare slots for other views) are ignored,
    2. the rest is cut into rows by height,
    3. in each row the cells nearest the body's centre line are taken and named left -> right."""
    xs = [(c["box"][1] + c["box"][3]) / 2 for c in panel]
    cx = (min(c["box"][1] for c in panel) + max(c["box"][3] for c in panel)) / 2.0
    d = sorted(abs(x - cx) for x in xs)
    cells = list(panel)
    for i in range(len(d) - 1, 0, -1):                        # a clear gap in distance-from-centre => outer spare slots
        if d[i] - d[i - 1] > 9 and (len(d) - i) >= 3:
            lim = d[i - 1] + 4.5; cells = [c for c in cells if abs(_centre(c)[0] - cx) <= lim]; break
    cells.sort(key=lambda c: _centre(c)[1])
    rows, cur = [], [cells[0]]
    for c in cells[1:]:
        if _centre(c)[1] - np.mean([_centre(k)[1] for k in cur]) > 8: rows.append(cur); cur = []
        cur.append(c)
    rows.append(cur)
    if len(rows) != len(BANDS):
        raise ValueError(f"expected {len(BANDS)} rows of parts but found {len(rows)} - is this the green-cell body layout?")
    out = {}
    for row, names in zip(rows, BANDS):
        row = sorted(sorted(row, key=lambda c: abs(_centre(c)[0] - cx))[:len(names)], key=lambda c: _centre(c)[0])
        if len(row) != len(names): raise ValueError(f"row {names} has {len(row)} cells, needs {len(names)}")
        for k, c in zip(names, row): out[k] = c
    return out


def read_sheet(path_or_img):
    """-> list of figures, left to right: dict(parts={key: rgba or None}, cells={key: cell}, n=number of non-empty parts)."""
    rgba = load_rgb(path_or_img); figs = []
    for panel in split_panels(find_cells(rgba)):
        try: cells = assign(panel)
        except ValueError as e: figs.append(dict(error=str(e), parts={}, cells={}, n=0)); continue
        parts = {k: cell_pixels(rgba, c) for k, c in cells.items()}
        figs.append(dict(parts=parts, cells=cells, n=sum(p is not None for p in parts.values())))
    return figs


def classify(figs):
    """Which figure is the living BODY and which is CORPSE data.  The body is the first one with every part drawn;
    the corpse is the next one. (The GUI lets the person swap them.)"""
    good = [f for f in figs if f.get("n", 0) > 0]
    return (good[0] if good else None), (good[1] if len(good) > 1 else None)
