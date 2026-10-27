"""rig_trim.py - manual fine-tuning of FINISHED rig parts: erase (or bring back) single pixels after the cut.

Why: a cut part sometimes carries material that looks fine while the limb hangs down but becomes a visible
crescent when the limb rotates (the puffed sleeve under a raised arm, the side of the underwear that rides along
with the thigh when the legs open ...).  A trim removes those pixels from the part for good.

How it is stored: per (view, part) a set of erased pixels, each as an offset (dx, dy) from the part's PIVOT in
sprite pixels.  Relative to the pivot (not to the sprite corner) so the trim still lands in the right place after
you CUT again and the sprite's bounding box has changed a little.  Nothing is ever destroyed: the raw cut is kept,
the erased pixels are only hidden, and the RESTORE brush brings them back - also in a later session.

PAINT: besides erasing, single pixels can be PAINTED with a colour (hair on an arm -> skin tone, a gap in the torso).
Paints are stored the same way, per (view, part): {(dx, dy): (r, g, b)} offsets from the pivot, in res.paints.  A pixel is
either erased or painted, never both: painting un-erases it, erasing or restoring removes the paint on it.

The trimmed sprite keeps exactly the same size and pivots as the raw one, so nothing else in the rig moves.
"""
import numpy as np
from PIL import Image, ImageDraw

SKIP_KINDS = ("placeholder", "disabled")


def origin(sp):
    """Integer pixel the offsets are measured from (the sprite's pivot, rounded)."""
    return int(round(float(sp.pj[0]))), int(round(float(sp.pj[1])))


def store(res):
    if getattr(res, "trims", None) is None: res.trims = {}
    return res.trims


def offsets(res, view, key, create=False):
    t = store(res)
    if create: return t.setdefault(view, {}).setdefault(key, set())
    return t.get(view, {}).get(key) or set()


def paints(res, view, key):
    return dict(((getattr(res, "paints", None) or {}).get(view, {}) or {}).get(key) or {})


def paint_count(res, view=None, key=None):
    t = getattr(res, "paints", None) or {}
    if view is not None and key is not None: return len(t.get(view, {}).get(key) or ())
    return sum(len(s) for d in t.values() for s in d.values())


def count(res, view=None, key=None):
    t = getattr(res, "trims", None) or {}
    if view is not None and key is not None: return len(t.get(view, {}).get(key) or ())
    return sum(len(s) for d in t.values() for s in d.values())


def erased_mask(shape_hw, sp, offs):
    """Boolean mask (h, w) of the erased pixels for sprite `sp`."""
    h, w = shape_hw; m = np.zeros((h, w), bool)
    if not offs: return m
    ox, oy = origin(sp)
    a = np.array(list(offs), np.int64).reshape(-1, 2)
    x, y = a[:, 0] + ox, a[:, 1] + oy
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    m[y[ok], x[ok]] = True
    return m


def trimmed_rgba(sp, offs, paint=None):
    """Copy of the sprite's pixels with the erased ones made transparent and the painted ones set to their colour."""
    out = np.array(sp.rgba, np.uint8, copy=True)
    if offs:
        m = erased_mask(out.shape[:2], sp, offs)
        out[m] = 0
    if paint:
        h, w = out.shape[:2]; ox, oy = origin(sp)
        for (dx, dy), c in paint.items():
            x, y = dx + ox, dy + oy
            if 0 <= x < w and 0 <= y < h: out[y, x] = (c[0], c[1], c[2], 255)
    return out


def trimmed_sprite(raw, offs, paint=None):
    """A new Sprite equal to `raw` but with the erased pixels removed / painted pixels set (same size, same pivots)."""
    n = type(raw)(); n.__dict__.update(raw.__dict__)
    n.rgba = trimmed_rgba(raw, offs, paint)
    return n


def apply_trims(res):
    """Call right after res.views has been (re)built from the raw cuts: remember the raw sprites in
    res.views_raw and swap in the trimmed copies."""
    res.views_raw = {v: dict(d) for v, d in res.views.items()}
    keys = {(v, k) for v, parts in (getattr(res, "trims", None) or {}).items() for k in parts}
    keys |= {(v, k) for v, parts in (getattr(res, "paints", None) or {}).items() for k in parts}
    for v, k in keys:
        offs, pnt = offsets(res, v, k), paints(res, v, k)
        raw = res.views_raw.get(v, {}).get(k)
        if raw is None or not (offs or pnt) or getattr(raw, "kind", "") in SKIP_KINDS: continue
        res.views[v][k] = trimmed_sprite(raw, offs, pnt)


def set_offsets(res, view, key, offs, paint=None):
    """Replace the erased set (and, when given, the painted pixels) of one part and rebuild that one trimmed sprite.
    paint=None keeps the part's current paint. Returns the new view sprite."""
    t = store(res)
    if offs: t.setdefault(view, {})[key] = set(offs)
    else:
        t.get(view, {}).pop(key, None)
        if view in t and not t[view]: t.pop(view)
    if paint is not None:
        if getattr(res, "paints", None) is None: res.paints = {}
        pt = res.paints
        if paint: pt.setdefault(view, {})[key] = dict(paint)
        else:
            pt.get(view, {}).pop(key, None)
            if view in pt and not pt[view]: pt.pop(view)
    pnt = paints(res, view, key)
    raw = res.views_raw[view][key]
    res.views[view][key] = trimmed_sprite(raw, offs, pnt) if (offs or pnt) else raw
    return res.views[view][key]


def set_state(res, view, key, state):
    """state = (erased set, paint dict) as returned by the *_state tools below / by get_state()."""
    return set_offsets(res, view, key, state[0], state[1])


def get_state(res, view, key):
    return set(offsets(res, view, key)), paints(res, view, key)


def _px(mask, raw):
    ox, oy = origin(raw); ys, xs = np.nonzero(mask)
    return set(zip((xs - ox).tolist(), (ys - oy).tolist()))


def erase_state(st, raw, mask):
    offs, pnt = st; gone = _px(mask, raw)
    return erase(offs, raw, mask), {k: c for k, c in pnt.items() if k not in gone}


def restore_state(st, raw, mask):
    offs, pnt = st; gone = _px(mask, raw)
    return restore(offs, raw, mask), {k: c for k, c in pnt.items() if k not in gone}


def keep_only_state(st, raw, mask):
    offs, pnt = st; gone = _px(~mask, raw)
    return keep_only(offs, raw, mask), {k: c for k, c in pnt.items() if k not in gone}


def paint_state(st, raw, mask, color):
    """Paint every pixel of `mask` (transparent ones too) with `color`; those pixels are no longer erased."""
    offs, pnt = st; new = _px(mask, raw); col = (int(color[0]), int(color[1]), int(color[2]))
    pnt = dict(pnt); pnt.update({k: col for k in new})
    return set(offs) - new, pnt


def _add(offs, raw, mask, erase):
    """mask: bool (h, w). erase=True adds the opaque masked pixels, erase=False removes them from the set."""
    ox, oy = origin(raw); h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0: return offs
    if erase:
        op = raw.rgba[ys, xs, 3] > 0
        ys, xs = ys[op], xs[op]
    new = set(zip((xs - ox).tolist(), (ys - oy).tolist()))
    return (offs | new) if erase else (offs - new)


def brush_mask(shape_hw, cx, cy, r):
    """Round brush centred on sprite coordinates (cx, cy) (floats, pixel centres at +0.5)."""
    h, w = shape_hw
    x0, x1 = max(0, int(np.floor(cx - r - 1))), min(w, int(np.ceil(cx + r + 2)))
    y0, y1 = max(0, int(np.floor(cy - r - 1))), min(h, int(np.ceil(cy + r + 2)))
    m = np.zeros((h, w), bool)
    if x1 <= x0 or y1 <= y0: return m
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2
    inside = d2 <= max(r, 0.5) ** 2
    m[y0:y1, x0:x1] = inside
    return m


def stroke_mask(shape_hw, p0, p1, r):
    """Brush dragged from p0 to p1 (no gaps)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(1, int(np.ceil(np.hypot(*(p1 - p0)) / max(0.35, r * 0.5))))
    m = np.zeros(shape_hw, bool)
    for t in np.linspace(0.0, 1.0, n + 1): m |= brush_mask(shape_hw, *(p0 + (p1 - p0) * t), r)
    return m


def polygon_mask(shape_hw, pts):
    h, w = shape_hw; im = Image.new("L", (w, h), 0)
    if len(pts) >= 3: ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in pts], fill=255, outline=255)
    return np.asarray(im) > 0


def erase(offs, raw, mask):   return _add(set(offs), raw, mask, True)
def restore(offs, raw, mask): return _add(set(offs), raw, mask, False)


def keep_only(offs, raw, mask):
    """Erase every opaque pixel OUTSIDE the mask."""
    return _add(set(offs), raw, ~mask, True)


# ---------------------------------------------------------------------------------------------- saving
def paints_to_meta(res):
    """JSON-able: {view: {key: [dx, dy, r, g, b, ...]}}"""
    out = {}
    for v, parts in (getattr(res, "paints", None) or {}).items():
        for k, d in parts.items():
            if d: out.setdefault(v, {})[k] = [int(c) for (p, c) in sorted(d.items()) for c in (*p, *c)]
    return out


def paints_from_meta(meta):
    out = {}
    for v, parts in (meta or {}).items():
        for k, flat in (parts or {}).items():
            try: d = {(int(flat[i]), int(flat[i + 1])): (int(flat[i + 2]), int(flat[i + 3]), int(flat[i + 4])) for i in range(0, len(flat) - 4, 5)}
            except Exception: continue
            if d: out.setdefault(v, {})[k] = d
    return out


def to_meta(res):
    """JSON-able: {view: {key: [dx0, dy0, dx1, dy1, ...]}}"""
    out = {}
    for v, parts in (getattr(res, "trims", None) or {}).items():
        for k, offs in parts.items():
            if offs: out.setdefault(v, {})[k] = [int(c) for p in sorted(offs) for c in p]
    return out


def from_meta(meta):
    out = {}
    for v, parts in (meta or {}).items():
        for k, flat in (parts or {}).items():
            try: pts = {(int(flat[i]), int(flat[i + 1])) for i in range(0, len(flat) - 1, 2)}
            except Exception: continue
            if pts: out.setdefault(v, {})[k] = pts
    return out


def final_rgba(res, view, key, sp):
    """The pixels to write for the per-part PNG: trimmed if this part has a trim, else the sprite's own array
    (the very same object, so callers can tell whether anything changed)."""
    offs = ((getattr(res, "trims", None) or {}).get(view, {}) or {}).get(key)
    pnt = paints(res, view, key)
    if not (offs or pnt) or getattr(sp, "kind", "") in SKIP_KINDS: return sp.rgba
    return trimmed_rgba(sp, offs or set(), pnt)


# ---------------------------------------------------------------------------------------------- test poses
def test_pose(view, name):
    """Angles (degrees, clockwise positive) that make leftover material obvious.  Same conventions as rig_motion."""
    sg = -1 if view == "back" else 1
    if name == "arms":
        return {"uarmR": -150, "uarmL": -150} if view == "side" else {"uarmR": 150 * sg, "uarmL": -150 * sg}
    if name == "legs":
        return {"thighR": -60, "thighL": 25} if view == "side" else {"thighR": 30 * sg, "thighL": -30 * sg}
    return {}


POSES = (("neutral", "NEUTRAL"), ("arms", "ARMS UP"), ("legs", "LEGS APART"))
