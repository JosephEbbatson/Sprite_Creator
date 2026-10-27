"""rig_trim.py - manual fine-tuning of FINISHED rig parts: erase (or bring back) single pixels after the cut.

Why: a cut part sometimes carries material that looks fine while the limb hangs down but becomes a visible
crescent when the limb rotates (the puffed sleeve under a raised arm, the side of the underwear that rides along
with the thigh when the legs open ...).  A trim removes those pixels from the part for good.

How it is stored: per (view, part) a set of erased pixels, each as an offset (dx, dy) from the part's PIVOT in
sprite pixels.  Relative to the pivot (not to the sprite corner) so the trim still lands in the right place after
you CUT again and the sprite's bounding box has changed a little.  Nothing is ever destroyed: the raw cut is kept,
the erased pixels are only hidden, and the RESTORE brush brings them back - also in a later session.

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


def trimmed_rgba(sp, offs):
    """Copy of the sprite's pixels with the erased ones made transparent."""
    out = np.array(sp.rgba, np.uint8, copy=True)
    if offs:
        m = erased_mask(out.shape[:2], sp, offs)
        out[m] = 0
    return out


def trimmed_sprite(raw, offs):
    """A new Sprite equal to `raw` but with the erased pixels removed (same size, same pivots)."""
    n = type(raw)(); n.__dict__.update(raw.__dict__)
    n.rgba = trimmed_rgba(raw, offs)
    return n


def apply_trims(res):
    """Call right after res.views has been (re)built from the raw cuts: remember the raw sprites in
    res.views_raw and swap in the trimmed copies."""
    res.views_raw = {v: dict(d) for v, d in res.views.items()}
    for v, parts in (getattr(res, "trims", None) or {}).items():
        for k, offs in parts.items():
            raw = res.views_raw.get(v, {}).get(k)
            if raw is None or not offs or getattr(raw, "kind", "") in SKIP_KINDS: continue
            res.views[v][k] = trimmed_sprite(raw, offs)


def set_offsets(res, view, key, offs):
    """Replace the erased set of one part and rebuild that one trimmed sprite. Returns the new view sprite."""
    t = store(res)
    if offs: t.setdefault(view, {})[key] = set(offs)
    else:
        t.get(view, {}).pop(key, None)
        if view in t and not t[view]: t.pop(view)
    raw = res.views_raw[view][key]
    res.views[view][key] = trimmed_sprite(raw, offs) if offs else raw
    return res.views[view][key]


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
    if not offs or getattr(sp, "kind", "") in SKIP_KINDS: return sp.rgba
    return trimmed_rgba(sp, offs)


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
