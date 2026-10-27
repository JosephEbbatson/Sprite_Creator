"""rig_extras.py - EXTRA PARTS for HUMAN.RIG.

The base rig is always the same 17 human parts.  An EXTRA part is something you add on top of it:
a tail, ears, horns, wings, a second pair of arms, a cape, a hat ...  Extras live in their own layer, so the
learned body model and the 17 base parts are never touched.

An extra part has
  * a sprite per view (front / side / back), made by CUTTING a region out of an image, or by COPYING / CUTTING
    one of the rig's own parts,
  * a PARENT  (any base part, or another extra - so a tail can be a chain of segments),
  * per view: an OFFSET from the parent's pivot (cm), a PIVOT inside the sprite (px), a LAYER and a visibility flag,
  * a SIZE multiplier and a REST ANGLE,
  * MOTION settings: rigid, or a damped spring (lag / bounce / inertia) with an optional idle sway.

Everything here is plain numpy / PIL, no tkinter.  Ids of extras always start with "x_" so they can never clash
with a base part key.
"""
import math, re, zlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw

VIEWS = ("front", "side", "back")
CENTER = ("head", "neck", "chest", "abdomen", "pelvis")
LAYERS = (("behind", "behind its parent"), ("front", "in front of its parent"),
          ("back_all", "behind the whole body"), ("front_all", "in front of the whole body"))
LAYER_KEYS = [k for k, _ in LAYERS]
LAYER_LABEL = dict(LAYERS)
BORROW = dict(front=("back", "side"), side=("front", "back"), back=("front", "side"))

# motion presets: freq = natural frequency (Hz, lower = lazier), damp = damping ratio (lower = bouncier),
# inertia = how much the pelvis' acceleration shoves it, follow = how much of the parent's rotation it copies,
# sway_amp / sway_hz = constant idle wobble (degrees, Hz)
MOTION_PRESETS = {
    "RIGID (glued on)":         dict(mode="rigid",  freq=6.0, damp=0.5,  inertia=0.0, follow=1.0, sway_amp=0.0,  sway_hz=1.0),
    "EAR (soft flick)":         dict(mode="spring", freq=8.0, damp=0.45, inertia=0.6, follow=1.0, sway_amp=0.0,  sway_hz=1.0),
    "TAIL (lazy swing)":        dict(mode="spring", freq=3.5, damp=0.35, inertia=1.0, follow=1.0, sway_amp=8.0,  sway_hz=0.7),
    "HAIR / CAPE (floppy)":     dict(mode="spring", freq=4.5, damp=0.30, inertia=1.4, follow=0.8, sway_amp=0.0,  sway_hz=1.0),
    "WING (flapping)":          dict(mode="spring", freq=6.0, damp=0.50, inertia=0.4, follow=1.0, sway_amp=25.0, sway_hz=2.0),
    "ANTENNA (springy)":        dict(mode="spring", freq=9.0, damp=0.15, inertia=0.8, follow=1.0, sway_amp=0.0,  sway_hz=1.0),
    "DANGLING (heavy)":         dict(mode="spring", freq=2.5, damp=0.60, inertia=1.2, follow=0.9, sway_amp=0.0,  sway_hz=1.0),
}
DEFAULT_PRESET = "RIGID (glued on)"
MOTION_KEYS = ("mode", "freq", "damp", "inertia", "follow", "sway_amp", "sway_hz")


def default_motion(preset=DEFAULT_PRESET):
    return dict(MOTION_PRESETS[preset])


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_") or "part"


def new_id(existing, name):
    base = "x_" + slug(name); i, out = 1, base
    taken = {e.id for e in existing}
    while out in taken:
        i += 1; out = f"{base}_{i}"
    return out


def swap_side(k):
    return k if (k in CENTER or not k or k[-1] not in "LR") else k[:-1] + ("L" if k[-1] == "R" else "R")


def is_extra(k):
    return isinstance(k, str) and k.startswith("x_")


# ---------------------------------------------------------------------------------------------- data model
class Extra:
    def __init__(self, id, name, parent, size=1.0, base_scale=1.0, rest=0.0, motion=None, twin=None):
        self.id, self.name, self.parent = id, name, parent
        self.size, self.base_scale, self.rest = float(size), float(base_scale), float(rest)
        self.motion = dict(motion) if motion else default_motion()
        self.twin = twin
        self.views = {}                     # view -> dict(rgba, pj, off, layer, show)

    def view_data(self, view):
        """Exactly this view's sprite, or None.  Views are NEVER borrowed from each other: a belly cut for the front
        must not appear in the side or back view - every view needs its own cut."""
        return self.views.get(view)

    def set_view(self, view, rgba, pj, off, layer="front", show=True, bs=None, size=None):
        """bs = picture-pixels -> rig-pixels factor of this view's source, size = the user's SIZE multiplier (per view)."""
        self.views[view] = dict(rgba=np.ascontiguousarray(rgba, np.uint8), pj=np.asarray(pj, float),
                                off=np.asarray(off, float), layer=layer, show=bool(show),
                                bs=float(self.base_scale if bs is None else bs), size=float(1.0 if size is None else size))

    def touch(self, view=None):
        for v in ([view] if view else list(self.views)):
            if v in self.views: self.views[v].pop("_c", None)


# ---------------------------------------------------------------------------------------------- cutting
def polygon_mask(w, h, pts):
    m = Image.new("L", (int(w), int(h)), 0)
    if pts is not None and len(pts) >= 3:
        ImageDraw.Draw(m).polygon([(float(x), float(y)) for x, y in pts], fill=255, outline=255)
    return np.asarray(m) > 0


def crop_to_content(rgba):
    """Trim transparent margins.  Returns (rgba, (x0, y0)) or (None, (0, 0)) when nothing is opaque."""
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    if len(ys) == 0: return None, (0, 0)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return np.ascontiguousarray(rgba[y0:y1, x0:x1]), (int(x0), int(y0))


def cut_polygon(rgba, pts=None):
    """Keep only what is inside the polygon (image pixel coordinates).  pts=None keeps everything.
    Returns (cropped rgba, (x0, y0)) or (None, ...) if the cut is empty."""
    out = np.array(rgba, np.uint8, copy=True)
    if pts is not None and len(pts) >= 3:
        m = polygon_mask(out.shape[1], out.shape[0], pts)
        out[..., 3] = np.where(m, out[..., 3], 0)
    out[out[..., 3] == 0, :3] = 0
    return crop_to_content(out)


def has_transparency(rgba, frac=0.02):
    return float((rgba[..., 3] < 250).mean()) > frac


def remove_background(rgba, tol=40):
    """Make the plain background transparent: every pixel connected to the image border whose colour is close to
    the border's dominant colour.  Works for cut-outs on a flat background; for anything else use the lasso."""
    from scipy import ndimage
    a = np.array(rgba, np.uint8, copy=True)
    h, w = a.shape[:2]
    edge = np.concatenate([a[0, :, :3], a[-1, :, :3], a[:, 0, :3], a[:, -1, :3]]).astype(int)
    q = (edge // 16)
    keys, cnt = np.unique(q, axis=0, return_counts=True)
    bg = edge[(q == keys[int(np.argmax(cnt))]).all(1)].mean(0)
    near = (np.abs(a[..., :3].astype(int) - bg).sum(2) <= tol * 3 / 2) | (a[..., 3] == 0)
    lab, n = ndimage.label(near)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    bgm = np.isin(lab, border[border > 0])
    a[bgm, 3] = 0; a[bgm, :3] = 0
    return a


def pivot_choices(rgba):
    """Handy pivot positions (sprite px): where a tail / ear / wing would normally hinge."""
    al = rgba[..., 3] > 0
    ys, xs = np.nonzero(al)
    if len(ys) == 0:
        h, w = rgba.shape[:2]; c = (w / 2.0, h / 2.0)
        return dict(center=c, top=c, bottom=c, left=c, right=c)
    h = ys.max() - ys.min() + 1; w = xs.max() - xs.min() + 1
    bh, bw = max(1, int(round(h * 0.12))), max(1, int(round(w * 0.12)))
    def band(m):
        yy, xx = np.nonzero(al & m)
        return (float(xx.mean()) + 0.5, float(yy.mean()) + 0.5) if len(yy) else (float(xs.mean()) + 0.5, float(ys.mean()) + 0.5)
    Y, X = np.mgrid[0:al.shape[0], 0:al.shape[1]]
    top = band(Y < ys.min() + bh); bot = band(Y > ys.max() - bh)
    lef = band(X < xs.min() + bw); rig = band(X > xs.max() - bw)
    return dict(center=(float(xs.mean()) + 0.5, float(ys.mean()) + 0.5),
                top=(top[0], float(ys.min())), bottom=(bot[0], float(ys.max() + 1)),
                left=(float(xs.min()), lef[1]), right=(float(xs.max() + 1), rig[1]))


def auto_base_scale(rgba, res, frac=0.16):
    """Scale that makes an imported cut roughly `frac` of the character's height (the user then adjusts SIZE)."""
    h, w = rgba.shape[:2]
    target = frac * float(res.H) / float(res.cpp)
    return float(np.clip(target / max(1, max(h, w)), 0.02, 8.0))


# ---------------------------------------------------------------------------------------------- rendering
def _resample(rgba, s):
    h, w = rgba.shape[:2]
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    if nw == w and nh == h:
        out = np.array(rgba, np.uint8, copy=True)
    else:
        mode = Image.Resampling.NEAREST if s >= 1.0 else Image.Resampling.BOX
        out = np.array(Image.fromarray(rgba).resize((nw, nh), mode))
    amax = int(out[..., 3].max()) if out.size else 0
    if amax == 0:
        return np.zeros((1, 1, 4), np.uint8), 1.0 / w, 1.0 / h
    keep = out[..., 3] >= min(128, amax)
    out[..., 3] = np.where(keep, 255, 0); out[~keep, :3] = 0
    return np.ascontiguousarray(out), nw / w, nh / h


def sprite_for(e, view, cpp=2.0):
    """The sprite of extra `e` in `view`, scaled to the rig's pixel grid.
    Returns SimpleNamespace(rgba, pj, pe, lever_cm, up) or None.  Cached on the view data."""
    vd = e.view_data(view)
    if vd is None: return None
    s = float(vd.get("bs", e.base_scale)) * float(vd.get("size", 1.0))
    c = vd.get("_c")
    if c is not None and c[0] == s and c[1] == cpp: return c[2]
    rgba, sx, sy = _resample(vd["rgba"], s)
    pj = np.asarray(vd["pj"], float) * np.array([sx, sy])
    ys, xs = np.nonzero(rgba[..., 3] > 0)
    cen = np.array([xs.mean() + 0.5, ys.mean() + 0.5]) if len(ys) else pj.copy()
    v = cen - pj; d = float(np.hypot(*v))
    u0 = v / d if d > 0.5 else np.array([0.0, 1.0])
    spr = SimpleNamespace(rgba=rgba, pj=pj, pe=cen, centre=cen, u0=u0, lever_cm=max(3.0, d * cpp))
    vd["_c"] = (s, cpp, spr)
    return spr


def default_offset(res, view, parent):
    """cm from the parent's pivot to the middle of the parent's sprite: a sensible first spot for a new part."""
    cpp = float(res.cpp)
    try:
        if is_extra(parent):
            pe = next(e for e in (getattr(res, "extras", None) or []) if e.id == parent)
            sp = sprite_for(pe, view, cpp)
            return (sp.centre - sp.pj) * cpp
        S = res.views[view][parent]
        ys, xs = np.nonzero(S.rgba[..., 3] > 0)
        c = np.array([xs.mean() + 0.5, ys.mean() + 0.5]) if len(ys) else np.asarray(S.pj, float)
        return (c - np.asarray(S.pj, float)) * cpp
    except Exception:
        return np.zeros(2)


def active_extras(res, view, base_keys):
    """Extras that exist in this view, ordered so every parent comes before its children.
    An extra whose parent part is switched off (or hidden) is dropped, with everything hanging from it."""
    ex = [e for e in (getattr(res, "extras", None) or []) if e.view_data(view) is not None]
    ok, out, base = set(), [], set(base_keys)
    for _ in range(len(ex) + 1):
        moved = False
        for e in ex:
            if e.id in ok: continue
            vd = e.view_data(view)
            if not vd.get("show", True): continue
            if e.parent in base or e.parent in ok:
                ok.add(e.id); out.append(e); moved = True
        if not moved: break
    return out


def merge_order(order, ex_list, view):
    """Insert the extras into the base back-to-front order according to each one's layer."""
    order = list(order)
    back_all, front_all = [], []
    for e in ex_list:
        lay = e.view_data(view).get("layer", "front")
        if lay == "back_all": back_all.append(e.id)
        elif lay == "front_all": front_all.append(e.id)
        else:
            i = order.index(e.parent) if e.parent in order else len(order) - 1
            order.insert(i if lay == "behind" else i + 1, e.id)
    return back_all + order + front_all


# ---------------------------------------------------------------------------------------------- building / editing
def make_extra(res, name, parent, rgba, views=VIEWS, pivot="center", base_scale=1.0, motion=None, layer="front", extras=None):
    """New extra from a cut sprite (RGBA, already cropped).  The same cut is used for every view in `views`."""
    extras = extras if extras is not None else (getattr(res, "extras", None) or [])
    e = Extra(new_id(extras, name), name, parent, base_scale=base_scale, motion=motion or default_motion())
    pj = pivot_choices(rgba)[pivot] if isinstance(pivot, str) else pivot
    for v in views:
        lay = layer
        if v == "back" and layer == "front" and parent in CENTER and parent in ("chest", "abdomen"): lay = "front"
        e.set_view(v, rgba, pj, default_offset(res, v, parent) if res is not None else (0, 0), lay, True, bs=base_scale)
    return e


def add_view(e, res, view, rgba, pivot="center", base_scale=1.0, layer="front"):
    """Give an existing extra its own sprite for `view` (replacing that view's sprite if it had one).
    Name, parent, motion and the other views are untouched."""
    old = e.views.get(view)
    pj = pivot_choices(rgba)[pivot] if isinstance(pivot, str) else pivot
    off = old["off"] if old is not None else default_offset(res, view, e.parent)
    e.set_view(view, rgba, pj, off, old.get("layer", layer) if old else layer, True, bs=base_scale, size=old.get("size", 1.0) if old else 1.0)


def views_label(e):
    """'F S -' : which views have their own sprite."""
    return " ".join(v[0].upper() if v in e.views else "-" for v in VIEWS)


def snapshot(extras):
    """Cheap copy of the whole editable state (pixel arrays are shared: they are never changed in place)."""
    return [dict(id=e.id, name=e.name, parent=e.parent, rest=float(e.rest), motion=dict(e.motion), twin=e.twin,
                 base_scale=e.base_scale, size=e.size,
                 views={v: dict(rgba=d["rgba"], pj=np.array(d["pj"], float), off=np.array(d["off"], float), layer=d.get("layer", "front"),
                                show=bool(d.get("show", True)), bs=d.get("bs"), size=d.get("size", 1.0)) for v, d in e.views.items()})
            for e in extras]


def snapshot_sig(snap):
    return tuple((s["id"], s["name"], s["parent"], round(s["rest"], 4), tuple(sorted(s["motion"].items())), s["twin"],
                  tuple((v, id(d["rgba"]), tuple(np.round(d["pj"], 3)), tuple(np.round(d["off"], 3)), d["layer"], d["show"], d["bs"], d["size"])
                        for v, d in sorted(s["views"].items()))) for s in snap)


def restore_snapshot(snap):
    out = []
    for s in snap:
        e = Extra(s["id"], s["name"], s["parent"], s["size"], s["base_scale"], s["rest"], s["motion"], s["twin"])
        for v, d in s["views"].items():
            e.set_view(v, d["rgba"], d["pj"].copy(), d["off"].copy(), d["layer"], d["show"], bs=d["bs"], size=d["size"])
        out.append(e)
    return out


def flip_view(e, view):
    vd = e.views.get(view)
    if vd is None: return
    w = vd["rgba"].shape[1]
    vd["rgba"] = np.ascontiguousarray(vd["rgba"][:, ::-1]); vd["pj"] = np.array([w - vd["pj"][0], vd["pj"][1]])
    vd.pop("_c", None)


def mirror_extra(e, extras):
    """Left/right twin: sprites flipped in front/back views, offsets mirrored, parent swapped to the other side."""
    twin_parent = swap_side(e.parent)
    if is_extra(e.parent):
        p = next((x for x in extras if x.id == e.parent), None)
        if p is not None and p.twin: twin_parent = p.twin
    name = e.name + ("_L" if e.name.endswith("_R") is False and not e.name.endswith("_L") else "")
    name = (e.name[:-2] + "_L") if e.name.endswith("_R") else ((e.name[:-2] + "_R") if e.name.endswith("_L") else e.name + "_twin")
    n = Extra(new_id(extras, name), name, twin_parent, base_scale=e.base_scale, rest=-e.rest, motion=e.motion, twin=e.id)
    e.twin = n.id
    for v, vd in e.views.items():
        rgba, pj, off = vd["rgba"], np.asarray(vd["pj"], float), np.asarray(vd["off"], float)
        if v in ("front", "back"):
            w = rgba.shape[1]
            rgba = np.ascontiguousarray(rgba[:, ::-1]); pj = np.array([w - pj[0], pj[1]]); off = np.array([-off[0], off[1]])
        show = vd.get("show", True)
        if v == "side" and e.parent in CENTER: show = False          # the far ear/horn is hidden behind the head in profile
        n.set_view(v, rgba, pj, off, vd.get("layer", "front"), show, bs=vd.get("bs"), size=vd.get("size"))
    return n


def duplicate_extra(e, extras):
    name = e.name + "_copy"
    n = Extra(new_id(extras, name), name, e.parent, base_scale=e.base_scale, rest=e.rest, motion=e.motion)
    for v, vd in e.views.items():
        n.set_view(v, vd["rgba"].copy(), vd["pj"].copy(), np.asarray(vd["off"], float) + np.array([3.0, 3.0]), vd.get("layer", "front"), vd.get("show", True), bs=vd.get("bs"), size=vd.get("size"))
    return n


def remove_extra(res, e):
    """Delete an extra; anything attached to it is re-attached to its parent."""
    lst = res.extras
    for c in lst:
        if c.parent == e.id: c.parent = e.parent
        if c.twin == e.id: c.twin = None
    lst.remove(e)


def descendants(extras, eid):
    out, todo = set(), [eid]
    while todo:
        x = todo.pop()
        for c in extras:
            if c.parent == x and c.id not in out: out.add(c.id); todo.append(c.id)
    return out


# ---------------------------------------------------------------------------------------------- persistence
def _crc(a): return zlib.crc32(np.ascontiguousarray(a).tobytes())


def save_extras(folder, extras):
    """Write the editable state: <folder>/src/<id>_<view>.png (native pixels).  Returns the JSON-able list."""
    folder = Path(folder); src = folder / "src"
    if not extras:
        if src.exists():
            for p in src.glob("*.png"): p.unlink()
        return []
    src.mkdir(parents=True, exist_ok=True)
    keep, out = set(), []
    for e in extras:
        d = dict(id=e.id, name=e.name, parent=e.parent, rest=round(e.rest, 3), motion=dict(e.motion), twin=e.twin, views={})
        for v, vd in e.views.items():
            fn = f"{e.id}_{v}.png"; keep.add(fn); h = _crc(vd["rgba"]); p = src / fn
            if vd.get("_h") != h or not p.exists():
                Image.fromarray(vd["rgba"]).save(p); vd["_h"] = h
            d["views"][v] = dict(file=fn, pj=[round(float(x), 3) for x in vd["pj"]], off=[round(float(x), 3) for x in vd["off"]],
                                 layer=vd.get("layer", "front"), show=bool(vd.get("show", True)),
                                 bs=round(float(vd.get("bs", e.base_scale)), 5), size=round(float(vd.get("size", 1.0)), 4))
        out.append(d)
    for p in src.glob("*.png"):
        if p.name not in keep: p.unlink()
    return out


def load_extras(folder, meta):
    out = []
    for d in (meta or []):
        try:
            e = Extra(d["id"], d.get("name", d["id"]), d["parent"], d.get("size", 1.0), d.get("base_scale", 1.0),
                      d.get("rest", 0.0), {**default_motion(), **(d.get("motion") or {})}, d.get("twin"))
            for v, vd in (d.get("views") or {}).items():
                p = Path(folder) / "src" / vd["file"]
                if not p.exists(): continue
                e.set_view(v, np.array(Image.open(p).convert("RGBA")), vd["pj"], vd["off"], vd.get("layer", "front"), vd.get("show", True),
                           bs=vd.get("bs", e.base_scale), size=vd.get("size", e.size))
            if e.views: out.append(e)
        except Exception:
            continue
    return out


def export_extras(res, folder):
    """Game-ready output: rig-resolution sprites in <folder>/<id>_<view>.png and a dict for rig.json."""
    folder = Path(folder); extras = getattr(res, "extras", None) or []
    info = {}
    if folder.exists():
        for p in folder.glob("*.png"): p.unlink()
    if not extras: return info
    folder.mkdir(parents=True, exist_ok=True)
    for e in extras:
        views = {}
        for v in VIEWS:
            vd = e.view_data(v); spr = sprite_for(e, v, res.cpp)
            if vd is None or spr is None: continue
            fn = f"{e.id}_{v}.png"; Image.fromarray(spr.rgba).save(folder / fn)
            views[v] = dict(file=f"extras/{fn}", pivot_px=[round(float(x), 2) for x in spr.pj],
                            offset_cm=[round(float(x), 2) for x in vd["off"]], layer=vd.get("layer", "front"),
                            visible=bool(vd.get("show", True)), size=round(float(vd.get("size", 1.0)), 3))
        info[e.id] = dict(name=e.name, parent=e.parent, rest_angle_deg=round(e.rest, 2),
                          motion=dict(e.motion), twin=e.twin, views=views)
    return info


# ---------------------------------------------------------------------------------------------- secondary motion
def _chain_angle(angles, k, parent_of):
    a = 0.0
    while k:
        a += float(angles.get(k, 0.0)); k = parent_of(k)
    return a


def simulate_extras(frames, extras, res, view, fps=24, loop=False, flex=1.0, bounce=1.0, inertia=1.0, speed=1.0, clip_dur=None):
    """Add the motion of the extra parts to frames made by rig_motion.simulate().
    Every extra becomes a key in frame['angles'] (degrees, local, relative to its parent - same convention as the
    base parts).  An extra follows its parent's ABSOLUTE angle like a damped spring, is shoved by the pelvis'
    acceleration, and can sway on its own.  flex=0 makes every extra rigid (used for the 'muscle' ghost)."""
    import rig_motion as RM
    if not frames or not extras: return frames
    base = set(RM.HIER)
    ex = active_extras(res, view, base)
    if not ex: return frames
    n = len(frames); cpp = float(res.cpp)
    pa = {k: np.array([_chain_angle(f["angles"], k, RM.parent_of) for f in frames]) for k in
          {e.parent for e in ex if e.parent in base}}
    R = np.array([f["root"] for f in frames], float)
    dtf = 1.0 / fps
    if n >= 3:
        nxt = np.roll(R, -1, 0) if loop else np.vstack([R[1:], R[-1:]])
        prv = np.roll(R, 1, 0) if loop else np.vstack([R[:1], R[:-1]])
        acc = np.clip((nxt - 2 * R + prv) / (dtf * dtf), -2500, 2500)
    else:
        acc = np.zeros_like(R)
    sub = max(4, int(math.ceil(360.0 / fps))); dt = dtf / sub
    rigid_all = flex < 0.02
    info = {}
    for e in ex:
        m = e.motion; spr = sprite_for(e, view, cpp)
        f0 = float(m.get("freq", 6.0)); z0 = float(m.get("damp", 0.5))
        f = float(np.clip(f0 / math.sqrt(max(flex, 0.02)), 0.8, 18.0)); z = float(np.clip(z0 / max(bounce, 0.05), 0.05, 1.6))
        hz = float(m.get("sway_hz", 1.0))
        T = (clip_dur / max(speed, 1e-3)) if clip_dur else n / fps
        if loop and T > 0: hz = max(1.0, round(hz * T)) / T          # a whole number of sway cycles per loop: no pop at the seam
        info[e.id] = dict(rigid=(rigid_all or m.get("mode") == "rigid"), ks=(2 * math.pi * f) ** 2, kd=2 * z * 2 * math.pi * f,
                          w=float(m.get("inertia", 0.0)) * inertia, follow=float(m.get("follow", 1.0)), amp=float(m.get("sway_amp", 0.0)),
                          hz=hz, ph=(zlib.crc32(e.id.encode()) % 628) / 100.0, rest=float(e.rest), lev=spr.lever_cm, u0=spr.u0)
    passes = 3 if loop else 1
    phi, om = {}, {}

    def parent_angle(e, i, j, u):
        if e.parent in pa: return pa[e.parent][i] * (1 - u) + pa[e.parent][j] * u
        return phi.get(e.parent, 0.0)

    def target(e, t):
        q = info[e.id]
        return q["follow"] * parent_angle(e, 0, 0, 0.0) + q["rest"] + (q["amp"] * math.sin(2 * math.pi * q["hz"] * t + q["ph"]) if q["amp"] else 0.0)

    for e in ex:                                           # start every part at rest on the first frame
        phi[e.id], om[e.id] = target(e, 0.0), 0.0
    for p in range(passes):
        for i in range(n):
            j = (i + 1) % n if loop else min(i + 1, n - 1)
            if p == passes - 1:                            # record the state at the start of this frame
                for e in ex: frames[i]["angles"][e.id] = float(phi[e.id] - parent_angle(e, i, j, 0.0))
            for s in range(sub):
                u = s / sub; t = (p * n + i + u) * dtf
                for e in ex:
                    q = info[e.id]
                    star = q["follow"] * parent_angle(e, i, j, u) + q["rest"] + (q["amp"] * math.sin(2 * math.pi * q["hz"] * t + q["ph"]) if q["amp"] else 0.0)
                    if q["rigid"]: phi[e.id], om[e.id] = star, 0.0; continue
                    r = math.radians(phi[e.id]); c, sn = math.cos(r), math.sin(r)
                    ux, uy = q["u0"][0] * c - q["u0"][1] * sn, q["u0"][0] * sn + q["u0"][1] * c
                    nx, ny = -uy, ux                       # direction the tip moves for a positive (clockwise) angle
                    al_in = math.degrees(-(acc[i][0] * nx + acc[i][1] * ny) * 1.5 / q["lev"] * q["w"])
                    al = q["ks"] * (star - phi[e.id]) - q["kd"] * om[e.id] + al_in
                    om[e.id] = float(np.clip(om[e.id] + al * dt, -2500, 2500)); phi[e.id] += om[e.id] * dt
    return frames
