"""rig_pose.py - pose renderer v2 for HUMAN.RIG.

What is new compared with the built-in renderer:
  * ANCHORS.  Every sprite remembers where each of its child joints sat inside it in the source frame.  A child is
    attached at exactly that spot instead of at an idealised position computed from average bone lengths, so the
    pieces line up the way they did in the photo.
  * MANUAL ALIGNMENT.  res.align[view][part] = {"off": [dx, dy] cm, "piv": [dx, dy] px}
        off  moves the part (and everything hanging from it) relative to its parent  - "move joint"
        piv  moves the part's rotation centre inside its sprite without moving the picture - "move pivot"
  * root motion for animation (pelvis offset in cm).
  * seam report + AUTO TIE (centres the cross-sections of stacked parts on each other).
The functions take G = the globals() of the main rig file so they use exactly its parts, hierarchy and helpers.
"""
import math
import numpy as np
from PIL import Image

VERTICAL_PAIRS = [("pelvis", "abdomen"), ("abdomen", "chest"), ("chest", "neck"), ("neck", "head"),
                  ("uarmR", "farmR"), ("uarmL", "farmL"), ("thighR", "shinR"), ("thighL", "shinL")]


def _al(al, k, what):
    return np.asarray((al.get(k) or {}).get(what, (0.0, 0.0)), float)


def layout(G, res, view, ang, root=(0.0, 0.0), use_align=True):
    """Forward kinematics. Returns th (deg), pos (cm, y down, pelvis at 0 + root), pj (effective pivot px)."""
    cpp, S = res.cpp, res.views[view]
    al = ((getattr(res, "align", None) or {}).get(view) or {}) if use_align else {}
    J0 = G["_pivots0"](res, view); rot = G["_rot"]
    mask = getattr(res, "mask", None)
    keys = G.get("RB").active_keys(mask) if G.get("RB") else G["PART_KEYS"]
    hier = G.get("RB").active_hier(mask) if G.get("RB") else G["HIER"]
    pj = {k: np.asarray(S[k].pj, float) + _al(al, k, "piv") for k in keys if k in S}
    th, pos = {}, {}
    for k in hier:
        p = G["parent_key"](k); a = float(ang.get(k, 0.0))
        if p is None:
            th[k] = a; pos[k] = J0[k] + np.asarray(root, float)
        else:
            sp = S[p]; anc = getattr(sp, "anc", None) or {}
            if sp.kind in ("own", "locked") and k in anc:
                base = (np.asarray(anc[k], float) - pj[p]) * cpp
            else:
                base = (J0[k] - J0[p]) - _al(al, p, "piv") * cpp
            off = base + _al(al, k, "off") + _al(al, k, "piv") * cpp
            pos[k] = pos[p] + rot(off, th[p]); th[k] = th[p] + a
    return th, pos, pj


def render_pose(G, res, view, angles=None, solid=False, shade=True, crop=True, rest=True, root=None):
    cpp, S = res.cpp, res.views[view]
    PART_KEYS, rot = G["PART_KEYS"], G["_rot"]
    ang = dict(G["rest_angles"](view)) if rest else {}
    if angles: ang.update({k: float(v) for k, v in angles.items()})
    root = np.zeros(2) if root is None else np.asarray(root, float)
    th, pos, pj = layout(G, res, view, ang, root)
    th0, pos0, _ = layout(G, res, view, {}, (0.0, 0.0))
    Sz = int(1.9 * res.H / cpp); ox, oy = Sz / 2.0, 0.55 * Sz
    canvas = Image.new("RGBA", (Sz, Sz), (0, 0, 0, 0))
    pivots, ends, d0 = {}, {}, {}
    order = G["draw_order"](view, getattr(res, "mask", None))
    for k in order:
        sp = S[k]; rgba = G["solidify_sprite"](sp.rgba, k) if solid else sp.rgba
        if shade and view == "side" and k.endswith("L"):
            rgba = rgba.copy(); rgba[..., :3] = (rgba[..., :3] * 0.72).astype(np.uint8)
        im = Image.fromarray(rgba); w, h = im.size; p0 = pj[k]
        pad = int(math.ceil(math.hypot(max(p0[0], w - p0[0]), max(p0[1], h - p0[1])))) + 2
        big = Image.new("RGBA", (2 * pad, 2 * pad), (0, 0, 0, 0))
        bx, by = pad - int(round(p0[0])), pad - int(round(p0[1])); big.paste(im, (bx, by))
        c = (bx + p0[0], by + p0[1])
        if abs(th[k]) > 1e-6: big = big.rotate(-th[k], resample=Image.Resampling.NEAREST, center=c)
        P = np.array([ox + pos[k][0] / cpp, oy + pos[k][1] / cpp])
        canvas.paste(big, (int(round(P[0] - c[0])), int(round(P[1] - c[1]))), big)
        pivots[k] = P; vec = np.asarray(sp.pe, float) - p0
        ends[k] = P + rot(vec, th[k]); d0[k] = math.degrees(math.atan2(vec[1], vec[0]))
    off = np.zeros(2)
    if crop:
        bb = canvas.getbbox() or (0, 0, Sz, Sz)
        bb = (max(0, bb[0] - 2), max(0, bb[1] - 2), min(Sz, bb[2] + 2), min(Sz, bb[3] + 2))
        canvas = canvas.crop(bb); off = np.array(bb[:2], float)
    r2 = lambda p: [round(float(p[0] - off[0]), 2), round(float(p[1] - off[1]), 2)]
    jp = {"pelvis": "pelvis", "neck": "neck", "head": "head"}
    for sd in "LR":
        jp.update({sd + "sho": "uarm" + sd, sd + "elb": "farm" + sd, sd + "wri": "hand" + sd,
                   sd + "hip": "thigh" + sd, sd + "knee": "shin" + sd, sd + "ank": "foot" + sd})
    meta = dict(view=view, note=f"{view} standing pose", pelvis_px=r2(pivots["pelvis"]), draw_order=order,
                pivots_px={k: r2(p) for k, p in pivots.items()}, ends_px={k: r2(p) for k, p in ends.items()},
                th=th, d0=d0, origin=[float(off[0]), float(off[1])], size=[Sz, Sz],
                joints_px={n: r2(pivots[k]) for n, k in jp.items()},
                joints_cm={n: [round(float(pos0[k][0] - pos0["pelvis"][0]), 1), round(float(pos0[k][1] - pos0["pelvis"][1]), 1)] for n, k in jp.items()},
                rest_angles=G["rest_angles"](view), parent={k: G["parent_key"](k) for k in PART_KEYS},
                lengths_cm={n: round(float(v), 1) for n, v in res.lengths.items()},
                shoulder_width_cm=round(res.widths["sho"], 1), hip_width_cm=round(res.widths["hip"], 1),
                alignment=((getattr(res, "align", None) or {}).get(view) or {}),
                parts={k: dict(pivot_world=r2(pivots[k]), parent=G["parent_key"](k), kind=S[k].kind) for k in PART_KEYS})
    return canvas, meta


# ---------------------------------------------------------------------------------------------- seams
def _place(G, res, view, k, th, pos, pj):
    """top-left (px, world) of sprite k in the neutral (unrotated) layout."""
    return pos[k] / res.cpp - pj[k]


def _cross_section(sp, tl, yj):
    """x-range (world px) of the sprite's opaque pixels in the rows around world row yj."""
    a = sp.rgba[..., 3] > 0; r = int(round(yj - tl[1])); h = a.shape[0]
    rows = a[max(0, r - 2):min(h, r + 3)]
    if rows.size == 0 or not rows.any(): return None
    cols = np.nonzero(rows.any(0))[0]
    return tl[0] + cols.min(), tl[0] + cols.max() + 1


def seam_report(G, res, view):
    """For every joint: lateral offset between the cross-sections of parent and child (px) and how well the
    two sprites cover a small disk around the joint (1 = no gap)."""
    S = res.views[view]; th, pos, pj = layout(G, res, view, {}, (0.0, 0.0))
    out = {}
    for k in G["HIER"]:
        p = G["parent_key"](k)
        if p is None or S[p].kind == "placeholder" or S[k].kind == "placeholder": continue
        tlp, tlk = _place(G, res, view, p, th, pos, pj), _place(G, res, view, k, th, pos, pj)
        jx, jy = pos[k] / res.cpp
        yy, xx = np.mgrid[-3:4, -3:4]; m = (xx ** 2 + yy ** 2) <= 9
        cov = 0.0; tot = 0
        for dy, dx in zip(yy[m], xx[m]):
            X, Y = jx + dx, jy + dy; tot += 1; hit = False
            for sp, tl in ((S[p], tlp), (S[k], tlk)):
                ix, iy = int(math.floor(X - tl[0])), int(math.floor(Y - tl[1]))
                if 0 <= iy < sp.rgba.shape[0] and 0 <= ix < sp.rgba.shape[1] and sp.rgba[iy, ix, 3] > 0: hit = True
            cov += hit
        dx = None
        if (p, k) in VERTICAL_PAIRS:
            cp, ck = _cross_section(S[p], tlp, jy), _cross_section(S[k], tlk, jy)
            if cp and ck and 0.5 < (cp[1] - cp[0]) / max(ck[1] - ck[0], 1e-6) < 2.0:
                dx = float((cp[0] + cp[1]) / 2 - (ck[0] + ck[1]) / 2)
        q = "ok"
        if cov / tot < 0.85: q = "gap"
        elif dx is not None and abs(dx) > 1.5: q = "off"
        elif dx is not None and abs(dx) > 0.75: q = "near"
        out[k] = dict(parent=p, dx=dx, cover=cov / tot, quality=q, world=(float(jx), float(jy)))
    return out


def auto_tie(G, res, view, max_shift=4.0, min_shift=0.5):
    """Centre every stacked pair on each other (top-down so the corrections do not fight).  Returns changed parts."""
    if not hasattr(res, "align") or res.align is None: res.align = {}
    al = res.align.setdefault(view, {}); changed = []
    for p, k in VERTICAL_PAIRS:
        rep = seam_report(G, res, view).get(k)
        if not rep or rep["dx"] is None: continue
        dx = float(rep["dx"])
        if min_shift <= abs(dx) <= max_shift:
            e = al.setdefault(k, {"off": [0.0, 0.0], "piv": [0.0, 0.0]})
            e["off"] = [round(e["off"][0] + dx * res.cpp, 3), e["off"][1]]; changed.append((k, round(float(dx), 2)))
    return changed
