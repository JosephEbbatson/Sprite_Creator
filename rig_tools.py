"""rig_tools.py - helpers for HUMAN.RIG: hidden-area filling (arm over body), hole repair, rig health check."""
import numpy as np
from collections import deque


def _seg_dist(P, a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ab = b - a; L2 = float(ab @ ab)
    if L2 < 1e-9: return np.hypot(P[:, 0] - a[0], P[:, 1] - a[1])
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    q = a + t[:, None] * ab
    return np.hypot(P[:, 0] - q[:, 0], P[:, 1] - q[:, 1])


def _nbsum(a):
    h, w = a.shape[:2]; p = np.pad(a, ((1, 1), (1, 1)) + ((0, 0),) * (a.ndim - 2))
    return sum(p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def _axis_ok(known, R, axis):
    """True where a known pixel exists within R on BOTH sides along the axis (the pixel is 'enclosed')."""
    k = known if axis == 1 else known.T
    idx = np.arange(k.shape[1])[None, :]
    left = np.maximum.accumulate(np.where(k, idx, -10 ** 6), axis=1)
    right = np.minimum.accumulate(np.where(k, idx, 10 ** 6)[:, ::-1], axis=1)[:, ::-1]
    ok = ((idx - left) <= R) & ((right - idx) <= R)
    return ok if axis == 1 else ok.T


def hidden_mask(lab, solid, known, front_ids, a, b, r, x0, y0, R, core=0.75):
    """Pixels that belong to the body of this part but are covered by a part drawn in FRONT of it
    (an arm hanging over the torso, the near leg over the far leg ...).  Only pixels that are
    surrounded by the part's own visible pixels are returned, so legitimate neighbours are left alone."""
    out = np.zeros(lab.shape, bool)
    if r < 0.01 or not front_ids: return out
    cand = solid & ~known & np.isin(lab, front_ids)
    if not cand.any(): return out
    ys, xs = np.nonzero(cand)
    P = np.stack([xs + x0 + 0.5, ys + y0 + 0.5], 1).astype(np.float32)
    keep = (_seg_dist(P, a, b) / r) < core
    out[ys[keep], xs[keep]] = True
    return out & (_axis_ok(known, R, 1) | _axis_ok(known, R, 0))


def inpaint(rgb, known, target, max_iter=120):
    """Fill `target` pixels with the colours growing in from `known` pixels. Returns (rgb, filled_mask)."""
    col = rgb.astype(np.float32).copy()
    kn = known.copy(); todo = target & ~kn; filled = np.zeros(kn.shape, bool)
    for _ in range(max_iter):
        if not todo.any(): break
        kf = kn.astype(np.float32)
        cnt = _nbsum(kf); acc = _nbsum(col * kf[..., None])
        now = todo & (cnt > 0)
        if not now.any(): break
        col[now] = acc[now] / cnt[now][:, None]
        kn |= now; filled |= now; todo &= ~now
    return np.clip(col, 0, 255).astype(np.uint8), filled


def enclosed_holes(rgba):
    """Transparent pixels that cannot reach the sprite border (holes inside the silhouette)."""
    op = rgba[..., 3] > 0
    if not op.any(): return np.zeros(op.shape, bool)
    h, w = op.shape; free = ~op
    outside = np.zeros_like(free); dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if free[y, x] and not outside[y, x]: outside[y, x] = True; dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if free[y, x] and not outside[y, x]: outside[y, x] = True; dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
            if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True; dq.append((ny, nx))
    return free & ~outside


def fill_holes(rgba, max_frac=0.25):
    """Close holes inside a sprite (e.g. where an arm used to be) with colours from around them."""
    try:
        holes = enclosed_holes(rgba)
        if not holes.any() or holes.sum() > max_frac * (rgba[..., 3] > 0).sum(): return rgba
        col, filled = inpaint(rgba[..., :3], rgba[..., 3] > 0, holes)
        out = rgba.copy(); out[..., :3] = np.where(filled[..., None], col, out[..., :3]); out[..., 3] = np.where(filled, 255, out[..., 3])
        return out
    except Exception:
        return rgba


def close_gaps(rgba, win_frac=0.35, max_frac=0.35):
    """Close what a covering part (an arm lying over the torso in a SIDE view, the near leg over the far one) left behind.
    enclosed_holes() only finds holes that are sealed on every side; this also fills
      * gaps that are bounded on both sides along a row or a column (even when the covering part ran the full width), and
      * 'bites' out of the silhouette edge that are shorter than win_frac of the part's height,
    with colours grown in from the surrounding pixels.  Real shape (waist, shoulders) is kept because only narrow dents
    are closed, and nothing is changed when it would add more than max_frac of the part's area."""
    try:
        from scipy.ndimage import minimum_filter1d, maximum_filter1d
        op = rgba[..., 3] > 0; h, w = op.shape
        if op.sum() < 12 or h < 5 or w < 3: return rgba
        target = np.zeros_like(op)
        # 1. a gap is only a hidden area when it is bounded along the row AND along the column (open notches such as a
        #    waist are real shape and are left to the dent test below)
        both = []
        for arr, lim in ((op, 0.6 * w), (op.T, 0.6 * h)):
            t = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                xs = np.nonzero(arr[i])[0]
                for g in (np.nonzero(np.diff(xs) > 1)[0] if len(xs) > 1 else []):
                    a_, b_ = xs[g] + 1, xs[g + 1]
                    if b_ - a_ <= lim: t[i, a_:b_] = True
            both.append(t)
        target |= both[0] & both[1].T
        # 2. dents in the left / right outline
        rows = np.nonzero(op.any(1))[0]
        if len(rows) >= 5:
            l = np.array([np.nonzero(op[y])[0].min() if op[y].any() else np.nan for y in range(h)], float)
            r = np.array([np.nonzero(op[y])[0].max() if op[y].any() else np.nan for y in range(h)], float)
            y0, y1 = rows.min(), rows.max(); idx = np.arange(y0, y1 + 1)
            ok = ~np.isnan(l[idx]); lv = np.interp(idx, idx[ok], l[idx][ok]); rv = np.interp(idx, idx[ok], r[idx][ok])
            win = max(3, int(win_frac * (y1 - y0 + 1))) | 1
            l2 = maximum_filter1d(minimum_filter1d(lv, win, mode="nearest"), win, mode="nearest")
            r2 = minimum_filter1d(maximum_filter1d(rv, win, mode="nearest"), win, mode="nearest")
            empty = np.array([not op[y].any() for y in idx])               # rows the covering part hid completely
            if empty.any() and empty.sum() <= 0.5 * len(idx):
                xs_ = np.arange(w)[None, :]
                target[y0:y1 + 1] |= empty[:, None] & (xs_ >= np.floor(lv)[:, None]) & (xs_ <= np.ceil(rv)[:, None])
            xs = np.arange(w)[None, :]; thr = max(2.0, 0.15 * w)       # only real bites, not the gentle curve of a waist
            dl, dr = (lv - l2) >= thr, (r2 - rv) >= thr
            lo_x = np.where(dl, np.floor(l2), np.ceil(lv)); hi_x = np.where(dr, np.ceil(r2), np.floor(rv))
            target[y0:y1 + 1] |= (xs >= lo_x[:, None]) & (xs <= hi_x[:, None])
        target &= ~op
        if not target.any() or target.sum() > max_frac * op.sum(): return rgba
        col, filled = inpaint(rgba[..., :3], op, target)
        out = rgba.copy(); out[..., :3] = np.where(filled[..., None], col, out[..., :3]); out[..., 3] = np.where(filled, 255, out[..., 3])
        return out
    except Exception:
        return rgba


def fill_gaps(rgba):
    """fill_holes + close_gaps: the one call the cutter uses for 'Fill hidden areas'."""
    return close_gaps(fill_holes(rgba))


def rig_check(res):
    """List what is weak in a cut rig.  Each item: dict(sev=0 bad/1 warn/2 info, view, key, msg)."""
    items = []
    add = lambda sev, v, k, m: items.append(dict(sev=sev, view=v, key=k, msg=m))
    for v, n in (getattr(res, "counts", None) or {}).items():
        if n == 0: add(1, v, None, f"No {v.upper()} source frame - every {v} part is borrowed or guessed. Add a {v} image or force a frame's view.")
    for v, S in res.views.items():
        for k, sp in S.items():
            if sp.kind == "placeholder": add(0, v, k, "MISSING - no usable cut, a placeholder is shown.")
            elif sp.kind == "synth": add(1, v, k, f"Guessed from the {sp.borrowed.upper()} view (not real data).")
            elif sp.kind == "borrowed": add(1, v, k, f"Borrowed from the {sp.borrowed.upper()} view.")
            elif sp.score < 0.2: add(1, v, k, f"Low confidence ({sp.score:.2f}) - compare the other candidates in PICK PARTS.")
            area = int((sp.rgba[..., 3] > 0).sum())
            if sp.kind not in ("placeholder",) and area < 30: add(1, v, k, f"Tiny sprite ({area}px) - probably a bad cut.")
            holes = int(enclosed_holes(sp.rgba).sum())
            if holes: add(1, v, k, f"{holes}px hole inside the sprite (something covered it). Turn on 'Fill hidden areas' and CUT again.")
    items.sort(key=lambda d: (d["sev"], d["view"], str(d["key"])))
    return items
