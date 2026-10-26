"""organism_model.py - the permanent, self-improving body model behind HUMAN.RIG.

One model per (organism, sex).  Sexless organisms (fungi, plants, jellyfish ...) share a single
"any" model.  Models live OUTSIDE the loadouts, in  ~/.humanrig/models/<organism>/<sex>/  (override
with the HUMANRIG_MODELS environment variable), so every cut of every loadout keeps building the same
brain.  Saves are atomic and a rolling backup is kept.

What it learns, all from the corrections you make in the editor (painted regions, dragged joints,
+/- overlap strokes, forced views, layer nudges) and from the cuts you confirm:

  1. PIXEL LABELLER   per-view softmax regression over body-frame position, distance to every part
                      capsule, the neighbourhood of the geometric guess, and colour.  It STARTS as an
                      exact copy of the geometric labeller (prior weights) and is pulled away from it only
                      as far as your corrections justify (L2 anchor to the prior).
  2. LAYERING         which part is drawn in front of which, per view, from who "wins" the pixels where two
                      parts overlap, plus explicit nudges from the pose page.
  3. JOINT CAPS       how far the overlap around each pivot really needs to reach (from +OVERLAP strokes).
  4. PROPORTIONS      waist / ribs / crotch / neck positions along the spine (prior for subjects whose
                      silhouette cannot be measured).
  5. JOINT BIAS       systematic offsets of the pose detector per view (e.g. hips always a bit high).
  6. CROSS-VIEW       real width ratios between the front / side / back version of every part, used when a
                      view has to be synthesised from another one.

Only numpy is required.  The module knows nothing about tkinter or the rig file: the host passes an
Anatomy (part keys, parents, base-name function) so the same code can serve other organisms later.
"""
import os, json, time, hashlib, shutil, threading
from pathlib import Path
import numpy as np

VIEWS = ("front", "side", "back")
SEXES = ("male", "female")
# organisms whose sexes look the same for rigging purposes -> one pooled "any" model
SEXLESS = {"fungus", "fungi", "mushroom", "plant", "plants", "tree", "fern", "moss", "algae", "sponge",
           "jellyfish", "coral", "earthworm", "snail", "starfish", "slug"}
KNOWN = ["human", "dog", "cat", "horse", "bird", "fish", "reptile", "insect", "fungus", "plant"]
SCHEMA_VERSION = 1


def known_organisms():
    return KNOWN


def model_key(organism, sex):
    o = (organism or "human").strip().lower().replace(" ", "_") or "human"
    if o in SEXLESS:
        return o, "any"
    s = (sex or "").strip().lower()
    return o, (s if s in SEXES else "any")


def models_root():
    return Path(os.environ.get("HUMANRIG_MODELS") or (Path.home() / ".humanrig" / "models"))


class Anatomy:
    def __init__(self, keys, parent, base_of, torso=("chest", "abdomen", "pelvis")):
        self.keys = list(keys)
        self.parent = dict(parent)
        self.base_of = base_of
        self.torso = tuple(torso)
        self.P = len(self.keys)
        self.idx = {k: i for i, k in enumerate(self.keys)}

    def schema(self):
        return hashlib.sha1((",".join(self.keys) + f"|v{SCHEMA_VERSION}").encode()).hexdigest()[:12]


def _seg_dist(P, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return np.hypot(P[:, 0] - a[0], P[:, 1] - a[1])
    t = np.clip(((P - a) @ ab) / L2, 0, 1)
    q = a + t[:, None] * ab
    return np.hypot(P[:, 0] - q[:, 0], P[:, 1] - q[:, 1])


def _dilate(m, r):
    out = m.copy()
    for _ in range(int(r)):
        o = out.copy()
        o[1:] |= out[:-1]; o[:-1] |= out[1:]; o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        out = o
    return out


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


class Registry:
    def __init__(self, anatomy, defaults, root=None):
        self.anatomy, self.defaults, self.root = anatomy, dict(defaults), Path(root) if root else None
        self._cache = {}

    def get(self, organism, sex):
        key = model_key(organism, sex)
        if key not in self._cache:
            self._cache[key] = BodyModel(self.anatomy, self.defaults, key[0], key[1], self.root or models_root())
        return self._cache[key]


class BodyModel:
    MIN_PX_TO_APPLY = 200       # painted pixels a view needs before the labeller is allowed to change anything
    CONF = 0.60                 # the learned label must beat the geometric one with at least this confidence
    LAM = 0.002                 # strength of the anchor to the geometric prior

    def __init__(self, anatomy, defaults, organism, sex, root):
        self.A, self.defaults = anatomy, dict(defaults)
        self.organism, self.sex = organism, sex
        self.dir = Path(root) / organism / sex
        self.P = anatomy.P
        self.F = 3 * self.P + 9
        self._lock = threading.RLock()
        self._reset_state()
        self.load()

    # ------------------------------------------------------------------ state / persistence
    def _prior(self):
        P = self.P
        W = np.zeros((P, self.F), np.float32)
        for k in range(P):
            W[k, k] = 3.0               # the geometric guess
            W[k, P + k] = 1.0           # its neighbourhood
            W[k, 2 * P + k] = 1.5       # closeness to the part's capsule
        return W

    def _reset_state(self):
        self.W0 = self._prior()
        self.W = {v: self.W0.copy() for v in VIEWS}
        self.G = {v: np.zeros_like(self.W0) for v in VIEWS}
        self.layers = {v: np.zeros((self.P, self.P), np.float32) for v in VIEWS}
        self.px = {v: 0 for v in VIEWS}                      # painted pixels learned per view
        self.props, self.caps, self.joff, self.xview = {}, {}, {}, {}
        self.hist, self.seen = [], []
        self.align_by = {}                                    # loadout name -> its manual alignment
        self.n_cuts = 0
        self.created = time.strftime("%Y-%m-%dT%H:%M:%S")

    def load(self):
        npz, js = self.dir / "model.npz", self.dir / "model.json"
        if not (npz.exists() and js.exists()):
            return False
        try:
            meta = json.loads(js.read_text())
            if meta.get("schema") != self.A.schema():
                raise ValueError("anatomy changed")
            z = np.load(npz)
            for v in VIEWS:
                W = z[f"W_{v}"]
                if W.shape != self.W0.shape:
                    raise ValueError("shape mismatch")
                self.W[v], self.G[v], self.layers[v] = W.astype(np.float32), z[f"G_{v}"].astype(np.float32), z[f"L_{v}"].astype(np.float32)
            self.px = {v: int(meta["px"].get(v, 0)) for v in VIEWS}
            self.props, self.caps, self.joff, self.xview = (meta.get(k, {}) for k in ("props", "caps", "joff", "xview"))
            self.hist, self.seen = meta.get("hist", []), meta.get("seen", [])
            self.align_by = meta.get("align_by", {})
            self.n_cuts, self.created = int(meta.get("n_cuts", 0)), meta.get("created", self.created)
            return True
        except Exception:
            # never destroy what we cannot read: park it and start clean
            stamp = time.strftime("%Y%m%d_%H%M%S")
            for p in (npz, js):
                if p.exists():
                    try: p.rename(p.with_name(f"{p.stem}.unreadable_{stamp}{p.suffix}"))
                    except Exception: pass
            self._reset_state()
            return False

    def save(self):
        with self._lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            npz, js = self.dir / "model.npz", self.dir / "model.json"
            if npz.exists():
                bk = self.dir / "backups"; bk.mkdir(exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                try:
                    shutil.copy2(npz, bk / f"model_{stamp}.npz")
                    if js.exists(): shutil.copy2(js, bk / f"model_{stamp}.json")
                    for old in sorted(bk.glob("model_*.npz"))[:-5]:
                        old.unlink(); old.with_suffix(".json").unlink(missing_ok=True)
                except Exception:
                    pass
            arrs = {}
            for v in VIEWS:
                arrs[f"W_{v}"], arrs[f"G_{v}"], arrs[f"L_{v}"] = self.W[v], self.G[v], self.layers[v]
            tmp = self.dir / "model.tmp.npz"
            np.savez_compressed(tmp, **arrs); os.replace(tmp, npz)
            meta = dict(schema=self.A.schema(), organism=self.organism, sex=self.sex, px=self.px, props=self.props,
                        caps=self.caps, joff=self.joff, xview=self.xview, hist=self.hist[-200:], seen=self.seen[-5000:], align_by=self.align_by,
                        n_cuts=self.n_cuts, created=self.created, saved=time.strftime("%Y-%m-%dT%H:%M:%S"))
            tj = self.dir / "model.tmp.json"
            tj.write_text(json.dumps(meta)); os.replace(tj, js)

    def export_zip(self, path):
        self.save()
        base = str(path)[:-4] if str(path).lower().endswith(".zip") else str(path)
        return shutil.make_archive(base, "zip", self.dir)

    def reset(self):
        """Forget everything (a backup of the current files is kept first)."""
        with self._lock:
            self.save()
            self._reset_state()
            self.save()

    def summary(self):
        tot = sum(self.px.values())
        acc = [h for h in self.hist if h.get("acc_model") is not None]
        s = f"{self.organism}/{self.sex}  -  {tot:,} corrected px, {self.n_cuts} cuts"
        if acc:
            s += f"\nlast cut: model {acc[-1]['acc_model']*100:.0f}% vs geometry {acc[-1]['acc_geo']*100:.0f}% agreement with you"
        else:
            s += "\nuntrained: behaves like the geometric labeller"
        return s

    # ------------------------------------------------------------------ small stat helper
    @staticmethod
    def _upd(store, key, val, w=1.0):
        m, n = store.get(key, (0.0, 0.0))
        n2 = n + w
        store[key] = [m + (val - m) * w / n2, n2]

    def _est(self, store, key, default, k=2.0):
        if key in store and store[key][1] >= 2:
            m, n = store[key]
            return (m * n + default * k) / (n + k)
        return default

    # ------------------------------------------------------------------ features
    def _dist(self, fd, pts):
        D = np.full((len(pts), self.P), 99.0, np.float32)
        caps = {c[0]: c for c in fd.caps}
        for i, k in enumerate(self.A.keys):
            _, a, b, r = caps[k]
            if r > 0.01:
                D[:, i] = _seg_dist(pts, np.asarray(a, np.float32), np.asarray(b, np.float32)) / r
        return D

    def _prep(self, fd):
        ys, xs = np.nonzero(fd.alpha > 127)
        rr = max(2, int(round(2.5 * fd.s)))
        h, w = fd.lab_geo.shape
        y0, y1 = max(0, ys.min() - rr), min(h, ys.max() + rr + 1)
        x0, x1 = max(0, xs.min() - rr), min(w, xs.max() + rr + 1)
        sub = fd.lab_geo[y0:y1, x0:x1]
        I = np.zeros((self.P, sub.shape[0] + 1, sub.shape[1] + 1), np.int32)
        for c in range(self.P):
            I[c, 1:, 1:] = (sub == c).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
        return dict(I=I, y0=y0, y1=y1, x0=x0, x1=x1, rr=rr)

    def _features(self, fd, prep, ys, xs):
        P, N = self.P, len(ys)
        pts = np.stack([xs, ys], 1).astype(np.float32) + 0.5
        D = self._dist(fd, pts)
        X = np.zeros((N, self.F), np.float32)
        lab = fd.lab_geo[ys, xs]
        ok = lab >= 0
        X[np.nonzero(ok)[0], lab[ok]] = 1.0
        # neighbourhood of the geometric labels
        rr, I = prep["rr"], prep["I"]
        ya0 = np.clip(ys - rr, prep["y0"], prep["y1"]) - prep["y0"]; ya1 = np.clip(ys + rr + 1, prep["y0"], prep["y1"]) - prep["y0"]
        xa0 = np.clip(xs - rr, prep["x0"], prep["x1"]) - prep["x0"]; xa1 = np.clip(xs + rr + 1, prep["x0"], prep["x1"]) - prep["x0"]
        cnt = I[:, ya1, xa1] - I[:, ya0, xa1] - I[:, ya1, xa0] + I[:, ya0, xa0]
        X[:, P:2 * P] = (cnt.T / float((2 * rr + 1) ** 2)).astype(np.float32)
        X[:, 2 * P:3 * P] = np.exp(-np.minimum(D, 8.0))
        # body-frame position (side views are normalised to face right)
        p0, n0 = np.asarray(fd.J["pelvis"], np.float32), np.asarray(fd.J["neck"], np.float32)
        S = n0 - p0; Ls = max(float(np.linalg.norm(S)), 1e-3); sh = S / Ls
        perp = np.array([sh[1], -sh[0]], np.float32)
        q = pts - p0
        t = np.clip((q @ sh) / Ls, -1.0, 2.0); wv = np.clip((q @ perp) / Ls, -1.5, 1.5)
        if fd.facing == "side": wv = wv * float(getattr(fd, "face_dir", 1))
        o = 3 * P
        X[:, o], X[:, o + 1], X[:, o + 2], X[:, o + 3], X[:, o + 4] = t, wv, t * t, wv * wv, t * wv
        X[:, o + 5:o + 8] = fd.rgb[ys, xs].astype(np.float32) / 255.0
        X[:, -1] = 1.0
        return X, D

    def _logits(self, view, X, D):
        return np.where(D > 6.0, -30.0, X @ self.W[view].T)

    # ------------------------------------------------------------------ labelling
    def relabel(self, fd):
        """Geometric labels, corrected where the learned model is clearly sure the geometry is wrong."""
        out = fd.lab_geo.copy()
        view = fd.facing
        if self.px.get(view, 0) < self.MIN_PX_TO_APPLY:
            return out
        try:
            ys, xs = np.nonzero(fd.alpha > 127)
            if len(ys) == 0: return out
            prep = self._prep(fd)
            for i in range(0, len(ys), 120000):
                yy, xx = ys[i:i + 120000], xs[i:i + 120000]
                X, D = self._features(fd, prep, yy, xx)
                Pr = _softmax(self._logits(view, X, D))
                pred, conf = Pr.argmax(1), Pr.max(1)
                geo = fd.lab_geo[yy, xx]
                take = (pred != geo) & (conf >= self.CONF) & (D.min(1) < 5.0)
                out[yy[take], xx[take]] = pred[take]
        except Exception:
            return fd.lab_geo.copy()
        return out.astype(np.int16)

    # ------------------------------------------------------------------ training
    @staticmethod
    def _sig(fd, owner):
        h = hashlib.sha1()
        h.update(f"{owner}|{fd.name}|{fd.view_override}|{fd.flip}".encode())
        h.update(fd.manual.tobytes()); h.update(fd.extm.tobytes())
        h.update(json.dumps(fd.jover, sort_keys=True).encode())
        return h.hexdigest()[:20]

    def _fit(self, view, X, D, y, w, lr=0.25, epochs=4):
        W, G, W0 = self.W[view], self.G[view], self.W0
        n = len(y)
        rng = np.random.default_rng(int(sum(self.px.values())) + n)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for i in range(0, n, 256):
                b = idx[i:i + 256]
                Dm = D[b].copy(); Dm[np.arange(len(b)), y[b]] = 0.0     # never forbid the label you taught
                p = _softmax(np.where(Dm > 6.0, -30.0, X[b] @ W.T))
                p[np.arange(len(b)), y[b]] -= 1.0
                p *= w[b][:, None]
                g = p.T @ X[b] / max(float(w[b].sum()), 1e-6) + self.LAM * (W - W0)
                G += g * g
                W -= lr * g / (np.sqrt(G) + 1e-6)

    def train(self, frames, log=lambda t: None, owner="", max_painted=14000):
        """Learn from the frames of a CUT.  Identical edits are only ever learned once."""
        with self._lock:
            data = {v: [] for v in VIEWS}
            hit_m = hit_g = tot = 0
            new_frames = 0
            sigs = []
            for fd in frames:
                sig = self._sig(fd, owner)
                if sig in self.seen: continue
                try:
                    view = fd.facing
                    edited = bool((fd.manual > -2).any() or fd.extm.any() or fd.jover or fd.view_override)
                    for k, v in (getattr(fd, "props_measured", None) or {}).items():
                        if k in ("waist", "crotch", "neck"): self._upd(self.props, k, float(v), 1.0)
                    if edited:
                        self._train_joints(fd, view)
                        self._train_caps(fd)
                        self._train_props(fd)
                        self._train_layers(fd, view)
                    prep = self._prep(fd)
                    alpha_ok = fd.alpha > 127
                    ys, xs = np.nonzero((fd.manual >= 0) & alpha_ok)
                    rng = np.random.default_rng(len(ys) + 7)
                    if len(ys) > max_painted:
                        sel = rng.choice(len(ys), max_painted, replace=False); ys, xs = ys[sel], xs[sel]
                    if len(ys):
                        X, D = self._features(fd, prep, ys, xs)
                        y = fd.manual[ys, xs].astype(np.int64)
                        pred = self._logits(view, X, D).argmax(1)
                        hit_m += int((pred == y).sum()); hit_g += int((fd.lab_geo[ys, xs] == y).sum()); tot += len(y)
                        data[view].append((X, D, y, np.full(len(y), 4.0, np.float32)))
                        self.px[view] += len(y)
                    # anchors: geometry you did NOT touch (so accepted), at random and right next to your strokes -
                    # without them a few painted pixels would drag the whole labeller toward the painted part
                    painted = fd.manual >= 0
                    free = alpha_ok & (fd.manual == -2) & (fd.lab_geo >= 0)
                    near = _dilate(painted, max(3, int(round(2.5 * fd.s)))) & free if painted.any() else np.zeros_like(free)
                    for msk, cap in ((free, 2500), (near, 2500)):
                        y2, x2 = np.nonzero(msk)
                        if len(y2) == 0: continue
                        sel = rng.choice(len(y2), min(cap, len(y2)), replace=False); y2, x2 = y2[sel], x2[sel]
                        X, D = self._features(fd, prep, y2, x2)
                        data[view].append((X, D, fd.lab_geo[y2, x2].astype(np.int64), np.full(len(y2), 1.0, np.float32)))
                    sigs.append(sig); new_frames += 1
                except Exception as e:
                    log(f"model: could not learn from {getattr(fd, 'name', '?')} ({e!r})")
            for v in VIEWS:
                if not data[v]: continue
                X = np.concatenate([d[0] for d in data[v]]); D = np.concatenate([d[1] for d in data[v]])
                y = np.concatenate([d[2] for d in data[v]]); w = np.concatenate([d[3] for d in data[v]])
                self._fit(v, X, D, y, w)
            if new_frames:
                self.seen.extend(sigs); self.n_cuts += 1
                rec = dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), frames=new_frames, painted=tot,
                           acc_model=(hit_m / tot if tot else None), acc_geo=(hit_g / tot if tot else None))
                self.hist.append(rec)
                if tot:
                    log(f"model {self.organism}/{self.sex}: learned {new_frames} frame(s), {tot} corrected px - "
                        f"before learning it agreed with you on {100*hit_m/tot:.0f}% (pure geometry {100*hit_g/tot:.0f}%)")
                else:
                    log(f"model {self.organism}/{self.sex}: learned {new_frames} frame(s) (no painted corrections)")
                self.save()
            return new_frames

    def _basis(self, fd):
        p0, n0 = np.asarray(fd.J["pelvis"], float), np.asarray(fd.J["neck"], float)
        S = n0 - p0; Ls = float(np.linalg.norm(S))
        sh = S / max(Ls, 1e-6)
        return p0, Ls, sh, np.array([sh[1], -sh[0]])

    def _train_joints(self, fd, view):
        det = getattr(fd, "J_det", None)
        if not det or not fd.jover: return
        p0, Ls, sh, perp = self._basis(fd)
        if Ls < 10: return
        face = float(getattr(fd, "face_dir", 1)) if view == "side" else 1.0
        for n in fd.jover:
            if n not in det: continue
            d = np.asarray(fd.J[n], float) - np.asarray(det[n], float)
            self._upd(self.joff, f"{view}|{n}|x", float(np.clip(d @ sh / Ls, -0.15, 0.15)))
            self._upd(self.joff, f"{view}|{n}|y", float(np.clip(d @ perp / Ls * face, -0.15, 0.15)))

    def adjust_joints(self, J, V, view, face_dir=1, protect=()):
        """Apply the learned detector bias to freshly detected joints (never to joints you placed)."""
        if not self.joff: return
        p0, n0 = np.asarray(J["pelvis"], float), np.asarray(J["neck"], float)
        S = n0 - p0; Ls = float(np.linalg.norm(S))
        if Ls < 10: return
        sh = S / Ls; perp = np.array([sh[1], -sh[0]]); face = float(face_dir) if view == "side" else 1.0
        for n in list(J.keys()):
            if n in protect: continue
            kx, ky = f"{view}|{n}|x", f"{view}|{n}|y"
            if kx not in self.joff or self.joff[kx][1] < 2: continue
            sh_k = self.joff[kx][1] / (self.joff[kx][1] + 4.0)
            dx, dy = self.joff[kx][0] * sh_k, self.joff[ky][0] * sh_k
            J[n] = np.asarray(J[n], float) + (dx * sh + dy * face * perp) * Ls

    def _train_caps(self, fd):
        A = self.A
        for pid, k in enumerate(A.keys):
            bit = np.uint32(1 << pid)
            m = ((fd.extm & bit) != 0) & (fd.alpha > 127)
            if m.sum() < 20: continue
            ys, xs = np.nonzero(m)
            pts = np.stack([xs, ys], 1).astype(np.float64) + 0.5
            centres = [("p:" + A.base_of(k), fd.pivots[k])] + [("d:" + A.base_of(c), fd.pivots[c]) for c in A.keys if A.parent[c] == k]
            dist = np.stack([np.hypot(pts[:, 0] - c[1][0], pts[:, 1] - c[1][1]) for c in centres], 1)
            near = dist.argmin(1)
            for i, (key, _) in enumerate(centres):
                sel = near == i
                if sel.sum() >= 20:
                    self._upd(self.caps, key, float(np.percentile(dist[sel, i], 90)) / fd.s)

    def cap_p(self, base, default): return self._est(self.caps, "p:" + base, default)
    def cap_d(self, base, default): return self._est(self.caps, "d:" + base, default)

    def _train_props(self, fd):
        A = self.A
        try: ch, ab, pe = (A.idx[n] for n in A.torso)
        except KeyError: return
        p0, Ls, sh, _ = self._basis(fd)
        if Ls < 30: return
        ys, xs = np.nonzero(fd.alpha > 127)
        lab = fd.lab[ys, xs]
        t = (((np.stack([xs, ys], 1) + 0.5) - p0) @ sh) / Ls
        tc, tp = t[lab == ch], t[lab == pe]
        if len(tc) < 80 or len(tp) < 80: return
        ribs, waist, crotch = float(np.percentile(tc, 3)), float(np.percentile(tp, 97)), float(np.percentile(tp, 2))
        if 0.12 <= waist <= 0.40: self._upd(self.props, "waist", waist, 3.0)
        if waist + 0.1 <= ribs <= 0.8: self._upd(self.props, "ribs", ribs, 3.0)
        if -0.45 <= crotch <= -0.04: self._upd(self.props, "crotch", crotch, 3.0)

    def props_prior(self):
        out = dict(self.defaults)
        for k in ("waist", "ribs", "crotch", "neck"):
            if k in self.props and k in out and out[k] is not None:
                out[k] = self._est(self.props, k, out[k])
        out["ribs"] = max(out["ribs"], out["waist"] + 0.14)
        return out

    # ------------------------------------------------------------------ layering
    def _train_layers(self, fd, view):
        ys, xs = np.nonzero(fd.alpha > 127)
        if len(ys) == 0: return
        if len(ys) > 25000:
            sel = np.random.default_rng(3).choice(len(ys), 25000, replace=False); ys, xs = ys[sel], xs[sel]
        pts = np.stack([xs, ys], 1).astype(np.float32) + 0.5
        D = self._dist(fd, pts)
        lab = fd.lab[ys, xs]
        inside = D < 0.9
        A1 = np.zeros((len(ys), self.P), np.float32)
        ok = lab >= 0
        A1[np.nonzero(ok)[0], lab[ok]] = 1.0
        # pixel labelled a, inside the capsule of a AND of b  ->  a won the overlap: a is in front of b
        cnt = (A1 * inside).T @ inside.astype(np.float32)
        np.fill_diagonal(cnt, 0)
        self.layers[view] += cnt

    def _prefer(self, view, a, b):
        L = self.layers[view]; n = L[a, b] + L[b, a]
        return 0.5 if n < 40 else float((L[a, b] + 1) / (n + 2))

    def draw_order(self, view, default):
        """Back-to-front order: the default, adjusted wherever the evidence clearly disagrees."""
        order = list(default)
        if view not in self.layers or not self.layers[view].any(): return order
        idx = self.A.idx
        for _ in range(len(order)):
            moved = False
            for i in range(len(order) - 1):
                a, b = order[i], order[i + 1]      # a is drawn behind b
                if self._prefer(view, idx[a], idx[b]) > 0.75:
                    order[i], order[i + 1] = b, a; moved = True
            if not moved: break
        return order

    def layer_nudge(self, view, key, direction, default, reach=3, weight=300.0):
        """direction +1 = bring `key` in front of the next parts, -1 = send it behind the previous ones."""
        idx = self.A.idx
        order = self.draw_order(view, default); i = order.index(key)
        others = order[i + 1:i + 1 + reach] if direction > 0 else order[max(0, i - reach):i]
        for o in others:
            if direction > 0: self.layers[view][idx[key], idx[o]] += weight
            else: self.layers[view][idx[o], idx[key]] += weight

    # ------------------------------------------------------------------ cross-view widths
    def xview_ratio(self, ov, v, base, default):
        """Learned width ratio when turning a sprite of view `ov` into view `v`."""
        st = self.xview
        k1, k2 = f"{ov}>{v}:{base}", f"{v}>{ov}:{base}"
        if k1 in st and st[k1][1] >= 2: r = self._est(st, k1, default)
        elif k2 in st and st[k2][1] >= 2: r = 1.0 / max(self._est(st, k2, 1.0 / default), 1e-3)
        else: return default
        return float(np.clip(r, 0.3, 3.0))

    def train_crossview(self, res, log=lambda t: None):
        """After a cut: compare the REAL (not synthesised) sprites of the same part in different views."""
        def width(sp):
            xs = np.nonzero(sp.rgba[..., 3].any(0))[0]
            return float(xs.max() - xs.min() + 1) if len(xs) else 0.0
        new = 0
        with self._lock:
            for k in self.A.keys:
                b = self.A.base_of(k)
                for a, c in (("front", "side"), ("back", "side"), ("front", "back")):
                    sa, sc = res.chosen.get(a, {}).get(k), res.chosen.get(c, {}).get(k)
                    if sa is None or sc is None: continue
                    if sa.kind not in ("own", "locked") or sc.kind not in ("own", "locked"): continue
                    if min(sa.score, sc.score) < 0.3 or not sa.cid or not sc.cid: continue
                    sig = f"xv|{sa.cid}|{sc.cid}|{k}"
                    if sig in self.seen: continue
                    wa, wc = width(sa), width(sc)
                    if wa < 3 or wc < 3: continue
                    self._upd(self.xview, f"{a}>{c}:{b}", wc / wa, min(sa.score, sc.score))
                    self.seen.append(sig); new += 1
            if new:
                log(f"model: learned {new} cross-view width ratio(s)")
                self.save()

    # ------------------------------------------------------------------ manual joint alignment
    def train_align(self, owner, align, log=lambda t: None):
        """Remember how YOU aligned this loadout.  One entry per loadout (re-exporting replaces it, never piles up)."""
        with self._lock:
            clean = {v: {k: dict(off=[float(x) for x in e.get("off", (0, 0))], piv=[float(x) for x in e.get("piv", (0, 0))])
                         for k, e in (d or {}).items() if any(abs(float(x)) > 1e-6 for x in list(e.get("off", (0, 0))) + list(e.get("piv", (0, 0))))}
                     for v, d in (align or {}).items()}
            clean = {v: d for v, d in clean.items() if d}
            if clean: self.align_by[owner or "?"] = clean
            else: self.align_by.pop(owner or "?", None)
            self.save()

    def align_prior(self, min_owners=2):
        """Average alignment over loadouts (needs >= min_owners that touched the same part in the same view)."""
        acc = {}
        for owner, al in self.align_by.items():
            for v, d in al.items():
                for k, e in d.items():
                    acc.setdefault((v, k), []).append(e)
        out = {}
        for (v, k), es in acc.items():
            if len(es) >= min_owners:
                out.setdefault(v, {})[k] = dict(off=[float(np.mean([e["off"][i] for e in es])) for i in (0, 1)],
                                                piv=[float(np.mean([e["piv"][i] for e in es])) for i in (0, 1)])
        return out