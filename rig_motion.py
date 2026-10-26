"""rig_motion.py - preset ACTIONS + SECONDARY MOTION (lag / overshoot / inertia) for the HUMAN.RIG puppet.

A clip is a set of TARGET angles per part (what the muscles ask for) plus a root (pelvis) path.  The pelvis
leads; every other part is a damped spring that follows its PARENT's actual angle, so lag accumulates down each
chain: hip -> thigh -> shin -> foot, chest -> shoulder -> elbow -> wrist -> hand.  Hands trail the hips by
design.  The pelvis' acceleration also shoves the loose parts (arms trailing when you start running).

Angles: degrees, clockwise positive on screen (y down).  A hanging limb swings FORWARD with a NEGATIVE angle
when the figure faces right (side view).  Upward-pointing parts (spine, neck, head) lean forward with POSITIVE.
"""
import math, json, os
import numpy as np

CENTER = ["head", "neck", "chest", "abdomen", "pelvis"]
HIER = ["pelvis", "abdomen", "chest", "neck", "head", "uarmR", "uarmL", "farmR", "farmL", "handR", "handL",
        "thighR", "thighL", "shinR", "shinL", "footR", "footL"]
PARENT = dict(pelvis=None, abdomen="pelvis", chest="abdomen", neck="chest", head="neck", uarm="chest",
              farm="uarm", hand="farm", thigh="pelvis", shin="thigh", foot="shin")
UPWARD = {"head", "neck", "chest", "abdomen"}          # body extends UP from the pivot
NS = 60                                               # native samples per second of a clip


def base_of(k): return k[:-1] if k[-1] in "LR" and k not in CENTER else k
def swap(k): return k if k in CENTER else k[:-1] + ("L" if k[-1] == "R" else "R")
def parent_of(k):
    p = PARENT[base_of(k)]
    return None if p is None else (p if p in CENTER else p + k[-1])


def rest_base(view):
    if view == "side": return {}
    sg = 1 if view == "front" else -1
    return {"uarmR": 7 * sg, "uarmL": -7 * sg, "thighR": 2 * sg, "thighL": -2 * sg}


# ---------------------------------------------------------------------------------------------- curves
def _sm(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def kf(t, times, vals):
    """Eased key-frame curve (zero velocity at each key)."""
    t = np.asarray(t, float); times = np.asarray(times, float); vals = np.asarray(vals, float)
    i = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    u = _sm((t - times[i]) / np.maximum(times[i + 1] - times[i], 1e-9))
    out = vals[i] + (vals[i + 1] - vals[i]) * u
    return np.where(t <= times[0], vals[0], np.where(t >= times[-1], vals[-1], out))

def plant(L, thigh, shin):
    """How far the pelvis must drop so a foot stays on the floor when the leg is bent (cm, y down)."""
    a, b = np.radians(thigh), np.radians(thigh + shin)
    return (L["thigh"] + L["shin"]) - (L["thigh"] * np.cos(a) + L["shin"] * np.cos(b))


# ---------------------------------------------------------------------------------------------- clips
class Clip:
    def __init__(self, key, label, view, dur, loop, ang, root, desc=""):
        self.key, self.label, self.view, self.dur, self.loop, self.desc = key, label, view, dur, loop, desc
        self.A = np.stack([ang[k] for k in HIER]); self.R = np.asarray(root, float); self.n = self.A.shape[1]
        self.ang = ang

    def _idx(self, t):
        if self.loop:
            x = (t % self.dur) / self.dur * self.n; i0 = int(x) % self.n; i1 = (i0 + 1) % self.n; f = x - int(x)
        else:
            x = min(max(t, 0.0), self.dur) * NS; i0 = min(int(x), self.n - 1); i1 = min(i0 + 1, self.n - 1); f = x - int(x)
        return i0, i1, f

    def sample(self, t):
        i0, i1, f = self._idx(t)
        return self.A[:, i0] * (1 - f) + self.A[:, i1] * f, self.R[i0] * (1 - f) + self.R[i1] * f


def _walk(view, L, run=False):
    dur = 0.56 if run else 1.0
    def make(t):
        ph = 2 * np.pi * t / dur; a = {}
        amp_t, amp_a = (48, 48) if run else (26, 24)
        for s, off in (("R", 0.0), ("L", np.pi)):
            p = ph + off
            th = -amp_t * np.sin(p); sh = (10 + 75 * np.maximum(0, np.cos(p + 0.4))) if run else (4 + 42 * np.maximum(0, np.cos(p)))
            a["thigh" + s], a["shin" + s] = th, sh
            a["foot" + s] = np.clip(-0.55 * (th + sh), -25, 30)
            ua = amp_a * np.sin(p)
            a["uarm" + s] = ua
            a["farm" + s] = (-80 + 0.5 * ua) if run else (-14 + 0.55 * ua)
            a["hand" + s] = 0.2 * a["farm" + s]
        a["chest"] = (12 if run else 4) + 1.5 * np.sin(2 * ph); a["abdomen"] = 2 + 0 * t
        a["head"] = -(8 if run else 3) + 0.8 * np.sin(2 * ph); a["pelvis"] = 1.5 * np.sin(ph)
        y = (2 - 5 * np.abs(np.sin(ph))) if run else -1.4 * np.cos(2 * ph)
        return a, y, 0 * t
    return dur, True, make

def _idle(view, L):
    dur = 3.2
    def make(t):
        br = np.sin(2 * np.pi * t / dur); a = {"chest": 1.0 * br, "abdomen": 0.5 * br, "head": -0.8 * br, "neck": 0.3 * br}
        sw = 0.6 * np.sin(2 * np.pi * t / (2 * dur))
        if view == "side":
            for s in "RL": a["uarm" + s] = 3 + 1.2 * br; a["farm" + s] = -5 + 0 * t
        else:
            a["uarmR"] = 7 + 1.0 * br; a["uarmL"] = -7 - 1.0 * br; a["pelvis"] = sw
        return a, -0.35 * br, (0.0 if view == "side" else 0.25 * sw)
    return dur, True, make

def _crouch_pose(view, c, a):
    if view == "side":
        a["thighR"] = a["thighL"] = -55 * c; a["shinR"] = a["shinL"] = 95 * c; a["footR"] = a["footL"] = -40 * c
        a["chest"] = 15 * c; a["head"] = -8 * c
    else:
        a["thighR"] = 2 + 48 * c; a["thighL"] = -2 - 48 * c; a["shinR"] = -50 * c; a["shinL"] = 50 * c
    th = a["thighR"]; sh = a["shinR"]
    return th, sh

def _jump(view, L):
    dur, T0, T1, air, h = 1.5, 0.30, 0.46, 0.50, 35.0
    T2 = T1 + air
    def make(t):
        c = kf(t, [0, T0, T1, T2, T2 + 0.12, T2 + 0.40, dur], [0, 1, 0, 0, 0.8, 0, 0])
        u = np.clip((t - T1) / air, 0, 1); airborne = ((t > T1) & (t < T2)).astype(float)
        tuck = np.sin(np.pi * u) * airborne
        a = {}
        th, sh = _crouch_pose(view, c, a)
        if view == "side":
            a["thighR"] = a["thighL"] = -55 * c - 38 * tuck; a["shinR"] = a["shinL"] = 95 * c + 62 * tuck
            ar = kf(t, [0, T0, T1, T1 + 0.15, T2, T2 + 0.2, dur], [0, 45, -20, -150, -120, 10, 0])
            for s in "RL": a["uarm" + s] = ar; a["farm" + s] = -10 - 25 * tuck; a["hand" + s] = -5 * tuck
            a["chest"] = 15 * c - 5 * tuck
        else:
            ar = kf(t, [0, T0, T1, T1 + 0.15, T2, T2 + 0.2, dur], [7, 10, 60, 165, 150, 30, 7])
            a["uarmR"], a["uarmL"] = ar, -ar; a["farmR"], a["farmL"] = 12 * tuck, -12 * tuck
            a["thighR"] = a["thighR"] + 10 * tuck; a["thighL"] = a["thighL"] - 10 * tuck
        ground = 1 - airborne
        drop = plant(L, a["thighR"] if view != "side" else -55 * c, a["shinR"] if view != "side" else 95 * c)
        y = drop * ground - h * 4 * u * (1 - u) * airborne
        return a, y, 0 * t
    return dur, False, make

def _crouch(view, L):
    dur = 1.8
    def make(t):
        c = kf(t, [0, 0.55, 1.1, dur], [0, 1, 1, 0]); a = {}
        th, sh = _crouch_pose(view, c, a)
        if view == "side":
            for s in "RL": a["uarm" + s] = -30 * c; a["farm" + s] = -20 * c
        return a, plant(L, a["thighR"] if view != "side" else -55 * c, a["shinR"] if view != "side" else 95 * c), 0 * t
    return dur, False, make

def _wave(view, L):
    dur = 2.0
    def make(t):
        up = kf(t, [0, 0.35, 1.65, dur], [0, 1, 1, 0]); w = np.sin(2 * np.pi * 3 * np.clip(t - 0.35, 0, 1.3) / 1.3) * up
        a = {}
        if view == "side":
            a["uarmR"] = -145 * up + 3 * (1 - up); a["farmR"] = -35 * up - 25 * w; a["handR"] = -12 * w
            a["uarmL"] = 3 + 0 * t; a["farmL"] = -5 + 0 * t
        else:
            a["uarmR"] = 7 + 143 * up; a["farmR"] = 35 * up + 25 * w; a["handR"] = 14 * w
        a["head"] = (-4 if view == "side" else 3) * up; return a, 0 * t, 0 * t
    return dur, False, make

def _kick(view, L):
    dur = 1.4
    def make(t):
        a = {}
        if view == "side":
            th = kf(t, [0, .22, .5, .68, .95, dur], [0, -90, -92, -92, -10, 0]); sh = kf(t, [0, .22, .5, .68, .95, dur], [0, 115, 5, 5, 100, 0])
            a.update(thighR=th, shinR=sh, footR=kf(t, [0, .5, .68, dur], [0, 20, 20, 0]), thighL=0 * t - 2, shinL=6 + 0 * t)
            lean = kf(t, [0, .22, .5, .68, .95, dur], [0, -4, -14, -14, -3, 0]); a["chest"] = lean; a["head"] = -lean * 0.4
            a["uarmR"] = kf(t, [0, .5, dur], [0, 45, 0]); a["uarmL"] = kf(t, [0, .5, dur], [0, -55, 0])
            a["farmR"] = a["farmL"] = -25 + 0 * t
        else:
            th = kf(t, [0, .3, .62, .85, dur], [2, 70, 80, 60, 2]); sh = kf(t, [0, .3, .62, .85, dur], [0, -45, -5, -50, 0])
            a.update(thighR=th, shinR=sh); a["chest"] = kf(t, [0, .62, dur], [0, 10, 0])
            a["uarmR"] = kf(t, [0, .62, dur], [7, 40, 7]); a["uarmL"] = kf(t, [0, .62, dur], [-7, -60, -7])
        return a, 0 * t, 0 * t
    return dur, False, make

def _punch(view, L):
    dur = 1.1
    def make(t):
        a = {"thighR": -15 + 0 * t, "shinR": 12 + 0 * t, "thighL": 18 + 0 * t, "shinL": 10 + 0 * t}
        gu, gf = -40, -100
        jab = kf(t, [0, .1, .24, .36, dur], [0, 0, 1, 0, 0]); crs = kf(t, [0, .5, .64, .76, dur], [0, 0, 1, 0, 0])
        a["uarmR"] = gu + (-48) * jab; a["farmR"] = gf + 100 * jab; a["handR"] = 0 * t
        a["uarmL"] = gu + (-48) * crs; a["farmL"] = gf + 100 * crs; a["handL"] = 0 * t
        a["chest"] = 6 + 8 * crs + 4 * jab; a["abdomen"] = 3 + 0 * t; a["head"] = -4 + 0 * t
        dr = 0.5 * (plant(L, a["thighR"], a["shinR"]) + plant(L, a["thighL"], a["shinL"]))
        return a, dr + 0 * t, 0 * t
    return dur, False, make

def _cheer(view, L):
    dur = 0.9
    def make(t):
        p = 2 * np.pi * t / dur; b = np.abs(np.sin(p / 2)); a = {}
        a["uarmR"] = 150 + 14 * np.sin(p); a["uarmL"] = -150 - 14 * np.sin(p + np.pi)
        a["farmR"] = 20 + 10 * np.sin(p); a["farmL"] = -20 - 10 * np.sin(p + np.pi)
        a["thighR"] = 4 + 6 * b; a["thighL"] = -4 - 6 * b; a["head"] = 3 * np.sin(p)
        return a, -4 * b + 1.0, 0 * t
    return dur, True, make

def _jacks(view, L):
    dur = 0.8
    def make(t):
        p = 2 * np.pi * t / dur; op = 0.5 - 0.5 * np.cos(p); a = {}
        a["uarmR"] = 7 + 158 * op; a["uarmL"] = -7 - 158 * op
        a["thighR"] = 2 + 22 * op; a["thighL"] = -2 - 22 * op
        y = plant(L, 22 * op, 0) - 4 * np.abs(np.sin(p))
        return a, y, 0 * t
    return dur, True, make


BUILDERS = dict(idle=_idle, walk=lambda v, L: _walk(v, L), run=lambda v, L: _walk(v, L, True), jump=_jump, kick=_kick,
                punch=_punch, crouch=_crouch, wave=_wave, cheer=_cheer, jacks=_jacks)
INFO = dict(
    idle=("IDLE", "Breathing and a slow weight shift.", ("torso", "head"), ("front", "side", "back")),
    walk=("WALK", "In-place walk cycle: legs swing, arms counter-swing, hips bob.", ("legs", "arms", "torso"), ("side",)),
    run=("RUN", "Fast cycle with a flight phase, bent elbows and a forward lean.", ("legs", "arms", "torso"), ("side",)),
    jump=("JUMP", "Crouch, launch, tuck, land.  Hips lead; arms and hands trail.", ("legs", "arms", "torso"), ("front", "side", "back")),
    kick=("KICK", "Chamber and extend one leg while the torso leans back.", ("legs", "torso"), ("side",)),
    punch=("JAB-CROSS", "Two punches with a forward lean.", ("arms", "torso"), ("side",)),
    crouch=("CROUCH", "Down, hold, back up.", ("legs", "torso"), ("front", "side", "back")),
    wave=("WAVE", "Raise the near arm and wave.", ("arms",), ("front", "side", "back")),
    cheer=("CHEER", "Arms up, little bounces.", ("arms", "legs"), ("front", "back")),
    jacks=("JUMPING JACKS", "Arms and legs open and close with a hop.", ("arms", "legs"), ("front", "back")),
    sidekick=("SIDE KICK", "Lateral kick with a counter-lean.", ("legs", "torso"), ("front", "back")))
BUILDERS["sidekick"] = _kick
WHY_NOT = dict(
    side="Needs the FRONT/BACK view: the movement is sideways and hidden in profile.",
    front="Needs the SIDE view: the legs/arms swing toward and away from the camera.")
GROUPS = dict(legs=("thigh", "shin", "foot"), arms=("uarm", "farm", "hand"), torso=("chest", "abdomen", "pelvis"), head=("head", "neck"))
ORDER = ["idle", "walk", "run", "jump", "kick", "punch", "sidekick", "crouch", "wave", "cheer", "jacks"]


def build_clip(key, view, L):
    """L: lengths in cm (thigh, shin, uarm, farm ...).  Back view = the front clip with sides swapped."""
    src = "front" if view == "back" else view
    dur, loop, make = BUILDERS[key](src, L)
    n = int(round(dur * NS)); n = n if loop else n + 1
    t = np.arange(n) / NS
    upd, y, x = make(t)
    base = rest_base(src); ang = {}
    for k in HIER:
        v = upd.get(k, base.get(k, 0.0)); ang[k] = np.broadcast_to(np.asarray(v, float), t.shape).copy()
    root = np.stack([np.broadcast_to(np.asarray(x, float), t.shape), np.broadcast_to(np.asarray(y, float), t.shape)], 1)
    if view == "back": ang = {swap(k): v for k, v in ang.items()}
    label, desc = INFO[key][0], INFO[key][1]
    return Clip(key, label, view, dur, loop, ang, root, desc)


def list_actions(res, view):
    """What this body can do in this view, and how trustworthy the result will look."""
    S = res.views[view]; out = []
    score = dict(own=1.0, locked=1.0, borrowed=0.5, synth=0.4, placeholder=0.0, disabled=0.0)
    def q(group):
        ks = [k for k in S if base_of(k) in GROUPS[group]]
        return (sum(score.get(S[k].kind, 0.5) for k in ks) / max(1, len(ks)), sum(1 for k in ks if S[k].kind in ("synth", "placeholder")), len(ks))
    for key in ORDER:
        label, desc, needs, views = INFO[key]
        if view not in views:
            out.append(dict(key=key, label=label, ok=False, quality=0.0, note=WHY_NOT["side" if view == "side" else "front"])); continue
        qs = [(g,) + q(g) for g in needs]; worst = min(qs, key=lambda z: z[1])
        notes = [f"{g}: {bad} of {n} parts are guessed/missing in the {view.upper()} view" for g, qq, bad, n in qs if bad]
        out.append(dict(key=key, label=label, ok=True, quality=worst[1], note=desc + ("  |  " + "; ".join(notes) if notes else "")))
    return out


# ---------------------------------------------------------------------------------------------- secondary motion
SPRING = dict(  # natural frequency (Hz), damping ratio, inertial coupling weight, lever length fallback (cm)
    abdomen=(9, .8, .10, 18), chest=(8, .8, .10, 30), neck=(6, .6, .40, 10), head=(5, .45, .60, 22),
    uarm=(6.5, .55, .50, 30), farm=(5, .45, .80, 26), hand=(4, .35, 1.00, 10),
    thigh=(9, .8, .20, 42), shin=(7, .6, .30, 42), foot=(6, .5, .50, 15))


def simulate(clip, L, flex=1.0, bounce=1.0, inertia=1.0, jitter=0.0, speed=1.0, fps=24, seed=3):
    """Run the follow-through physics.  Returns [{'t', 'angles': {part: local deg}, 'root': (x, y) cm}]."""
    T = clip.dur / max(speed, 1e-3); dt = 1.0 / 240.0
    pre = 2 * T if clip.loop else 0.0
    rec = T if clip.loop else T + 0.6
    nf = max(2, int(round(rec * fps))); times = pre + np.arange(nf) / fps
    total = int((pre + rec) / dt) + 2
    n = len(HIER); idx = {k: i for i, k in enumerate(HIER)}; par = [idx[parent_of(k)] if parent_of(k) else -1 for k in HIER]
    rig = flex < 0.02
    ks = np.zeros(n); kd = np.zeros(n); w = np.zeros(n); lev = np.ones(n); up = np.zeros(n, bool)
    for i, k in enumerate(HIER):
        b = base_of(k)
        if b == "pelvis": continue
        f0, z0, w0, l0 = SPRING[b]
        f = max(0.8, f0 / math.sqrt(max(flex, 0.02))); z = float(np.clip(z0 / max(bounce, 0.05), 0.05, 1.6))
        ks[i] = (2 * math.pi * f) ** 2; kd[i] = 2 * z * 2 * math.pi * f; w[i] = w0
        lev[i] = L.get(b, l0) if b in ("uarm", "farm", "thigh", "shin") else l0; up[i] = b in UPWARD
    rng = np.random.default_rng(seed); ph = rng.uniform(0, 2 * np.pi, (n, 2)); fr = np.stack([1.3 + 0.37 * np.arange(n), 3.1 + 0.51 * np.arange(n)], 1)
    h = 1.0 / 60.0
    def root_at(tau): return clip.sample(tau * speed)[1]
    tgt0, _ = clip.sample(0.0)
    phi = np.zeros(n); om = np.zeros(n)
    for i in range(n):                       # start at rest on the first frame
        phi[i] = tgt0[i] + (phi[par[i]] if par[i] >= 0 else 0.0)
    out = []; fi = 0
    for s in range(total):
        tau = s * dt
        tgt, root = clip.sample(tau * speed)
        if jitter > 0:
            tgt = tgt + jitter * w * (np.sin(2 * math.pi * fr[:, 0] * tau + ph[:, 0]) + 0.6 * np.sin(2 * math.pi * fr[:, 1] * tau + ph[:, 1]))
        a = np.clip((root_at(tau + h) - 2 * root + root_at(max(tau - h, 0.0))) / (h * h), -2500, 2500)
        for i in range(n):
            p = par[i]
            if p < 0: phi[i] = tgt[i]; continue
            star = phi[p] + tgt[i]
            if rig: phi[i] = star; om[i] = 0.0; continue
            r = math.radians(phi[i]); c, sn = math.cos(r), math.sin(r)
            nx, ny = (c, sn) if up[i] else (-c, -sn)
            al_in = -(a[0] * nx + a[1] * ny) * 1.5 / lev[i] * w[i] * inertia
            al = ks[i] * (star - phi[i]) - kd[i] * om[i] + math.degrees(al_in)
            om[i] = float(np.clip(om[i] + al * dt, -2500, 2500)); phi[i] += om[i] * dt
        while fi < nf and tau + 1e-9 >= times[fi]:
            ang = {k: float(phi[i] - (phi[par[i]] if par[i] >= 0 else 0.0)) for i, k in enumerate(HIER)}
            out.append(dict(t=float(times[fi] - pre), angles=ang, root=(float(root[0]), float(root[1])))); fi += 1
        if fi >= nf: break
    return out


# ---------------------------------------------------------------------------------------------- export
def _gif_frame(im, bg=None):
    from PIL import Image
    im = im.convert("RGBA"); a = np.asarray(im)[..., 3]
    p = im.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
    arr = np.array(p); arr[a < 128] = 255; p = Image.fromarray(arr, "P"); pal = p.getpalette() or []
    p.putpalette(pal + [0] * (768 - len(pal))); p.info["transparency"] = 255
    return p


def export_animation(images, frames, fps, folder, name, meta=None):
    """images: list of RGBA PIL frames (same size).  Writes <name>.png (APNG), <name>.gif, <name>_sheet.png, <name>.json."""
    from PIL import Image
    os.makedirs(folder, exist_ok=True)
    bb = None
    for im in images:
        b = im.getbbox()
        if b: bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
    bb = bb or (0, 0, images[0].width, images[0].height)
    bb = (max(0, bb[0] - 2), max(0, bb[1] - 2), min(images[0].width, bb[2] + 2), min(images[0].height, bb[3] + 2))
    ims = [im.crop(bb) for im in images]; ms = int(round(1000 / fps)); w, h = ims[0].size
    ims[0].save(os.path.join(folder, f"{name}.png"), save_all=True, append_images=ims[1:], duration=ms, loop=0, disposal=2)
    g = [_gif_frame(i) for i in ims]
    g[0].save(os.path.join(folder, f"{name}.gif"), save_all=True, append_images=g[1:], duration=ms, loop=0, disposal=2, transparency=255)
    cols = min(len(ims), 8); rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * w, rows * h), (0, 0, 0, 0))
    for i, im in enumerate(ims): sheet.alpha_composite(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(os.path.join(folder, f"{name}_sheet.png"))
    data = dict(name=name, fps=fps, frame_size=[w, h], crop_origin=[bb[0], bb[1]], sheet_cols=cols, frames=frames, **(meta or {}))
    with open(os.path.join(folder, f"{name}.json"), "w") as f: json.dump(data, f, indent=1)
    return folder
