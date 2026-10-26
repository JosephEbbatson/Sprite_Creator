"""HUMAN.RIG v3  -  rig cutter + paint editor + part picker, built to feed the game.

Pipeline
  1. ANALYSE  detect a skeleton in every source frame (MediaPipe) and label every pixel by body part.
  2. EDIT     fix bad assumptions: paint part regions, drag joints, force a frame's view, disable frames.
              Every edit is saved inside the loadout and survives re-analysis.
  3. CUT      cut every part for FRONT, SIDE and BACK separately (frames are sorted by the view they show).
  4. PICK     choose the best candidate per part/view from a card inventory (locks survive re-cuts).
  5. EXPORT   rig.png (sheet: exploded parts + assembled figures for all 3 views), atlas.png (clean,
              transparent, same coordinates), rig.json (pivots, bones, rects) and per-part PNGs.

Side-view convention: sprites always face RIGHT.  In the SIDE view  R = near limb, L = far limb.
1 px = 2 cm by default.  Needs: numpy, pillow, (mediapipe + rembg optional), tkinter for the GUI.
"""
import os, sys, math, json, threading, queue, traceback, shutil, datetime, re
from pathlib import Path
from collections import deque
import numpy as np
from PIL import Image, ImageDraw
import organism_model as OM
import rig_tools as RT
import rig_pose as RP
import rig_motion as RM
import rig_gui_extra as RGX
import rig_body as RB

CM_PER_PX = 2.0
SS = 3
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
MAX_SRC_S = 3.5
MAX_GIF_FRAMES = 30
LOADOUTS_DIR = Path("loadouts")
VIEWS = ("front", "side", "back")
MIN_SCORE = 0.05
KEEP_CANDS = 14

# ----------------------------------------------------------------------------------------------
# Skeleton / parts
# ----------------------------------------------------------------------------------------------
PRIOR = dict(spine=.29, neck=.12, sho=.129, uarm=.186, farm=.146, hip=.095, thigh=.245, shin=.246)
BONES = [("pelvis", "neck", "spine"), ("neck", "head", "neck"),
         ("Lsho", "Lelb", "uarm"), ("Lelb", "Lwri", "farm"), ("Rsho", "Relb", "uarm"),
         ("Relb", "Rwri", "farm"), ("Lhip", "Lknee", "thigh"), ("Lknee", "Lank", "shin"),
         ("Rhip", "Rknee", "thigh"), ("Rknee", "Rank", "shin")]
HAND_LEN, FOOT_EXT = 0.105, 0.045

CENTER = ["head", "neck", "chest", "abdomen", "pelvis"]
PART_KEYS = CENTER + ["uarmR", "uarmL", "farmR", "farmL", "handR", "handL",
                      "thighR", "thighL", "shinR", "shinL", "footR", "footL"]
STRETCH = ("uarm", "farm", "thigh", "shin")
SPEC = dict(head=(13, 3, 4), neck=(9, 6, 7), chest=(30, 8, 10), abdomen=(24, 8, 9), pelvis=(30, 8, 8),
            uarm=(11, 9, 6), farm=(10, 7, 5), hand=(9, 6, 6), thigh=(15, 13, 8), shin=(12, 9, 7),
            foot=(11, 8, 8))
# the window only clips; side views need room front/back of the spine and toes pointing sideways
SPEC_VIEW = dict(side=dict(head=(16, 3, 4), neck=(10, 6, 7), chest=(19, 8, 10), abdomen=(18, 8, 9),
                           pelvis=(19, 8, 8), hand=(10, 6, 6), thigh=(15, 13, 8), foot=(17, 8, 8)))
# Overlap ("joint caps").  Every part also owns a round patch of its neighbours around its PIVOT
# (CAP_P, cm) and the parent keeps a patch of each child around the child's pivot (CAPD, cm), so a
# rotated limb exposes real material instead of a wedge-shaped gap.
CAP_P = dict(head=8, neck=6, chest=10, abdomen=9, pelvis=0, uarm=8, farm=5.5, hand=5, thigh=11, shin=8, foot=7)
CAPD = dict(abdomen=8, chest=8, neck=7, head=5, uarm=7, farm=5, hand=4, thigh=13, shin=7, foot=6)
# fraction of the axis (from its A end) where the pivot of a centre part sits
# Proportions along the spine (0 = hip centre P, 1 = shoulder centre N).  These are only the DEFAULTS:
# measure_props() fits waist / crotch / neck to each subject's own silhouette.
DEF_PROPS = dict(waist=0.22, ribs=0.52, crotch=-0.14, neck=0.6, neck_half=None, neck_wc=0.0)
PIV_FRAC = dict(head=0.613, neck=1.0, chest=1.0, abdomen=1.0, pelvis=0.611)
RADIUS = dict(head=9, neck=5.5, chest=18, abdomen=15, pelvis=15, uarm=5, farm=4.2, hand=4,
              thigh=9, shin=6.2, foot=5.5)
# In a side view the torso is as deep as it is wide/2: front-view radii would swallow the arm hanging in front of it.
RADIUS_SIDE = dict(head=9, neck=5, chest=12.5, abdomen=10.5, pelvis=11.5, uarm=5, farm=4.2, hand=4,
                   thigh=8.5, shin=6, foot=5.5)
OCC_R = dict(head=8, neck=5, chest=14, abdomen=12.5, pelvis=12, uarm=4.5, farm=3.8, hand=3.5,
             thigh=8, shin=5.5, foot=5)
PARENT = dict(pelvis=None, abdomen="pelvis", chest="abdomen", neck="chest", head="neck",
              uarm="chest", farm="uarm", hand="farm", thigh="pelvis", shin="thigh", foot="shin")
JOINTS = dict(head=("head",), neck=("neck", "head"), chest=("Lsho", "Rsho"),
              abdomen=("Lsho", "Rsho", "Lhip", "Rhip"), pelvis=("Lhip", "Rhip"),
              uarm=("{s}sho", "{s}elb"), farm=("{s}elb", "{s}wri"), hand=("{s}wri", "{s}mid"),
              thigh=("{s}hip", "{s}knee"), shin=("{s}knee", "{s}ank"), foot=("{s}ank", "{s}toe"))

def base_of(k): return k[:-1] if k[-1] in "LR" and k not in CENTER else k
def side_of(k): return k[-1] if k not in CENTER else ""
def swap_side(k): return k if k in CENTER else k[:-1] + ("L" if k[-1] == "R" else "R")
def spec_for(view, b): return SPEC_VIEW.get(view, {}).get(b, SPEC[b])
def parent_key(k):
    p = PARENT[base_of(k)]
    return None if p is None else (p if p in CENTER else p + side_of(k))
def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v * 0.0
def pretty(k):
    b = base_of(k); s = side_of(k)
    return (b.upper() + ("." + s if s else ""))

def _adj():
    A = set()
    def add(a, b): A.add(frozenset((a, b)))
    for a, b in (("head","neck"),("neck","chest"),("chest","abdomen"),("abdomen","pelvis"),
                 ("head","chest"),("chest","pelvis")): add(a,b)
    for s in "LR":
        for p in ("chest","neck","abdomen"): add(p, "uarm"+s)
        add("uarm"+s,"farm"+s); add("farm"+s,"hand"+s)
        add("pelvis","thigh"+s); add("abdomen","thigh"+s)
        add("thigh"+s,"shin"+s); add("shin"+s,"foot"+s)
    return A
ADJ = _adj()

# ---- learned body model (permanent, per organism + sex; see organism_model.py) ---------------------------------
ANATOMY = OM.Anatomy(PART_KEYS, {k: parent_key(k) for k in PART_KEYS}, base_of)
REGISTRY = OM.Registry(ANATOMY, DEF_PROPS)
ACTIVE_MODEL = None
def set_active_model(m):
    global ACTIVE_MODEL
    ACTIVE_MODEL = m
def _cap(kind, b, default):
    if ACTIVE_MODEL is None: return default
    return ACTIVE_MODEL.cap_p(b, default) if kind == "p" else ACTIVE_MODEL.cap_d(b, default)
def _xv(ov, v, b, default):
    return default if ACTIVE_MODEL is None else ACTIVE_MODEL.xview_ratio(ov, v, b, default)
def draw_order(view, mask=None):
    """Back-to-front order, filtered by the loadout part mask."""
    base = RB.active_order(view, mask)
    if ACTIVE_MODEL is None:
        return base
    # model may reorder; keep only keys that are still active
    ordered = ACTIVE_MODEL.draw_order(view, base)
    active = set(base)
    return [k for k in ordered if k in active] + [k for k in base if k not in ordered]
def relearn_frame(fd):
    """Recompute the labels of one frame after the model changed (cheap: no re-measuring)."""
    fd.lab_learn = ACTIVE_MODEL.relabel(fd) if ACTIVE_MODEL else fd.lab_geo.copy()
    fd.lab_auto = fd.lab_learn.copy()
    cm = getattr(fd, "cmodel", None)
    if cm is not None: apply_color_model(fd, cm)
    merge_manual(fd)

class FrameData: pass
class Sprite: pass

def make_joints(il, w, h):
    """33 MediaPipe landmarks -> named joints.  Centre joints keep the BEST visibility of their
    landmarks (in a side view one shoulder/hip is always hidden)."""
    def P(i): return np.array([il[i][0]*w, il[i][1]*h], float)
    J, V = {}, {}
    def put(n, idx, agg=min):
        J[n] = np.mean([P(i) for i in idx], 0)
        V[n] = float(agg(il[i][2] for i in idx))
    put("head",(7,8),max); put("neck",(11,12),max); put("pelvis",(23,24),max)
    V["head"] = max(V["head"], 0.9*float(il[0][2]))
    put("nose",(0,)); put("mouth",(9,10))
    for s,o in (("L",0),("R",1)):
        put(s+"sho",(11+o,)); put(s+"elb",(13+o,)); put(s+"wri",(15+o,))
        put(s+"hip",(23+o,)); put(s+"knee",(25+o,)); put(s+"ank",(27+o,))
        put(s+"mid",(17+o,19+o)); put(s+"toe",(31+o,)); put(s+"heel",(29+o,))
    return J, V

def make_axes(J, s, H, props=None):
    P,N,Hd = J["pelvis"],J["neck"],J["head"]
    S,u = N-P, Hd-N
    pr = props or DEF_PROPS
    ax = dict(head=(Hd+0.55*u,N), neck=(N+pr["neck"]*u,N), chest=(N,P+pr["ribs"]*S),
              abdomen=(P+pr["ribs"]*S,P+pr["waist"]*S), pelvis=(P+pr["waist"]*S,P+pr["crotch"]*S))
    for sd in "LR":
        ax["uarm"+sd] = (J[sd+"sho"], J[sd+"elb"])
        ax["farm"+sd] = (J[sd+"elb"], J[sd+"wri"])
        f = unit(J[sd+"wri"]-J[sd+"elb"])
        m = J[sd+"mid"]-J[sd+"wri"]
        d = unit(unit(m)*0.6 + f*0.4) if np.linalg.norm(m) > 0.012*H*s else f
        ax["hand"+sd] = (J[sd+"wri"], J[sd+"wri"] + d*HAND_LEN*H*s)
        ax["thigh"+sd] = (J[sd+"hip"], J[sd+"knee"])
        ax["shin"+sd] = (J[sd+"knee"], J[sd+"ank"])
        sh = unit(J[sd+"ank"]-J[sd+"knee"])
        ax["foot"+sd] = (J[sd+"ank"], J[sd+"ank"] + sh*FOOT_EXT*H*s)
    return ax

def make_caps(J, s, H, V=None, near=None, props=None):
    P,N,Hd = J["pelvis"],J["neck"],J["head"]
    S,u = N-P, Hd-N
    pr = props or DEF_PROPS
    ax = make_axes(J,s,H,pr)
    c = dict(head=(N+(pr["neck"]+0.05)*u,Hd+0.55*u), neck=(N,N+pr["neck"]*u),
             chest=(P+pr["ribs"]*S,N+0.04*S), abdomen=(P+pr["waist"]*S,P+pr["ribs"]*S),
             pelvis=(P+pr["crotch"]*S,P+pr["waist"]*S))
    ext = FOOT_EXT*H*s
    for sd in "LR":
        c["uarm"+sd] = ax["uarm"+sd]; c["farm"+sd] = ax["farm"+sd]
        a,b = ax["hand"+sd]; c["hand"+sd] = (a, b+0.2*(b-a))
        c["thigh"+sd] = ax["thigh"+sd]; c["shin"+sd] = ax["shin"+sd]
        a,b = ax["foot"+sd]
        # side views: the toes point sideways, so label the foot toward the toe landmark
        if V is not None and V.get(sd+"toe", 0) > 0.35:
            toe = J[sd+"toe"]; d = float(np.linalg.norm(toe-a))
            if 0.4*ext < d < 3.5*ext: b = a + (toe-a)*1.1
        c["foot"+sd] = (a,b)
    rad = dict(RADIUS)
    hidden = set()
    if near:                                    # side view
        rad.update(RADIUS_SIDE); far = "L" if near == "R" else "R"; sc = H*s
        sep = lambda n: float(np.linalg.norm(J[far+n]-J[near+n]))/sc
        if (sep("knee")+sep("ank"))/2 < 0.06: hidden |= {"thigh"+far, "shin"+far, "foot"+far}
        if (sep("elb")+sep("wri"))/2 < 0.06: hidden |= {"uarm"+far, "farm"+far, "hand"+far}
    return [(k, np.asarray(c[k][0],float), np.asarray(c[k][1],float),
             (1e-3 if k in hidden else rad[base_of(k)]*s)) for k in PART_KEYS]

def make_pivots(J, props=None):
    """The joint every part rotates about (known BEFORE cutting).  Centre parts hang from the spine:
    pelvis=root, abdomen/chest/neck pivot at their lower end, head at the top of the neck."""
    P, N, Hd = J["pelvis"], J["neck"], J["head"]; S = N-P; u = Hd-N
    pr = props or DEF_PROPS
    pv = dict(head=N+pr["neck"]*u, neck=N.copy(), chest=P+pr["ribs"]*S, abdomen=P+pr["waist"]*S, pelvis=P.copy())
    for sd in "LR":
        pv["uarm"+sd] = J[sd+"sho"]; pv["farm"+sd] = J[sd+"elb"]; pv["hand"+sd] = J[sd+"wri"]
        pv["thigh"+sd] = J[sd+"hip"]; pv["shin"+sd] = J[sd+"knee"]; pv["foot"+sd] = J[sd+"ank"]
    return {k: np.asarray(v, float) for k, v in pv.items()}

def make_ends(axes):
    """Far end of each part from its pivot (the handle used when posing)."""
    return {k: np.asarray(axes[k][1] if k not in CENTER else axes[k][0], float) for k in PART_KEYS}

def seg_dist(P, a, b):
    ab = b-a; L2 = float(ab@ab)
    if L2 < 1e-9: return np.hypot(P[:,0]-a[0], P[:,1]-a[1])
    t = np.clip(((P-a)@ab)/L2, 0, 1)
    q = a + t[:,None]*ab
    return np.hypot(P[:,0]-q[:,0], P[:,1]-q[:,1])

def label_pixels(alpha, caps, maxd=3.5):
    ys, xs = np.nonzero(alpha > 127)
    lab = np.full(alpha.shape, -1, np.int16)
    if len(ys) == 0: return lab
    P = np.stack([xs,ys],1).astype(np.float32)+0.5
    best = np.full(len(P), np.inf, np.float32)
    arg = np.zeros(len(P), np.int16)
    for i,(_,a,b,r) in enumerate(caps):
        d = seg_dist(P, a.astype(np.float32), b.astype(np.float32)) / r
        m = d < best; best[m] = d[m]; arg[m] = i
    lab[ys,xs] = np.where(best < maxd, arg, -1)
    return lab

def measure_props(alpha, lab, J, s, V=None):
    """Fit neck / waist / crotch to THIS subject's silhouette.  Returns only what could be measured
    reliably; everything else keeps its default."""
    out = {}
    P, N, Hd = J["pelvis"], J["neck"], J["head"]; S = N-P; Ls = float(np.linalg.norm(S))
    if Ls < 30: return out
    sh = S/Ls; perp = np.array([sh[1], -sh[0]]); u = Hd-N; Lu = float(np.linalg.norm(u))
    ys, xs = np.nonzero(alpha > 127)
    if len(ys) < 300: return out
    armids = [i for i, k in enumerate(PART_KEYS) if base_of(k) in ("uarm", "farm", "hand")]
    keep = ~np.isin(lab[ys, xs], armids)                       # arms hide the torso edge: leave them out
    Q = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)+0.5-P
    if len(Q) < 200: return out
    d, w = Q@sh, Q@perp
    d0, w0 = int(np.floor(d.min())), int(np.floor(w.min()))
    occ = np.zeros((int(d.max())-d0+2, int(w.max())-w0+2), bool)
    occ[(d-d0).astype(int), (w-w0).astype(int)] = True
    o2 = occ.copy(); o2[:, 1:] |= occ[:, :-1]; o2[:, :-1] |= occ[:, 1:]       # close 1px specks
    runs = (o2[:, 1:] & ~o2[:, :-1]).sum(1) + o2[:, 0]
    first = o2.argmax(1); last = o2.shape[1]-1-o2[:, ::-1].argmax(1)
    width = np.where(o2.any(1), last-first+1-2, 0).astype(float)             # -2: undo the dilation
    dd = d0 + np.arange(len(runs))
    rows = lambda lo, hi: np.nonzero((dd >= lo) & (dd <= hi))[0]
    # crotch: where the silhouette below the hips first splits into two legs
    sel = rows(-0.5*Ls, 0.04*Ls)[::-1]
    need = max(8, int(0.12*Ls)); gap_ok = np.zeros(len(runs), bool)
    for i in sel:                                               # is there a central gap between two runs in this row?
        if runs[i] < 2: continue
        row = o2[i, first[i]:last[i]+1]; z = np.concatenate([[1], (~row).astype(int), [1]]); st = np.nonzero(np.diff(z) == 1)[0]
        en = np.nonzero(np.diff(z) == -1)[0]
        for a0, b0 in zip(st, en):
            if a0 > 0 and b0 < len(row) and (b0-a0) >= 3:
                cx = first[i]+(a0+b0)/2.0+w0
                if abs(cx) < 0.35*Ls: gap_ok[i] = True
    for k in range(max(0, len(sel)-need)):
        if gap_ok[sel[k:k+need]].all():
            out["crotch"] = float(np.clip(dd[sel[k]]/Ls, -0.45, -0.06)); break
    # waist: narrowest point of the torso between hips and ribs
    idx = rows(0.10*Ls, 0.40*Ls)
    if len(idx) > 10:
        wd = np.convolve(width[idx], np.ones(5)/5, mode="same")[3:-3]
        tw = dd[idx[int(np.argmin(wd))+3]]/Ls
        if wd.min() < 0.94*wd.mean() and 0.17 <= tw <= 0.33: out["waist"] = float(tw)   # a real narrowing, not the window edge
    # neck: from the shoulder line up to the chin (landmarks), as wide as the silhouette is there
    if Lu > 10:
        chin_d = None
        if V is not None and V.get("mouth", 0) > 0.5 and V.get("nose", 0) > 0.5:
            chin = J["mouth"] + (J["mouth"]-J["nose"]); chin_d = float((chin-P)@sh)
        lo = Ls + 0.15*Lu
        hi = (chin_d - 0.03*Lu) if chin_d is not None else (Ls + 0.85*Lu)
        idx = rows(lo, hi)
        if len(idx) > 5 and hi > Ls + 0.3*Lu:
            wd = np.convolve(width[idx], np.ones(3)/3, mode="same"); j = int(np.argmin(wd[1:-1]))+1; wmin = wd[j]
            top = None
            if chin_d is not None: top = hi
            else:                                                            # no chin: trust only a clear narrowing
                hi_rows = rows(Ls+0.95*Lu, Ls+1.3*Lu); wh = width[hi_rows].max() if len(hi_rows) else 0
                if wmin > 0 and wh > 0 and wmin < 0.8*wh:
                    k2 = next((jj for jj in range(j, len(wd)) if wd[jj] >= 1.3*wmin), None)
                    if k2 is not None: top = dd[idx[k2]]
            if top is not None and wmin > 0:
                row = idx[j]
                out.update(neck=float(np.clip((top-Ls)/Lu, 0.35, 0.8)),
                           neck_half=float(np.clip(wmin/2, 3.0*s, 5.8*s)),
                           neck_wc=float((first[row]+last[row])/2.0+w0))
    return out

def refine_torso(lab, caps, J, props, alpha, s):
    """Split the torso at the subject's waist / ribs, make the hip block span waist -> crotch at the
    subject's real width, and cut the neck out of the silhouette at the subject's real neck width."""
    ids = {c[0]: i for i, c in enumerate(caps)}
    P, N, Hd = J["pelvis"], J["neck"], J["head"]; S = N-P; Ls = float(np.linalg.norm(S))
    if Ls < 1e-6: return lab
    sh = S/Ls; perp = np.array([sh[1], -sh[0]], np.float32)
    ys, xs = np.nonzero(alpha > 127)
    if len(ys) == 0: return lab
    Q = np.stack([xs, ys], 1).astype(np.float32)+0.5-P
    d, w = Q@sh, Q@perp; t = d/Ls
    L0 = lab[ys, xs].astype(np.int16); new = L0.copy()
    torso = [ids["chest"], ids["abdomen"], ids["pelvis"]]; thighs = [ids["thighL"], ids["thighR"]]
    waist, ribs, crotch = props["waist"], props["ribs"], props["crotch"]
    m = np.isin(L0, torso)
    new[m] = np.where(t[m] < waist, ids["pelvis"], np.where(t[m] < ribs, ids["abdomen"], ids["chest"]))
    # hip block: everything between crotch and waist (inside the hip width) is pelvis
    hipx = abs(float((J["Lhip"]-J["Rhip"])@perp))/2 + (RADIUS["thigh"]*0.9+3)*s
    armish = np.isin(L0, [ids[k] for k in PART_KEYS if base_of(k) in ("uarm", "farm", "hand", "shin", "foot")])
    blk = (t >= crotch) & (t < waist) & (np.abs(w) <= hipx) & ~armish & np.isin(L0, torso+thighs)
    new[blk] = ids["pelvis"]
    # below the crotch the torso gives way to the legs
    low = (t < crotch) & np.isin(new, torso)
    if low.any():
        hid = [k for k in ("thighL", "thighR") if caps[ids[k]][3] < 0.01]
        if hid: new[low] = ids["thighR" if hid[0] == "thighL" else "thighL"]
        else:
            dl = seg_dist(Q[low]+P, caps[ids["thighL"]][1], caps[ids["thighL"]][2])
            dr = seg_dist(Q[low]+P, caps[ids["thighR"]][1], caps[ids["thighR"]][2])
            new[low] = np.where(dl < dr, ids["thighL"], ids["thighR"])
    # neck: cut at the measured neck width; hair beside it belongs to the head
    nh = props.get("neck_half")
    if nh:
        Lu = float(np.linalg.norm(Hd-N)); d_top = Ls + props["neck"]*Lu; wc = props.get("neck_wc", 0.0)
        band = (d >= 0.985*Ls) & (d <= d_top); inside = np.abs(w-wc) <= nh*1.1
        new[band & inside & np.isin(L0, [ids["head"], ids["neck"], ids["chest"]])] = ids["neck"]
        new[band & ~inside & (L0 == ids["neck"])] = ids["head"]
        new[(d > d_top) & (L0 == ids["neck"])] = ids["head"]
    lab[ys, xs] = new
    return lab

def occlusion(caps, s, behind=(), front=()):
    """Fraction of each part covered by non-adjacent parts.  Parts in `behind` (the far limbs of a
    side view) are hidden by everything, but never hide anything themselves."""
    caps = [(k,a,b,OCC_R[base_of(k)]*s) for k,a,b,_ in caps]
    pts = np.array([c[1] for c in caps] + [c[2] for c in caps])
    rmax = max(c[3] for c in caps)
    lo,hi = pts.min(0)-rmax, pts.max(0)+rmax
    step = max(max(hi-lo)/200.0, 1e-3)
    gx,gy = np.meshgrid(np.arange(lo[0],hi[0],step), np.arange(lo[1],hi[1],step))
    G = np.stack([gx.ravel(),gy.ravel()],1)
    M = {k: seg_dist(G,a,b)<=r for k,a,b,r in caps}
    out = {}
    for k in M:
        other = np.zeros(len(G), bool)
        for k2 in M:
            if k2 == k or frozenset((k,k2)) in ADJ: continue
            if k2 in behind and k not in behind: continue
            if k in front and k2 not in front: continue        # near limb: nothing behind it can cover it
            other |= M[k2]
        n = M[k].sum()
        out[k] = float((M[k]&other).sum()/n) if n else 1.0
    return out

def ext_disks(fd, k, scale=1.0):
    """[(centre, radius_px, allowed_label_ids)] describing the automatic overlap of part k."""
    b, s = base_of(k), fd.s
    kids = [c for c in PART_KEYS if parent_key(c) == k]
    out = []
    rp = _cap("p", b, CAP_P[b])*s*scale
    if rp > 0:
        allowed = [PART_KEYS.index(o) for o in PART_KEYS if o != k and o not in kids and frozenset((k,o)) in ADJ]
        if allowed: out.append((fd.pivots[k], rp, allowed))
    for c in kids:
        out.append((fd.pivots[c], _cap("d", base_of(c), CAPD[base_of(c)])*s*scale, [PART_KEYS.index(c)]))
    return out

def part_ext_mask(fd, k, x0, y0, x1, y1, auto=True, scale=1.0):
    """Pixels that are NOT labelled k but also belong to its sprite (automatic caps + painted extension)."""
    pid = PART_KEYS.index(k); lab = fd.lab[y0:y1, x0:x1]
    m = np.zeros(lab.shape, bool)
    if auto:
        yy = (np.arange(y0, y1)+0.5)[:, None]; xx = (np.arange(x0, x1)+0.5)[None, :]
        for pt, rad, allowed in ext_disks(fd, k, scale):
            m |= (((xx-pt[0])**2 + (yy-pt[1])**2) <= rad*rad) & np.isin(lab, allowed)
    m |= (fd.extm[y0:y1, x0:x1] & np.uint32(1 << pid)) != 0
    m &= (lab != pid) & (fd.alpha[y0:y1, x0:x1] > 127)
    return m

# ----------------------------------------------------------------------------------------------
# Frames
# ----------------------------------------------------------------------------------------------
def estimate_scale(J, V, H):
    r = []
    for a,b,n in BONES:
        if min(V[a],V[b])>.5: r.append(np.linalg.norm(J[b]-J[a])/(PRIOR[n]*H))
    return float(np.percentile(r,75)) if len(r)>=4 else None

def front_raw(J, V, H, s):
    f = [(J["Lsho"][0]-J["Rsho"][0])/(0.23*H*s)]
    if min(V["Lhip"],V["Rhip"])>.4: f.append((J["Lhip"][0]-J["Rhip"][0])/(0.10*H*s))
    return float(np.mean(f))

def facing_label(raw):
    if raw >= 0.60: return "front"
    if raw <= -0.45: return "back"
    return "side"

def near_side(J, V, face_dir, flip=False):
    """Which limbs face the camera in a side view ('R' when the subject looks right, unless the
    landmark visibilities clearly say otherwise).  FLIP FACING inverts it."""
    vis = {sd: sum(V[sd+n] for n in ("sho", "elb", "wri", "hip", "knee", "ank")) for sd in "LR"}
    if abs(vis["L"]-vis["R"]) > 1.0: near = "L" if vis["L"] > vis["R"] else "R"
    else: near = "R" if face_dir > 0 else "L"
    if flip and abs(vis["L"]-vis["R"]) > 1.0: near = "L" if near == "R" else "R"
    return near

def face_direction(J, V, H, s):
    """+1 = the subject looks toward image-right, -1 = image-left (only meaningful for side views)."""
    sc, t = H*s, 0.0
    if V.get("nose",0) > 0.3:
        t += float(np.clip((J["nose"][0]-J["head"][0])/(0.045*sc), -1, 1))
    for sd in "LR":
        if min(V[sd+"toe"], V[sd+"heel"]) > 0.3:
            t += 0.7*float(np.clip((J[sd+"toe"][0]-J[sd+"heel"][0])/(0.06*sc), -1, 1))
    return 1 if t >= 0 else -1

def view_conf(fd):
    """How well does this frame really match the view it was assigned to (1 = perfect)?"""
    if getattr(fd, "view_override", None): return 1.0
    r = fd.front_raw
    if fd.facing == "front": return float(np.clip((r-0.25)/0.6, 0.25, 1))
    if fd.facing == "back":  return float(np.clip((-r-0.10)/0.6, 0.25, 1))
    return float(1 - np.clip(abs(r)/0.7, 0, 0.7))

SOLID_COL = {
    "head":(240,220,60),"neck":(200,180,50),"chest":(140,80,180),"abdomen":(130,70,170),"pelvis":(120,60,160),
    "uarmR":(60,180,100),"uarmL":(60,180,100),"farmR":(50,160,90),"farmL":(50,160,90),
    "handR":(40,140,80),"handL":(40,140,80),"thighR":(220,120,160),"thighL":(220,120,160),
    "shinR":(200,100,140),"shinL":(200,100,140),"footR":(180,80,120),"footL":(180,80,120)}
REGION_COL = [(230,25,75),(60,180,75),(255,225,25),(0,130,200),(245,130,48),
              (145,30,180),(70,240,240),(240,50,230),(210,245,60),(250,190,212),
              (0,128,128),(220,190,255),(170,110,40),(255,250,200),(128,0,0),
              (170,255,195),(128,128,0)]
PART_COL = {k: REGION_COL[i] for i, k in enumerate(PART_KEYS)}

def solidify_sprite(rgba, key):
    col = SOLID_COL.get(key, (180,180,180))
    out = rgba.copy(); m = out[...,3]>0
    out[m,0]=col[0]; out[m,1]=col[1]; out[m,2]=col[2]
    return out

def default_landmarks(im):
    a = np.asarray(im.convert("RGBA"))[...,3]>127
    w,h = im.size
    ys,xs = np.nonzero(a)
    if len(ys)<300: x0,x1,y0,y1 = 0,w,0,h
    else: x0,x1,y0,y1 = xs.min(),xs.max()+1,ys.min(),ys.max()+1
    Hp,cx,bot = float(y1-y0),(x0+x1)/2.0,float(y1)
    L = np.zeros((33,3))
    def put(i,x,y): L[i] = [(cx+x*Hp)/w, (bot-y*Hp)/h, 1.0]
    hip_y,neck_y = .53,.82
    for s_,sg,o in (("L",1,0),("R",-1,1)):
        put(7+o,sg*.035,.94); put(11+o,sg*.105,neck_y)
        put(13+o,sg*.115,neck_y-.186); put(15+o,sg*.12,neck_y-.332)
        put(17+o,sg*.125,neck_y-.40); put(19+o,sg*.125,neck_y-.41)
        put(23+o,sg*.05,hip_y); put(25+o,sg*.055,hip_y-.245); put(27+o,sg*.055,hip_y-.491)
        put(29+o,sg*.055,.015); put(31+o,sg*.055,.008)
    put(0,0,.925)
    return L

def merge_manual(fd):
    fd.lab = np.where(fd.manual > -2, fd.manual, fd.lab_auto).astype(np.int16)

def refresh_frame(fd):
    """Recompute everything derived from the joints / overrides of one frame."""
    J, V, H = fd.J, fd.V, fd.H
    s = estimate_scale(J, V, H)
    if s: fd.s = s; fd.s_orig = s / fd.rescale
    s = fd.s
    fd.front_raw = front_raw(J, V, H, s)
    fd.front = float(np.clip(fd.front_raw, 0, 1))
    fd.facing_auto = facing_label(fd.front_raw)
    fd.facing = fd.view_override or fd.facing_auto
    fd.face_auto = face_direction(J, V, H, s)
    fd.face_dir = -fd.face_auto if fd.flip else fd.face_auto
    fd.near = near_side(J, V, fd.face_dir, fd.flip) if fd.facing == "side" else None
    # 1) rough labelling with default proportions -> measure THIS subject's neck / waist / crotch
    prior = ACTIVE_MODEL.props_prior() if ACTIVE_MODEL else dict(DEF_PROPS)
    lab0 = label_pixels(fd.alpha, make_caps(J, s, H, V, fd.near, prior))
    fd.props_measured = measure_props(fd.alpha, lab0, J, s, V)
    props = dict(prior); props.update(fd.props_measured)
    props["ribs"] = max(props["ribs"], props["waist"]+0.14)
    fd.props = props
    # 2) real axes / pivots / labels from the fitted proportions
    fd.axes = make_axes(J, s, H, props)
    fd.caps = make_caps(J, s, H, V, fd.near, props)
    fd.hidden = [k for k, _, _, r in fd.caps if r < 0.01]
    fd.pivots = make_pivots(J, props); fd.ends = make_ends(fd.axes)
    fd.lab_geo = refine_torso(label_pixels(fd.alpha, fd.caps), fd.caps, J, props, fd.alpha, s)
    fd.lab_auto = fd.lab_geo.copy()
    fd.lab_learn = ACTIVE_MODEL.relabel(fd) if ACTIVE_MODEL else fd.lab_geo.copy()   # learned correction of the geometry
    fd.lab_auto = fd.lab_learn.copy()
    cm = getattr(fd, "cmodel", None)
    if cm is not None: apply_color_model(fd, cm)          # what the model learned from all frames
    behind = front = ()
    if fd.facing == "side":
        far = "L" if fd.near == "R" else "R"
        behind = tuple(b+far for b in ("uarm","farm","hand","thigh","shin","foot"))
        front = tuple(b+fd.near for b in ("uarm","farm","hand","thigh","shin","foot"))
    fd.occ = occlusion(fd.caps, s, behind, front)
    merge_manual(fd)

def build_frame(name, img, il, H, edits=None):
    edits = edits or {}
    w,h = img.size
    il = np.asarray(il, float)
    if il.shape!=(33,3) or not np.isfinite(il).all(): return None
    J,V = make_joints(il,w,h)
    Jdet = {k: v.copy() for k, v in J.items()}
    jover = {}
    for n,(nx,ny) in (edits.get("jover") or {}).items():
        if n in J:
            J[n] = np.array([nx*w, ny*h], float); V[n] = 1.0; jover[n] = [float(nx), float(ny)]
    s = estimate_scale(J,V,H)
    if s is None or min(V["neck"],V["pelvis"])<.4: return None
    if ACTIVE_MODEL is not None:
        _vw = edits.get("view") if edits.get("view") in VIEWS else facing_label(front_raw(J, V, H, s))
        ACTIVE_MODEL.adjust_joints(J, V, _vw, face_direction(J, V, H, s), protect=set(jover))
        s = estimate_scale(J, V, H) or s
    s_orig, rescale = s, 1.0
    if s > MAX_SRC_S:
        f = MAX_SRC_S/s
        img = img.convert("RGBa").resize((max(1,round(w*f)), max(1,round(h*f))), Image.Resampling.BOX).convert("RGBA")
        rescale = img.width/w
        J = {k:v*rescale for k,v in J.items()}
        Jdet = {k: v*rescale for k, v in Jdet.items()}
        s *= rescale
    arr = np.asarray(img)
    fd = FrameData()
    fd.name,fd.J,fd.V,fd.s,fd.s_orig,fd.H = name,J,V,s,s_orig,H
    fd.rescale = rescale
    fd.rgb = np.ascontiguousarray(arr[...,:3])
    fd.alpha = np.ascontiguousarray(arr[...,3])
    if (fd.alpha>127).sum()<300: return None
    fd.il = il; fd.jover = jover; fd.J_det = Jdet
    fd.view_override = edits.get("view") if edits.get("view") in VIEWS else None
    fd.flip = bool(edits.get("flip", False))
    fd.enabled = bool(edits.get("enabled", True))
    m = edits.get("manual")
    fd.manual = m.astype(np.int16) if (m is not None and m.shape == fd.alpha.shape) \
        else np.full(fd.alpha.shape, -2, np.int16)
    x = edits.get("extm")
    fd.extm = x.astype(np.uint32) if (x is not None and x.shape == fd.alpha.shape) else np.zeros(fd.alpha.shape, np.uint32)
    fd.undo, fd.redo = [], []
    fd.warn = ""
    fd.cmodel = None
    refresh_frame(fd)
    return fd

def fit_lengths(frames, H):
    m = {}
    for fd in frames:
        for a,b,n in BONES:
            if min(fd.V[a],fd.V[b])>.5:
                m.setdefault(n,[]).append(np.linalg.norm(fd.J[b]-fd.J[a])/fd.s)
    L = {}
    for n,p in PRIOR.items():
        base = p*H
        L[n] = float(np.clip(np.percentile(m[n],80),.8*base,1.25*base)) if len(m.get(n,[]))>=2 else base
    return L

def fit_widths(frames, H):
    sho,hip = [],[]
    for fd in frames:
        if abs(fd.front_raw) < 0.5: continue          # side frames say nothing about width
        if min(fd.V["Lsho"],fd.V["Rsho"])>.5: sho.append(abs(fd.J["Lsho"][0]-fd.J["Rsho"][0])/fd.s)
        if min(fd.V["Lhip"],fd.V["Rhip"])>.5: hip.append(abs(fd.J["Lhip"][0]-fd.J["Rhip"][0])/fd.s)
    return dict(sho=float(np.clip(np.percentile(sho,75),0.18*H,0.32*H)) if sho else 0.23*H,
                hip=float(np.clip(np.percentile(hip,75),0.14*H,0.28*H)) if hip else 0.19*H)

def nominal_cm(b, L, H):
    """Typical length in cm of a part (limbs come from the fitted lengths)."""
    if b in L: return L[b]
    sp = L.get("spine", PRIOR["spine"]*H)
    return dict(head=0.13*H, hand=HAND_LEN*H, foot=FOOT_EXT*H, chest=0.48*sp,
                abdomen=0.30*sp, pelvis=0.36*sp).get(b, 0.1*H)

def render_part(fd, k, pid, A, B, spec, Ltrue, cpp=CM_PER_PX, athr=110, auto_ext=True, ext_scale=1.0, fill=True):
    """Cut part k upright (axis pointing down).  Returns (rgba, pa, pb, pivot, end) in sprite pixels."""
    halfw, e0, e1 = spec
    s = fd.s
    d = B-A; Lp = float(np.hypot(*d))
    if Lp < 1e-3: return None
    v = d/Lp; u = np.array([v[1], -v[0]])
    s_par = float(np.clip(Lp/Ltrue, 0.7*s, 1.2*s)) if Ltrue else s
    Lc = Lp/s_par
    Wf = int(math.ceil(2*halfw/cpp)); Wf += Wf % 2
    Hf = int(math.ceil((e0+Lc+e1)/cpp))
    kk = cpp/SS
    a, b = u[0]*s*kk, v[0]*s_par*kk
    c = A[0]-u[0]*s*(Wf/2)*cpp - v[0]*s_par*e0
    d_, e = u[1]*s*kk, v[1]*s_par*kk
    f = A[1]-u[1]*s*(Wf/2)*cpp - v[1]*s_par*e0
    corners = [(0,0),(Wf*SS,0),(0,Hf*SS),(Wf*SS,Hf*SS)]
    xs = [a*x+b*y+c for x,y in corners]; ys = [d_*x+e*y+f for x,y in corners]
    H0, W0 = fd.alpha.shape
    x0 = max(0, int(math.floor(min(xs)))-3); x1 = min(W0, int(math.ceil(max(xs)))+3)
    y0 = max(0, int(math.floor(min(ys)))-3); y1 = min(H0, int(math.ceil(max(ys)))+3)
    if x1-x0 < 2 or y1-y0 < 2: return None
    own = fd.lab[y0:y1, x0:x1] == pid
    if own.sum() < 12: return None
    sel = own | part_ext_mask(fd, k, x0, y0, x1, y1, auto_ext, ext_scale)
    rgbc = fd.rgb[y0:y1, x0:x1]
    if fill:                       # material hidden behind a part drawn in FRONT (arm over torso, near leg over far leg)
        try:
            cap = next(c for c in fd.caps if c[0] == k)
            _o = draw_order(fd.facing); _fr = [PART_KEYS.index(q) for q in _o[_o.index(k)+1:]]
            hid = RT.hidden_mask(fd.lab[y0:y1, x0:x1], fd.alpha[y0:y1, x0:x1] > 127, sel, _fr,
                                 cap[1], cap[2], cap[3], x0, y0, 14*s)
            if hid.any() and hid.sum() <= 0.6*max(1, int(sel.sum())):
                rgbc, _filled = RT.inpaint(rgbc, sel, hid); sel = sel | _filled
        except Exception: pass
    al = (fd.alpha[y0:y1, x0:x1]*sel).astype(np.uint8)
    im = Image.fromarray(np.dstack([rgbc, al])).convert("RGBa")
    r = max(1, int(s*cpp/SS))
    if r > 1: im = im.reduce(r)
    co = (a/r, b/r, (c-x0)/r, d_/r, e/r, (f-y0)/r)
    out = im.transform((Wf*SS, Hf*SS), Image.Transform.AFFINE, co, Image.Resampling.BILINEAR)
    out = np.array(out.reduce(SS).convert("RGBA"))
    out[..., 3] = np.where(out[..., 3] >= athr, 255, 0)
    out[out[..., 3] == 0, :3] = 0
    if out[..., 3].sum() == 0: return None
    def to_sprite(Q):
        dd = np.asarray(Q, float)-A
        return np.array([(dd@u)/s/cpp + Wf/2, ((dd@v)/s_par + e0)/cpp])
    pa = np.array([Wf/2, e0/cpp]); pb = np.array([Wf/2, (e0+Lc)/cpp])
    return out, pa, pb, to_sprite(fd.pivots[k]), to_sprite(fd.ends[k]), {c: to_sprite(fd.pivots[c]) for c in PART_KEYS if parent_key(c) == k}

def part_score(fd, k, L):
    """Quality of this part in this frame (view match included)."""
    b,sd = base_of(k), side_of(k)
    agg = max if (fd.facing == "side" and k in CENTER) else min
    vis = agg(fd.V[j.format(s=sd)] for j in JOINTS[b])
    qv = float(np.clip(vis/0.55,0,1))
    qf = 1.0
    if b in STRETCH:
        A,B = fd.axes[k]
        ratio = np.linalg.norm(B-A)/(L[b]*fd.s)
        qf = float(np.clip(ratio/0.92,0.05,1))**2
    qo = (1-fd.occ[k])**2
    qr = min(1.0, fd.s_orig/1.2)**0.5
    return qv*qf*qo*qr*view_conf(fd)

def fuse_candidates(cs, blend, gapfill):
    """Combine the candidates of one part.  blend=True: weighted CONSENSUS of up to 10 frames (the shape
    is what most good frames agree on, colours are a weighted mean after throwing out outliers), so
    every extra image makes the part fit the subject better.  Otherwise: best frame, holes filled."""
    best = cs[0]; img = best["img"].copy()
    use = [c for c in cs[1:10] if c["score"] >= 0.4*best["score"] and c["img"].shape == img.shape]
    if not use or not (blend or gapfill): return img, 1
    if blend:
        allc = [best]+use
        Wt = np.array([max(c["score"], 1e-3)**2 for c in allc]); Wt /= Wt.sum()
        A = np.stack([c["img"][..., 3] > 0 for c in allc])
        C = np.stack([c["img"][..., :3] for c in allc]).astype(np.float32)
        cover = (Wt[:, None, None]*A).sum(0)
        mask = cover >= (0.34 if gapfill else 0.5)
        if not mask.any(): return img, 1
        aw = Wt[:, None, None]*A
        m1 = (aw[..., None]*C).sum(0)/np.maximum(aw.sum(0), 1e-6)[..., None]
        inl = A & (np.abs(C-m1[None]).sum(-1) <= 100)
        aw2 = Wt[:, None, None]*inl
        den = aw2.sum(0)
        m2 = np.where((den > 0)[..., None], (aw2[..., None]*C).sum(0)/np.maximum(den, 1e-6)[..., None], m1)
        out = np.zeros_like(img); out[..., :3] = np.where(mask[..., None], np.clip(m2, 0, 255), 0).astype(np.uint8)
        out[..., 3] = np.where(mask, 255, 0)
        return out, len(allc)
    nused = 1; ys, xs = np.nonzero(img[..., 3])
    for c in use:
        hole = img[..., 3] == 0; box = np.zeros_like(hole)
        if len(ys): box[max(0, ys.min()-2):ys.max()+3, max(0, xs.min()-2):xs.max()+3] = True
        take = hole & box & (c["img"][..., 3] > 0); img[take] = c["img"][take]; nused += 1
    return img, nused

# ---- edge smoother: parts should feel grown, not cut ---------------------------------------------------
def _nb8(m):
    h, w = m.shape; p = np.pad(m.astype(np.uint8), 1)
    return sum(p[1+dy:1+dy+h, 1+dx:1+dx+w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)

def _nbsum(a):
    h, w = a.shape[:2]; pad = ((1, 1), (1, 1)) + ((0, 0),)*(a.ndim-2); p = np.pad(a, pad)
    return sum(p[1+dy:1+dy+h, 1+dx:1+dx+w] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)

def smooth_sprite(rgba, level=2):
    """level 0 off | 1 clean specks + pinholes | 2 + fill notches | 3 + round the hard corners left by cut lines.
    Also removes the background halo along the rim so the silhouette takes the colour of the body."""
    level = int(level)
    if level <= 0: return rgba
    keep_n, fill_n, passes = {1: (2, 7, 1), 2: (3, 5, 1), 3: (4, 5, 2)}[min(3, level)]
    h, w = rgba.shape[:2]; a = rgba[..., 3] > 0; m = a.copy()
    for _ in range(passes):
        n = _nb8(m); m = np.where(m, n >= keep_n, n >= fill_n)
    if not m.any(): return rgba
    rgb = rgba[..., :3].astype(np.float32); af = a.astype(np.float32)
    nb_cnt = _nbsum(af); nb_col = _nbsum(rgb*af[..., None])/np.maximum(nb_cnt, 1)[..., None]
    col = np.where(a[..., None], rgb, nb_col)                       # new pixels borrow from their neighbours
    p4 = np.pad(m, 1, constant_values=False)
    edge = m & ~(p4[0:h, 1:w+1] & p4[2:h+2, 1:w+1] & p4[1:h+1, 0:w] & p4[1:h+1, 2:w+2])
    fi = (m & ~edge).astype(np.float32); icnt = _nbsum(fi)
    avg = _nbsum(col*fi[..., None])/np.maximum(icnt, 1)[..., None]
    mix = edge & (icnt > 0)
    col = np.where(mix[..., None], 0.4*col+0.6*avg, col)             # rim leans toward the body colour
    out = np.zeros_like(rgba)
    out[..., :3] = np.where(m[..., None], np.clip(col, 0, 255), 0).astype(np.uint8)
    out[..., 3] = np.where(m, 255, 0)
    return out

# ---- the model that learns from every frame --------------------------------------------------------
COMPAT = np.zeros((len(PART_KEYS), len(PART_KEYS)), bool)
for _i, _a in enumerate(PART_KEYS):
    for _j, _b in enumerate(PART_KEYS):
        COMPAT[_i, _j] = frozenset((_a, _b)) in ADJ

def _color_bins(rgb):
    return ((rgb[..., 0] >> 4).astype(np.int32) << 8) | ((rgb[..., 1] >> 4).astype(np.int32) << 4) | (rgb[..., 2] >> 4).astype(np.int32)

def _interior(lab):
    ok = lab >= 0
    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2),(1,1),(-1,-1),(1,-1),(-1,1)):
        ok &= np.roll(lab, (dy, dx), (0, 1)) == lab
    return ok

class ColorModel: pass

def train_color_model(frames):
    """Per-part colour statistics from the CONFIDENT pixels (deep inside a part) of every frame.
    Pixels you painted count 4x: your corrections teach the model."""
    Hh = np.zeros((len(PART_KEYS), 4096), np.float64)
    for fd in frames:
        lab = np.where(fd.manual > -2, fd.manual, fd.lab_learn)
        ok = _interior(lab) & (fd.alpha > 127); wgt = np.where(fd.manual > -2, 4.0, 1.0); b = _color_bins(fd.rgb)
        for k in range(len(PART_KEYS)):
            sel = ok & (lab == k)
            if sel.any(): Hh[k] += np.bincount(b[sel], weights=wgt[sel], minlength=4096)
    H4 = Hh.reshape(-1, 16, 16, 16)
    for ax in (1, 2, 3):                                              # a little smoothing between nearby colours
        a = np.moveaxis(H4, ax, -1); pad = np.pad(a, [(0, 0)]*3+[(1, 1)], mode="edge")
        H4 = np.moveaxis(0.25*pad[..., :-2]+0.5*pad[..., 1:-1]+0.25*pad[..., 2:], -1, ax)
    Hh = H4.reshape(len(PART_KEYS), 4096)
    m = ColorModel(); m.p = (Hh+0.05)/(Hh.sum(1, keepdims=True)+0.05*4096); m.n = int(sum(f.alpha.size for f in frames) > 0)
    return m

def apply_color_model(fd, model, passes=3, ratio=4.0):
    """Let ambiguous boundary pixels move to the neighbouring part whose colours they match far better."""
    lab = fd.lab_auto.copy(); free = (fd.manual == -2) & (fd.alpha > 127); b = _color_bins(fd.rgb); moved = 0
    for _ in range(passes):
        bestp = np.where(lab >= 0, model.p[np.clip(lab, 0, None), b], 0.0); best = lab.copy(); changed = False
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            nl = np.roll(lab, (dy, dx), (0, 1))
            ok = (nl != lab) & (nl >= 0) & (lab >= 0) & free & COMPAT[np.clip(lab, 0, None), np.clip(nl, 0, None)]
            pn = model.p[np.clip(nl, 0, None), b]; take = ok & (pn > bestp*ratio)
            if take.any(): best = np.where(take, nl, best); bestp = np.where(take, pn, bestp); changed = True
        if not changed: break
        moved += int((best != lab).sum()); lab = best
    fd.lab_auto = lab.astype(np.int16)
    return moved

def learn_colors(frames, log=lambda t: None):
    """(Re)train the colour model on ALL frames and re-label each frame with it."""
    if not frames: return
    model = train_color_model(frames); moved = 0
    for fd in frames:
        fd.cmodel = model; fd.lab_auto = fd.lab_learn.copy(); moved += apply_color_model(fd, model); merge_manual(fd)
    log(f"learned part colours from {len(frames)} frame(s); {moved} boundary pixels re-assigned")

def unlearn_colors(frames):
    for fd in frames:
        if getattr(fd, "cmodel", None) is not None:
            fd.cmodel = None; fd.lab_auto = fd.lab_learn.copy(); merge_manual(fd)

def fit_props(frames):
    """The subject's own proportions: the median of what each frame measured."""
    out = dict(DEF_PROPS)
    for k in ("waist", "ribs", "neck"):
        v = [fd.props[k] for fd in frames if getattr(fd, "props", None)]
        if v: out[k] = float(np.median(v))
    out["ribs"] = max(out["ribs"], out["waist"]+0.14)
    return out

def make_cid(fname, src_key, flip, twin):
    return f"{fname}|{src_key}|{int(flip)}|{int(twin)}"

def sprite_from_img(k, img, c, L, cpp, nused=1, kind="own"):
    ys, xs = np.nonzero(img[..., 3])
    if len(ys) == 0: return None
    x0, x1, y0, y1 = xs.min(), xs.max()+1, ys.min(), ys.max()+1
    sp = Sprite()
    sp.key, sp.score, sp.frame, sp.nused, sp.kind = k, float(c["score"]), str(c.get("fname", "")), nused, kind
    sp.cid, sp.flip, sp.twin = c.get("cid", ""), bool(c.get("flip", False)), bool(c.get("twin", False))
    sp.mirrored = sp.flip or sp.twin
    sp.rgba = np.ascontiguousarray(img[y0:y1, x0:x1])
    off = np.array([x0, y0], float)
    sp.pa, sp.pb = np.asarray(c["pa"], float)-off, np.asarray(c["pb"], float)-off
    sp.pj = np.asarray(c.get("pj", c["pa"]), float)-off          # pivot: the joint it rotates about
    sp.pe = np.asarray(c.get("pe", c["pb"]), float)-off          # far end (pose handle)
    b = base_of(k)
    sp.length_cm = round(float(L[b]) if b in STRETCH else float(np.linalg.norm(sp.pb-sp.pa)*cpp), 1)
    sp.anc = {ck: np.asarray(p, float)-off for ck, p in (c.get("anc") or {}).items()}
    sp.borrowed = ""
    return sp

def apply_palette(rgba, pal):
    m = rgba[...,3] > 0
    if not m.any() or len(pal) == 0: return
    px = rgba[m][:,:3].astype(np.int32)
    d = ((px[:,None,:]-pal[None,:,:].astype(np.int32))**2).sum(2)
    rgba[m,:3] = pal[np.argmin(d,1)]
    rgba[~m,:3] = 0

def build_palette(sprites, n):
    px = [sp.rgba[sp.rgba[...,3]>0][:,:3] for sp in sprites]
    px = [p for p in px if len(p)]
    if not px: return []
    allpx = np.concatenate(px)
    if n and n > 0 and len(np.unique(allpx,axis=0)) > n:
        try:
            q = Image.fromarray(np.ascontiguousarray(allpx.reshape(1,-1,3))).quantize(
                colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
            pal = np.array(q.getpalette()[:3*n], np.uint8).reshape(-1,3)
            used = sorted(set(np.asarray(q).ravel().tolist()))
            pal = pal[used]
        except Exception:
            pal = np.unique(allpx,axis=0)[:n]
        for sp in sprites: apply_palette(sp.rgba, pal)
        allpx = np.concatenate([sp.rgba[sp.rgba[...,3]>0][:,:3] for sp in sprites])
    cols,cnt = np.unique(allpx,axis=0,return_counts=True)
    return [tuple(int(x) for x in cols[i]) for i in np.argsort(-cnt)]

def _make_placeholder(k, L, cpp, col, view="front"):
    b = base_of(k); halfw, e0, e1 = spec_for(view, b)
    length = nominal_cm(b, L, 172.0)
    W = max(4, int(math.ceil(2*halfw/cpp))); H = max(4, int(math.ceil((e0+length+e1)/cpp)))
    img = np.zeros((H, W, 4), np.uint8); cx = W//2; r = max(1, int(halfw/cpp*0.5))
    img[:, max(0, cx-r):cx+r+1] = (*col, 220)
    sp = Sprite(); sp.key, sp.rgba = k, img
    sp.pa = np.array([W/2., e0/cpp]); sp.pb = np.array([W/2., (e0+length)/cpp])
    if k in CENTER:
        sp.pj = np.array([W/2., (e0+PIV_FRAC[k]*length)/cpp]); sp.pe = sp.pa.copy()
    else: sp.pj, sp.pe = sp.pa.copy(), sp.pb.copy()
    sp.length_cm = round(length, 1); sp.score = 0.0; sp.mirrored = sp.flip = sp.twin = False
    sp.frame, sp.cid, sp.nused, sp.kind, sp.borrowed = "placeholder", "", 0, "placeholder", ""
    return sp

# ---- synthesised views: used when a view (or a part of it) has no real source ----------------------
SIDE_FX = dict(head=.88, neck=.80, chest=.62, abdomen=.62, pelvis=.66, uarm=.85, farm=.85, hand=.80,
               thigh=.75, shin=.80, foot=1.9)

def _scale_sprite(sp, fx, fy):
    h, w = sp.rgba.shape[:2]; nw, nh = max(2, int(round(w*fx))), max(2, int(round(h*fy)))
    im = Image.fromarray(sp.rgba).resize((nw, nh), Image.Resampling.NEAREST)
    n = Sprite(); n.__dict__.update(sp.__dict__); n.rgba = np.array(im)
    sx, sy = nw/w, nh/h
    for a in ("pa", "pb", "pj", "pe"):
        p = getattr(sp, a); setattr(n, a, np.array([p[0]*sx, p[1]*sy]))
    n.anc = {ck: np.array([p[0]*sx, p[1]*sy]) for ck, p in (getattr(sp, "anc", None) or {}).items()}
    return n

def _hair_head(rgba):
    """Back of a head from a front head: keep the silhouette, fill it with the hair colour."""
    out = rgba.copy(); a = out[..., 3] > 0; ys, xs = np.nonzero(a)
    if len(ys) == 0: return out
    top, bot = ys.min(), ys.max(); cut = top+max(2, int((bot-top)*0.30))
    band = a.copy(); band[cut:, :] = False
    hair = (np.median(out[band][:, :3], axis=0) if band.any() else out[a][:, :3].mean(0))*0.85
    yy = np.arange(out.shape[0])[:, None]*np.ones((1, out.shape[1]))
    shade = 1.0-0.12*(yy-top)/max(1, bot-top)
    for c in range(3): out[..., c] = np.where(a, np.clip(hair[c]*shade, 0, 255), 0).astype(np.uint8)
    return out

def synthesize(sp, k, ov, v):
    """Make the best guess for part k of view v from the real sprite `sp` of view ov."""
    b = base_of(k); n = Sprite(); n.__dict__.update(sp.__dict__)
    if ov == "side" and v in ("front", "back"):                       # widen a side sprite
        n = _scale_sprite(sp, _xv(ov, v, b, min(2.2, 1/SIDE_FX[b])), 1.0)
    if v == "side":                                                     # squash a front/back sprite
        n = _scale_sprite(sp, _xv(ov, v, b, SIDE_FX[b]), 0.85 if b == "foot" else 1.0)
        if b == "foot": n.pj = np.array([n.pj[0]*0.6, n.pj[1]]); n.pe = np.array([n.pe[0]*0.6, n.pe[1]])
    elif v == "back" and ov != "back":
        n.rgba = _hair_head(n.rgba) if b == "head" else n.rgba.copy()
        if b != "head":
            m = n.rgba[..., 3] > 0; n.rgba[m, :3] = (n.rgba[m, :3]*0.88).astype(np.uint8)
    n.kind, n.borrowed, n.frame = "synth", ov, f"synth<{ov}"
    n.score = float(sp.score)*0.5; n.cid = ""; n.flip = n.twin = n.mirrored = False
    return n

BORROW_ORDER = dict(front=("back", "side"), side=("front", "back"), back=("front", "side"))

def finalize_views(res):
    """res.chosen[view][key] (real cuts) -> res.views[view][key] (always complete): a missing part is
    synthesised from another view when possible (purple), else drawn as a placeholder (red).
    Parts turned off by the loadout mask become transparent disabled placeholders."""
    synth = getattr(res, "synth", True)
    mask = RB.normalize_mask(getattr(res, "mask", None))
    res.mask = mask
    res.scale = RB.normalize_scale(getattr(res, "scale", None))
    res.views = {v: {} for v in VIEWS}
    skin = [c for c in (res.palette or []) if c[0] > 80 and c[1] > 40 and c[2] > 30]
    for v in VIEWS:
        for k in PART_KEYS:
            if not mask.get(k, True):
                # disabled: tiny transparent placeholder so downstream code stays simple
                col = (0, 0, 0)
                ph = _make_placeholder(k, res.lengths, res.cpp, col, v)
                ph.kind = "disabled"
                ph.rgba = ph.rgba.copy(); ph.rgba[..., 3] = 0
                res.views[v][k] = ph
                continue
            sp = res.chosen.get(v, {}).get(k)
            if (sp is not None and sp.score >= MIN_SCORE) or (sp is not None and sp.kind == "locked"):
                res.views[v][k] = sp; continue
            alt = None
            for ov in BORROW_ORDER[v]:
                o = res.chosen.get(ov, {}).get(k)
                if o is not None and o.score >= MIN_SCORE: alt = (ov, o); break
            if alt is None and sp is not None: alt = (v, sp)       # weak but real beats a guess
            if alt is None:                                         # last resort: any weak real cut elsewhere
                for ov in BORROW_ORDER[v]:
                    o = res.chosen.get(ov, {}).get(k)
                    if o is not None: alt = (ov, o); break
            if alt is not None:
                ov, o = alt
                if ov == v: b = Sprite(); b.__dict__.update(o.__dict__)
                elif synth: b = synthesize(o, k, ov, v)
                else:
                    b = Sprite(); b.__dict__.update(o.__dict__); b.kind, b.borrowed = "borrowed", ov
                res.views[v][k] = b; continue
            col = SOLID_COL.get(k, (160, 160, 160))
            if skin and base_of(k) in ("head", "neck", "uarm", "farm", "hand", "thigh", "shin", "foot"):
                col = skin[len(skin)//2]
            res.views[v][k] = _make_placeholder(k, res.lengths, res.cpp, col, v)

def _tkey(ck, view, near):
    """target key of a part in a side view (R = near limb)"""
    if view == "side" and ck not in CENTER: return base_of(ck) + ("R" if side_of(ck) == near else "L")
    return ck

def _flip_pts(c, W):
    for key in ("pa", "pb", "pj", "pe"):
        p = np.asarray(c[key], float); c[key] = np.array([W-p[0], p[1]])
    if c.get("anc"): c["anc"] = {ck: np.array([W-p[0], p[1]]) for ck, p in c["anc"].items()}

def cut_rig(frames, H, cpp=CM_PER_PX, mirror=True, gapfill=True, ncolors=32, athr=110,
            blend=True, locks=None, rejects=None, overlap=True, overlap_scale=1.0, synth=True, smooth=2, learn=True,
            model=None, learn_model=True, owner="", fill_hidden=True, align=None,
            mask=None, scale=None,
            log=lambda t: None, progress=lambda i, n: None, **_ignored):
    locks, rejects = locks or {}, rejects or {}
    mask = RB.normalize_mask(mask)
    scale = RB.normalize_scale(scale)
    frames = [f for f in frames if getattr(f, "enabled", True)]
    if not frames: raise ValueError("No enabled frames to cut.")
    if model is not None:
        set_active_model(model)
        if learn_model:
            try:
                model.train(frames, log, owner=owner)
                for fd in frames: relearn_frame(fd)
            except Exception as e: log(f"model training skipped ({e!r})")
    if learn:
        try: learn_colors(frames, log)
        except Exception as e: log(f"colour learning skipped ({e!r})")
    else: unlearn_colors(frames)
    L, widths = fit_lengths(frames, H), fit_widths(frames, H)
    L = RB.apply_scale_to_lengths(L, scale)
    props = fit_props(frames)
    cands = {v: {k: [] for k in PART_KEYS} for v in VIEWS}
    issues = []; active = RB.active_keys(mask)
    total = max(1, len(frames) * len(active)); done = 0
    counts = {v: 0 for v in VIEWS}
    for fd in frames:
        view = fd.facing; counts[view] += 1
        flip = (view == "side" and fd.face_dir < 0)       # normalise: side sprites face RIGHT
        for pid, k in enumerate(PART_KEYS):
            if not mask.get(k, True):
                continue
            done += 1; progress(done, total)
            try:
                sc = part_score(fd, k, L)
                if sc <= 0.02: continue
                b = base_of(k); A, B = fd.axes[k]
                r = render_part(fd, k, pid, A, B, spec_for(view, b), (L[b] if b in STRETCH else None),
                                cpp, athr, overlap, overlap_scale, fill_hidden)
                if r is None: continue
                img, pa, pb, pj, pe, anc = r
                tk = k
                c = dict(score=sc, img=img, pa=pa, pb=pb, pj=pj, pe=pe, anc=anc, fname=fd.name, src_key=k,
                         flip=flip, twin=False, view=view)
                if view == "side" and k not in CENTER:
                    tk = base_of(k) + ("R" if side_of(k) == fd.near else "L")
                if flip:
                    c["img"] = img[:, ::-1].copy(); _flip_pts(c, img.shape[1])
                c["anc"] = {_tkey(ck, view, fd.near): v for ck, v in c["anc"].items()}
                c["cid"] = make_cid(fd.name, k, flip, False)
                cands[view][tk].append(c)
            except Exception as e:
                issues.append(f"{fd.name}: could not cut {k} ({e!r})")
    for v in VIEWS:
        if counts[v] == 0:
            log(f"No {v.upper()} frame - {v} parts will be "
                f"{'synthesized from the other views (purple)' if synth else 'borrowed from another view'}.")
    if mirror:
        for v in VIEWS:
            for k in PART_KEYS:
                if k in CENTER: continue
                o = swap_side(k)
                for c in list(cands[v][o]):
                    if c["twin"]: continue
                    fl = (v != "side")        # front/back: mirror image.  side: same silhouette, other limb
                    t = dict(c, img=(c["img"][:, ::-1].copy() if fl else c["img"].copy()),
                             score=c["score"]*0.85, twin=True, flip=(c["flip"] != fl))
                    if fl: _flip_pts(t, c["img"].shape[1])
                    t["anc"] = {swap_side(ck): v for ck, v in (t.get("anc") or {}).items()}
                    t["cid"] = make_cid(c["fname"], c["src_key"], t["flip"], True)
                    cands[v][k].append(t)
    chosen = {v: {} for v in VIEWS}
    for v in VIEWS:
        for k in PART_KEYS:
            bad = set(rejects.get(v, {}).get(k, []))
            cs = sorted((c for c in cands[v][k] if c["cid"] not in bad), key=lambda c: -c["score"])
            cands[v][k] = cs[:KEEP_CANDS]
            if not cs: continue
            lock = locks.get(v, {}).get(k)
            hit = next((c for c in cs if c["cid"] == lock), None) if lock else None
            try:
                if hit is not None: img, nused, best, kind = hit["img"].copy(), 1, hit, "locked"
                else: img, nused = fuse_candidates(cs, blend, gapfill); best, kind = cs[0], "own"
            except Exception as e:
                issues.append(f"{v}/{k}: blending failed ({e!r})"); img, nused, best, kind = cs[0]["img"].copy(), 1, cs[0], "own"
            sp = sprite_from_img(k, (RT.fill_holes(smooth_sprite(img, smooth)) if fill_hidden else smooth_sprite(img, smooth)), best, L, cpp, nused, kind)
            if sp is not None:
                chosen[v][k] = sp
                log(f"{v:5s} {k:8s} <- {nused} frame(s) score {sp.score:.2f}{'  [locked]' if kind == 'locked' else ''}")
    allsp = [sp for v in VIEWS for sp in chosen[v].values()]
    try: palette = build_palette(allsp, ncolors)
    except Exception as e: issues.append(f"palette failed ({e!r})"); palette = []
    res = Sprite()
    res.chosen, res.cands, res.palette, res.lengths, res.widths = chosen, cands, palette, L, widths
    res.H, res.cpp, res.issues, res.counts, res.cands_dirty, res.synth = H, cpp, issues, counts, True, synth
    res.smooth, res.props = smooth, props
    res.mask, res.scale = mask, scale
    res.model_tag = (f"{model.organism}/{model.sex}" if model is not None else "")
    res.align = align if align else (model.align_prior() if model is not None else {})
    if model is not None and learn_model:
        try: model.train_crossview(res, log)
        except Exception as e: log(f"cross-view learning skipped ({e!r})")
    finalize_views(res)
    render_outputs(res)
    return res

# ----------------------------------------------------------------------------------------------
# Pixel font (3x5) - used for the sheet and the GUI logo
# ----------------------------------------------------------------------------------------------
GREEN,RED,YELLOW,BLUE=(44,222,81),(237,28,36),(255,242,0),(0,162,232)
GLYPHS = {
 "0":"111101101101111","1":"010110010010111","2":"111001111100111","3":"111001111001111",
 "4":"101101111001001","5":"111100111001111","6":"111100111101111","7":"111001001001001",
 "8":"111101111101111","9":"111101111001111",
 "A":"010101111101101","B":"110101110101110","C":"011100100100011","D":"110101101101110",
 "E":"111100110100111","F":"111100110100100","G":"011100101101011","H":"101101111101101",
 "I":"111010010010111","J":"001001001101010","K":"101101110101101","L":"100100100100111",
 "M":"101111111101101","N":"110101101101101","O":"010101101101010","P":"110101110100100",
 "Q":"010101101111011","R":"110101110101101","S":"011100010001110","T":"111010010010010",
 "U":"101101101101111","V":"101101101101010","W":"101101111111101","X":"101101010101101",
 "Y":"101101010010010","Z":"111001010100111",
 ".":"000000000000010","-":"000000111000000","/":"001001010100100",":":"000010000010000",
 "+":"000010111010000","_":"000000000000111","#":"101111101111101","<":"001010100010001",
 ">":"100010001010100","(":"010100100100010",")":"010001001001010","%":"101001010100101",
 "'":"010010000000000"," ":"0"*15}
def draw_text(d, x, y, text, color, sc=1):
    for ch in str(text).upper():
        g = GLYPHS.get(ch, GLYPHS[" "])
        for i, bit in enumerate(g):
            if bit == "1":
                px, py = x+(i%3)*sc, y+(i//3)*sc
                d.rectangle([px,py,px+sc-1,py+sc-1], fill=color)
        x += 4*sc
    return x
def text_w(text, sc=1): return max(0, 4*sc*len(str(text))-sc)

# ----------------------------------------------------------------------------------------------
# Assembly (per view)
# ----------------------------------------------------------------------------------------------
# draw orders: children over parents (their joint caps then hide every seam); side view: far limbs behind
POSE_ORDER = dict(
    front=["pelvis","abdomen","chest","thighR","thighL","shinR","shinL","footR","footL",
           "uarmR","uarmL","farmR","farmL","handR","handL","neck","head"],
    side=["uarmL","farmL","handL","thighL","shinL","footL","pelvis","abdomen","chest","thighR","shinR",
          "footR","uarmR","farmR","handR","neck","head"])
POSE_ORDER["back"] = POSE_ORDER["front"]
HIER = ["pelvis","abdomen","chest","neck","head","uarmR","uarmL","farmR","farmL","handR","handL",
        "thighR","thighL","shinR","shinL","footR","footL"]          # parents always before children
ASM_ORDER = POSE_ORDER

def rest_angles(view):
    """Neutral standing pose: a slight A-pose so arms and legs do not sit on the torso."""
    if view == "side": return {}
    sg = 1 if view == "front" else -1
    return {"uarmR": 7*sg, "uarmL": -7*sg, "thighR": 2*sg, "thighL": -2*sg}

def close_seams(img, maxgap=2):
    a = np.array(img)
    for axis in (0,1):
        b = a if axis == 0 else a.transpose(1,0,2)
        op = b[...,3] > 0
        n = b.shape[0]
        for g in range(1, maxgap+1):
            for i in range(1, n-g):
                ok = (~op[i:i+g]).all(0) & op[i-1] & op[i+g]
                if ok.any():
                    for o in range(g):
                        b[i+o][ok] = b[i-1][ok]; op[i+o][ok] = True
    return Image.fromarray(np.ascontiguousarray(a))

def _pivots0(res, view):
    """Neutral world position (cm, y down, pelvis at 0,0) of every part's pivot."""
    L, wd = res.lengths, res.widths
    Ls, Ln = L["spine"], L["neck"]
    pr = getattr(res, "props", None) or DEF_PROPS
    J = dict(pelvis=(0,0), abdomen=(0,-pr["waist"]*Ls), chest=(0,-pr["ribs"]*Ls), neck=(0,-Ls), head=(0,-Ls-pr["neck"]*Ln))
    for sd, sg in (("R",-1), ("L",1)):
        if view == "side":
            shx = hx = (0.0 if sd == "R" else -2.0)      # near limb on the spine, far limb slightly behind
        else:
            if view == "back": sg = -sg                   # subject's left is image-left from behind
            shx, hx = sg*wd["sho"]/2, sg*wd["hip"]/2
        J.update({"uarm"+sd:(shx,-Ls), "farm"+sd:(shx,-Ls+L["uarm"]), "hand"+sd:(shx,-Ls+L["uarm"]+L["farm"]),
                  "thigh"+sd:(hx,0), "shin"+sd:(hx,L["thigh"]), "foot"+sd:(hx,L["thigh"]+L["shin"])})
    return {k: np.array(v, float) for k, v in J.items()}

def _rot(v, deg):
    r = math.radians(deg); c, s_ = math.cos(r), math.sin(r)
    return np.array([v[0]*c-v[1]*s_, v[0]*s_+v[1]*c])      # y is down: positive = clockwise on screen

def _render_pose_v1(res, view, angles=None, solid=False, shade=True, crop=True, rest=True):
    """Forward kinematics: every sprite rotates about its pivot, children follow their parent.
    angles: {part: degrees relative to its parent, clockwise positive}.  Returns (image, meta)."""
    cpp, S = res.cpp, res.views[view]
    ang = dict(rest_angles(view)) if rest else {}
    if angles: ang.update({k: float(v) for k, v in angles.items()})
    J0 = _pivots0(res, view); th, pos = {}, {}
    for k in HIER:
        p = parent_key(k); a = float(ang.get(k, 0.0))
        if p is None: th[k] = a; pos[k] = J0[k].copy()
        else:
            pos[k] = pos[p] + _rot(J0[k]-J0[p], th[p]); th[k] = th[p]+a
    Sz = int(1.9*res.H/cpp); ox, oy = Sz/2.0, 0.55*Sz
    canvas = Image.new("RGBA", (Sz, Sz), (0,0,0,0))
    pivots, ends, d0 = {}, {}, {}
    for k in draw_order(view):
        sp = S[k]; rgba = solidify_sprite(sp.rgba, k) if solid else sp.rgba
        if shade and view == "side" and k.endswith("L"):
            rgba = rgba.copy(); rgba[..., :3] = (rgba[..., :3]*0.72).astype(np.uint8)
        im = Image.fromarray(rgba); w, h = im.size; pj = np.asarray(sp.pj, float)
        pad = int(math.ceil(math.hypot(max(pj[0], w-pj[0]), max(pj[1], h-pj[1]))))+2
        big = Image.new("RGBA", (2*pad, 2*pad), (0,0,0,0))
        bx, by = pad-int(round(pj[0])), pad-int(round(pj[1])); big.paste(im, (bx, by))
        c = (bx+pj[0], by+pj[1])
        if abs(th[k]) > 1e-6: big = big.rotate(-th[k], resample=Image.Resampling.NEAREST, center=c)
        P = np.array([ox+pos[k][0]/cpp, oy+pos[k][1]/cpp])
        canvas.paste(big, (int(round(P[0]-c[0])), int(round(P[1]-c[1]))), big)
        pivots[k] = P; vec = np.asarray(sp.pe, float)-pj
        ends[k] = P + _rot(vec, th[k]); d0[k] = math.degrees(math.atan2(vec[1], vec[0]))
    off = np.zeros(2)
    if crop:
        bb = canvas.getbbox() or (0, 0, Sz, Sz)
        bb = (max(0, bb[0]-2), max(0, bb[1]-2), min(Sz, bb[2]+2), min(Sz, bb[3]+2))
        canvas = canvas.crop(bb); off = np.array(bb[:2], float)
    r2 = lambda p: [round(float(p[0]-off[0]), 2), round(float(p[1]-off[1]), 2)]
    jp = {"pelvis": "pelvis", "neck": "neck", "head": "head"}
    for sd in "LR":
        jp.update({sd+"sho": "uarm"+sd, sd+"elb": "farm"+sd, sd+"wri": "hand"+sd,
                   sd+"hip": "thigh"+sd, sd+"knee": "shin"+sd, sd+"ank": "foot"+sd})
    meta = dict(view=view, note=f"{view} standing pose", pelvis_px=r2(pivots["pelvis"]), draw_order=draw_order(view),
                pivots_px={k: r2(p) for k, p in pivots.items()}, ends_px={k: r2(p) for k, p in ends.items()},
                th=th, d0=d0, origin=[float(off[0]), float(off[1])], size=[Sz, Sz],
                joints_px={n: r2(pivots[k]) for n, k in jp.items()},
                joints_cm={n: [round(float(J0[k][0]), 1), round(float(J0[k][1]), 1)] for n, k in jp.items()},
                rest_angles=rest_angles(view), parent={k: parent_key(k) for k in PART_KEYS},
                lengths_cm={n: round(float(v), 1) for n, v in res.lengths.items()},
                shoulder_width_cm=round(res.widths["sho"], 1), hip_width_cm=round(res.widths["hip"], 1),
                parts={k: dict(pivot_world=r2(pivots[k]), parent=parent_key(k), kind=S[k].kind) for k in PART_KEYS})
    return canvas, meta

def render_pose(res, view, angles=None, solid=False, shade=True, crop=True, rest=True, root=None):
    return RP.render_pose(globals(), res, view, angles, solid, shade, crop, rest, root)

def assemble(res, solid=False, view="front", shade=True):
    img, meta = render_pose(res, view, None, solid=solid, shade=shade, crop=True, rest=True)
    return img, meta

def compose_views(res, solid=False, bones=False):
    """All three assembled views side by side on one stage (pelvis-aligned)."""
    ims = {v: assemble(res, solid=solid, view=v) for v in VIEWS}
    gap, lab = 10, 12
    up = max(m["pelvis_px"][1] for _,m in ims.values())
    down = max(i.height-m["pelvis_px"][1] for i,m in ims.values())
    W = sum(i.width for i,_ in ims.values()) + gap*(len(VIEWS)+1)
    Hh = int(math.ceil(up+down)) + lab + gap
    stage = Image.new("RGBA", (W, Hh), (11,14,22,255))
    d = ImageDraw.Draw(stage); x = gap
    d.line([0,int(up+down)+2,W,int(up+down)+2], fill=(40,56,92))
    for v in VIEWS:
        i, m = ims[v]; y = int(round(up - m["pelvis_px"][1]))
        stage.alpha_composite(i, (x,y))
        if bones:
            for a,b in (("pelvis","neck"),("neck","head"),("Lsho","Lelb"),("Lelb","Lwri"),("Rsho","Relb"),
                        ("Relb","Rwri"),("Lhip","Lknee"),("Lknee","Lank"),("Rhip","Rknee"),("Rknee","Rank")):
                pa_, pb_ = m["joints_px"][a], m["joints_px"][b]
                d.line([x+pa_[0],y+pa_[1],x+pb_[0],y+pb_[1]], fill=(0,229,255))
            for p in m["joints_px"].values(): d.rectangle([x+p[0]-1,y+p[1]-1,x+p[0]+1,y+p[1]+1], fill=YELLOW)
        draw_text(d, x+(i.width-text_w(v.upper()))//2, int(up+down)+5, v.upper(), (120,140,180))
        x += i.width + gap
    return stage

# ----------------------------------------------------------------------------------------------
# Rig sheet: exploded parts for FRONT / SIDE / BACK + assembled figures + ruler + palette + legend
# ----------------------------------------------------------------------------------------------
PANEL_BG, PANEL_LINE = (14,18,28), (40,52,80)
CELL_BG, CELL_LINE, BONE_COL = (26,34,52), (60,76,110), (70,120,210)
KIND_COL = dict(own=CELL_LINE, locked=(255,200,61), borrowed=(255,150,40), synth=(166,107,255), placeholder=(255,77,77))
LABEL_H, CGAP = 7, 3

def _panel_layout(S, view):
    w = lambda k: S[k].rgba.shape[1]
    h = lambda k: S[k].rgba.shape[0]
    step = lambda k: h(k)+2+LABEL_H+CGAP
    pos, y = {}, 0
    for k in CENTER:
        pos[k] = [-w(k)/2.0, y]; y += step(k)
    body_w = max(w(k) for k in CENTER)
    y_chest = pos["chest"][1]; y_pelvis_end = pos["pelvis"][1]+step("pelvis")
    arm_w = max(w(b+s) for b in ("uarm","farm","hand") for s in "LR")
    leg_w = max(w(b+s) for b in ("thigh","shin","foot") for s in "LR")
    armx = max(body_w/2.0 + 5 + arm_w/2.0, 38)       # labels are ~27px wide: keep columns apart
    legx = max(leg_w/2.0+2, 17)
    right_side = "R" if view == "back" else "L"
    for s in "LR":
        sg = 1 if s == right_side else -1
        yy = y_chest
        for b in ("uarm","farm","hand"):
            k = b+s; pos[k] = [sg*armx-w(k)/2.0, yy]; yy += step(k)
        yy = y_pelvis_end
        for b in ("thigh","shin","foot"):
            k = b+s; pos[k] = [sg*legx-w(k)/2.0, yy]; yy += step(k)
    xs0, xs1, ys1 = [], [], []
    for k in list(pos):
        x, y = int(round(pos[k][0])), int(round(pos[k][1])); pos[k] = [x,y]
        lw = text_w(pretty(k)); cx = x+w(k)/2.0
        xs0 += [x-1, cx-lw/2.0]; xs1 += [x+w(k)+1, cx+lw/2.0]; ys1.append(y+h(k)+2+LABEL_H)
    minx = int(math.floor(min(xs0))); maxx = int(math.ceil(max(xs1)))
    PAD = 4
    for k in pos: pos[k] = (pos[k][0]-minx+PAD, pos[k][1]+PAD+1)
    return pos, (maxx-minx+2*PAD, max(ys1)+2*PAD+1)

def build_rig_sheet(res):
    cpp, H, S_all = res.cpp, res.H, res.views
    lay = {v:_panel_layout(S_all[v], v) for v in VIEWS}
    MARGIN, RW, TITLE_H, HEADER_H, PG = 6, 24, 22, 20, 6
    x = RW+MARGIN; px0 = {}
    for v in VIEWS: px0[v] = x; x += lay[v][1][0]+PG
    Wd = max(x-PG+MARGIN, 220)
    panel_h = max(lay[v][1][1] for v in VIEWS)+TITLE_H
    asm_h = max(res.asm[v].height for v in VIEWS)+16
    pal = list(res.palette)[:160]
    per_row = max(1, (Wd-2*MARGIN)//5); rows = int(math.ceil(len(pal)/per_row)) if pal else 0
    foot_h = 8+rows*5+(8 if rows else 0)+10
    Ht = HEADER_H+panel_h+4+asm_h+foot_h
    img = Image.new("RGBA", (Wd,Ht), (8,10,16,255)); atlas = Image.new("RGBA", (Wd,Ht), (0,0,0,0))
    d = ImageDraw.Draw(img)
    draw_text(d, MARGIN, 5, "HUMAN.RIG", (0,229,255), 2)
    sub = f"{getattr(res,'name','')}  {getattr(res,'model_tag','')}  {H:.0f}CM  {cpp:g}CM/PX"
    draw_text(d, Wd-MARGIN-text_w(sub), 8, sub, (130,150,190))
    d.line([0,HEADER_H-3,Wd,HEADER_H-3], fill=PANEL_LINE)
    views_meta = {}
    for v in VIEWS:
        S = S_all[v]; pos, (pw,ph) = lay[v]; ox, oy = px0[v], HEADER_H
        d.rectangle([ox,oy,ox+pw-1,oy+panel_h-1], fill=PANEL_BG, outline=PANEL_LINE)
        n = (getattr(res,"counts",None) or {}).get(v)
        draw_text(d, ox+4, oy+3, v.upper(), (230,237,247), 2)
        if n is not None:
            t = f"{n} FRAME" + ("" if n == 1 else "S"); col = (130,150,190) if n else (255,77,77)
            draw_text(d, ox+4, oy+15, t, col)
        base_y = oy+TITLE_H
        R = {k:(ox+pos[k][0], base_y+pos[k][1], S[k].rgba.shape[1], S[k].rgba.shape[0]) for k in PART_KEYS}
        # bones first (under the cells)
        def line(p, q): d.line([p[0],p[1],q[0],q[1]], fill=BONE_COL)
        for a,b in (("head","neck"),("neck","chest"),("chest","abdomen"),("abdomen","pelvis")):
            ax_,ay,aw,ah = R[a]; bx_,by,bw,bh = R[b]
            line((ax_+aw//2,ay+ah+2+LABEL_H),(bx_+bw//2,by-2))
        for s in "LR":
            ax_,ay,aw,ah = R["pelvis"]; bx_,by,bw,bh = R["thigh"+s]
            ym = by-3; cx0, cx1 = ax_+aw//2, bx_+bw//2
            line((cx0,ay+ah+2+LABEL_H),(cx0,ym)); line((cx0,ym),(cx1,ym)); line((cx1,ym),(cx1,by-2))
            ax_,ay,aw,ah = R["chest"]; bx_,by,bw,bh = R["uarm"+s]
            if bx_ > ax_: line((ax_+aw+1,by+4),(bx_-2,by+4))
            else:         line((bx_+bw+1,by+4),(ax_-2,by+4))
            for a,b in (("uarm","farm"),("farm","hand"),("thigh","shin"),("shin","foot")):
                ax_,ay,aw,ah = R[a+s]; bx_,by,bw,bh = R[b+s]
                line((ax_+aw//2,ay+ah+2+LABEL_H),(bx_+bw//2,by-2))
        vm = {}
        for k in PART_KEYS:
            sp = S[k]; X,Y,w_,h_ = R[k]
            d.rectangle([X-1,Y-1,X+w_,Y+h_], fill=CELL_BG, outline=KIND_COL.get(sp.kind,CELL_LINE))
            lc = KIND_COL[sp.kind] if sp.kind != "own" else (130,150,190)
            if sp.kind == "own": lc = (110,130,170)
            lab = pretty(k); draw_text(d, X+w_//2-text_w(lab)//2, Y+h_+3, lab, lc)
        for k in PART_KEYS:
            sp = S[k]; X,Y,w_,h_ = R[k]
            spr = Image.fromarray(sp.rgba)
            img.alpha_composite(spr,(X,Y)); atlas.alpha_composite(spr,(X,Y))
        for k in PART_KEYS:
            sp = S[k]; X,Y,w_,h_ = R[k]
            for p,col in ((sp.pj,(0,229,255)),(sp.pe,(255,61,129))):
                qx, qy = int(round(X+p[0])), int(round(Y+p[1]))
                if 0 <= qx-X < w_ and 0 <= qy-Y < h_: d.point((qx,qy), fill=col)
            vm[k] = dict(rect=[X,Y,w_,h_], pivot=[round(float(sp.pj[0]),2),round(float(sp.pj[1]),2)],
                end=[round(float(sp.pe[0]),2),round(float(sp.pe[1]),2)],
                pivot_sheet=[round(X+float(sp.pj[0]),2),round(Y+float(sp.pj[1]),2)], pivot_a=[round(float(sp.pa[0]),2),round(float(sp.pa[1]),2)],
                pivot_b=[round(float(sp.pb[0]),2),round(float(sp.pb[1]),2)],
                pivot_a_sheet=[round(X+float(sp.pa[0]),2),round(Y+float(sp.pa[1]),2)],
                length_cm=sp.length_cm, parent=parent_key(k), side=side_of(k) or None,
                kind=sp.kind, source=sp.frame, flip=bool(getattr(sp,"flip",False)),
                twin=bool(getattr(sp,"twin",False)), borrowed_from=getattr(sp,"borrowed","") or None,
                score=round(float(sp.score),3))
        views_meta[v] = dict(panel=[ox,oy,pw,panel_h], parts=vm)
    # assembled row + ruler
    y0 = HEADER_H+panel_h+4; ground = y0+8+max(res.asm[v].height for v in VIEWS)
    for v in VIEWS:
        a = res.asm[v]; pw = lay[v][1][0]; ox = px0[v]
        d.rectangle([ox,y0,ox+pw-1,y0+asm_h-1], fill=(10,13,20), outline=PANEL_LINE)
        img.alpha_composite(a, (ox+(pw-a.width)//2, ground-a.height))
        draw_text(d, ox+4, y0+asm_h-8, v.upper()+" ASSEMBLED", (90,108,150))
    gy = ground-2
    d.line([RW-3,gy,RW-3,gy-int(round(H/cpp))], fill=(60,76,110))
    for cm in range(0, int(H)+1, 10):
        yy = gy-int(round(cm/cpp)); big = cm % 50 == 0
        d.line([RW-3-(5 if big else 2),yy,RW-3,yy], fill=(130,150,190))
        if big and cm: draw_text(d, RW-7-text_w(str(cm)), yy-2, str(cm), (130,150,190))
    hy = gy-int(round(H/cpp)); d.line([RW-9,hy,RW-3,hy], fill=RED)
    # palette + legend
    fy = HEADER_H+panel_h+4+asm_h+5
    for i,c in enumerate(pal):
        sx, sy = MARGIN+(i%per_row)*5, fy+(i//per_row)*5
        d.rectangle([sx,sy,sx+3,sy+3], fill=tuple(c))
    ly = fy+rows*5+(6 if rows else 0); lx = MARGIN
    for name,key in (("OWN CUT","own"),("LOCKED","locked"),("SYNTH","synth"),("MISSING","placeholder")):
        d.rectangle([lx,ly,lx+4,ly+4], outline=KIND_COL[key], fill=CELL_BG)
        lx = draw_text(d, lx+8, ly, name, (110,130,170))+6
    meta = dict(layout="rig3", name=getattr(res,"name",""), cm_per_px=cpp, subject_height_cm=H,
                sheet_size=[Wd,Ht],
                note="sprites face RIGHT in the side view; side view R = near limb, L = far limb; "
                     "L/R are the subject's own left/right",
                palette=["#%02x%02x%02x"%tuple(c) for c in res.palette],
                lengths_cm={n:round(float(v),2) for n,v in res.lengths.items()},
                widths_cm={n:round(float(v),2) for n,v in res.widths.items()}, views=views_meta)
    return img, atlas, meta

def render_outputs(res):
    res.asm, res.asm_meta = {}, {}
    for v in VIEWS: res.asm[v], res.asm_meta[v] = assemble(res, view=v)
    res.sheet, res.atlas, res.meta = build_rig_sheet(res)
    res.combo = compose_views(res)
    res.assembled, res.assembled_side, res.assembled_back = res.asm["front"], res.asm["side"], res.asm["back"]

# ----------------------------------------------------------------------------------------------
# Edits (per frame, persisted in the loadout)
# ----------------------------------------------------------------------------------------------
def _safe(tag): return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)

def save_edits(edits_dir, fd):
    edits_dir = Path(edits_dir); edits_dir.mkdir(parents=True, exist_ok=True)
    base = _safe(fd.name)
    with open(edits_dir/f"{base}.json","w") as f:
        json.dump(dict(view=fd.view_override, flip=fd.flip, enabled=fd.enabled, jover=fd.jover), f)
    npz = edits_dir/f"{base}.npz"
    if (fd.manual > -2).any() or fd.extm.any(): np.savez_compressed(npz, m=fd.manual, x=fd.extm)
    elif npz.exists(): npz.unlink()

def load_edits(edits_dir, tag):
    if not edits_dir: return {}
    base = Path(edits_dir)/_safe(tag)
    out = {}
    try:
        if base.with_suffix(".json").exists():
            with open(base.with_suffix(".json")) as f: out = json.load(f)
        if base.with_suffix(".npz").exists():
            z = np.load(base.with_suffix(".npz")); out["manual"] = z["m"]
            if "x" in z.files: out["extm"] = z["x"]
    except Exception:
        out = {}
    return out

# ----------------------------------------------------------------------------------------------
# Loadout
# ----------------------------------------------------------------------------------------------
class Loadout:
    def __init__(self, name):
        self.name = name.strip().replace(" ","_") or "subject"
        self.path = LOADOUTS_DIR / self.name
        self.meta = {"name":self.name,"height_cm":172.0,"cm_per_px":CM_PER_PX,"sources":[],"views":{},
                     "palette":[],"lengths":{},"widths":{},"locks":{},"rejects":{},"settings":{},
                     "mask":{},"scale":{},
                     "created":None,"updated":None}
        self.chosen = {v:{} for v in VIEWS}
    @property
    def edits_dir(self): return self.path/"edits"
    def ensure_dirs(self):
        for v in VIEWS: (self.path/"parts"/v).mkdir(parents=True, exist_ok=True)
        (self.path/"sources").mkdir(parents=True, exist_ok=True)
        self.edits_dir.mkdir(parents=True, exist_ok=True)
    def write_meta(self):
        self.ensure_dirs()
        now = datetime.datetime.now().isoformat(timespec="seconds")
        if not self.meta.get("created"): self.meta["created"] = now
        self.meta["updated"] = now
        with open(self.path/"meta.json","w") as f: json.dump(self.meta, f, indent=2)
    def save(self, res=None):
        self.ensure_dirs()
        if res is not None:
            res.name = self.name
            for v in VIEWS:
                d = self.path/"parts"/v
                for p in d.glob("*.png"): p.unlink()
                self.meta["views"][v] = {}
                for k,sp in res.chosen.get(v,{}).items():
                    Image.fromarray(sp.rgba).save(d/f"{k}.png")
                    self.meta["views"][v][k] = dict(pivot_a=[round(float(x),2) for x in sp.pa],
                        pivot_b=[round(float(x),2) for x in sp.pb], length_cm=sp.length_cm,
                        pivot_j=[round(float(x),2) for x in sp.pj], pivot_e=[round(float(x),2) for x in sp.pe],
                        anchors={ck: [round(float(x),2) for x in p] for ck, p in (getattr(sp, "anc", None) or {}).items()},
                        score=round(float(sp.score),4), source=sp.frame, cid=getattr(sp,"cid",""),
                        flip=bool(getattr(sp,"flip",False)), twin=bool(getattr(sp,"twin",False)))
            render_outputs(res)             # refresh the title (name) on the sheet
            res.sheet.save(self.path/"rig.png"); res.atlas.save(self.path/"atlas.png")
            res.sheet.resize((res.sheet.width*4,res.sheet.height*4), Image.Resampling.NEAREST).save(self.path/"rig_x4.png")
            for v in VIEWS: res.asm[v].save(self.path/f"assembled_{v}.png")
            res.combo.save(self.path/"assembled_all.png")
            self.meta["palette"] = res.meta["palette"]
            self.meta["align"] = getattr(res, "align", None) or {}
            self.meta["props"] = {k: round(float(v), 4) for k, v in (getattr(res, "props", None) or {}).items() if v is not None and k in ("waist", "ribs", "neck", "crotch")}
            self.meta["lengths"] = {k:round(float(x),3) for k,x in res.lengths.items()}
            self.meta["widths"] = {k:round(float(x),3) for k,x in res.widths.items()}
            self.meta["mask"] = {k: bool(v) for k, v in RB.normalize_mask(getattr(res, "mask", None)).items()}
            self.meta["scale"] = {k: round(float(v), 3) for k, v in RB.normalize_scale(getattr(res, "scale", None)).items()}
            rig = dict(name=self.name, height_cm=self.meta["height_cm"], cm_per_px=self.meta["cm_per_px"],
                       sheet=res.meta, assembled=res.asm_meta,
                       hierarchy={k: parent_key(k) for k in PART_KEYS}, alignment=getattr(res, "align", None) or {}, rotation="degrees, clockwise positive, about each part's pivot")
            with open(self.path/"rig.json","w") as f: json.dump(rig, f, indent=2)
            if getattr(res, "cands_dirty", False): self.write_cands(res); res.cands_dirty = False
        self.write_meta()
    def write_cands(self, res):
        arrays, index = {}, {}
        for v in VIEWS:
            index[v] = {}
            for k in PART_KEYS:
                lst = []
                for i,c in enumerate(res.cands[v][k]):
                    name = f"{v}__{k}__{i}"; arrays[name] = c["img"]
                    lst.append(dict(file=name, cid=c["cid"], score=float(c["score"]), fname=c["fname"],
                        src_key=c["src_key"], flip=bool(c["flip"]), twin=bool(c["twin"]),
                        pa=[float(x) for x in c["pa"]], pb=[float(x) for x in c["pb"]],
                        pj=[float(x) for x in c["pj"]], pe=[float(x) for x in c["pe"]],
                        anc={ck: [float(x) for x in p] for ck, p in (c.get("anc") or {}).items()}))
                index[v][k] = lst
        np.savez_compressed(self.path/"cands.npz", **arrays)
        with open(self.path/"cands.json","w") as f: json.dump(index, f)
    def load_cands(self):
        cj, cz = self.path/"cands.json", self.path/"cands.npz"
        cands = {v:{k:[] for k in PART_KEYS} for v in VIEWS}
        if not (cj.exists() and cz.exists()): return cands
        try:
            with open(cj) as f: index = json.load(f)
            z = np.load(cz)
            for v in VIEWS:
                for k in PART_KEYS:
                    bad = set(self.meta.get("rejects",{}).get(v,{}).get(k,[]))
                    for e in index.get(v,{}).get(k,[]):
                        if e["cid"] in bad: continue
                        cands[v][k].append(dict(img=z[e["file"]], score=e["score"], fname=e["fname"],
                            src_key=e["src_key"], flip=e["flip"], twin=e["twin"], view=v, cid=e["cid"],
                            pa=np.array(e["pa"]), pb=np.array(e["pb"]),
                            pj=np.array(e.get("pj", e["pa"])), pe=np.array(e.get("pe", e["pb"])),
                            anc={ck: np.array(p, float) for ck, p in (e.get("anc") or {}).items()}))
        except Exception:
            pass
        return cands
    def load(self):
        mp = self.path/"meta.json"
        if not mp.exists(): return False
        with open(mp) as f: self.meta.update(json.load(f))
        for k0,dv in (("views",{}),("locks",{}),("rejects",{}),("settings",{}),("sources",[])):
            self.meta.setdefault(k0, dv)
        self.chosen = {v:{} for v in VIEWS}
        for v in VIEWS:
            d = self.path/"parts"/v
            if not d.exists() and v == "front" and (self.path/"parts"/"standing").exists():
                d = self.path/"parts"/"standing"           # loadouts made by the old version
            if not d.exists(): continue
            for p in d.glob("*.png"):
                k = p.stem
                if k not in PART_KEYS: continue
                arr = np.array(Image.open(p).convert("RGBA"))
                info = self.meta["views"].get(v,{}).get(k) or (self.meta.get("parts",{}).get(k,{}) if v=="front" else {})
                sp = Sprite(); sp.key, sp.rgba = k, arr
                sp.pa = np.array(info.get("pivot_a",[arr.shape[1]/2,0]),float)
                sp.pb = np.array(info.get("pivot_b",[arr.shape[1]/2,arr.shape[0]]),float)
                sp.length_cm = info.get("length_cm",0); sp.score = float(info.get("score",0.5))
                sp.frame = info.get("source",""); sp.cid = info.get("cid","")
                sp.flip, sp.twin = bool(info.get("flip",False)), bool(info.get("twin",False))
                sp.pj = np.array(info.get("pivot_j", info.get("pivot_a", [arr.shape[1]/2,0])), float)
                sp.pe = np.array(info.get("pivot_e", info.get("pivot_b", [arr.shape[1]/2,arr.shape[0]])), float)
                sp.mirrored = sp.flip or sp.twin; sp.nused = 1; sp.borrowed = ""; sp.anc = {ck: np.array(p, float) for ck, p in (info.get("anchors") or {}).items()}
                lock = self.meta["locks"].get(v,{}).get(k)
                sp.kind = "locked" if lock else "own"
                self.chosen[v][k] = sp
        return True
    def to_result(self):
        """Rebuild a full result (views, sheet, assembled, candidates) from what is on disk."""
        H = float(self.meta.get("height_cm",172.0))
        res = Sprite(); res.name = self.name
        res.chosen, res.H, res.cpp = self.chosen, H, float(self.meta.get("cm_per_px",CM_PER_PX))
        res.lengths = {n:float(self.meta.get("lengths",{}).get(n, p*H)) for n,p in PRIOR.items()}
        wd = self.meta.get("widths",{}) or {}
        res.widths = dict(sho=float(wd.get("sho",0.23*H)), hip=float(wd.get("hip",0.19*H)))
        res.palette = [tuple(int(c[i:i+2],16) for i in (1,3,5)) for c in self.meta.get("palette",[])]
        res.issues, res.counts, res.cands_dirty = [], None, False
        res.synth = bool((self.meta.get("settings") or {}).get("synth", True))
        res.smooth = int((self.meta.get("settings") or {}).get("smooth", 2))
        res.props = dict(DEF_PROPS); res.props.update({k: float(v) for k, v in (self.meta.get("props") or {}).items()})
        res.cands = self.load_cands()
        res.align = self.meta.get("align", {}) or {}
        res.mask = RB.normalize_mask(self.meta.get("mask"))
        res.scale = RB.normalize_scale(self.meta.get("scale"))
        finalize_views(res); render_outputs(res)
        return res
    def add_source(self, src_path):
        self.ensure_dirs()
        src = Path(src_path); dest = self.path/"sources"/src.name
        if not dest.exists():
            try: os.link(src, dest)
            except Exception: shutil.copy2(src, dest)
        if str(dest) not in self.meta["sources"]: self.meta["sources"].append(str(dest))
        self.write_meta()
    def remove_source(self, p):
        if p in self.meta["sources"]: self.meta["sources"].remove(p)
        self.write_meta()
    def list_sources(self): return list(self.meta.get("sources",[]))

def list_loadouts():
    if not LOADOUTS_DIR.exists(): return []
    return sorted(p.name for p in LOADOUTS_DIR.iterdir() if p.is_dir() and (p/"meta.json").exists())

# ----------------------------------------------------------------------------------------------
# Source loading + analysis
# ----------------------------------------------------------------------------------------------
def load_images(path, maxgif=MAX_GIF_FRAMES):
    im = Image.open(path); n = getattr(im,"n_frames",1); out = []
    if n > 1:
        for i in sorted(set(np.linspace(0,n-1,min(n,maxgif)).round().astype(int).tolist())):
            im.seek(int(i)); out.append((int(i), im.convert("RGBA")))
    else: out.append((0, im.convert("RGBA")))
    return out

def ensure_alpha(img, cache):
    a = np.asarray(img)[...,3]
    if (a<200).mean() > .02: return img
    if cache.get("rb") is None:
        try:
            from rembg import remove, new_session
            cache["rb"] = (remove, new_session("u2net_human_seg"))
        except Exception: cache["rb"] = False
    if cache["rb"]:
        rm, ses = cache["rb"]; big = max(img.size)
        sm = img if big<=1024 else img.resize((round(img.width*1024/big),round(img.height*1024/big)),Image.Resampling.LANCZOS)
        al = rm(sm,session=ses).getchannel("A").resize(img.size,Image.Resampling.BILINEAR)
        out = img.copy(); out.putalpha(al); return out
    rgb = np.asarray(img.convert("RGB")).astype(int)
    border = np.concatenate([rgb[0],rgb[-1],rgb[:,0],rgb[:,-1]])
    key = np.median(border,0); diff = np.abs(rgb-key).sum(2)
    out = img.copy(); out.putalpha(Image.fromarray(np.where(diff>60,255,0).astype(np.uint8)))
    return out

def analyse_sources(paths, H, det, log=lambda t:None, progress=lambda i,n:None, edits_dir=None, learn=True):
    frames, failed, cache = [], [], {}
    for pi,p in enumerate(paths):
        progress(pi,len(paths)); base = os.path.splitext(os.path.basename(p))[0]
        try: imgs = load_images(p)
        except Exception as e:
            log(f"{base}: cannot read ({e})"); failed.append(dict(tag=base,reason=str(e),im=None,il=None)); continue
        for i,im in imgs:
            tag = base if len(imgs)==1 else f"{base}#{i}"
            try:
                big = max(im.size)
                if big > 1280: im = im.resize((round(im.width*1280/big),round(im.height*1280/big)),Image.Resampling.LANCZOS)
                im = ensure_alpha(im, cache)
                bg = Image.new("RGBA", im.size, (127,127,127,255)); bg.alpha_composite(im)
                r = det(np.asarray(bg.convert("RGB")), np.asarray(im)[...,3])
                if r is None:
                    log(f"{tag}: no person found"); failed.append(dict(tag=tag,reason="no person found",im=im,il=default_landmarks(im))); continue
                fd = build_frame(tag, im, r[0], H, load_edits(edits_dir, tag))
                if fd is None:
                    log(f"{tag}: body not visible enough"); failed.append(dict(tag=tag,reason="body not visible enough",im=im,il=default_landmarks(im))); continue
                fd.src = (tag, im, r[0])
                n_ed = int((fd.manual>-2).sum()) + len(fd.jover)
                log(f"{tag}: {fd.s_orig:.2f} px/cm, {fd.facing.upper()}{' (forced)' if fd.view_override else ''}"
                    f"{', ' + str(n_ed) + ' saved edits' if n_ed else ''}")
                frames.append(fd)
            except Exception as e:
                log(f"{tag}: failed ({e!r})"); failed.append(dict(tag=tag,reason=repr(e),im=None,il=None))
    if learn and frames:
        try: learn_colors(frames, log)
        except Exception as e: log(f"colour learning skipped ({e!r})")
    progress(len(paths),len(paths)); return frames, failed

# Detector ---------------------------------------------------------------------------------------
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_NAME = "pose_landmarker_heavy.task"
DATA_DIR = os.path.join(os.path.expanduser("~"), ".humanrig")
def ensure_model():
    for d in (DATA_DIR, os.path.join(os.path.expanduser("~"),".pixelscale_master")):
        p = os.path.join(d, MODEL_NAME)
        if os.path.exists(p) and os.path.getsize(p) > 1_000_000: return p
    import urllib.request
    os.makedirs(DATA_DIR, exist_ok=True); p = os.path.join(DATA_DIR, MODEL_NAME)
    urllib.request.urlretrieve(MODEL_URL, p+".part"); os.replace(p+".part", p); return p

class Detector:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=ensure_model()), num_poses=3))
    def __call__(self, rgb, alpha=None):
        r = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        if not r.pose_landmarks: return None
        k = int(np.argmax([max(p.y for p in q)-min(p.y for p in q) for q in r.pose_landmarks]))
        return np.array([[p.x,p.y,p.visibility] for p in r.pose_landmarks[k]]), None

class FallbackDetector:
    """Used when MediaPipe is unavailable: a standing template that you then fix with the joint tool."""
    def __call__(self, rgb, alpha=None):
        a = alpha if alpha is not None else np.full(rgb.shape[:2], 255, np.uint8)
        return default_landmarks(Image.fromarray(np.dstack([rgb, a]))), None

def make_detector(owner=None): return Detector()

# ----------------------------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------------------------
_GUI_CACHE = {}

def open_rig(app):
    w = getattr(app, "rig_win", None)
    if w is not None and w.winfo_exists():
        return w.lift()
    app.rig_win = _build_gui_class()(app)
    return app.rig_win

def _hex(c): return "#%02x%02x%02x" % tuple(int(x) for x in c)
def _dark(c, f=0.35): return _hex(tuple(int(x*f) for x in c))

def _build_gui_class(standalone=False):
    standalone = bool(standalone)
    if standalone in _GUI_CACHE: return _GUI_CACHE[standalone]
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    import tkinter.font as tkfont
    from PIL import ImageTk

    BG0,BG1,BG2,BG3 = "#080b12","#0e131d","#161d2b","#212b40"
    LINE,LINE2 = "#26324a","#38496f"
    TEXT,MUTED,DIM = "#e6edf7","#8593ad","#55617c"
    ACC,ACC_HI,PINK,GOLD,GOOD,WARN,BAD,PURP,BLUE2 = ("#00e5ff","#5cf0ff","#ff3d81","#ffc83d","#3ddc84",
                                                      "#ff9a28","#ff4d4d","#a66bff","#4da3ff")
    VIEW_COL = dict(front=ACC, side=WARN, back=PINK)
    STAGE = ("ANALYSE","EDIT","CUT","PICK","EXPORT")
    LUT = np.array([(190,190,205)] + [REGION_COL[i] for i in range(len(PART_KEYS))], np.float32)
    M = 12                                             # canvas margin around the editor image

    def rarity(sc):
        if sc >= 0.85: return "LEGENDARY", GOLD
        if sc >= 0.65: return "EPIC", PURP
        if sc >= 0.40: return "RARE", BLUE2
        if sc >= 0.20: return "UNCOMMON", GOOD
        return "COMMON", MUTED

    POSE_PRESETS = dict(
        front={"REST": {}, "T-POSE": dict(uarmR=90, uarmL=-90), "ARMS UP": dict(uarmR=165, uarmL=-165),
               "LEG RAISE": dict(thighR=90, shinR=-90), "WAVE": dict(uarmR=135, farmR=45, handR=10),
               "SIT": {}, "WALK A": dict(thighR=8, thighL=-6, uarmR=10, uarmL=-10),
               "WALK B": dict(thighL=-8, thighR=6, uarmR=10, uarmL=-10), "KICK": dict(thighR=40, shinR=-30, uarmR=20, uarmL=-20)},
        side={"REST": {}, "T-POSE": dict(uarmR=-90, uarmL=-90), "ARMS UP": dict(uarmR=-165, uarmL=-165),
              "LEG RAISE": dict(thighR=-90, shinR=90), "WAVE": dict(uarmR=-150, farmR=-30),
              "SIT": dict(thighR=-90, thighL=-90, shinR=90, shinL=90),
              "WALK A": dict(thighR=-28, shinR=12, footR=-5, thighL=22, shinL=28, footL=8, uarmR=22, farmR=-18, uarmL=-22, farmL=-28),
              "WALK B": dict(thighL=-28, shinL=12, footL=-5, thighR=22, shinR=28, footR=8, uarmL=22, farmL=-18, uarmR=-22, farmR=-28),
              "KICK": dict(thighR=-80, shinR=20, uarmL=30, uarmR=-30)})

    class RigWindow(tk.Tk if standalone else tk.Toplevel):
        # ------------------------------------------------------------------ setup
        def __init__(self, app=None):
            if standalone: tk.Tk.__init__(self)
            else: tk.Toplevel.__init__(self, app)
            self.app = app
            self.title("HUMAN.RIG")
            self.configure(bg=BG0)
            self.geometry("1560x940"); self.minsize(1280, 780)
            fams = set(tkfont.families(self))
            ui = next((f for f in ("Segoe UI","SF Pro Text","Helvetica Neue","Ubuntu","DejaVu Sans","Arial") if f in fams), "TkDefaultFont")
            mono = next((f for f in ("Consolas","Menlo","DejaVu Sans Mono","Courier New") if f in fams), "TkFixedFont")
            self.F = dict(ui=(ui,10), b=(ui,10,"bold"), h=(ui,11,"bold"), s=(ui,9), mono=(mono,9),
                          big=(ui,14,"bold"), card=(ui,8,"bold"), cardb=(ui,9,"bold"))
            self.q = queue.Queue()
            self.sources, self.frames, self.failed = [], [], []
            self.res = None; self.det = None; self.busy = False
            self.loadout = None; self.cur = None
            self.locks, self.rejects = {}, {}
            self.dirty = False; self.exported = False
            self.photos = {}
            self.ed_box = None; self.ed_base = None; self.ed_bg = None; self.ed_zoom = 1.0
            self._stroke = None; self._grab = None; self._pan = False; self._ed_pending = False
            self.guide_view, self.guide_key = "front", PART_KEYS[0]
            self.card_photos, self.inv_photos, self.ctx_photo = [], [], None
            self._persist_job = None; self.fit = dict(sheet=True, asm=True)
            self.z = dict(sheet=3.0, asm=3.0)

            self.v_name = tk.StringVar(value="subject")
            self.v_h = tk.StringVar(value="172")
            self.v_cpp = tk.StringVar(value=str(CM_PER_PX))
            self.v_mirror = tk.BooleanVar(value=True); self.v_fill = tk.BooleanVar(value=True)
            self.v_blend = tk.BooleanVar(value=True)
            self.v_cols = tk.IntVar(value=32); self.v_athr = tk.IntVar(value=110)
            self.v_tool = tk.StringVar(value="brush"); self.v_part = tk.StringVar(value="chest")
            self.v_brush = tk.DoubleVar(value=6.0); self.v_tint = tk.IntVar(value=55)
            self.v_regions = tk.BooleanVar(value=True); self.v_borders = tk.BooleanVar(value=True)
            self.v_skel = tk.BooleanVar(value=False); self.v_only = tk.BooleanVar(value=True)
            self.v_enabled = tk.BooleanVar(value=True); self.v_autoadv = tk.BooleanVar(value=True)
            self.v_solid = tk.BooleanVar(value=False); self.v_bones = tk.BooleanVar(value=False)
            self.v_overlap = tk.BooleanVar(value=True); self.v_ovpct = tk.IntVar(value=100)
            self.v_synth = tk.BooleanVar(value=True)
            self.v_smooth = tk.IntVar(value=2); self.v_learn = tk.BooleanVar(value=True)
            self.v_handles = tk.BooleanVar(value=True); self.v_ppiv = tk.BooleanVar(value=False)
            self.v_org = tk.StringVar(value="human"); self.v_sex = tk.StringVar(value="male")
            self.v_usemodel = tk.BooleanVar(value=True); self.v_trainmodel = tk.BooleanVar(value=True)
            self.model = None; self._psel = None; self._align = {}
            self.body_mask = RB.default_mask()
            self.body_scale = RB.default_scale()
            self.v_body_group = {g: tk.BooleanVar(value=True) for g in RB.LIMB_GROUPS}
            self.v_body_scale = {b: tk.DoubleVar(value=1.0) for b in RB.SCALE_BASES}
            self.v_fillhidden = tk.BooleanVar(value=True); self.last_saved = None
            self.v_psolid = tk.BooleanVar(value=False)
            self.pose_view = "front"; self.pose_ang = {k: 0.0 for k in PART_KEYS}; self.pose_vars = {}
            self.pose_lbl = {}; self.pose_meta = None; self.pose_zoom = 3; self.pose_fit_on = True
            self._pgrab = None; self._pose_pending = False; self._pose_lock = False; self.pose_photo = None

            self.style_ttk()
            self.build_ui()
            self.refresh_roster()
            self.set_tool("brush"); self.set_part("chest"); self.pose_reset("front"); self.show_page("editor")
            self.update_stage()
            self.refresh_model()
            self.bind("<KeyPress>", self.on_key)
            self.protocol("WM_DELETE_WINDOW", self.destroy)
            self.after(50, self._tick)
            self.log("Welcome. Add source images, then ANALYSE. Every edit is saved with the loadout.", "info")

        def update_model_chip(self):
            m = self.model
            if m is None: return
            col = {"male": BLUE2, "female": PINK}.get(m.sex, GOOD)
            on = self.v_usemodel.get(); px = sum(m.px.values())
            self.chip_model.config(text=f"BODY MODEL   {m.organism.upper()} · {m.sex.upper()}   ({px:,} px learned){'' if on else '   OFF'}",
                                   fg=col if on else DIM, highlightbackground=col if on else LINE)
            self.title(f"HUMAN.RIG  -  {m.organism}/{m.sex} model" + (f"  -  {self.loadout.name}" if self.loadout else ""))
        def coach_state(self):
            lo = self.loadout; locks = sum(len(d) for d in self.locks.values())
            if lo is None: return ("1  START", "Create a character. Click NEW (top-left), type a name, then add your images.", "NEW", self.new_loadout)
            if not self.sources: return ("1  ADD IMAGES", "Add your source images (cut-out, full body). Include FRONT, SIDE and BACK shots if you can.", "+ FILES", self.add_files)
            if not self.frames and self.res is None: return ("2  ANALYSE", "Click ANALYSE. The program finds the skeleton and colours every body part in every image.", "ANALYSE", self.analyse)
            if self.dirty: return ("3  CUT", "You changed labels or joints. They are already saved with the character, but the rig does not use them until you click CUT.", "CUT", self.cut)
            if self.res is None: return ("2  EDIT", "Each colour is a body part. Pick a part chip on the right, then paint wrong areas (arm, torso, legs). Check every frame, then click CUT.", "CUT", self.cut)
            if not self.exported: return ("4  CHECK & SAVE", "Your rig is cut and AUTOSAVED. Review RIG SHEET / ASSEMBLED / POSE, press RIG CHECK for weak parts, then SAVE & EXPORT for the final files.", "RIG CHECK", self.run_check)
            return ("✓  DONE", f"Exported to {lo.path.resolve()}. Keep improving any time: fix labels, CUT again - the model keeps learning.", "OPEN FOLDER", self.open_folder)
        def update_coach(self):
            try:
                step, txt, btn, cmd = self.coach_state()
                self.coach_step.config(text=step); self.coach_txt.config(text=txt)
                self.coach_btn.config(text=btn); self.coach_btn.unbind("<Button-1>"); self.coach_btn.bind("<Button-1>", lambda e: cmd())
            except Exception: pass
        def refresh_save_chip(self):
            t = self.last_saved or "-"
            if self.res is None and not self.frames: txt, col = "nothing to save yet", DIM
            elif self.dirty: txt, col = "✎ edits saved - NOT in the rig yet: press CUT", WARN
            elif self.exported: txt, col = f"✓ EXPORTED {t}", GOOD
            else: txt, col = f"✓ autosaved {t}  (SAVE & EXPORT = final)", GOOD
            self.save_chip.config(text=txt, fg=col)
        def snapshot(self):
            import shutil as _sh
            if self.loadout is None: return self.toast("Nothing to snapshot yet.", "warn")
            lo = self.ensure_loadout(); dst = lo.path.parent / "_snapshots" / f"{lo.name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                _sh.copytree(lo.path, dst, ignore=_sh.ignore_patterns("sources", "cands.npz"))
                self.toast(f"Snapshot saved: {dst}", "ok")
            except Exception as ex: self.toast(f"Snapshot failed: {ex}", "err")
        def run_check(self):
            if self.res is None: return self.toast("CUT a rig first, then run the check.", "warn")
            items = RT.rig_check(self.res)
            w = tk.Toplevel(self); w.title("RIG CHECK"); w.configure(bg=BG0); w.geometry("760x460")
            head = "All good - nothing weak found." if not items else f"{len(items)} thing(s) worth a look.  Double-click one to open it in PICK PARTS."
            tk.Label(w, text=head, bg=BG0, fg=GOOD if not items else WARN, font=self.F["h"], anchor="w").pack(fill="x", padx=12, pady=(10,4))
            lb = self.mk_list(w, 18); lb.pack(fill="both", expand=True, padx=12, pady=(0,12))
            for i, it in enumerate(items):
                lab = f"{it['view'].upper():5s} {pretty(it['key']) if it['key'] else '-':10s} {it['msg']}"
                lb.insert("end", lab); lb.itemconfig(i, fg=(BAD, WARN, MUTED)[it["sev"]])
            def go(_e=None):
                s = lb.curselection()
                if not s or not items[s[0]]["key"]: return
                self.guide_view, self.guide_key = items[s[0]]["view"], items[s[0]]["key"]; self.show_page("guide"); w.destroy()
            lb.bind("<Double-Button-1>", go)
        def refresh_model(self, *_):
            try:
                self.model = REGISTRY.get(self.v_org.get(), self.v_sex.get())
                set_active_model(self.model if self.v_usemodel.get() else None)
                self.update_model_chip(); self.model_lbl.config(text=self.model.summary() + ("" if self.v_usemodel.get() else "\n(model OFF)"))
            except Exception as ex: self.log(f"model load failed: {ex!r}", "err")
        def model_export(self):
            if self.model is None: return
            p = filedialog.asksaveasfilename(parent=self, defaultextension=".zip", initialfile=f"humanrig_model_{self.model.organism}_{self.model.sex}.zip")
            if p: self.toast(f"Model exported to {self.model.export_zip(p)}", "ok")
        def model_reset(self):
            if self.model is None: return
            if messagebox.askyesno("Reset model", f"Forget everything learned for {self.model.organism}/{self.model.sex}?\n(a backup is kept in the model folder)", parent=self):
                self.model.reset(); self.refresh_model(); self.toast("Model reset (backup kept).", "warn")
        def pose_layer(self, d):
            k = getattr(self, "_psel", None)
            if not k or self.model is None: return self.toast("Drag/click a part handle first, then use the layer buttons.", "warn")
            self.model.layer_nudge(self.pose_view, k, d, POSE_ORDER[self.pose_view]); self.model.save()
            if self.res is not None: render_outputs(self.res)
            self.pose_refresh(); self.toast(f"{pretty(k)} moved {'forward' if d > 0 else 'back'} in the {self.pose_view.upper()} view (learned).", "ok")
        def style_ttk(self):
            st = ttk.Style(self)
            try: st.theme_use("clam")
            except Exception: pass
            for orient in ("Vertical", "Horizontal"):
                st.configure(f"{orient}.TScrollbar", background=BG3, troughcolor=BG0, bordercolor=BG0,
                             arrowcolor=MUTED, lightcolor=BG3, darkcolor=BG3, relief="flat")
                st.map(f"{orient}.TScrollbar", background=[("active", LINE2)])

        # ------------------------------------------------------------------ tiny widget kit
        def mk_btn(self, parent, text, cmd, kind="n", font=None, padx=10, pady=5, anchor="center"):
            pal = dict(n=(BG3, TEXT, LINE2), p=(ACC_HI, "#001a20", "#9af6ff"), g=(GOOD, "#04210f", "#7ff0ad"),
                       d=("#3a1620", "#ff9aa8", "#55202e"), t=(BG2, MUTED, BG3))
            bg, fg, hov = pal[kind]
            b = tk.Label(parent, text=text, bg=bg, fg=fg, font=font or self.F["b"], padx=padx, pady=pady,
                         cursor="hand2", anchor=anchor, highlightthickness=1, highlightbackground=LINE)
            b._bg, b._fg, b._hov, b._act = bg, fg, hov, False
            def enter(_): b.config(bg=b._hov if not b._act else PINK)
            def leave(_): b.config(bg=(PINK if b._act else b._bg))
            b.bind("<Enter>", enter); b.bind("<Leave>", leave)
            b.bind("<Button-1>", lambda e: cmd())
            return b
        def set_active(self, b, on):
            b._act = on; b.config(bg=PINK if on else b._bg, fg="#fff" if on else b._fg)
        def mk_panel(self, parent, title, accent=ACC):
            outer = tk.Frame(parent, bg=BG1, highlightthickness=1, highlightbackground=LINE)
            head = tk.Frame(outer, bg=BG1); head.pack(fill="x")
            tk.Frame(head, bg=accent, width=4, height=14).pack(side="left", padx=(6,6), pady=5)
            tk.Label(head, text=title, bg=BG1, fg=TEXT, font=self.F["cardb"]).pack(side="left")
            inner = tk.Frame(outer, bg=BG1); inner.pack(fill="both", expand=True, padx=8, pady=(0,7))
            return outer, inner
        def mk_label(self, parent, text="", fg=MUTED, font=None, **kw):
            return tk.Label(parent, text=text, bg=kw.pop("bg", BG1), fg=fg, font=font or self.F["s"], **kw)
        def mk_entry(self, parent, var, width=8):
            return tk.Entry(parent, textvariable=var, width=width, bg=BG0, fg=TEXT, insertbackground=ACC,
                            relief="flat", highlightthickness=1, highlightbackground=LINE, highlightcolor=ACC,
                            font=self.F["mono"], disabledbackground=BG2, disabledforeground=MUTED, readonlybackground=BG2)
        def mk_check(self, parent, text, var, cmd=None):
            return tk.Checkbutton(parent, text=text, variable=var, command=cmd, bg=BG1, fg=TEXT, selectcolor=BG0,
                                  activebackground=BG1, activeforeground=ACC, font=self.F["s"], highlightthickness=0,
                                  bd=0, anchor="w")
        def mk_list(self, parent, height):
            return tk.Listbox(parent, height=height, bg=BG0, fg=TEXT, selectbackground=BG3, selectforeground=ACC,
                              relief="flat", highlightthickness=1, highlightbackground=LINE, highlightcolor=ACC,
                              font=self.F["mono"], activestyle="none", exportselection=False)
        def mk_scale(self, parent, var, lo, hi, cmd=None, res=1):
            return tk.Scale(parent, variable=var, from_=lo, to=hi, orient="horizontal", resolution=res, command=cmd,
                            bg=BG1, fg=TEXT, troughcolor=BG0, highlightthickness=0, activebackground=ACC,
                            font=self.F["s"], showvalue=True, sliderrelief="flat", bd=0, length=200)

        # ------------------------------------------------------------------ layout
        def build_ui(self):
            top = tk.Frame(self, bg=BG0); top.pack(fill="x", padx=10, pady=(8,4))
            self.logo = tk.Canvas(top, width=250, height=42, bg=BG0, highlightthickness=0); self.logo.pack(side="left")
            x = 4
            for ch in "HUMAN.RIG":
                for i,bit in enumerate(GLYPHS.get(ch, GLYPHS[" "])):
                    if bit == "1":
                        px, py = x+(i%3)*5, 6+(i//3)*5
                        self.logo.create_rectangle(px+2,py+2,px+6,py+6, fill=PINK, outline="")
                        self.logo.create_rectangle(px,py,px+4,py+4, fill=ACC, outline="")
                x += 20 if ch != "." else 12
            self.stage_fr = tk.Frame(top, bg=BG0); self.stage_fr.pack(side="left", padx=20)
            self.stage_lbl = []
            for i,n in enumerate(STAGE):
                l = tk.Label(self.stage_fr, text=f"{i+1}  {n}", bg=BG2, fg=DIM, font=self.F["cardb"], padx=12, pady=6,
                             highlightthickness=1, highlightbackground=LINE)
                l.pack(side="left", padx=(0,3)); self.stage_lbl.append(l)
            self.chip_lo = tk.Label(top, text="NO LOADOUT", bg=BG2, fg=MUTED, font=self.F["cardb"], padx=12, pady=6,
                                    highlightthickness=1, highlightbackground=LINE)
            self.chip_lo.pack(side="right")
            self.chip_model = tk.Label(top, text="", bg=BG2, fg=TEXT, font=self.F["cardb"], padx=12, pady=6, highlightthickness=2, highlightbackground=LINE)
            self.chip_model.pack(side="right", padx=8)

            self.coach = tk.Frame(self, bg=BG2, highlightthickness=1, highlightbackground=ACC); self.coach.pack(fill="x", padx=10, pady=(2,2))
            self.coach_step = tk.Label(self.coach, text="", bg=ACC, fg="#001a20", font=self.F["cardb"], padx=12, pady=8); self.coach_step.pack(side="left")
            self.coach_txt = tk.Label(self.coach, text="", bg=BG2, fg=TEXT, font=self.F["ui"], anchor="w", justify="left", wraplength=1000); self.coach_txt.pack(side="left", fill="x", expand=True, padx=12)
            self.coach_btn = self.mk_btn(self.coach, "", lambda: None, "p", pady=6, padx=16); self.coach_btn.pack(side="right", padx=8, pady=4)
            body = tk.Frame(self, bg=BG0); body.pack(fill="both", expand=True, padx=10, pady=4)
            # scrollable left sidebar (fits small screens / short windows)
            left_wrap = tk.Frame(body, bg=BG0, width=348); left_wrap.pack(side="left", fill="y", padx=(0,8)); left_wrap.pack_propagate(False)
            left_cv = tk.Canvas(left_wrap, bg=BG0, highlightthickness=0, width=330)
            left_sb = ttk.Scrollbar(left_wrap, orient="vertical", command=left_cv.yview)
            left_sb.pack(side="right", fill="y"); left_cv.pack(side="left", fill="both", expand=True)
            left_cv.configure(yscrollcommand=left_sb.set)
            left = tk.Frame(left_cv, bg=BG0)
            left_win = left_cv.create_window((0, 0), window=left, anchor="nw")
            def _left_cfg(_e=None):
                left_cv.configure(scrollregion=left_cv.bbox("all"))
                left_cv.itemconfigure(left_win, width=left_cv.winfo_width() or 330)
            left.bind("<Configure>", _left_cfg); left_cv.bind("<Configure>", _left_cfg)
            def _left_wheel(e):
                delta = -1 if (getattr(e, "delta", 0) > 0 or getattr(e, "num", 0) == 4) else 1
                left_cv.yview_scroll(delta, "units")
            for w in (left_cv, left, left_wrap):
                w.bind("<MouseWheel>", _left_wheel)
                w.bind("<Button-4>", _left_wheel); w.bind("<Button-5>", _left_wheel)
            self.build_left(left)
            main = tk.Frame(body, bg=BG0); main.pack(side="left", fill="both", expand=True)
            self.tabbar = tk.Frame(main, bg=BG0); self.tabbar.pack(fill="x")
            self.stack = tk.Frame(main, bg=BG1, highlightthickness=1, highlightbackground=LINE)
            self.stack.pack(fill="both", expand=True)
            self.stack.grid_rowconfigure(0, weight=1); self.stack.grid_columnconfigure(0, weight=1)
            self.pages, self.tabs = {}, {}
            for key,title in (("editor","EDITOR"),("sheet","RIG SHEET"),("asm","ASSEMBLED"),("pose","POSE"),
                              ("guide","PICK PARTS"),("align","ALIGN"),("anim","ANIMATE"),("log","LOG / HELP")):
                f = tk.Frame(self.stack, bg=BG1); f.grid(row=0, column=0, sticky="nsew"); self.pages[key] = f
                t = tk.Label(self.tabbar, text=title, bg=BG2, fg=MUTED, font=self.F["cardb"], padx=16, pady=7, cursor="hand2",
                             highlightthickness=1, highlightbackground=LINE)
                t.pack(side="left", padx=(0,2)); t.bind("<Button-1>", lambda e,k=key: self.show_page(k)); self.tabs[key] = t
            self.build_editor(self.pages["editor"]); self.build_sheet_page(self.pages["sheet"])
            self.build_asm_page(self.pages["asm"]); self.build_pose(self.pages["pose"])
            self.build_guide(self.pages["guide"]); self.build_log(self.pages["log"])
            self.build_align(self.pages["align"]); self.build_anim(self.pages["anim"])

            bar = tk.Frame(self, bg=BG1, highlightthickness=1, highlightbackground=LINE); bar.pack(fill="x", padx=10, pady=(4,8))
            self.status = tk.Label(bar, text="", bg=BG1, fg=ACC, font=self.F["ui"], anchor="w")
            self.status.pack(side="left", fill="x", expand=True, padx=8, pady=4)
            self.pb = tk.Canvas(bar, width=260, height=10, bg=BG0, highlightthickness=0); self.pb.pack(side="right", padx=8)
            self.save_chip = tk.Label(bar, text="", bg=BG1, fg=DIM, font=self.F["cardb"], padx=10); self.save_chip.pack(side="right")
            self.bind_all("<F1>", lambda e: self.show_page("editor")); self.bind_all("<F2>", lambda e: self.show_page("sheet"))
            self.bind_all("<F3>", lambda e: self.show_page("asm")); self.bind_all("<F4>", lambda e: self.show_page("pose"))
            self.bind_all("<F5>", lambda e: self.start_guide()); self.bind_all("<F6>", lambda e: self.show_page("log"))

        def build_left(self, L):
            p, f = self.mk_panel(L, "ROSTER"); p.pack(fill="x", pady=(0,6))
            row = tk.Frame(f, bg=BG1); row.pack(fill="x")
            self.lb_lo = self.mk_list(row, 4); self.lb_lo.pack(side="left", fill="x", expand=True)
            self.lb_lo.bind("<<ListboxSelect>>", lambda e: self.on_select_loadout())
            bt = tk.Frame(row, bg=BG1); bt.pack(side="left", padx=(6,0))
            self.mk_btn(bt, "NEW", self.new_loadout, pady=3).pack(fill="x", pady=1)
            self.mk_btn(bt, "FOLDER", self.open_folder, "t", pady=3).pack(fill="x", pady=1)
            g = tk.Frame(f, bg=BG1); g.pack(fill="x", pady=(6,0))
            for c,(t,v,w) in enumerate((("NAME",self.v_name,10),("HEIGHT CM",self.v_h,5),("CM/PX",self.v_cpp,4))):
                self.mk_label(g, t).grid(row=0, column=c, sticky="w", padx=(0 if c==0 else 6,0))
                e = self.mk_entry(g, v, w); e.grid(row=1, column=c, sticky="w", padx=(0 if c==0 else 6,0))
                if c == 0: self.e_name = e
            p, f = self.mk_panel(L, "SOURCES"); p.pack(fill="x", pady=(0,6))
            self.lb = self.mk_list(f, 3); self.lb.config(selectmode="extended"); self.lb.pack(fill="x")
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(5,0))
            self.mk_btn(r, "+ FILES", self.add_files, pady=3).pack(side="left")
            self.mk_btn(r, "+ FOLDER", self.add_folder, pady=3).pack(side="left", padx=4)
            self.mk_btn(r, "REMOVE", self.remove, "d", pady=3).pack(side="left")
            p, f = self.mk_panel(L, "FRAMES   (click to edit)", PINK); p.pack(fill="both", expand=True, pady=(0,6))
            self.fl = self.mk_list(f, 6); self.fl.pack(fill="both", expand=True)
            self.fl.bind("<<ListboxSelect>>", lambda e: self.pick())
            self.mk_label(f, "● on  ○ off     F front  S side  B back", DIM).pack(anchor="w", pady=(3,0))
            p, f = self.mk_panel(L, "CUT SETTINGS"); p.pack(fill="x", pady=(0,6))
            r = tk.Frame(f, bg=BG1); r.pack(fill="x")
            self.mk_check(r, "Twin limbs", self.v_mirror).pack(side="left")
            self.mk_check(r, "Gap fill", self.v_fill).pack(side="left", padx=6)
            self.mk_check(r, "Blend", self.v_blend).pack(side="left")
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4,0))
            self.mk_label(r, "COLOURS").pack(side="left")
            tk.Spinbox(r, from_=0, to=128, width=4, textvariable=self.v_cols, bg=BG0, fg=TEXT, relief="flat",
                       buttonbackground=BG3, insertbackground=ACC, font=self.F["mono"]).pack(side="left", padx=(4,12))
            self.mk_label(r, "EDGE").pack(side="left")
            tk.Spinbox(r, from_=20, to=240, width=4, textvariable=self.v_athr, bg=BG0, fg=TEXT, relief="flat",
                       buttonbackground=BG3, insertbackground=ACC, font=self.F["mono"]).pack(side="left", padx=4)
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4,0))
            self.mk_check(r, "Overlap", self.v_overlap).pack(side="left")
            self.mk_label(r, "SIZE %").pack(side="left", padx=(4,0))
            tk.Spinbox(r, from_=40, to=200, increment=10, width=4, textvariable=self.v_ovpct, bg=BG0, fg=TEXT, relief="flat",
                       buttonbackground=BG3, insertbackground=ACC, font=self.F["mono"]).pack(side="left", padx=4)
            self.mk_check(r, "Synth views", self.v_synth).pack(side="left", padx=4)
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4,0))
            self.mk_label(r, "EDGE SMOOTH").pack(side="left")
            tk.Spinbox(r, from_=0, to=3, width=3, textvariable=self.v_smooth, bg=BG0, fg=TEXT, relief="flat",
                       buttonbackground=BG3, insertbackground=ACC, font=self.F["mono"]).pack(side="left", padx=4)
            self.mk_check(r, "Learn colours from all frames", self.v_learn).pack(side="left", padx=6)
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4,0))
            self.mk_check(r, "Fill hidden areas (arm over body)", self.v_fillhidden).pack(side="left")
            p, f = self.mk_panel(L, "BODY MODEL  (permanent, learns from your edits)", GOLD); p.pack(fill="x", pady=(0,6))
            r = tk.Frame(f, bg=BG1); r.pack(fill="x")
            self.mk_label(r, "ORGANISM").pack(side="left")
            cb = ttk.Combobox(r, textvariable=self.v_org, width=9, values=OM.known_organisms()); cb.pack(side="left", padx=4)
            cb.bind("<<ComboboxSelected>>", self.refresh_model); cb.bind("<Return>", self.refresh_model)
            self.mk_label(r, "SEX").pack(side="left", padx=(6,0))
            cs = ttk.Combobox(r, textvariable=self.v_sex, width=7, values=("male","female","any"), state="readonly"); cs.pack(side="left", padx=4)
            cs.bind("<<ComboboxSelected>>", self.refresh_model)
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3,0))
            self.mk_check(r, "Use model", self.v_usemodel, self.refresh_model).pack(side="left")
            self.mk_check(r, "Train on CUT", self.v_trainmodel).pack(side="left", padx=6)
            self.model_lbl = self.mk_label(f, "", MUTED, justify="left", wraplength=300); self.model_lbl.pack(anchor="w")
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3,0))
            self.mk_btn(r, "EXPORT MODEL", self.model_export, "t", pady=2, font=self.F["card"]).pack(side="left")
            self.mk_btn(r, "RESET", self.model_reset, "d", pady=2, font=self.F["card"]).pack(side="left", padx=4)
            # ---- BODY: enable/disable limbs + local scale ----
            p, f = self.mk_panel(L, "BODY  (add / remove parts)", PINK); p.pack(fill="x", pady=(0,6))
            self.body_sum = self.mk_label(f, "full body, normal proportions", MUTED, justify="left", wraplength=300)
            self.body_sum.pack(anchor="w", pady=(0,4))
            g = tk.Frame(f, bg=BG1); g.pack(fill="x")
            for i, (grp, lab) in enumerate(RB.GROUP_LABELS.items()):
                cb = tk.Checkbutton(g, text=lab, variable=self.v_body_group[grp],
                                    command=lambda g=grp: self._body_toggle(g),
                                    bg=BG1, fg=TEXT, selectcolor=BG0, activebackground=BG1,
                                    activeforeground=ACC, font=self.F["s"], anchor="w", highlightthickness=0)
                cb.grid(row=i//2, column=i%2, sticky="w", padx=2, pady=1)
            self.mk_label(f, "SCALE  (0.55 – 1.55)", DIM).pack(anchor="w", pady=(6,0))
            # only the most useful scales in the panel to keep it compact
            for b in ("neck", "chest", "abdomen", "uarm", "thigh"):
                row = tk.Frame(f, bg=BG1); row.pack(fill="x", pady=1)
                self.mk_label(row, RB.SCALE_LABELS[b], width=12).pack(side="left")
                s = self.mk_scale(row, self.v_body_scale[b], RB.SCALE_MIN, RB.SCALE_MAX, res=0.05)
                s.config(length=140, command=lambda _=None, bb=b: self._body_scale_changed(bb))
                s.pack(side="left", fill="x", expand=True)
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4,0))
            self.mk_btn(r, "RESET BODY", self._body_reset, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")
            self.mk_label(f, "Turn a limb off, then CUT again. Scale changes length + thickness of that region.",
                          DIM, wraplength=300, justify="left").pack(anchor="w", pady=(2,0))

            a = tk.Frame(L, bg=BG0); a.pack(fill="x")
            self.mk_btn(a, "▶   ANALYSE", self.analyse, "p", pady=8).pack(fill="x", pady=2)
            self.mk_btn(a, "◆   CUT / UPDATE RIG", self.cut, "p", pady=8).pack(fill="x", pady=2)
            self.mk_btn(a, "★   PICK PARTS", self.start_guide, "p", pady=8).pack(fill="x", pady=2)
            self.mk_btn(a, "⬇   SAVE & EXPORT", self.export, "g", pady=8).pack(fill="x", pady=2)
            rr = tk.Frame(a, bg=BG0); rr.pack(fill="x", pady=2)
            self.mk_btn(rr, "✔ RIG CHECK", self.run_check, "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(rr, "⎘ SNAPSHOT", self.snapshot, "n", pady=5).pack(side="left", expand=True, fill="x")

        def show_page(self, key):
            self.pages[key].tkraise()
            for k,t in self.tabs.items():
                t.config(bg=BG1 if k == key else BG2, fg=ACC if k == key else MUTED)
            self.cur_page = key
            if key in ("sheet","asm"): self.after(30, self.refresh_view_pages)
            if key == "guide": self.after(30, self.guide_redraw)
            if key == "pose": self.after(30, self.pose_refresh)
            if key == "align": self.after(30, self.align_refresh)
            if key == "anim": self.after(30, self.anim_refresh)

        # ------------------------------------------------------------------ editor page
        def build_editor(self, page):
            right = tk.Frame(page, bg=BG1, width=282); right.pack(side="right", fill="y", padx=8, pady=8); right.pack_propagate(False)
            left = tk.Frame(page, bg=BG1); left.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
            bar = tk.Frame(left, bg=BG1); bar.pack(fill="x", pady=(0,4))
            self.ed_title = tk.Label(bar, text="no frame - ANALYSE first", bg=BG1, fg=TEXT, font=self.F["h"]); self.ed_title.pack(side="left")
            self.ed_dirty = tk.Label(bar, text="", bg=BG1, fg=WARN, font=self.F["cardb"]); self.ed_dirty.pack(side="left", padx=12)
            self.mk_btn(bar, "FIT", self.ed_fit_render, "t", pady=2, padx=8).pack(side="right")
            self.mk_btn(bar, "+", lambda: self.ed_zoom_by(1.25), "t", pady=2, padx=8).pack(side="right", padx=2)
            self.mk_btn(bar, "-", lambda: self.ed_zoom_by(0.8), "t", pady=2, padx=8).pack(side="right")
            self.ed_coord = tk.Label(bar, text="", bg=BG1, fg=MUTED, font=self.F["mono"]); self.ed_coord.pack(side="right", padx=10)
            fr = tk.Frame(left, bg=BG1); fr.pack(fill="both", expand=True)
            cv = tk.Canvas(fr, bg="#05070b", highlightthickness=1, highlightbackground=LINE, cursor="crosshair")
            hs = ttk.Scrollbar(fr, orient="horizontal", command=cv.xview); vs = ttk.Scrollbar(fr, orient="vertical", command=cv.yview)
            cv.config(xscrollcommand=hs.set, yscrollcommand=vs.set)
            hs.pack(side="bottom", fill="x"); vs.pack(side="right", fill="y"); cv.pack(side="left", fill="both", expand=True)
            self.ed_cv = cv
            cv.bind("<ButtonPress-1>", lambda e: self.ed_press(e, 1)); cv.bind("<B1-Motion>", self.ed_motion)
            cv.bind("<ButtonRelease-1>", self.ed_release)
            for b in (3, 2) if sys.platform == "darwin" else (3,):
                cv.bind(f"<ButtonPress-{b}>", lambda e: self.ed_press(e, 3)); cv.bind(f"<B{b}-Motion>", self.ed_motion)
                cv.bind(f"<ButtonRelease-{b}>", self.ed_release)
            if sys.platform != "darwin":
                cv.bind("<ButtonPress-2>", self.ed_pan_start); cv.bind("<B2-Motion>", self.ed_pan_move)
            cv.bind("<Motion>", self.ed_cursor)
            cv.bind("<MouseWheel>", self.ed_wheel); cv.bind("<Button-4>", self.ed_wheel); cv.bind("<Button-5>", self.ed_wheel)
            self.mk_label(left, "Left-drag paint  |  right-drag erase  |  Shift-drag or middle-drag pan  |  wheel zoom  |  "
                                "B brush  E erase  R restore  G fill  I pick  J joints  X +overlap  U -overlap  [ ] size  Ctrl+Z undo", DIM, wraplength=760, justify="left").pack(anchor="w", pady=(4,0))

            p, f = self.mk_panel(right, "TOOLS"); p.pack(fill="x", pady=(0,6))
            self.tool_btns = {}
            grid = tk.Frame(f, bg=BG1); grid.pack(fill="x")
            for i,(k,t) in enumerate((("brush","BRUSH"),("erase","ERASE"),("restore","AUTO"),("fill","FILL"),("pick","PICK"),("joints","JOINTS"),("extend","+OVERLAP"),("unext","-OVERLAP"))):
                b = self.mk_btn(grid, t, lambda k=k: self.set_tool(k), "n", pady=4, padx=6, font=self.F["card"])
                b.grid(row=i//3, column=i%3, sticky="ew", padx=1, pady=1); grid.grid_columnconfigure(i%3, weight=1); self.tool_btns[k] = b
            self.mk_label(f, "BRUSH SIZE  (cm)").pack(anchor="w", pady=(6,0))
            self.mk_scale(f, self.v_brush, 1, 30, res=0.5).pack(fill="x")
            self.mk_label(f, "REGION TINT  (%)").pack(anchor="w")
            self.mk_scale(f, self.v_tint, 0, 100, cmd=lambda v: self.ed_soon()).pack(fill="x")
            r = tk.Frame(f, bg=BG1); r.pack(fill="x")
            self.mk_check(r, "Regions", self.v_regions, self.ed_soon).pack(side="left")
            self.mk_check(r, "Borders", self.v_borders, self.ed_soon).pack(side="left", padx=4)
            self.mk_check(r, "Skeleton", self.v_skel, self.ed_soon).pack(side="left")
            self.mk_check(f, "Paint only on the subject", self.v_only).pack(anchor="w")
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4,0))
            self.mk_btn(r, "UNDO", self.ed_undo, "t", pady=3).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(r, "REDO", self.ed_redo, "t", pady=3).pack(side="left", expand=True, fill="x")

            p, f = self.mk_panel(right, "PAINT WITH", PINK); p.pack(fill="x", pady=(0,6))
            self.chips = {}
            grid = tk.Frame(f, bg=BG1); grid.pack(fill="x")
            for i,k in enumerate(PART_KEYS):
                c = PART_COL[k]
                l = tk.Label(grid, text=pretty(k), bg=_dark(c, 0.30), fg="#fff", font=self.F["card"], pady=3, cursor="hand2",
                             highlightthickness=2, highlightbackground=BG1)
                l.grid(row=i//3, column=i%3, sticky="ew", padx=1, pady=1); grid.grid_columnconfigure(i%3, weight=1)
                l.bind("<Button-1>", lambda e,k=k: self.set_part(k)); self.chips[k] = l

            p, f = self.mk_panel(right, "THIS FRAME"); p.pack(fill="both", expand=True)
            self.mk_label(f, "VIEW  (override the auto-detected view)").pack(anchor="w")
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=2)
            self.view_btns = {}
            for v in (None,) + VIEWS:
                b = self.mk_btn(r, "AUTO" if v is None else v.upper(), lambda v=v: self.set_view_override(v), "n", pady=3, padx=5, font=self.F["card"])
                b.pack(side="left", expand=True, fill="x", padx=1); self.view_btns[v] = b
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=2)
            self.mk_btn(r, "FLIP FACING", self.toggle_flip, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_check(r, "Use in cut", self.v_enabled, self.toggle_enabled).pack(side="left", padx=4)
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=2)
            self.mk_btn(r, "CLEAR PAINT", self.clear_paint, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(r, "RESET JOINTS", self.reset_joints, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")
            self.frame_info = tk.Label(f, text="", bg=BG1, fg=MUTED, font=self.F["mono"], justify="left", anchor="nw", wraplength=250)
            self.frame_info.pack(fill="both", expand=True, pady=(6,0))

        def set_tool(self, k):
            self.v_tool.set(k)
            for n,b in self.tool_btns.items(): self.set_active(b, n == k)
            self.ed_cv.config(cursor={"brush":"crosshair","erase":"crosshair","restore":"crosshair","fill":"spraycan",
                                      "pick":"target","joints":"fleur"}.get(k, "crosshair"))
            if k == "joints": self.v_skel.set(True); self.ed_soon()
            self.ed_cv.delete("cur")
        def set_part(self, k):
            self.v_part.set(k)
            for n,l in self.chips.items(): l.config(highlightbackground=(_hex(PART_COL[n]) if n == k else BG1),
                                                   bg=(_dark(PART_COL[n],0.55) if n == k else _dark(PART_COL[n],0.30)))
        def active_pid(self): return PART_KEYS.index(self.v_part.get())

        def on_key(self, e):
            w = self.focus_get()
            if isinstance(w, (tk.Entry, tk.Spinbox, tk.Text)): return
            ch = (e.char or "").lower()
            if getattr(self, "cur_page", "") == "align" and self.align_key(e): return
            if getattr(self, "cur_page", "") == "guide":
                if e.keysym == "Left": return self.guide_step(-1)
                if e.keysym == "Right": return self.guide_step(1)
                if e.keysym == "Return": return self.g_lock_best()
            if e.state & 0x4 and ch in ("\x1a", "z"): return self.ed_undo()
            if e.state & 0x4 and ch in ("\x19", "y"): return self.ed_redo()
            tools = dict(b="brush", e="erase", r="restore", g="fill", i="pick", j="joints", x="extend", u="unext")
            if ch in tools and self.cur_page == "editor": self.set_tool(tools[ch])
            elif ch == "[": self.v_brush.set(max(1, self.v_brush.get()-1))
            elif ch == "]": self.v_brush.set(min(30, self.v_brush.get()+1))

        # ------------------------------------------------------------------ sheet / assembled pages
        def build_view_page(self, page, key, extra):
            bar = tk.Frame(page, bg=BG1); bar.pack(fill="x", padx=8, pady=8)
            self.mk_btn(bar, "-", lambda: self.zoom_page(key, 0.8), "t", pady=2, padx=10).pack(side="left")
            self.mk_btn(bar, "+", lambda: self.zoom_page(key, 1.25), "t", pady=2, padx=10).pack(side="left", padx=2)
            self.mk_btn(bar, "FIT", lambda: self.fit_page(key), "t", pady=2, padx=10).pack(side="left")
            extra(bar)
            info = tk.Label(bar, text="", bg=BG1, fg=MUTED, font=self.F["mono"]); info.pack(side="right")
            fr = tk.Frame(page, bg=BG1); fr.pack(fill="both", expand=True, padx=8, pady=(0,8))
            cv = tk.Canvas(fr, bg="#05070b", highlightthickness=1, highlightbackground=LINE)
            hs = ttk.Scrollbar(fr, orient="horizontal", command=cv.xview); vs = ttk.Scrollbar(fr, orient="vertical", command=cv.yview)
            cv.config(xscrollcommand=hs.set, yscrollcommand=vs.set)
            hs.pack(side="bottom", fill="x"); vs.pack(side="right", fill="y"); cv.pack(side="left", fill="both", expand=True)
            cv.bind("<Configure>", lambda e: self.after_idle(lambda: self.refresh_view_pages(only=key) if self.fit[key] else None))
            setattr(self, f"{key}_cv", cv); setattr(self, f"{key}_info", info)
        def build_sheet_page(self, page):
            def extra(bar):
                self.mk_btn(bar, "OPEN FOLDER", self.open_folder, "t", pady=2, padx=10).pack(side="left", padx=10)
                self.mk_label(bar, "rig.png = exploded parts + assembled figures for FRONT / SIDE / BACK. "
                                   "atlas.png has the same layout, clean and transparent.", DIM).pack(side="left", padx=6)
            self.build_view_page(page, "sheet", extra)
        def build_asm_page(self, page):
            def extra(bar):
                self.mk_check(bar, "Solid colours", self.v_solid, self.refresh_view_pages).pack(side="left", padx=(12,4))
                self.mk_check(bar, "Joints", self.v_bones, self.refresh_view_pages).pack(side="left")
                self.mk_label(bar, "Side view: sprites face right, far limbs are drawn darker (preview only).", DIM).pack(side="left", padx=10)
            self.build_view_page(page, "asm", extra)
        def zoom_page(self, key, f):
            self.fit[key] = False; self.z[key] = float(np.clip(self.z[key]*f, 0.5, 16)); self.refresh_view_pages(only=key)
        def fit_page(self, key): self.fit[key] = True; self.refresh_view_pages(only=key)
        def refresh_view_pages(self, only=None):
            if self.res is None: return
            if only is None: self.pose_refresh()
            for key in ("sheet","asm"):
                if only and key != only: continue
                cv = getattr(self, f"{key}_cv")
                if key == "sheet": im = self.res.sheet
                else:
                    try: im = compose_views(self.res, self.v_solid.get(), self.v_bones.get())
                    except Exception as ex: self.log(f"assembled preview failed: {ex!r}", "warn"); continue
                if self.fit[key]:
                    cw, ch = max(200, cv.winfo_width()-24), max(200, cv.winfo_height()-24)
                    z = min(cw/im.width, ch/im.height); self.z[key] = float(int(z) if z >= 1 else max(0.5, z))
                z = self.z[key]; w, h = max(1,int(im.width*z)), max(1,int(im.height*z))
                self.photos[key] = ImageTk.PhotoImage(im.resize((w,h), Image.Resampling.NEAREST))
                cv.delete("all"); cv.create_image(10, 10, image=self.photos[key], anchor="nw")
                cv.config(scrollregion=(0,0,w+20,h+20))
                getattr(self, f"{key}_info").config(text=f"{im.width}x{im.height}px   x{z:.2g}")

        # ------------------------------------------------------------------ pick-parts page
        def build_guide(self, page):
            top = tk.Frame(page, bg=BG1); top.pack(fill="x", padx=10, pady=(10,4))
            self.view_tab = {}
            for v in VIEWS:
                b = self.mk_btn(top, v.upper(), lambda v=v: self.guide_set_view(v), "n", pady=5, padx=18); b.pack(side="left", padx=(0,3)); self.view_tab[v] = b
            self.g_head = tk.Label(top, text="", bg=BG1, fg=TEXT, font=self.F["h"]); self.g_head.pack(side="left", padx=16)
            self.g_xp = tk.Canvas(top, width=220, height=12, bg=BG0, highlightthickness=1, highlightbackground=LINE); self.g_xp.pack(side="right")
            self.g_xp_lbl = tk.Label(top, text="", bg=BG1, fg=MUTED, font=self.F["card"]); self.g_xp_lbl.pack(side="right", padx=8)
            mid = tk.Frame(page, bg=BG1); mid.pack(fill="both", expand=True, padx=10, pady=4)
            inv = tk.Frame(mid, bg=BG1, width=268); inv.pack(side="left", fill="y"); inv.pack_propagate(False)
            self.mk_label(inv, "INVENTORY", ACC, self.F["cardb"]).pack(anchor="w")
            self.inv_cv = tk.Canvas(inv, bg=BG0, highlightthickness=1, highlightbackground=LINE, width=262)
            self.inv_cv.pack(fill="both", expand=True)
            ctr = tk.Frame(mid, bg=BG1); ctr.pack(side="left", fill="both", expand=True, padx=8)
            self.g_sub = tk.Label(ctr, text="", bg=BG1, fg=MUTED, font=self.F["s"], anchor="w"); self.g_sub.pack(fill="x")
            fr = tk.Frame(ctr, bg=BG1); fr.pack(fill="both", expand=True)
            self.card_cv = tk.Canvas(fr, bg=BG0, highlightthickness=1, highlightbackground=LINE)
            vs = ttk.Scrollbar(fr, orient="vertical", command=self.card_cv.yview); self.card_cv.config(yscrollcommand=vs.set)
            vs.pack(side="right", fill="y"); self.card_cv.pack(side="left", fill="both", expand=True)
            self.card_cv.bind("<Configure>", lambda e: self.after_idle(self.draw_cards))
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"): self.card_cv.bind(ev, self.card_wheel)
            side = tk.Frame(mid, bg=BG1, width=282); side.pack(side="left", fill="y"); side.pack_propagate(False)
            self.mk_label(side, "SOURCE CONTEXT  (hover a card)", ACC, self.F["cardb"]).pack(anchor="w")
            self.ctx_cv = tk.Canvas(side, bg=BG0, highlightthickness=1, highlightbackground=LINE, width=276, height=360); self.ctx_cv.pack()
            self.ctx_info = tk.Label(side, text="", bg=BG1, fg=MUTED, font=self.F["mono"], justify="left", anchor="nw", wraplength=270)
            self.ctx_info.pack(fill="x", pady=4)
            r = tk.Frame(side, bg=BG1); r.pack(fill="x", pady=2)
            self.mk_btn(r, "◀ PREV", lambda: self.guide_step(-1), "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(r, "NEXT ▶", lambda: self.guide_step(1), "n", pady=5).pack(side="left", expand=True, fill="x")
            self.mk_btn(side, "✓  LOCK THE ONE IN USE   (Enter)", self.g_lock_best, "g", pady=6).pack(fill="x", pady=2)
            self.mk_btn(side, "AUTO  (unlock this part)", self.g_auto, "n", pady=5).pack(fill="x", pady=2)
            self.mk_btn(side, "BRING BACK REJECTED", self.g_unreject, "t", pady=4).pack(fill="x", pady=2)
            self.mk_check(side, "Auto-advance after a pick", self.v_autoadv).pack(anchor="w", pady=(4,0))
            self.mk_label(side, "Click a card = use + lock it (survives re-cuts).\nRight-click = reject that candidate.\n"
                                "Arrow keys move between parts.", DIM, justify="left").pack(anchor="w", pady=4)

        def card_wheel(self, e):
            d = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
            self.card_cv.yview_scroll(d*2, "units")

        def start_guide(self):
            if self.res is None or not getattr(self.res, "cands", None) or not any(
                    self.res.cands[v][k] for v in VIEWS for k in PART_KEYS):
                return self.toast("Run CUT first - there are no candidates to pick from yet.", "warn")
            self.show_page("guide")

        def guide_set_view(self, v): self.guide_view = v; self.guide_redraw()
        def guide_select_part(self, k): self.guide_key = k; self.guide_redraw()
        def guide_step(self, d):
            i = PART_KEYS.index(self.guide_key) + d
            if i >= len(PART_KEYS): i = 0; self.guide_view = VIEWS[(VIEWS.index(self.guide_view)+1) % 3]
            elif i < 0: i = len(PART_KEYS)-1; self.guide_view = VIEWS[(VIEWS.index(self.guide_view)-1) % 3]
            self.guide_key = PART_KEYS[i]; self.guide_redraw()
        def g_cands(self): return self.res.cands.get(self.guide_view, {}).get(self.guide_key, []) if self.res else []

        def guide_redraw(self):
            if self.res is None: return
            for v,b in self.view_tab.items(): self.set_active(b, v == self.guide_view)
            sp = self.res.views[self.guide_view][self.guide_key]
            self.g_head.config(text=f"{self.guide_view.upper()}  /  {pretty(self.guide_key)}")
            note = {"own":"auto-picked (best score, blended)", "locked":"LOCKED by you", "borrowed":f"BORROWED from the {sp.borrowed.upper()} view",
                    "synth":f"SYNTHESIZED from the {sp.borrowed.upper()} view (a guess, not real data)", "placeholder":"MISSING - no usable candidate; placeholder shown"}[sp.kind]
            self.g_sub.config(text=f"in use: {note}   |   {len(self.g_cands())} candidates")
            n = sum(len(self.locks.get(v, {})) for v in VIEWS); tot = len(VIEWS)*len(PART_KEYS)
            self.g_xp.delete("all"); self.g_xp.create_rectangle(0,0,int(220*n/tot),12, fill=GOLD, outline="")
            self.g_xp_lbl.config(text=f"LOCKED {n}/{tot}")
            self.draw_inventory(); self.draw_cards()

        def draw_inventory(self):
            cv = self.inv_cv; cv.delete("all"); self.inv_photos = []
            RH = 40; v = self.guide_view
            for i,k in enumerate(PART_KEYS):
                y = i*RH; sp = self.res.views[v][k]; sel = k == self.guide_key; tag = f"inv{i}"
                kc = dict(locked=GOLD, own=ACC, borrowed=WARN, synth=PURP, placeholder=BAD)[sp.kind]
                cv.create_rectangle(1, y+1, 260, y+RH-2, fill=BG3 if sel else BG2, outline=ACC if sel else LINE, tags=(tag,))
                cv.create_rectangle(1, y+1, 6, y+RH-2, fill=_hex(PART_COL[k]), outline="", tags=(tag,))
                im = Image.fromarray(sp.rgba); ys, xs = np.nonzero(sp.rgba[...,3])
                if len(ys): im = im.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
                s = min(30/max(1,im.width), 30/max(1,im.height)); s = int(s) if s >= 1 else s
                ph = ImageTk.PhotoImage(im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.Resampling.NEAREST))
                self.inv_photos.append(ph); cv.create_image(26, y+RH//2, image=ph, tags=(tag,))
                cv.create_text(48, y+12, text=pretty(k), anchor="w", fill=TEXT, font=self.F["cardb"], tags=(tag,))
                sub = {"own":f"score {sp.score:.2f}", "locked":"locked", "borrowed":f"from {sp.borrowed}", "synth":f"synth < {sp.borrowed}", "placeholder":"missing"}[sp.kind]
                cv.create_text(48, y+27, text=sub, anchor="w", fill=MUTED, font=self.F["card"], tags=(tag,))
                glyph = dict(locked="✓", own="●", borrowed="◐", synth="◈", placeholder="✕")[sp.kind]
                cv.create_text(246, y+RH//2, text=glyph, fill=kc, font=self.F["h"], tags=(tag,))
                cv.tag_bind(tag, "<Button-1>", lambda e,k=k: self.guide_select_part(k))
            cv.config(scrollregion=(0,0,262,len(PART_KEYS)*RH))

        def draw_cards(self):
            cv = self.card_cv; cv.delete("all"); self.card_photos = []
            if self.res is None: return
            v, k = self.guide_view, self.guide_key
            cands = self.g_cands()
            W = max(360, cv.winfo_width()); CW, CH, PAD = 156, 214, 12
            if not cands:
                cv.create_text(W//2, 120, text="No candidates for this part in this view.\nAdd a frame that shows it (or force a frame's view in the editor)\nand CUT again.",
                               fill=MUTED, font=self.F["ui"], justify="center"); return
            cols = max(1, (W-PAD)//(CW+PAD)); lock = self.locks.get(v, {}).get(k); inuse = self.res.views[v][k].cid
            for i,c in enumerate(cands):
                x, y = PAD+(i%cols)*(CW+PAD), PAD+(i//cols)*(CH+PAD); tag = f"card{i}"
                rn, rc = rarity(c["score"]); sel = c["cid"] == lock; used = c["cid"] == inuse and not sel
                edge = GOLD if sel else (ACC if used else rc)
                cv.create_rectangle(x, y, x+CW, y+CH, fill=BG2, outline=edge, width=3 if (sel or used) else 1, tags=(tag,))
                cv.create_text(x+8, y+11, text=rn, anchor="w", fill=rc, font=self.F["card"], tags=(tag,))
                cv.create_rectangle(x+8, y+24, x+CW-8, y+144, fill="#05070b", outline=LINE, tags=(tag,))
                im = Image.fromarray(c["img"]); ys, xs = np.nonzero(c["img"][...,3])
                if len(ys): im = im.crop((xs.min(), ys.min(), xs.max()+1, ys.max()+1))
                s = min((CW-24)/max(1,im.width), 112/max(1,im.height)); s = int(s) if s >= 1 else s
                ph = ImageTk.PhotoImage(im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.Resampling.NEAREST))
                self.card_photos.append(ph); cv.create_image(x+CW//2, y+84, image=ph, tags=(tag,))
                cv.create_rectangle(x+8, y+152, x+CW-8, y+158, fill=BG0, outline="", tags=(tag,))
                cv.create_rectangle(x+8, y+152, x+8+int((CW-16)*min(1,c["score"])), y+158, fill=rc, outline="", tags=(tag,))
                cv.create_text(x+8, y+170, text=f"{c['score']:.2f}", anchor="w", fill=TEXT, font=self.F["cardb"], tags=(tag,))
                tags = [t for t,on in (("LOCKED",sel),("IN USE",used),("FLIP",c["flip"]),("TWIN",c["twin"])) if on]
                cv.create_text(x+CW-8, y+170, text="  ".join(tags), anchor="e", fill=GOLD if sel else (ACC if used else MUTED), font=self.F["card"], tags=(tag,))
                cv.create_text(x+8, y+192, text=str(c["fname"])[:22], anchor="w", fill=MUTED, font=self.F["card"], tags=(tag,))
                cv.tag_bind(tag, "<Button-1>", lambda e,cid=c["cid"]: self.g_pick(cid))
                cv.tag_bind(tag, "<Button-3>", lambda e,cid=c["cid"]: self.g_reject(cid))
                cv.tag_bind(tag, "<Button-2>", lambda e,cid=c["cid"]: self.g_reject(cid))
                cv.tag_bind(tag, "<Enter>", lambda e,c=c: self.ctx_show(c))
            rows = (len(cands)+cols-1)//cols
            cv.config(scrollregion=(0, 0, W, PAD+rows*(CH+PAD)))

        def ctx_show(self, c):
            cv = self.ctx_cv; cv.delete("all")
            fd = next((f for f in self.frames if f.name == c["fname"]), None)
            self.ctx_info.config(text=f"{c['fname']}\nscore {c['score']:.3f}   cut as {c['src_key']}\n"
                                      f"{'flipped  ' if c['flip'] else ''}{'twin of opposite limb' if c['twin'] else ''}")
            if fd is None:
                cv.create_text(138, 180, text="Run ANALYSE to see where\nthis candidate was cut from.", fill=MUTED, font=self.F["ui"], justify="center"); return
            ys, xs = np.nonzero(fd.alpha > 127); m = int(6*fd.s)
            x0, x1 = max(0,xs.min()-m), min(fd.alpha.shape[1], xs.max()+m+1); y0, y1 = max(0,ys.min()-m), min(fd.alpha.shape[0], ys.max()+m+1)
            rgb = fd.rgb[y0:y1,x0:x1].astype(np.float32); a = fd.alpha[y0:y1,x0:x1] > 127
            lab = fd.lab[y0:y1,x0:x1]; hit = lab == PART_KEYS.index(c["src_key"])
            out = np.full(rgb.shape, 10, np.float32); out[a] = rgb[a]*0.30
            out[hit] = rgb[hit]*0.6 + np.array(PART_COL[c["src_key"]], np.float32)*0.4
            im = Image.fromarray(out.astype(np.uint8)); s = min(272/im.width, 354/im.height)
            im = im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.Resampling.NEAREST if s >= 1 else Image.Resampling.BILINEAR)
            self.ctx_photo = ImageTk.PhotoImage(im); cv.create_image(138, 180, image=self.ctx_photo)

        def _apply_choice(self, v, k, sp):
            if self.res.palette and len(self.res.palette) <= 256: apply_palette(sp.rgba, np.array(self.res.palette, np.uint8))
            self.res.chosen[v][k] = sp
            finalize_views(self.res); render_outputs(self.res); self.res.cands_dirty = False
            self.schedule_persist(); self.refresh_view_pages(); self.update_stage()

        def g_pick(self, cid):
            v, k = self.guide_view, self.guide_key
            c = next((c for c in self.g_cands() if c["cid"] == cid), None)
            if c is None: return
            sp = sprite_from_img(k, RT.fill_holes(smooth_sprite(c["img"], getattr(self.res, "smooth", 2))), c, self.res.lengths, self.res.cpp, 1, "locked")
            if sp is None: return
            self.locks.setdefault(v, {})[k] = cid
            self._apply_choice(v, k, sp)
            self.toast(f"{v.upper()} {pretty(k)}: locked ({c['score']:.2f})")
            if self.v_autoadv.get(): self.guide_step(1)
            else: self.guide_redraw()

        def g_lock_best(self):
            if self.res is None: return
            cands = self.g_cands()
            if not cands: return self.toast("Nothing to lock for this part.", "warn")
            cur = self.res.views[self.guide_view][self.guide_key].cid
            self.g_pick(cur if any(c["cid"] == cur for c in cands) else cands[0]["cid"])

        def g_auto(self):
            if self.res is None: return
            v, k = self.guide_view, self.guide_key
            self.locks.get(v, {}).pop(k, None); cands = self.g_cands()
            if cands:
                sp = sprite_from_img(k, RT.fill_holes(smooth_sprite(cands[0]["img"], getattr(self.res, "smooth", 2))), cands[0], self.res.lengths, self.res.cpp, 1, "own")
                if sp is not None: self._apply_choice(v, k, sp)
            else:
                self.res.chosen[v].pop(k, None); finalize_views(self.res); render_outputs(self.res); self.schedule_persist()
            self.toast(f"{v.upper()} {pretty(k)}: back to auto"); self.guide_redraw()

        def g_reject(self, cid):
            v, k = self.guide_view, self.guide_key
            self.rejects.setdefault(v, {}).setdefault(k, []).append(cid)
            self.res.cands[v][k] = [c for c in self.res.cands[v][k] if c["cid"] != cid]
            was_lock = self.locks.get(v, {}).get(k) == cid
            if was_lock: self.locks[v].pop(k, None)
            if was_lock or self.res.views[v][k].cid == cid: self.g_auto()
            else: self.guide_redraw(); self.schedule_persist()
            self.toast("Candidate rejected (it stays hidden after re-cuts).")

        def g_unreject(self):
            v, k = self.guide_view, self.guide_key
            if self.rejects.get(v, {}).pop(k, None) is not None:
                self.toast("Rejected candidates return on the next CUT."); self.schedule_persist()
            else: self.toast("Nothing rejected for this part.")

        # ------------------------------------------------------------------ pose page
        def build_pose(self, page):
            right = tk.Frame(page, bg=BG1, width=560); right.pack(side="right", fill="y", padx=8, pady=8); right.pack_propagate(False)
            left = tk.Frame(page, bg=BG1); left.pack(side="left", fill="both", expand=True, padx=(8,0), pady=8)
            bar = tk.Frame(left, bg=BG1); bar.pack(fill="x", pady=(0,4))
            self.pose_title = tk.Label(bar, text="POSE LAB", bg=BG1, fg=TEXT, font=self.F["h"]); self.pose_title.pack(side="left")
            self.mk_btn(bar, "FIT", self.pose_fit, "t", pady=2, padx=8).pack(side="right")
            self.mk_btn(bar, "+", lambda: self.pose_zoom_by(1), "t", pady=2, padx=8).pack(side="right", padx=2)
            self.mk_btn(bar, "-", lambda: self.pose_zoom_by(-1), "t", pady=2, padx=8).pack(side="right")
            cv = tk.Canvas(left, bg="#05070b", highlightthickness=1, highlightbackground=LINE, cursor="hand2")
            cv.pack(fill="both", expand=True); self.pose_cv = cv
            cv.bind("<ButtonPress-1>", self.pose_press); cv.bind("<B1-Motion>", self.pose_motion)
            cv.bind("<ButtonRelease-1>", lambda e: setattr(self, "_pgrab", None))
            cv.bind("<Configure>", lambda e: self.after_idle(self.pose_refresh) if self.pose_fit_on else None)
            self.mk_label(left, "Drag a round handle to swing that part about its pivot  |  children follow their parent  |  "
                                "watch the joints for gaps: the overlap caps should hide them.", DIM, wraplength=760, justify="left").pack(anchor="w", pady=(4,0))
            p, f = self.mk_panel(right, "VIEW"); p.pack(fill="x", pady=(0,6))
            r = tk.Frame(f, bg=BG1); r.pack(fill="x")
            self.pose_vbtn = {}
            for v in VIEWS:
                b = self.mk_btn(r, v.upper(), lambda v=v: self.pose_reset(v), "n", pady=4); b.pack(side="left", expand=True, fill="x", padx=1); self.pose_vbtn[v] = b
            p, f = self.mk_panel(right, "PRESETS", PINK); p.pack(fill="x", pady=(0,6))
            g = tk.Frame(f, bg=BG1); g.pack(fill="x")
            for i,n in enumerate(POSE_PRESETS["front"]):
                b = self.mk_btn(g, n, lambda n=n: self.pose_preset(n), "n", pady=3, padx=4, font=self.F["card"])
                b.grid(row=i//3, column=i%3, sticky="ew", padx=1, pady=1); g.grid_columnconfigure(i%3, weight=1)
            p, f = self.mk_panel(right, "JOINT ANGLES   (degrees, clockwise +)"); p.pack(fill="both", expand=True, pady=(0,6))
            cols = tk.Frame(f, bg=BG1); cols.pack(fill="both", expand=True)
            groups = (("BODY", CENTER), ("RIGHT", [k for k in PART_KEYS if k.endswith("R")]), ("LEFT", [k for k in PART_KEYS if k.endswith("L")]))
            for gi,(gname, keys) in enumerate(groups):
                col = tk.Frame(cols, bg=BG1); col.grid(row=0, column=gi, sticky="n", padx=(0 if gi == 0 else 6, 0))
                self.mk_label(col, gname, ACC, self.F["cardb"]).pack(anchor="w")
                for k in keys:
                    row = tk.Frame(col, bg=BG1); row.pack(anchor="w", pady=1)
                    tk.Label(row, text=pretty(k), bg=BG1, fg=_hex(PART_COL[k]), font=self.F["card"], width=8, anchor="w").pack(side="left")
                    self.pose_vars[k] = tk.DoubleVar(value=0.0)
                    tk.Scale(row, variable=self.pose_vars[k], from_=-180, to=180, orient="horizontal", showvalue=False, length=84,
                             sliderlength=12, width=9, bg=BG1, troughcolor=BG0, highlightthickness=0, bd=0, activebackground=ACC,
                             command=lambda v, k=k: self.pose_slider(k)).pack(side="left")
                    l = tk.Label(row, text="0", bg=BG1, fg=MUTED, font=self.F["mono"], width=4, anchor="e"); l.pack(side="left"); self.pose_lbl[k] = l
            p, f = self.mk_panel(right, "SHOW"); p.pack(fill="x", pady=(0,6))
            r = tk.Frame(f, bg=BG1); r.pack(fill="x")
            self.mk_check(r, "Handles", self.v_handles, self.pose_soon).pack(side="left")
            self.mk_check(r, "Pivots", self.v_ppiv, self.pose_soon).pack(side="left", padx=6)
            self.mk_check(r, "Solid colours", self.v_psolid, self.pose_soon).pack(side="left")
            r = tk.Frame(right, bg=BG1); r.pack(fill="x")
            self.mk_btn(r, "RESET POSE", lambda: self.pose_reset(), "d", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(r, "◀ BEHIND", lambda: self.pose_layer(-1), "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(r, "FRONT ▶", lambda: self.pose_layer(1), "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(r, "MIRROR", self.pose_mirror, "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))
            self.mk_btn(r, "SAVE PNG", self.pose_save, "g", pady=5).pack(side="left", expand=True, fill="x")

        def pose_base(self, v):
            d = {k: 0.0 for k in PART_KEYS}; d.update({k: float(a) for k, a in rest_angles(v).items()}); return d
        def pose_reset(self, view=None):
            self.pose_view = view or self.pose_view; self.pose_ang = self.pose_base(self.pose_view)
            self.pose_sync(); self.pose_refresh()
        def pose_sync(self):
            self._pose_lock = True
            for k in PART_KEYS:
                self.pose_vars[k].set(self.pose_ang[k]); self.pose_lbl[k].config(text=f"{self.pose_ang[k]:.0f}")
            self._pose_lock = False
            for v,b in self.pose_vbtn.items(): self.set_active(b, v == self.pose_view)
        def pose_slider(self, k):
            if self._pose_lock: return
            self.pose_ang[k] = float(self.pose_vars[k].get()); self.pose_lbl[k].config(text=f"{self.pose_ang[k]:.0f}"); self.pose_soon()
        def pose_preset(self, name):
            v = self.pose_view
            tab = POSE_PRESETS["side" if v == "side" else "front"][name]
            if v == "back": tab = {swap_side(k): a for k, a in tab.items()}     # same picture, seen from behind
            self.pose_ang = self.pose_base(v); self.pose_ang.update({k: float(a) for k, a in tab.items()})
            self.pose_sync(); self.pose_refresh()
        def pose_mirror(self):
            if self.pose_view == "side": return self.toast("Mirror works for FRONT / BACK. The side view always faces right.", "warn")
            self.pose_ang = {swap_side(k): -a for k, a in self.pose_ang.items()}; self.pose_sync(); self.pose_refresh()
        def pose_soon(self, *_):
            if self._pose_pending: return
            self._pose_pending = True; self.after(16, self._pose_flush)
        def _pose_flush(self):
            self._pose_pending = False; self.pose_refresh()
        def pose_zoom_by(self, d):
            self.pose_fit_on = False; self.pose_zoom = float(np.clip(self.pose_zoom+0.5*d, 1, 12)); self.pose_refresh()
        def pose_fit(self): self.pose_fit_on = True; self.pose_refresh()
        def pose_refresh(self):
            cv = self.pose_cv
            if self.res is None:
                cv.delete("all"); cv.create_text(320, 220, text="CUT a rig first, then pose it here.", fill=DIM, font=self.F["h"]); return
            try: img, meta = render_pose(self.res, self.pose_view, self.pose_ang, solid=self.v_psolid.get(), crop=False, rest=False)
            except Exception as ex: self.log(f"pose render failed: {ex!r}", "warn"); return
            self.pose_meta = meta
            if self.pose_fit_on:
                cw, ch = max(300, cv.winfo_width()-24), max(300, cv.winfo_height()-24)
                self.pose_zoom = max(1.0, int(2*min(cw/img.width, ch/img.height))/2.0)
            z = self.pose_zoom
            self.pose_photo = ImageTk.PhotoImage(img.resize((int(img.width*z), int(img.height*z)), Image.Resampling.NEAREST))
            cv.delete("all")
            L = self.res.lengths; gy = 10+(0.56*img.height+(L["thigh"]+L["shin"])/self.res.cpp+4)*z
            cv.create_line(10, gy, 10+img.width*z, gy, fill=LINE2, dash=(4,4))
            cv.create_image(10, 10, image=self.pose_photo, anchor="nw")
            self.pose_title.config(text=f"POSE LAB  -  {self.pose_view.upper()}")
            if self.v_handles.get() or self.v_ppiv.get():
                for k in PART_KEYS:
                    P, E = meta["pivots_px"][k], meta["ends_px"][k]; col = _hex(PART_COL[k])
                    px, py, ex, ey = 10+P[0]*z, 10+P[1]*z, 10+E[0]*z, 10+E[1]*z
                    if self.v_handles.get():
                        cv.create_line(px, py, ex, ey, fill=col, width=1)
                        cv.create_oval(ex-6, ey-6, ex+6, ey+6, outline=col, width=2, fill=BG0, tags=("h:"+k,))
                    cv.create_oval(px-3, py-3, px+3, py+3, outline="#fff", fill=col)
        def pose_press(self, e):
            if self.pose_meta is None: return
            z, c = self.pose_zoom, self.pose_cv; mx, my = c.canvasx(e.x), c.canvasy(e.y); best, bd = None, 18
            for k in PART_KEYS:
                E = self.pose_meta["ends_px"][k]; d = math.hypot(10+E[0]*z-mx, 10+E[1]*z-my)
                if d < bd: best, bd = k, d
            self._pgrab = best; self._psel = best or self._psel
        def pose_motion(self, e):
            k = self._pgrab
            if k is None or self.pose_meta is None: return
            z, c, m = self.pose_zoom, self.pose_cv, self.pose_meta
            P = m["pivots_px"][k]; mx, my = c.canvasx(e.x), c.canvasy(e.y)
            phi = math.degrees(math.atan2(my-(10+P[1]*z), mx-(10+P[0]*z)))
            par = parent_key(k); par_th = m["th"][par] if par else 0.0
            loc = ((phi-m["d0"][k]-par_th+180) % 360)-180
            self.pose_ang[k] = loc; self._pose_lock = True; self.pose_vars[k].set(loc); self._pose_lock = False
            self.pose_lbl[k].config(text=f"{loc:.0f}"); self.pose_soon()
        def pose_save(self):
            if self.res is None: return self.toast("Nothing to pose yet - CUT first.", "warn")
            lo = self.ensure_loadout(); d = lo.path/"poses"; d.mkdir(parents=True, exist_ok=True)
            n = len(list(d.glob("pose_*_x4.png")))+1; stem = f"pose_{n:03d}_{self.pose_view}"
            img, _ = render_pose(self.res, self.pose_view, self.pose_ang, solid=self.v_psolid.get(), crop=True, rest=False)
            img.save(d/f"{stem}.png"); img.resize((img.width*4, img.height*4), Image.Resampling.NEAREST).save(d/f"{stem}_x4.png")
            with open(d/f"{stem}.json", "w") as f: json.dump(dict(view=self.pose_view, angles=self.pose_ang), f, indent=2)
            self.toast(f"Saved {stem}.png in {d}", "ok")

        # ------------------------------------------------------------------ log / help page
        def build_log(self, page):
            top = tk.Frame(page, bg=BG1); top.pack(fill="x", padx=10, pady=10)
            p, f = self.mk_panel(top, "NEEDS HELP  -  frames the detector could not read", WARN); p.pack(fill="x")
            self.fail_lb = self.mk_list(f, 5); self.fail_lb.pack(fill="x")
            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(5,0))
            self.mk_btn(r, "RESCUE AS MANUAL FRAME", self.rescue, "p", pady=4).pack(side="left")
            self.mk_label(r, "  adds a standing template pose - then drag the joints onto the body (JOINTS tool).", DIM).pack(side="left")
            p, f = self.mk_panel(page, "LOG"); p.pack(fill="both", expand=True, padx=10, pady=(0,10))
            self.logbox = tk.Text(f, bg=BG0, fg=TEXT, relief="flat", wrap="word", font=self.F["mono"], state="disabled",
                                  highlightthickness=1, highlightbackground=LINE, padx=8, pady=6)
            self.logbox.pack(fill="both", expand=True)
            for t,c in (("info",MUTED),("ok",GOOD),("warn",WARN),("err",BAD)): self.logbox.tag_config(t, foreground=c)

        def log(self, msg, kind="info"):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self.logbox.config(state="normal"); self.logbox.insert("end", f"{ts}  {msg}\n", kind)
            self.logbox.see("end"); self.logbox.config(state="disabled")
        def toast(self, msg, kind="info"):
            self.status.config(text=msg, fg={"info":ACC,"warn":WARN,"err":BAD,"ok":GOOD}.get(kind, ACC)); self.log(msg, kind)
        def set_progress(self, f):
            self.pb.delete("all"); self.pb.create_rectangle(0,0,int(260*max(0,min(1,f))),10, fill=ACC, outline="")

        # ------------------------------------------------------------------ roster / loadouts
        def cpp(self):
            try: return max(0.25, float(self.v_cpp.get()))
            except Exception: return CM_PER_PX
        def get_h(self):
            try: return max(40.0, float(self.v_h.get()))
            except Exception: return 172.0
        def refresh_roster(self):
            self.lb_lo.delete(0, "end")
            for n in list_loadouts(): self.lb_lo.insert("end", n)
        def ensure_loadout(self):
            if self.loadout is None:
                self.loadout = Loadout(self.v_name.get())
                if (self.loadout.path/"meta.json").exists(): self.loadout.load()
                self.v_name.set(self.loadout.name); self.e_name.config(state="readonly")
                self.chip_lo.config(text=f"LOADOUT  {self.loadout.name.upper()}", fg=ACC)
            self.sync_meta(); return self.loadout
        def sync_meta(self):
            m = self.loadout.meta
            m["height_cm"], m["cm_per_px"] = self.get_h(), self.cpp()
            m["organism"], m["sex"] = self.v_org.get(), self.v_sex.get()
            m["locks"], m["rejects"] = self.locks, self.rejects
            m["settings"] = dict(mirror=self.v_mirror.get(), fill=self.v_fill.get(), blend=self.v_blend.get(),
                                 colours=int(self.v_cols.get()), edge=int(self.v_athr.get()),
                                 overlap=self.v_overlap.get(), ovpct=int(self.v_ovpct.get()), synth=self.v_synth.get(),
                                 smooth=int(self.v_smooth.get()), learn=self.v_learn.get())
        def reset_session(self):
            self.frames, self.failed, self.res, self.cur = [], [], None, None
            self.sources = []; self.lb.delete(0,"end"); self.fl.delete(0,"end"); self.fail_lb.delete(0,"end")
            self.dirty = False; self.exported = False; self.ed_render(); self.card_cv.delete("all"); self.inv_cv.delete("all")
            for k in ("sheet","asm"): getattr(self, f"{k}_cv").delete("all")
        def new_loadout(self):
            name = simpledialog.askstring("New loadout", "Character name:", parent=self)
            if not name: return
            self.reset_session(); self.loadout = None; self.locks, self.rejects = {}, {}
            self.e_name.config(state="normal"); self.v_name.set(name.strip().replace(" ","_")); self.ensure_loadout()
            self.loadout.write_meta(); self.refresh_roster(); self.update_stage()
            self.toast(f"Created loadout '{self.loadout.name}'. Add source images.", "ok")
        def on_select_loadout(self):
            sel = self.lb_lo.curselection()
            if not sel: return
            name = self.lb_lo.get(sel[0]); lo = Loadout(name)
            if not lo.load(): return
            self.reset_session(); self.loadout = lo
            self.e_name.config(state="normal"); self.v_name.set(name); self.e_name.config(state="readonly")
            self.v_h.set(str(lo.meta.get("height_cm", 172.0))); self.v_cpp.set(str(lo.meta.get("cm_per_px", CM_PER_PX)))
            self.locks, self.rejects = lo.meta.get("locks", {}) or {}, lo.meta.get("rejects", {}) or {}
            self.v_org.set(lo.meta.get("organism", "human")); self.v_sex.set(lo.meta.get("sex", "male")); self.refresh_model(); self._align = lo.meta.get("align", {}) or {}
            self._body_load_from_meta(lo.meta)
            st = lo.meta.get("settings", {}) or {}
            self.v_mirror.set(st.get("mirror", True)); self.v_fill.set(st.get("fill", True)); self.v_blend.set(st.get("blend", True))
            self.v_cols.set(st.get("colours", 32)); self.v_athr.set(st.get("edge", 110))
            self.v_smooth.set(st.get("smooth", 2)); self.v_learn.set(st.get("learn", True))
            self.v_overlap.set(st.get("overlap", True)); self.v_ovpct.set(st.get("ovpct", 100)); self.v_synth.set(st.get("synth", True))
            self.sources = [p for p in lo.list_sources() if os.path.exists(p)]
            for p in self.sources: self.lb.insert("end", os.path.basename(p))
            self.chip_lo.config(text=f"LOADOUT  {name.upper()}", fg=ACC)
            if any(lo.chosen[v] for v in VIEWS):
                try:
                    self.res = lo.to_result(); self.res.name = name
                    self.toast(f"Loaded '{name}'. ANALYSE again to edit frames or see source context.", "ok")
                    self.show_page("sheet"); self.refresh_view_pages()
                except Exception as ex: self.log(f"could not rebuild '{name}': {ex!r}", "err")
            else: self.toast(f"Loaded '{name}' (nothing cut yet).")
            self.update_stage()
        def open_folder(self):
            if not self.loadout: return messagebox.showinfo("No loadout", "Create or open one first.", parent=self)
            path = self.loadout.path.resolve(); path.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32": os.startfile(path)
            elif sys.platform == "darwin": os.system(f'open "{path}"')
            else: os.system(f'xdg-open "{path}"')

        # ------------------------------------------------------------------ sources
        def add_files(self):
            d = getattr(self.app, "folder", None) or os.getcwd()
            ps = filedialog.askopenfilenames(parent=self, initialdir=d, title="Add images",
                                             filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif")])
            self.add(ps)
        def add_folder(self):
            d = filedialog.askdirectory(parent=self)
            if d: self.add(sorted(os.path.join(d,n) for n in os.listdir(d) if n.lower().endswith(IMG_EXT)))
        def add(self, paths):
            lo = self.ensure_loadout(); added = 0
            for p in paths:
                if p and str(p).lower().endswith(IMG_EXT):
                    lo.add_source(p); dest = str(lo.path/"sources"/Path(p).name)
                    if dest not in self.sources and p not in self.sources: self.sources.append(dest); added += 1
            self.lb.delete(0,"end")
            for p in self.sources: self.lb.insert("end", os.path.basename(p))
            self.refresh_roster(); self.update_stage()
            if added: self.toast(f"Added {added} source image(s).")
        def remove(self):
            lo = self.ensure_loadout()
            for i in reversed(self.lb.curselection()):
                lo.remove_source(self.sources[i]); del self.sources[i]
            self.lb.delete(0,"end")
            for p in self.sources: self.lb.insert("end", os.path.basename(p))

        # ------------------------------------------------------------------ analyse / cut
        def analyse(self):
            if self.busy: return
            if not self.sources: return self.toast("Add source images first.", "warn")
            lo = self.ensure_loadout(); H = self.get_h(); paths = list(self.sources); edits = lo.edits_dir; learn = self.v_learn.get()
            self.busy = True; self.set_progress(0); self.toast("Analysing...")
            def work():
                try:
                    if self.det is None:
                        try: self.det = make_detector()
                        except Exception as e:
                            self.q.put(("log", f"MediaPipe unavailable ({e!r}). Using template poses: fix the joints in the editor.", "warn"))
                            self.det = FallbackDetector()
                    fr, failed = analyse_sources(paths, H, self.det, lambda t: self.q.put(("log", t)),
                                                 lambda i,n: self.q.put(("prog", i, n)), edits, learn)
                    self.q.put(("analysed", fr, failed))
                except Exception as e: self.q.put(("err", repr(e), traceback.format_exc()))
            threading.Thread(target=work, daemon=True).start()


        # ------------------------------------------------------------------ body mask + scale
        def _body_sync_vars(self):
            """Push current mask/scale into the tk variables (and summary label)."""
            for g in RB.LIMB_GROUPS:
                self.v_body_group[g].set(RB.group_state(self.body_mask, g))
            for b in RB.SCALE_BASES:
                if b in self.v_body_scale:
                    self.v_body_scale[b].set(float(self.body_scale.get(b, 1.0)))
            if hasattr(self, "body_sum"):
                self.body_sum.config(text=RB.body_summary(self.body_mask, self.body_scale))

        def _body_toggle(self, group):
            on = bool(self.v_body_group[group].get())
            self.body_mask = RB.set_group(self.body_mask, group, on)
            self._body_sync_vars()
            self.dirty = True
            if hasattr(self, "ed_dirty"):
                self.ed_dirty.config(text="● BODY CHANGED – press CUT / UPDATE")
            self.update_stage()

        def _body_scale_changed(self, base):
            try:
                v = float(self.v_body_scale[base].get())
            except Exception:
                return
            self.body_scale[base] = RB.clamp_scale(v)
            if hasattr(self, "body_sum"):
                self.body_sum.config(text=RB.body_summary(self.body_mask, self.body_scale))
            self.dirty = True
            if hasattr(self, "ed_dirty"):
                self.ed_dirty.config(text="● BODY CHANGED – press CUT / UPDATE")

        def _body_reset(self):
            self.body_mask = RB.default_mask()
            self.body_scale = RB.default_scale()
            self._body_sync_vars()
            self.dirty = True
            if hasattr(self, "ed_dirty"):
                self.ed_dirty.config(text="● BODY RESET – press CUT / UPDATE")
            self.toast("Body restored to full human, normal proportions.", "ok")

        def _body_load_from_meta(self, meta):
            self.body_mask = RB.normalize_mask((meta or {}).get("mask"))
            self.body_scale = RB.normalize_scale((meta or {}).get("scale"))
            self._body_sync_vars()

        def cut(self):

            if self.busy: return
            frames = [f for f in self.frames if f.enabled]
            if not frames: return self.toast("Analyse first (and keep at least one frame switched on).", "warn")
            lo = self.ensure_loadout(); self.busy = True; self.set_progress(0); self.toast("Cutting...")
            args = dict(model=(self.model if self.v_usemodel.get() else None), learn_model=self.v_trainmodel.get(), owner=lo.name, fill_hidden=self.v_fillhidden.get(), align=(self._align or None), H=self.get_h(), cpp=self.cpp(), mirror=self.v_mirror.get(), gapfill=self.v_fill.get(),
                        ncolors=int(self.v_cols.get()), athr=int(self.v_athr.get()), blend=self.v_blend.get(),
                        locks={v:dict(d) for v,d in self.locks.items()}, rejects={v:{k:list(x) for k,x in d.items()} for v,d in self.rejects.items()})
            args.update(overlap=self.v_overlap.get(), overlap_scale=int(self.v_ovpct.get())/100.0, synth=self.v_synth.get(),
                        smooth=int(self.v_smooth.get()), learn=self.v_learn.get(),
                        mask=dict(self.body_mask), scale=dict(self.body_scale))
            def work():
                try:
                    r = cut_rig(frames, log=lambda t: self.q.put(("log", t)), progress=lambda i,n: self.q.put(("prog", i, n)), **args)
                    self.q.put(("cut", r))
                except Exception as e: self.q.put(("err", repr(e), traceback.format_exc()))
            threading.Thread(target=work, daemon=True).start()

        def _tick(self):
            try:
                while True:
                    m = self.q.get_nowait()
                    if m[0] == "prog": self.set_progress(m[1]/max(1, m[2]))
                    elif m[0] == "log":
                        self.status.config(text=m[1], fg=WARN if (len(m) > 2 and m[2] == "warn") else ACC)
                        self.log(m[1], m[2] if len(m) > 2 else "info")
                    elif m[0] == "analysed": self.on_analysed(m[1], m[2])
                    elif m[0] == "cut": self.on_cut(m[1])
                    elif m[0] == "err":
                        self.busy = False; self.set_progress(0); self.log(m[1] + "\n" + m[2], "err")
                        self.toast("Error: " + m[1], "err"); messagebox.showerror("Error", m[1], parent=self)
            except queue.Empty: pass
            except Exception: self.log(traceback.format_exc(), "err")
            try: self.after(50, self._tick)
            except Exception: pass

        def on_analysed(self, frames, failed):
            self.busy = False; self.set_progress(0)
            self.frames, self.failed = frames, failed
            self.refresh_fail_list(); self.refresh_frame_list()
            self.toast(f"{len(frames)} frame(s) ready, {len(failed)} need help." if frames else
                       "No usable frames. See LOG / HELP to rescue them manually.", "ok" if frames else "warn")
            if frames:
                best = max(range(len(frames)), key=lambda i: frames[i].front)
                self.fl.selection_clear(0, "end"); self.fl.selection_set(best); self.pick()
                for v in VIEWS:
                    if not any(f.facing == v for f in frames):
                        self.log(f"No {v.upper()} frame detected yet. Add one, or force a frame's view in the editor.", "warn")
            self.update_stage()

        def on_cut(self, res):
            self.busy = False; self.set_progress(0)
            try:
                lo = self.ensure_loadout(); res.name = lo.name; self.res = res; lo.chosen = res.chosen; self._align = res.align
                lo.save(res); self.last_saved = datetime.datetime.now().strftime("%H:%M:%S")
            except Exception: self.log("save failed:\n" + traceback.format_exc(), "err")
            self.dirty = False; self.ed_dirty.config(text=""); self.exported = False; self.refresh_model()
            miss = [v for v in VIEWS if res.counts.get(v, 0) == 0]
            for i in res.issues[:6]: self.log(i, "warn")
            n_b = sum(1 for v in VIEWS for sp in res.views[v].values() if sp.kind in ("borrowed", "synth"))
            n_p = sum(1 for v in VIEWS for sp in res.views[v].values() if sp.kind == "placeholder")
            self.toast(f"Rig cut: {res.counts['front']} front / {res.counts['side']} side / {res.counts['back']} back frames"
                       f"{'  -  ' + str(n_b) + ' synthesized' if n_b else ''}{'  -  ' + str(n_p) + ' MISSING' if n_p else ''}",
                       "warn" if (miss or n_p) else "ok")
            if self.cur is not None: self.ed_render()          # labels may have been refined by the colour model
            self.refresh_roster(); self.show_page("sheet"); self.refresh_view_pages(); self.update_stage()

        def schedule_persist(self):
            if self._persist_job: self.after_cancel(self._persist_job)
            self._persist_job = self.after(500, self._persist_now)
        def _persist_now(self):
            self._persist_job = None
            if self.res is None: return
            try:
                lo = self.ensure_loadout(); lo.chosen = self.res.chosen; lo.save(self.res); self.exported = False; self.last_saved = datetime.datetime.now().strftime("%H:%M:%S")
            except Exception: self.log("autosave failed:\n" + traceback.format_exc(), "err")

        def export(self):
            if self.res is None: return self.toast("Nothing to export yet - CUT first.", "warn")
            try:
                if self._persist_job: self.after_cancel(self._persist_job); self._persist_job = None
                self.align_learn(); lo = self.ensure_loadout(); lo.chosen = self.res.chosen; self.res.cands_dirty = True; lo.save(self.res)
            except Exception: return self.toast("Export failed - see LOG.", "err") or self.log(traceback.format_exc(), "err")
            self.exported = True; self.last_saved = datetime.datetime.now().strftime("%H:%M:%S"); self.update_stage()
            self.toast(f"Exported to {lo.path.resolve()}  (rig.png, atlas.png, rig.json, assembled_*.png, parts/)", "ok")

        # ------------------------------------------------------------------ frames list / rescue
        def fmt_frame(self, fd):
            return f"{'●' if fd.enabled else '○'} {fd.name[:15]:15s} {fd.facing[0].upper()}{'*' if fd.view_override else ' '} {fd.front_raw:+.2f}" \
                   f"{' ✎' if ((fd.manual > -2).any() or fd.extm.any() or fd.jover) else ''}"
        def refresh_frame_list(self, keep=None):
            self.fl.delete(0, "end")
            for i,fd in enumerate(self.frames):
                self.fl.insert("end", self.fmt_frame(fd))
                self.fl.itemconfig(i, fg=(VIEW_COL[fd.facing] if fd.enabled else DIM))
            if keep is not None and 0 <= keep < len(self.frames): self.fl.selection_set(keep)
        def pick(self):
            s = self.fl.curselection()
            if s and s[0] < len(self.frames):
                self.show_page("editor"); self.ed_set_frame(self.frames[s[0]])
        def cur_index(self):
            return self.frames.index(self.cur) if self.cur in self.frames else None
        def refresh_fail_list(self):
            self.fail_lb.delete(0, "end")
            for f in self.failed: self.fail_lb.insert("end", f"{f['tag'][:26]:26s} {f['reason']}")
        def rescue(self):
            s = self.fail_lb.curselection()
            if not s: return self.toast("Select a failed frame in the list first.", "warn")
            f = self.failed[s[0]]
            if f.get("im") is None: return self.toast("That file could not be read at all - nothing to rescue.", "warn")
            lo = self.ensure_loadout()
            fd = build_frame(f["tag"], f["im"], default_landmarks(f["im"]), self.get_h(), load_edits(lo.edits_dir, f["tag"]))
            if fd is None: return self.toast("Even the template pose does not fit this image (is the figure cut out?).", "err")
            fd.src = (f["tag"], f["im"], None); self.frames.append(fd); del self.failed[s[0]]
            self.refresh_fail_list(); self.refresh_frame_list(keep=len(self.frames)-1); self.set_tool("joints")
            self.show_page("editor"); self.ed_set_frame(fd)
            self.toast("Rescued. Drag the yellow joints onto the body (JOINTS tool), then CUT.", "ok")

        # ------------------------------------------------------------------ editor core
        def ed_set_frame(self, fd, fit=True):
            self.cur = fd
            if fd is None: self.ed_render(); return
            h, w = fd.alpha.shape; ys, xs = np.nonzero(fd.alpha > 127); m = int(round(8*fd.s))
            x0, x1 = max(0,int(xs.min())-m), min(w,int(xs.max())+m+1); y0, y1 = max(0,int(ys.min())-m), min(h,int(ys.max())+m+1)
            self.ed_box = (x0,y0,x1,y1)
            self.ed_base = np.dstack([fd.rgb[y0:y1,x0:x1], fd.alpha[y0:y1,x0:x1]])
            yy, xx = np.mgrid[0:y1-y0, 0:x1-x0]; chk = (((yy//8)+(xx//8)) % 2).astype(np.uint8)
            self.ed_bg = np.repeat(np.where(chk == 0, 16, 26).astype(np.uint8)[...,None], 3, axis=2)
            if fit: self.ed_fit()
            self.ed_render(); self.update_frame_panel()
        def ed_fit(self):
            if self.ed_box is None: return
            self.update_idletasks()
            x0,y0,x1,y1 = self.ed_box; cw, ch = max(320, self.ed_cv.winfo_width()), max(320, self.ed_cv.winfo_height())
            self.ed_zoom = float(np.clip(min((cw-2*M)/(x1-x0), (ch-2*M)/(y1-y0)), 0.3, 14))
        def ed_fit_render(self): self.ed_fit(); self.ed_render()
        def ed_zoom_by(self, f):
            self.ed_zoom = float(np.clip(self.ed_zoom*f, 0.3, 24)); self.ed_render()
        def ed_wheel(self, e):
            if self.cur is None: return
            up = getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0
            sx, sy = self.ed_src(e); f = 1.15 if up else 1/1.15
            self.ed_zoom = float(np.clip(self.ed_zoom*f, 0.3, 24)); self.ed_render()
            x0, y0, x1, y1 = self.ed_box; z = self.ed_zoom
            tw, th = (x1-x0)*z+2*M, (y1-y0)*z+2*M
            self.ed_cv.xview_moveto(max(0, (M+(sx-x0)*z-e.x)/tw)); self.ed_cv.yview_moveto(max(0, (M+(sy-y0)*z-e.y)/th))
        def ed_soon(self, *_):
            if self._ed_pending: return
            self._ed_pending = True; self.after(16, self._ed_flush)
        def _ed_flush(self):
            self._ed_pending = False; self.ed_render()
        def ed_src(self, e):
            c = self.ed_cv; x0, y0 = self.ed_box[0], self.ed_box[1]
            return x0 + (c.canvasx(e.x)-M)/self.ed_zoom, y0 + (c.canvasy(e.y)-M)/self.ed_zoom

        def ed_render(self):
            c = self.ed_cv; fd = self.cur
            if fd is None or self.ed_box is None:
                c.delete("all")
                c.create_text(300, 200, text="ANALYSE your sources, then pick a frame on the left.", fill=DIM, font=self.F["h"]); return
            x0,y0,x1,y1 = self.ed_box; a = self.ed_base[...,3] > 127
            out = self.ed_bg.astype(np.float32); out[a] = self.ed_base[...,:3][a]
            lab = fd.lab[y0:y1,x0:x1]
            if self.v_regions.get():
                op = self.v_tint.get()/100.0; tint = LUT[lab+1]
                out[a] = out[a]*(1-op) + tint[a]*op
                if self.v_borders.get():
                    e = np.zeros(lab.shape, bool); e[:,1:] |= lab[:,1:] != lab[:,:-1]; e[1:,:] |= lab[1:,:] != lab[:-1,:]; e &= a
                    out[e] = out[e]*0.35 + 255*0.65
            if self.v_tool.get() in ("extend", "unext"):           # show what the active part currently overlaps
                em = part_ext_mask(fd, self.v_part.get(), x0, y0, x1, y1, self.v_overlap.get(), self.v_ovpct.get()/100.0) & a
                out[em] = out[em]*0.35 + np.array(PART_COL[self.v_part.get()], np.float32)*0.65
            im = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)); z = self.ed_zoom
            W, H = max(1,int(im.width*z)), max(1,int(im.height*z))
            im = im.resize((W,H), Image.Resampling.NEAREST if z >= 1 else Image.Resampling.BILINEAR)
            self.ed_photo = ImageTk.PhotoImage(im)
            c.delete("img"); c.create_image(M, M, image=self.ed_photo, anchor="nw", tags=("img",)); c.tag_lower("img")
            c.config(scrollregion=(0, 0, W+2*M, H+2*M)); self.ed_overlay()
        def ed_overlay(self):
            c = self.ed_cv; c.delete("ov"); fd = self.cur
            if fd is None or not (self.v_skel.get() or self.v_tool.get() == "joints"): return
            x0, y0 = self.ed_box[0], self.ed_box[1]; z = self.ed_zoom
            P = lambda p: (M+(p[0]-x0)*z, M+(p[1]-y0)*z)
            for a,b,_ in BONES + [("Lsho","Rsho",""), ("Lhip","Rhip","")]:
                pa, pb = P(fd.J[a]), P(fd.J[b]); c.create_line(*pa, *pb, fill=ACC, width=2, tags=("ov",))
            for n,p in fd.J.items():
                q = P(p); edited = n in fd.jover
                c.create_oval(q[0]-4, q[1]-4, q[0]+4, q[1]+4, fill=(PINK if edited else GOLD), outline="#000", tags=("ov", "j:"+n))
            for k,p in fd.pivots.items():                          # the joint each part rotates about
                q = P(p); c.create_rectangle(q[0]-3, q[1]-3, q[0]+3, q[1]+3, fill=_hex(PART_COL[k]), outline="#fff", tags=("ov",))
            c.tag_raise("cur")
        def ed_cursor(self, e):
            c = self.ed_cv; fd = self.cur
            if fd is None or self.ed_box is None: return
            sx, sy = self.ed_src(e); ix, iy = int(sx), int(sy); h, w = fd.lab.shape
            if 0 <= ix < w and 0 <= iy < h:
                l = int(fd.lab[iy,ix]); self.ed_coord.config(text=f"{ix},{iy}  {PART_KEYS[l] if l >= 0 else '-'}")
            tool = self.v_tool.get(); c.delete("cur")
            if tool in ("brush","erase","restore","extend","unext"):
                r = max(1.0, self.v_brush.get()*fd.s/2)*self.ed_zoom; cx, cy = c.canvasx(e.x), c.canvasy(e.y)
                col = {"brush":_hex(PART_COL[self.v_part.get()]), "erase":BAD, "restore":GOOD, "extend":_hex(PART_COL[self.v_part.get()]), "unext":PINK}[tool]
                c.create_oval(cx-r, cy-r, cx+r, cy+r, outline=col, width=2, tags=("cur",))
            elif tool == "joints":
                n = self.ed_nearest_joint(e)
                if n:
                    cx, cy = c.canvasx(e.x), c.canvasy(e.y)
                    c.create_text(cx+12, cy-12, text=n, fill="#fff", anchor="w", font=self.F["cardb"], tags=("cur",))

        def ed_pan_start(self, e): self.ed_cv.scan_mark(e.x, e.y); self._pan = True
        def ed_pan_move(self, e): self.ed_cv.scan_dragto(e.x, e.y, gain=1)

        def ed_press(self, e, btn):
            fd = self.cur
            if fd is None or self.ed_box is None: return
            if btn == 1 and (e.state & 0x1): self.ed_cv.scan_mark(e.x, e.y); self._pan = True; return
            sx, sy = self.ed_src(e); tool = "erase" if btn == 3 else self.v_tool.get()
            self._stroke = None
            if tool == "joints":
                if btn == 1: self._grab = self.ed_nearest_joint(e)
                return
            if tool == "pick":
                ix, iy = int(sx), int(sy)
                if 0 <= iy < fd.lab.shape[0] and 0 <= ix < fd.lab.shape[1] and fd.lab[iy,ix] >= 0: self.set_part(PART_KEYS[int(fd.lab[iy,ix])])
                return
            self.ed_push_undo()
            if tool == "fill":
                self.ed_fill(sx, sy, self.active_pid() if btn == 1 else -1); self.ed_commit(); return
            pid = {"brush": self.active_pid(), "erase": -1, "restore": -2, "extend": self.active_pid(), "unext": self.active_pid()}[tool]
            self._stroke = dict(pid=pid, last=None, tool=tool); self.ed_paint_to(sx, sy)
        def ed_motion(self, e):
            if self._pan: self.ed_cv.scan_dragto(e.x, e.y, gain=1); return
            fd = self.cur
            if fd is None or self.ed_box is None: return
            sx, sy = self.ed_src(e)
            if self._grab:
                fd.J[self._grab] = np.array([sx, sy]); self.ed_overlay()
            elif self._stroke: self.ed_paint_to(sx, sy)
            self.ed_cursor(e)
        def ed_release(self, e):
            if self._pan: self._pan = False; return
            fd = self.cur
            if self._grab and fd is not None:
                n = self._grab; self._grab = None; h, w = fd.alpha.shape
                fd.jover[n] = [float(fd.J[n][0]/w), float(fd.J[n][1]/h)]; fd.V[n] = 1.0
                self.config(cursor="watch"); self.update_idletasks()
                try: refresh_frame(fd)
                finally: self.config(cursor="")
                self.ed_render(); self.ed_commit()
            elif self._stroke: self._stroke = None; self.ed_commit()
        def ed_nearest_joint(self, e):
            fd = self.cur; c = self.ed_cv; x0, y0 = self.ed_box[0], self.ed_box[1]; z = self.ed_zoom
            cx, cy = c.canvasx(e.x), c.canvasy(e.y); best, bd = None, 16
            for n,p in fd.J.items():
                d = math.hypot(M+(p[0]-x0)*z-cx, M+(p[1]-y0)*z-cy)
                if d < bd: best, bd = n, d
            return best

        def ed_paint_to(self, sx, sy):
            fd, st = self.cur, self._stroke; r = max(1.0, self.v_brush.get()*fd.s/2)
            if st["last"] is None: pts = [(sx, sy)]
            else:
                lx, ly = st["last"]; n = max(1, int(math.hypot(sx-lx, sy-ly)/(r*0.5)))
                pts = [(lx+(sx-lx)*i/n, ly+(sy-ly)*i/n) for i in range(1, n+1)]
            for p in pts: self._stamp(fd, p[0], p[1], r, st["pid"], st["tool"])
            st["last"] = (sx, sy); self.ed_soon()
        def _stamp(self, fd, sx, sy, r, pid, tool="brush"):
            h, w = fd.manual.shape
            x0, x1 = int(max(0, math.floor(sx-r))), int(min(w, math.ceil(sx+r)+1)); y0, y1 = int(max(0, math.floor(sy-r))), int(min(h, math.ceil(sy+r)+1))
            if x1 <= x0 or y1 <= y0: return
            yy, xx = np.ogrid[y0:y1, x0:x1]; m = (xx-sx)**2 + (yy-sy)**2 <= r*r
            if self.v_only.get() or tool in ("extend", "unext"): m &= fd.alpha[y0:y1, x0:x1] > 127
            if tool in ("extend", "unext"):                        # overlap layer: "this pixel also belongs to part pid"
                se = fd.extm[y0:y1, x0:x1]; bit = np.uint32(1 << pid)
                se[m] = (se[m] | bit) if tool == "extend" else (se[m] & ~bit)
                return
            sm, sl = fd.manual[y0:y1, x0:x1], fd.lab[y0:y1, x0:x1]
            if pid == -2: sm[m] = -2; sl[m] = fd.lab_auto[y0:y1, x0:x1][m]
            else: sm[m] = pid; sl[m] = pid
        def ed_fill(self, sx, sy, pid):
            fd = self.cur; h, w = fd.lab.shape; x, y = int(sx), int(sy)
            if not (0 <= x < w and 0 <= y < h): return
            only = self.v_only.get()
            if only and fd.alpha[y,x] <= 127: return
            mask = fd.lab == fd.lab[y,x]
            if only: mask &= fd.alpha > 127
            try:
                from scipy import ndimage
                lbl, _ = ndimage.label(mask); reg = lbl == lbl[y,x]
            except Exception:
                reg = np.zeros_like(mask); dq = deque([(x,y)]); reg[y,x] = True
                while dq:
                    px, py = dq.popleft()
                    for nx, ny in ((px+1,py),(px-1,py),(px,py+1),(px,py-1)):
                        if 0 <= nx < w and 0 <= ny < h and mask[ny,nx] and not reg[ny,nx]: reg[ny,nx] = True; dq.append((nx,ny))
            fd.manual[reg] = pid; fd.lab[reg] = pid; self.ed_render()
        def ed_push_undo(self):
            fd = self.cur; fd.undo.append((fd.manual.copy(), fd.extm.copy())); del fd.undo[:-25]; fd.redo.clear()
        def ed_undo(self):
            fd = self.cur
            if fd is None or not fd.undo: return
            fd.redo.append((fd.manual.copy(), fd.extm.copy())); fd.manual, fd.extm = fd.undo.pop(); merge_manual(fd); self.ed_render(); self.ed_commit()
        def ed_redo(self):
            fd = self.cur
            if fd is None or not fd.redo: return
            fd.undo.append((fd.manual.copy(), fd.extm.copy())); fd.manual, fd.extm = fd.redo.pop(); merge_manual(fd); self.ed_render(); self.ed_commit()
        def ed_commit(self):
            fd = self.cur
            if fd is None: return
            self.dirty = True; self.ed_dirty.config(text="● EDITS NOT CUT YET - press CUT / UPDATE")
            try: save_edits(self.ensure_loadout().edits_dir, fd)
            except Exception as ex: self.log(f"could not save edits: {ex}", "warn")
            self.update_frame_panel(); i = self.cur_index()
            if i is not None:
                self.fl.delete(i); self.fl.insert(i, self.fmt_frame(fd)); self.fl.itemconfig(i, fg=(VIEW_COL[fd.facing] if fd.enabled else DIM))
                self.fl.selection_clear(0, "end"); self.fl.selection_set(i)
            self.update_stage()

        # frame-level controls
        def update_frame_panel(self):
            fd = self.cur
            if fd is None: self.frame_info.config(text="No frame selected."); self.ed_title.config(text="no frame - ANALYSE first"); return
            self.ed_title.config(text=fd.name, fg=VIEW_COL[fd.facing])
            for v,b in self.view_btns.items(): self.set_active(b, fd.view_override == v)
            self.v_enabled.set(fd.enabled)
            self.frame_info.config(text=f"detected   {fd.facing_auto.upper()}  (raw {fd.front_raw:+.2f})\nused as    {fd.facing.upper()}"
                f"{'  (forced)' if fd.view_override else ''}\nfaces      {'RIGHT' if fd.face_dir > 0 else 'LEFT'}"
                f"{'  (flipped)' if fd.flip else ''}\n{('near limbs ' + fd.near + ('   hidden: ' + ','.join(sorted({k[:-1] for k in fd.hidden})) if fd.hidden else '')) if fd.near else ''}\nscale      {fd.s_orig:.2f} px/cm\npainted    {int((fd.manual > -2).sum())} px\n"
                f"joints     {len(fd.jover)} moved\n\nSide frames are flipped to face right automatically; use FLIP FACING if the arrow is wrong.")
        def set_view_override(self, v):
            fd = self.cur
            if fd is None: return
            fd.view_override = v; self.busy_refresh(fd)
        def toggle_flip(self):
            fd = self.cur
            if fd is None: return
            fd.flip = not fd.flip; self.busy_refresh(fd)
        def toggle_enabled(self):
            fd = self.cur
            if fd is None: return
            fd.enabled = bool(self.v_enabled.get()); self.ed_commit()
        def busy_refresh(self, fd):
            self.config(cursor="watch"); self.update_idletasks()
            try: refresh_frame(fd)
            finally: self.config(cursor="")
            self.ed_render(); self.ed_commit()
        def clear_paint(self):
            fd = self.cur
            if fd is None: return
            self.ed_push_undo(); fd.manual[:] = -2; fd.extm[:] = 0; merge_manual(fd); self.ed_render(); self.ed_commit()
        def reset_joints(self):
            fd = self.cur
            if fd is None: return
            h, w = fd.alpha.shape; J, V = make_joints(fd.il, w, h); fd.J, fd.V, fd.jover = J, V, {}
            self.busy_refresh(fd)

        # ------------------------------------------------------------------ stage tracker
        def update_stage(self):
            edited = any(((f.manual > -2).any() or f.extm.any() or f.jover or f.view_override) for f in self.frames)
            locks = sum(len(d) for d in self.locks.values())
            done = [bool(self.frames) or self.res is not None, edited, self.res is not None and not self.dirty, locks > 0, self.exported]
            cur = next((i for i in (0,2,3,4) if not done[i]), None)
            for i,l in enumerate(self.stage_lbl):
                if i == cur: l.config(bg=ACC, fg="#001a20", text=f"{i+1}  {STAGE[i]}")
                elif done[i]: l.config(bg=BG2, fg=GOOD, text=f"✓  {STAGE[i]}")
                else: l.config(bg=BG2, fg=DIM, text=f"{i+1}  {STAGE[i]}")
            self.update_coach(); self.refresh_save_chip()

    RGX.install(RigWindow, globals(), dict(BG0=BG0, BG1=BG1, BG2=BG2, BG3=BG3, LINE=LINE, LINE2=LINE2, TEXT=TEXT, MUTED=MUTED, DIM=DIM,
                                         ACC=ACC, PINK=PINK, GOLD=GOLD, GOOD=GOOD, WARN=WARN, BAD=BAD), tk, ttk, ImageTk, filedialog, messagebox)
    _GUI_CACHE[standalone] = RigWindow
    return RigWindow

def __getattr__(name):
    if name == "RigWindow": return _build_gui_class()
    raise AttributeError(name)

def main():
    _build_gui_class(standalone=True)().mainloop()

if __name__ == "__main__":
    main()
