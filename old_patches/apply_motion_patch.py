"""Patch 3 for HUMAN.RIG: joint ANCHORS (parts attach where they sat in the source image), ALIGN page,
ANIMATE page (actions + secondary motion + export), learned alignment.
Run AFTER apply_model_patch.py and apply_ux_patch.py.  Needs next to your rig file:
    organism_model.py (updated)  rig_tools.py  rig_pose.py  rig_motion.py  rig_gui_extra.py
usage:  python apply_motion_patch.py human_rig.py"""
import sys, shutil
E = []
def edit(old, new): E.append((old, new))

edit("import rig_tools as RT", "import rig_tools as RT\nimport rig_pose as RP\nimport rig_motion as RM\nimport rig_gui_extra as RGX")

# ---- renderer v2 (anchors + manual alignment + root motion) -------------------------------------------------------------
edit("def render_pose(res, view, angles=None, solid=False, shade=True, crop=True, rest=True):",
     "def _render_pose_v1(res, view, angles=None, solid=False, shade=True, crop=True, rest=True):")
edit('def assemble(res, solid=False, view="front", shade=True):',
     'def render_pose(res, view, angles=None, solid=False, shade=True, crop=True, rest=True, root=None):\n'
     '    return RP.render_pose(globals(), res, view, angles, solid, shade, crop, rest, root)\n\n'
     'def assemble(res, solid=False, view="front", shade=True):')

# ---- anchors: where every child joint sits inside its parent's sprite ---------------------------------------------------
edit("    return out, pa, pb, to_sprite(fd.pivots[k]), to_sprite(fd.ends[k])",
     "    return out, pa, pb, to_sprite(fd.pivots[k]), to_sprite(fd.ends[k]), {c: to_sprite(fd.pivots[c]) for c in PART_KEYS if parent_key(c) == k}")
edit("img, pa, pb, pj, pe = r", "img, pa, pb, pj, pe, anc = r")
edit("pj=pj, pe=pe, fname=fd.name, src_key=k,", "pj=pj, pe=pe, anc=anc, fname=fd.name, src_key=k,")
edit('c["cid"] = make_cid(fd.name, k, flip, False)',
     'c["anc"] = {_tkey(ck, view, fd.near): v for ck, v in c["anc"].items()}\n                c["cid"] = make_cid(fd.name, k, flip, False)')
edit('t["cid"] = make_cid(c["fname"], c["src_key"], t["flip"], True)',
     't["anc"] = {swap_side(ck): v for ck, v in (t.get("anc") or {}).items()}\n                    t["cid"] = make_cid(c["fname"], c["src_key"], t["flip"], True)')
edit("def _flip_pts(c, W):",
     'def _tkey(ck, view, near):\n    """target key of a part in a side view (R = near limb)"""\n'
     '    if view == "side" and ck not in CENTER: return base_of(ck) + ("R" if side_of(ck) == near else "L")\n    return ck\n\n'
     "def _flip_pts(c, W):")
edit("p = np.asarray(c[key], float); c[key] = np.array([W-p[0], p[1]])",
     "p = np.asarray(c[key], float); c[key] = np.array([W-p[0], p[1]])\n"
     '    if c.get("anc"): c["anc"] = {ck: np.array([W-p[0], p[1]]) for ck, p in c["anc"].items()}')
edit("sp.length_cm = round(float(L[b]) if b in STRETCH else float(np.linalg.norm(sp.pb-sp.pa)*cpp), 1)",
     "sp.length_cm = round(float(L[b]) if b in STRETCH else float(np.linalg.norm(sp.pb-sp.pa)*cpp), 1)\n"
     '    sp.anc = {ck: np.asarray(p, float)-off for ck, p in (c.get("anc") or {}).items()}')
edit("p = getattr(sp, a); setattr(n, a, np.array([p[0]*sx, p[1]*sy]))",
     "p = getattr(sp, a); setattr(n, a, np.array([p[0]*sx, p[1]*sy]))\n"
     '    n.anc = {ck: np.array([p[0]*sx, p[1]*sy]) for ck, p in (getattr(sp, "anc", None) or {}).items()}')

# ---- persistence ----------------------------------------------------------------------------------------------------------
edit("pivot_j=[round(float(x),2) for x in sp.pj], pivot_e=[round(float(x),2) for x in sp.pe],",
     "pivot_j=[round(float(x),2) for x in sp.pj], pivot_e=[round(float(x),2) for x in sp.pe],\n"
     '                        anchors={ck: [round(float(x),2) for x in p] for ck, p in (getattr(sp, "anc", None) or {}).items()},')
edit('self.meta["palette"] = res.meta["palette"]',
     'self.meta["palette"] = res.meta["palette"]\n            self.meta["align"] = getattr(res, "align", None) or {}')
edit("hierarchy={k: parent_key(k) for k in PART_KEYS}, rotation=\"degrees, clockwise positive, about each part's pivot\")",
     "hierarchy={k: parent_key(k) for k in PART_KEYS}, alignment=getattr(res, \"align\", None) or {}, rotation=\"degrees, clockwise positive, about each part's pivot\")")
edit('sp.mirrored = sp.flip or sp.twin; sp.nused = 1; sp.borrowed = ""',
     'sp.mirrored = sp.flip or sp.twin; sp.nused = 1; sp.borrowed = ""; sp.anc = {ck: np.array(p, float) for ck, p in (info.get("anchors") or {}).items()}')
edit('pj=[float(x) for x in c["pj"]], pe=[float(x) for x in c["pe"]]))',
     'pj=[float(x) for x in c["pj"]], pe=[float(x) for x in c["pe"]],\n'
     '                        anc={ck: [float(x) for x in p] for ck, p in (c.get("anc") or {}).items()}))')
edit('pj=np.array(e.get("pj", e["pa"])), pe=np.array(e.get("pe", e["pb"]))))',
     'pj=np.array(e.get("pj", e["pa"])), pe=np.array(e.get("pe", e["pb"])),\n'
     '                            anc={ck: np.array(p, float) for ck, p in (e.get("anc") or {}).items()}))')
edit("res.cands = self.load_cands()", 'res.cands = self.load_cands()\n        res.align = self.meta.get("align", {}) or {}')
edit('model=None, learn_model=True, owner="", fill_hidden=True,', 'model=None, learn_model=True, owner="", fill_hidden=True, align=None,')
edit('    res.model_tag = (f"{model.organism}/{model.sex}" if model is not None else "")',
     '    res.model_tag = (f"{model.organism}/{model.sex}" if model is not None else "")\n'
     '    res.align = align if align else (model.align_prior() if model is not None else {})')

# ---- GUI ----------------------------------------------------------------------------------------------------------------------
edit("self.model = None; self._psel = None", "self.model = None; self._psel = None; self._align = {}")
edit('("guide","PICK PARTS"),("log","LOG / HELP")):', '("guide","PICK PARTS"),("align","ALIGN"),("anim","ANIMATE"),("log","LOG / HELP")):')
edit('self.build_guide(self.pages["guide"]); self.build_log(self.pages["log"])',
     'self.build_guide(self.pages["guide"]); self.build_log(self.pages["log"])\n'
     '            self.build_align(self.pages["align"]); self.build_anim(self.pages["anim"])')
edit('if key == "pose": self.after(30, self.pose_refresh)',
     'if key == "pose": self.after(30, self.pose_refresh)\n'
     '            if key == "align": self.after(30, self.align_refresh)\n'
     '            if key == "anim": self.after(30, self.anim_refresh)')
edit('if getattr(self, "cur_page", "") == "guide":',
     'if getattr(self, "cur_page", "") == "align" and self.align_key(e): return\n'
     '            if getattr(self, "cur_page", "") == "guide":')
edit('self.v_org.set(lo.meta.get("organism", "human")); self.v_sex.set(lo.meta.get("sex", "male")); self.refresh_model()',
     'self.v_org.set(lo.meta.get("organism", "human")); self.v_sex.set(lo.meta.get("sex", "male")); self.refresh_model(); self._align = lo.meta.get("align", {}) or {}')
edit("lo = self.ensure_loadout(); res.name = lo.name; self.res = res; lo.chosen = res.chosen",
     "lo = self.ensure_loadout(); res.name = lo.name; self.res = res; lo.chosen = res.chosen; self._align = res.align")
edit("owner=lo.name, fill_hidden=self.v_fillhidden.get(), H=self.get_h(),",
     "owner=lo.name, fill_hidden=self.v_fillhidden.get(), align=(self._align or None), H=self.get_h(),")
edit("lo = self.ensure_loadout(); lo.chosen = self.res.chosen; self.res.cands_dirty = True; lo.save(self.res)",
     "self.align_learn(); lo = self.ensure_loadout(); lo.chosen = self.res.chosen; self.res.cands_dirty = True; lo.save(self.res)")
edit("_GUI_CACHE[standalone] = RigWindow",
     "RGX.install(RigWindow, globals(), dict(BG0=BG0, BG1=BG1, BG2=BG2, BG3=BG3, LINE=LINE, LINE2=LINE2, TEXT=TEXT, MUTED=MUTED, DIM=DIM,\n"
     "                                         ACC=ACC, PINK=PINK, GOLD=GOLD, GOOD=GOOD, WARN=WARN, BAD=BAD), tk, ttk, ImageTk, filedialog, messagebox)\n"
     "    _GUI_CACHE[standalone] = RigWindow")

def main(path):
    src = open(path, encoding="utf-8").read()
    bad = [(i, o[:80]) for i, (o, _) in enumerate(E, 1) if src.count(o) != 1]
    if bad:
        for i, o in bad: print(f"anchor #{i} matched {src.count(E[i-1][0])}x (need exactly 1): {o!r}")
        sys.exit("Nothing written. Were the two earlier patches applied, and is this the right file?")
    shutil.copy2(path, path + ".before_motion.bak")
    spans = sorted(((src.index(o), len(o), n) for o, n in E), reverse=True)
    for i, ln, n in spans: src = src[:i] + n + src[i+ln:]
    open(path, "w", encoding="utf-8").write(src)
    print(f"patched {path} ({len(E)} edits). Backup: {path}.before_motion.bak")

if __name__ == "__main__":
    if len(sys.argv) != 2: sys.exit(__doc__)
    main(sys.argv[1])
