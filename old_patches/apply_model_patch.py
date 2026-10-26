"""Wires organism_model.py into your HUMAN.RIG file.
usage:  python apply_model_patch.py path/to/your_rig_file.py
Every anchor must match exactly once or NOTHING is written.  A .bak copy is made first."""
import sys, shutil

E = []   # (anchor, replacement)
def edit(old, new): E.append((old, new))

# 1 ---- import + global glue ------------------------------------------------------------------------------
edit("from PIL import Image, ImageDraw", "from PIL import Image, ImageDraw\nimport organism_model as OM")

edit("ADJ = _adj()", '''ADJ = _adj()

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
def draw_order(view):
    d = POSE_ORDER[view]
    return list(d) if ACTIVE_MODEL is None else ACTIVE_MODEL.draw_order(view, d)
def relearn_frame(fd):
    """Recompute the labels of one frame after the model changed (cheap: no re-measuring)."""
    fd.lab_learn = ACTIVE_MODEL.relabel(fd) if ACTIVE_MODEL else fd.lab_geo.copy()
    fd.lab_auto = fd.lab_learn.copy()
    cm = getattr(fd, "cmodel", None)
    if cm is not None: apply_color_model(fd, cm)
    merge_manual(fd)''')

# 2 ---- learned joint-cap sizes -------------------------------------------------------------------------------
edit("    rp = CAP_P[b]*s*scale", '    rp = _cap("p", b, CAP_P[b])*s*scale')
edit("out.append((fd.pivots[c], CAPD[base_of(c)]*s*scale, [PART_KEYS.index(c)]))",
     'out.append((fd.pivots[c], _cap("d", base_of(c), CAPD[base_of(c)])*s*scale, [PART_KEYS.index(c)]))')

# 3 ---- build_frame: remember the raw detection, apply the learned detector bias ---------------------------------
edit("    J,V = make_joints(il,w,h)", "    J,V = make_joints(il,w,h)\n    Jdet = {k: v.copy() for k, v in J.items()}")
edit('    if s is None or min(V["neck"],V["pelvis"])<.4: return None',
'''    if s is None or min(V["neck"],V["pelvis"])<.4: return None
    if ACTIVE_MODEL is not None:
        _vw = edits.get("view") if edits.get("view") in VIEWS else facing_label(front_raw(J, V, H, s))
        ACTIVE_MODEL.adjust_joints(J, V, _vw, face_direction(J, V, H, s), protect=set(jover))
        s = estimate_scale(J, V, H) or s''')
edit("        J = {k:v*rescale for k,v in J.items()}",
     "        J = {k:v*rescale for k,v in J.items()}\n        Jdet = {k: v*rescale for k, v in Jdet.items()}")
edit("    fd.il = il; fd.jover = jover", "    fd.il = il; fd.jover = jover; fd.J_det = Jdet")

# 4 ---- refresh_frame: learned proportions prior + learned labels ---------------------------------------------------
edit("    lab0 = label_pixels(fd.alpha, make_caps(J, s, H, V, fd.near, DEF_PROPS))",
     "    prior = ACTIVE_MODEL.props_prior() if ACTIVE_MODEL else dict(DEF_PROPS)\n"
     "    lab0 = label_pixels(fd.alpha, make_caps(J, s, H, V, fd.near, prior))")
edit("    props = dict(DEF_PROPS); props.update(measure_props(fd.alpha, lab0, J, s, V))",
     "    fd.props_measured = measure_props(fd.alpha, lab0, J, s, V)\n    props = dict(prior); props.update(fd.props_measured)")
edit('    cm = getattr(fd, "cmodel", None)',
     '    fd.lab_learn = ACTIVE_MODEL.relabel(fd) if ACTIVE_MODEL else fd.lab_geo.copy()   # learned correction of the geometry\n'
     '    fd.lab_auto = fd.lab_learn.copy()\n'
     '    cm = getattr(fd, "cmodel", None)')

# 5 ---- colour model starts from the learned labels ------------------------------------------------------------------
edit("        lab = np.where(fd.manual > -2, fd.manual, fd.lab_geo)", "        lab = np.where(fd.manual > -2, fd.manual, fd.lab_learn)")
edit("fd.cmodel = model; fd.lab_auto = fd.lab_geo.copy(); moved += apply_color_model(fd, model); merge_manual(fd)",
     "fd.cmodel = model; fd.lab_auto = fd.lab_learn.copy(); moved += apply_color_model(fd, model); merge_manual(fd)")
edit("fd.cmodel = None; fd.lab_auto = fd.lab_geo.copy(); merge_manual(fd)",
     "fd.cmodel = None; fd.lab_auto = fd.lab_learn.copy(); merge_manual(fd)")

# 6 ---- cut_rig: learn first, cut with the improved model, learn cross-view ratios ----------------------------------------
edit("blend=True, locks=None, rejects=None, overlap=True, overlap_scale=1.0, synth=True, smooth=2, learn=True,",
     'blend=True, locks=None, rejects=None, overlap=True, overlap_scale=1.0, synth=True, smooth=2, learn=True,\n            model=None, learn_model=True, owner="",')
edit('    if not frames: raise ValueError("No enabled frames to cut.")',
'''    if not frames: raise ValueError("No enabled frames to cut.")
    if model is not None:
        set_active_model(model)
        if learn_model:
            try:
                model.train(frames, log, owner=owner)
                for fd in frames: relearn_frame(fd)
            except Exception as e: log(f"model training skipped ({e!r})")''')
edit("    res.smooth, res.props = smooth, props",
'''    res.smooth, res.props = smooth, props
    if model is not None and learn_model:
        try: model.train_crossview(res, log)
        except Exception as e: log(f"cross-view learning skipped ({e!r})")''')

# 7 ---- synthesised views use the learned width ratios -------------------------------------------------------------------
edit("n = _scale_sprite(sp, min(2.2, 1/SIDE_FX[b]), 1.0)", "n = _scale_sprite(sp, _xv(ov, v, b, min(2.2, 1/SIDE_FX[b])), 1.0)")
edit('n = _scale_sprite(sp, SIDE_FX[b], 0.85 if b == "foot" else 1.0)', 'n = _scale_sprite(sp, _xv(ov, v, b, SIDE_FX[b]), 0.85 if b == "foot" else 1.0)')

# 8 ---- learned layering ----------------------------------------------------------------------------------------------------
edit("    for k in POSE_ORDER[view]:", "    for k in draw_order(view):")
edit("draw_order=POSE_ORDER[view]", "draw_order=draw_order(view)")

# 9 ---- GUI -------------------------------------------------------------------------------------------------------------------
edit("self.v_handles = tk.BooleanVar(value=True); self.v_ppiv = tk.BooleanVar(value=False)",
     "self.v_handles = tk.BooleanVar(value=True); self.v_ppiv = tk.BooleanVar(value=False)\n"
     '            self.v_org = tk.StringVar(value="human"); self.v_sex = tk.StringVar(value="male")\n'
     "            self.v_usemodel = tk.BooleanVar(value=True); self.v_trainmodel = tk.BooleanVar(value=True)\n"
     "            self.model = None; self._psel = None")
edit('self.bind("<KeyPress>", self.on_key)', 'self.refresh_model()\n            self.bind("<KeyPress>", self.on_key)')
edit("        def style_ttk(self):", '''        def refresh_model(self, *_):
            try:
                self.model = REGISTRY.get(self.v_org.get(), self.v_sex.get())
                set_active_model(self.model if self.v_usemodel.get() else None)
                self.model_lbl.config(text=self.model.summary() + ("" if self.v_usemodel.get() else "\\n(model OFF)"))
            except Exception as ex: self.log(f"model load failed: {ex!r}", "err")
        def model_export(self):
            if self.model is None: return
            p = filedialog.asksaveasfilename(parent=self, defaultextension=".zip", initialfile=f"humanrig_model_{self.model.organism}_{self.model.sex}.zip")
            if p: self.toast(f"Model exported to {self.model.export_zip(p)}", "ok")
        def model_reset(self):
            if self.model is None: return
            if messagebox.askyesno("Reset model", f"Forget everything learned for {self.model.organism}/{self.model.sex}?\\n(a backup is kept in the model folder)", parent=self):
                self.model.reset(); self.refresh_model(); self.toast("Model reset (backup kept).", "warn")
        def pose_layer(self, d):
            k = getattr(self, "_psel", None)
            if not k or self.model is None: return self.toast("Drag/click a part handle first, then use the layer buttons.", "warn")
            self.model.layer_nudge(self.pose_view, k, d, POSE_ORDER[self.pose_view]); self.model.save()
            if self.res is not None: render_outputs(self.res)
            self.pose_refresh(); self.toast(f"{pretty(k)} moved {'forward' if d > 0 else 'back'} in the {self.pose_view.upper()} view (learned).", "ok")
        def style_ttk(self):''')
edit('a = tk.Frame(L, bg=BG0); a.pack(fill="x")', '''p, f = self.mk_panel(L, "BODY MODEL  (permanent, learns from your edits)", GOLD); p.pack(fill="x", pady=(0,6))
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
            a = tk.Frame(L, bg=BG0); a.pack(fill="x")''')
edit('m["height_cm"], m["cm_per_px"] = self.get_h(), self.cpp()',
     'm["height_cm"], m["cm_per_px"] = self.get_h(), self.cpp()\n            m["organism"], m["sex"] = self.v_org.get(), self.v_sex.get()')
edit('self.locks, self.rejects = lo.meta.get("locks", {}) or {}, lo.meta.get("rejects", {}) or {}',
     'self.locks, self.rejects = lo.meta.get("locks", {}) or {}, lo.meta.get("rejects", {}) or {}\n'
     '            self.v_org.set(lo.meta.get("organism", "human")); self.v_sex.set(lo.meta.get("sex", "male")); self.refresh_model()')
edit("args = dict(H=self.get_h(), cpp=self.cpp(),",
     "args = dict(model=(self.model if self.v_usemodel.get() else None), learn_model=self.v_trainmodel.get(), owner=lo.name, H=self.get_h(), cpp=self.cpp(),")
edit('self.dirty = False; self.ed_dirty.config(text=""); self.exported = False',
     'self.dirty = False; self.ed_dirty.config(text=""); self.exported = False; self.refresh_model()')
edit("            self._pgrab = best", "            self._pgrab = best; self._psel = best or self._psel")
edit('self.mk_btn(r, "RESET POSE", lambda: self.pose_reset(), "d", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))',
     'self.mk_btn(r, "RESET POSE", lambda: self.pose_reset(), "d", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))\n'
     '            self.mk_btn(r, "◀ BEHIND", lambda: self.pose_layer(-1), "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))\n'
     '            self.mk_btn(r, "FRONT ▶", lambda: self.pose_layer(1), "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))')

def main(path):
    src = open(path, encoding="utf-8").read()
    bad = [(i, o[:70]) for i, (o, _) in enumerate(E, 1) if src.count(o) != 1]
    if bad:
        for i, o in bad: print(f"anchor #{i} matched {src.count(E[i-1][0])}x (need exactly 1): {o!r}")
        sys.exit("Nothing written. Your file differs from the version this patch was made for.")
    shutil.copy2(path, path + ".bak")
    spans = sorted(((src.index(o), len(o), n) for o, n in E), reverse=True)   # splice on the ORIGINAL text
    for i, ln, n in spans: src = src[:i] + n + src[i+ln:]
    open(path, "w", encoding="utf-8").write(src)
    print(f"patched {path} ({len(E)} edits). Backup: {path}.bak. Keep organism_model.py next to it.")

if __name__ == "__main__":
    if len(sys.argv) != 2: sys.exit(__doc__)
    main(sys.argv[1])
