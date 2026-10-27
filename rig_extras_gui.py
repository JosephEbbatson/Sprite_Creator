"""rig_extras_gui.py - the EXTRAS page for the HUMAN.RIG window (installed into RigWindow by install()).

Three steps, three canvas modes:
  1  CUT     load a source (one of the rig's own parts, or any image file), draw a lasso around the bit you want,
             press MAKE PART.  Use the whole picture if you want a plain copy.
  2  PIVOT   drag the dot to where the part should rotate (the base of a tail, the root of an ear ...).
  3  PLACE   drag the part on the body to where it belongs, per view.
Then give it a size, a layer, a parent and a MOTION (rigid / ear / tail / wing / cape ...).  Watch it move on ANIMATE.
"""
import math
import numpy as np
from PIL import Image
import rig_extras as RX

VIEWS = ("front", "side", "back")
M = 14                                                    # canvas margin


def install(cls, G, K, tk, ttk, ImageTk, filedialog, messagebox):
    BG0, BG1, BG2, BG3, LINE, LINE2 = K["BG0"], K["BG1"], K["BG2"], K["BG3"], K["LINE"], K["LINE2"]
    TEXT, MUTED, DIM, ACC, PINK, GOLD, GOOD, WARN, BAD = (K[n] for n in ("TEXT", "MUTED", "DIM", "ACC", "PINK", "GOLD", "GOOD", "WARN", "BAD"))
    PART_KEYS = G["PART_KEYS"]
    PRESETS = list(RX.MOTION_PRESETS.keys()); CUSTOM = "CUSTOM"
    LAYER_NAMES = [RX.LAYER_LABEL[k] for k in RX.LAYER_KEYS]
    PIVOTS = ("center", "top", "bottom", "left", "right")

    def add(fn):
        setattr(cls, fn.__name__, fn); return fn

    # ------------------------------------------------------------------ small helpers
    @add
    def _ex_list(self):
        if self.res is None: return []
        if getattr(self.res, "extras", None) is None: self.res.extras = []
        return self.res.extras

    @add
    def _ex_cur(self):
        for e in self._ex_list():
            if e.id == self.ex_sel: return e
        return None

    @add
    def _ex_scrollframe(self, parent, width):
        outer = tk.Frame(parent, bg=BG1, width=width); outer.pack_propagate(False)
        cv = tk.Canvas(outer, bg=BG1, highlightthickness=0, width=width - 16)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview); cv.configure(yscrollcommand=sb.set)
        inner = tk.Frame(cv, bg=BG1); win = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win, width=e.width))
        sb.pack(side="right", fill="y"); cv.pack(side="left", fill="both", expand=True)

        def wheel(e):
            d = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
            cv.yview_scroll(d * 2, "units")
        def on(_):
            outer.bind_all("<MouseWheel>", wheel); outer.bind_all("<Button-4>", wheel); outer.bind_all("<Button-5>", wheel)
        def off(_):
            outer.unbind_all("<MouseWheel>"); outer.unbind_all("<Button-4>"); outer.unbind_all("<Button-5>")
        outer.bind("<Enter>", on); outer.bind("<Leave>", off)
        outer._cv = cv
        return outer, inner

    @add
    def _ex_wheel_guard(self, root, outer):
        """The mouse wheel must scroll the side panel and NEVER change a slider, dropdown or spin box that happens to be
        under the pointer (that is how settings got changed by accident)."""
        cv = outer._cv
        def scroll(e):
            d = -1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1
            cv.yview_scroll(d * 2, "units"); return "break"
        def walk(w):
            for c in w.winfo_children():
                if isinstance(c, (tk.Scale, ttk.Combobox, tk.Spinbox, tk.Listbox)):
                    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"): c.bind(seq, scroll)
                walk(c)
        walk(root)

    @add
    def _ex_combo(self, parent, var, values, cmd=None, width=22):
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        if cmd: cb.bind("<<ComboboxSelected>>", lambda e: cmd())
        return cb

    # ================================================================== PAGE
    @add
    def build_extras(self, page):
        self.ex_mode, self.ex_view, self.ex_sel = "cut", "front", None
        self.ex_src, self.ex_src_name, self.ex_src_kind = None, "", "file"
        self.ex_pts, self.ex_closed, self.ex_drag, self.ex_zoom_mult = [], False, None, 1.0
        self.ex_photo, self._ex_z, self._ex_meta, self._ex_loading, self._ex_pending = None, 1.0, None, False, False
        self._ex_pmap, self._ex_pdisp, self._ex_base, self._ex_dim, self._ex_grab = {}, {}, None, None, None
        sv, bv, dv = tk.StringVar, tk.BooleanVar, tk.DoubleVar
        self.v_ex_name, self.v_ex_parent, self.v_ex_layer = sv(value=""), sv(value=""), sv(value=LAYER_NAMES[1])
        self.v_ex_show, self.v_ex_size, self.v_ex_rest = bv(value=True), dv(value=1.0), dv(value=0.0)
        self.v_ex_preset, self.v_ex_spring = sv(value=PRESETS[0]), bv(value=False)
        self.v_ex_m = {k: dv(value=v) for k, v in dict(freq=6.0, damp=0.5, inertia=0.0, follow=1.0, sway_amp=0.0, sway_hz=1.0).items()}
        self.v_ex_srcview, self.v_ex_srcpart = sv(value="front"), sv(value="head")
        self.v_ex_whole = bv(value=False)
        self.v_mk_name, self.v_mk_parent = sv(value="tail"), sv(value="")
        self.v_mk_views = {v: bv(value=(v == "front")) for v in VIEWS}
        self.v_mk_pivot, self.v_mk_motion = sv(value="top"), sv(value=PRESETS[0])
        self.v_mk_target = sv(value="new"); self._ex_hist, self._ex_hpos, self._ex_hist_owner = [], -1, None

        right_outer, R = self._ex_scrollframe(page, 340)
        right_outer.pack(side="right", fill="y", padx=8, pady=8)
        left = tk.Frame(page, bg=BG1); left.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)

        # ---- left: mode tabs, view tabs, canvas
        bar = tk.Frame(left, bg=BG1); bar.pack(fill="x", pady=(0, 4))
        self.ex_mbtn = {}
        for key, txt in (("cut", "1  CUT"), ("sprite", "2  PIVOT"), ("place", "3  PLACE")):
            b = self.mk_btn(bar, txt, lambda k=key: self.ex_set_mode(k), "n", pady=4, padx=12); b.pack(side="left", padx=(0, 2)); self.ex_mbtn[key] = b
        self.ex_title = tk.Label(bar, text="", bg=BG1, fg=MUTED, font=self.F["s"]); self.ex_title.pack(side="left", padx=12)
        self.mk_btn(bar, "+", lambda: self.ex_zoom_by(1.25), "t", pady=2, padx=8).pack(side="right", padx=2)
        self.mk_btn(bar, "-", lambda: self.ex_zoom_by(0.8), "t", pady=2, padx=8).pack(side="right")
        vb = tk.Frame(left, bg=BG1); vb.pack(fill="x", pady=(0, 4)); self.ex_vbtn = {}
        for v in VIEWS:
            b = self.mk_btn(vb, v.upper(), lambda v=v: self.ex_set_view(v), "n", pady=3); b.pack(side="left", expand=True, fill="x", padx=1); self.ex_vbtn[v] = b
        cv = tk.Canvas(left, bg="#05070b", highlightthickness=1, highlightbackground=LINE, cursor="crosshair"); cv.pack(fill="both", expand=True); self.ex_cv = cv
        cv.bind("<ButtonPress-1>", self.ex_press); cv.bind("<B1-Motion>", self.ex_motion); cv.bind("<ButtonRelease-1>", self.ex_release)
        cv.bind("<Double-Button-1>", self.ex_double); cv.bind("<Button-3>", self.ex_rclick)
        cv.bind("<Configure>", lambda e: self.after_idle(self.ex_redraw))
        self.ex_hint = self.mk_label(left, "", DIM, wraplength=760, justify="left"); self.ex_hint.pack(anchor="w", pady=(4, 0))

        # ---- right 0: the parts list
        p, f = self.mk_panel(R, "EXTRA PARTS ON THIS CHARACTER", PINK); p.pack(fill="x", pady=(0, 6))
        self.ex_lb = self.mk_list(f, 5); self.ex_lb.pack(fill="x"); self.ex_lb.bind("<<ListboxSelect>>", self.ex_pick)
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4, 0))
        self.mk_btn(r, "↶ UNDO", self.ex_hist_undo, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "↷ REDO", self.ex_hist_redo, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3, 0))
        self.mk_btn(r, "MIRROR TWIN", self.ex_mirror, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "DUPLICATE", self.ex_dup, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "DELETE", self.ex_delete, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")

        # ---- right 1: cut / copy
        p, f = self.mk_panel(R, "1  CUT OR COPY A PART", ACC); p.pack(fill="x", pady=(0, 6))
        self.mk_label(f, "Copy a part the rig already has:", MUTED).pack(anchor="w")
        r = tk.Frame(f, bg=BG1); r.pack(fill="x")
        self._ex_combo(r, self.v_ex_srcview, list(VIEWS), width=7).pack(side="left")
        self._ex_combo(r, self.v_ex_srcpart, list(PART_KEYS), width=9).pack(side="left", padx=4)
        self.mk_btn(r, "LOAD", self.ex_load_part, "n", pady=2, font=self.F["card"]).pack(side="left")
        self.mk_label(f, "...or import any picture:", MUTED).pack(anchor="w", pady=(4, 0))
        r = tk.Frame(f, bg=BG1); r.pack(fill="x")
        self.mk_btn(r, "IMPORT IMAGE...", self.ex_load_file, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "REMOVE FLAT BACKGROUND", self.ex_remove_bg, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")
        self.ex_src_lbl = self.mk_label(f, "nothing loaded", DIM, wraplength=300, justify="left"); self.ex_src_lbl.pack(anchor="w", pady=(2, 0))
        self.mk_label(f, "Lasso: click around the part (or drag). Click the first dot or double-click to close, right-click to undo.", DIM, wraplength=300, justify="left").pack(anchor="w", pady=(4, 0))
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(2, 0))
        self.mk_btn(r, "CLOSE SHAPE", self.ex_close, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "UNDO POINT", self.ex_undo, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "CLEAR", self.ex_clear, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")
        self.mk_check(f, "No lasso: use the whole picture (plain copy)", self.v_ex_whole, self.ex_redraw).pack(anchor="w")
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4, 0))
        self.mk_label(r, "NAME").pack(side="left"); self.mk_entry(r, self.v_mk_name, 14).pack(side="left", padx=4)
        self.mk_label(f, "ATTACH TO").pack(anchor="w", pady=(3, 0))
        self.mk_parent_cb = self._ex_combo(f, self.v_mk_parent, [], width=30); self.mk_parent_cb.pack(anchor="w")
        self.mk_label(f, "Every view needs its own cut (front, side and back are different pictures). Make the part for one view, then load "
                         "the next view's picture and choose 'add as a view of the selected part'.", DIM, wraplength=300, justify="left").pack(anchor="w", pady=(4, 0))
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3, 0)); self.mk_label(r, "FOR VIEW").pack(side="left")
        for v in VIEWS: self.mk_check(r, v, self.v_mk_views[v], self._ex_update_make_btn).pack(side="left", padx=3)
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3, 0)); self.mk_label(r, "PIVOT").pack(side="left")
        self._ex_combo(r, self.v_mk_pivot, list(PIVOTS), width=8).pack(side="left", padx=4)
        self.mk_label(r, "MOTION").pack(side="left", padx=(6, 0))
        self._ex_combo(r, self.v_mk_motion, PRESETS, width=19).pack(side="left", padx=4)
        for val, txt in (("new", "make a NEW part"), ("add", "add as a VIEW of the selected part")):
            tk.Radiobutton(f, text=txt, value=val, variable=self.v_mk_target, command=self._ex_update_make_btn, bg=BG1, fg=TEXT, selectcolor=BG0,
                           activebackground=BG1, activeforeground=ACC, font=self.F["s"], anchor="w", highlightthickness=0).pack(fill="x")
        self.ex_make_btn = self.mk_btn(f, "✂   MAKE PART", self.ex_make, "g", pady=7); self.ex_make_btn.pack(fill="x", pady=(4, 0))

        # ---- right 2: the selected part
        p, f = self.mk_panel(R, "2  THIS PART", GOLD); p.pack(fill="x", pady=(0, 6))
        self.ex_sel_lbl = self.mk_label(f, "no part selected", DIM); self.ex_sel_lbl.pack(anchor="w")
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(2, 0))
        self.mk_label(r, "NAME").pack(side="left"); en = self.mk_entry(r, self.v_ex_name, 16); en.pack(side="left", padx=4)
        en.bind("<Return>", lambda e: self.ex_apply(True)); en.bind("<FocusOut>", lambda e: self.ex_apply(True))
        self.mk_label(f, "ATTACHED TO").pack(anchor="w", pady=(3, 0))
        self.ex_parent_cb = self._ex_combo(f, self.v_ex_parent, [], self.ex_parent_changed, width=30); self.ex_parent_cb.pack(anchor="w")
        self.mk_label(f, "LAYER  (this view)").pack(anchor="w", pady=(3, 0))
        self._ex_combo(f, self.v_ex_layer, LAYER_NAMES, lambda: self.ex_apply(True), width=30).pack(anchor="w")
        self.mk_check(f, "Show in this view", self.v_ex_show, lambda: self.ex_apply(True)).pack(anchor="w")
        self.mk_label(f, "SIZE  (this view only)").pack(anchor="w")
        s = self.mk_scale(f, self.v_ex_size, 0.1, 4.0, self.ex_slider, res=0.05); s.config(length=300); s.pack(fill="x"); s.bind("<ButtonRelease-1>", lambda e: self.ex_commit())
        self.ex_size_lbl = self.mk_label(f, "", DIM); self.ex_size_lbl.pack(anchor="w")
        self.mk_label(f, "ROTATION  (rest angle, degrees)").pack(anchor="w")
        s = self.mk_scale(f, self.v_ex_rest, -180, 180, self.ex_slider, res=1); s.config(length=300); s.pack(fill="x"); s.bind("<ButtonRelease-1>", lambda e: self.ex_commit())
        self.mk_label(f, "PIVOT PRESETS  (or drag the dot in step 2)").pack(anchor="w", pady=(3, 0))
        r = tk.Frame(f, bg=BG1); r.pack(fill="x")
        for n in PIVOTS: self.mk_btn(r, n, lambda n=n: self.ex_pivot_preset(n), "n", pady=2, padx=4, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=1)
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3, 0))
        self.mk_btn(r, "FLIP ↔", self.ex_flip, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "COPY TO ALL VIEWS", self.ex_copy_view, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "RESET SPOT", self.ex_reset_spot, "n", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")

        # ---- right 3: motion
        p, f = self.mk_panel(R, "3  MOTION", GOOD); p.pack(fill="x", pady=(0, 6))
        self.mk_label(f, "PRESET").pack(anchor="w")
        self.ex_preset_cb = self._ex_combo(f, self.v_ex_preset, PRESETS + [CUSTOM], self.ex_preset_changed, width=30); self.ex_preset_cb.pack(anchor="w")
        self.mk_check(f, "Springy (lag, bounce, shoved by the hips)", self.v_ex_spring, self.ex_motion_changed).pack(anchor="w", pady=(3, 0))
        for lab, key, lo, hi, res_ in (("STIFFNESS  (low = lazy and floppy)", "freq", 0.8, 14.0, 0.1), ("DAMPING  (low = bouncy)", "damp", 0.05, 1.2, 0.05),
                                       ("INERTIA  (pushed by the hips)", "inertia", 0.0, 2.0, 0.05), ("FOLLOW PARENT  (0 = hangs free)", "follow", 0.0, 1.2, 0.05),
                                       ("IDLE SWAY  (degrees)", "sway_amp", 0.0, 45.0, 1.0), ("SWAY SPEED  (Hz)", "sway_hz", 0.2, 6.0, 0.1)):
            self.mk_label(f, lab).pack(anchor="w")
            s = self.mk_scale(f, self.v_ex_m[key], lo, hi, lambda v: self.ex_motion_changed(), res=res_); s.config(length=300); s.pack(fill="x")
            s.bind("<ButtonRelease-1>", lambda e: self.ex_commit())
        self.mk_btn(f, "▶   SEE IT MOVE  (ANIMATE page)", lambda: self.show_page("anim"), "p", pady=6).pack(fill="x", pady=(6, 0))
        self._ex_wheel_guard(R, right_outer)
        self.ex_set_mode("cut", refresh=False); self.ex_set_view("front", refresh=False)

    # ================================================================== state refresh
    @add
    def extras_refresh(self, *_):
        ex = self._ex_list(); keep = self.ex_sel
        self.ex_lb.delete(0, "end")
        for i, e in enumerate(ex):
            par = self._ex_label_of(e.parent)
            self.ex_lb.insert("end", f"{e.name:13s} -> {par}  [{RX.views_label(e)}]")
        self.ex_sel = keep if any(e.id == keep for e in ex) else (ex[0].id if ex else None)
        if self.ex_sel:
            self.ex_lb.selection_clear(0, "end"); self.ex_lb.selection_set([e.id for e in ex].index(self.ex_sel))
        self._ex_rebuild_parents(); self.ex_load_vars(); self._ex_update_make_btn()
        if self.res is not None and self._ex_hist_owner is not self.res:
            self._ex_hist_owner = self.res; snap = RX.snapshot(ex); self._ex_hist, self._ex_hpos = [(snap, RX.snapshot_sig(snap))], 0
        self.ex_redraw()

    @add
    def _ex_update_make_btn(self, *_):
        e = self._ex_cur(); vs = [v.upper() for v in VIEWS if self.v_mk_views[v].get()]
        if self.v_mk_target.get() == "add":
            self.ex_make_btn.config(text=(f"✂   ADD TO '{e.name}'  ({', '.join(vs) or 'pick a view'})" if e else "✂   select a part first"))
        else:
            self.ex_make_btn.config(text=f"✂   MAKE PART  ({', '.join(vs) or 'pick a view'})")

    @add
    def _ex_label_of(self, key):
        if RX.is_extra(key):
            e = next((x for x in self._ex_list() if x.id == key), None)
            return "★ " + (e.name if e else key)
        return G["pretty"](key)

    @add
    def _ex_rebuild_parents(self, for_extra=None):
        """Parent choices: every active base part, plus the other extras (never itself or what hangs from it)."""
        base = list(G["RB"].active_keys(getattr(self.res, "mask", None))) if self.res is not None else list(PART_KEYS)
        banned = set()
        cur = for_extra or self._ex_cur()
        if cur is not None: banned = {cur.id} | RX.descendants(self._ex_list(), cur.id)
        keys = base + [e.id for e in self._ex_list() if e.id not in banned]
        self._ex_pmap = {self._ex_label_of(k): k for k in keys}; self._ex_pdisp = {k: d for d, k in self._ex_pmap.items()}
        vals = list(self._ex_pmap.keys())
        self.ex_parent_cb.config(values=vals)
        self.mk_parent_cb.config(values=[self._ex_label_of(k) for k in base + [e.id for e in self._ex_list()]])
        full = {self._ex_label_of(k): k for k in base + [e.id for e in self._ex_list()]}
        self._ex_mkmap = full
        if self.v_mk_parent.get() not in full: self.v_mk_parent.set(self._ex_label_of("head" if "head" in base else base[0]))

    @add
    def ex_load_vars(self):
        e = self._ex_cur(); self._ex_loading = True
        try:
            if e is None:
                self.ex_sel_lbl.config(text="no part selected - make one in step 1", fg=DIM); self.v_ex_name.set(""); return
            self.v_ex_name.set(e.name); self.v_ex_parent.set(self._ex_pdisp.get(e.parent, self._ex_label_of(e.parent)))
            vd = e.view_data(self.ex_view) or {}
            self.v_ex_layer.set(RX.LAYER_LABEL.get(vd.get("layer", "front"), LAYER_NAMES[1])); self.v_ex_show.set(bool(vd.get("show", True)))
            self.v_ex_size.set(float(vd.get("size", 1.0))); self.v_ex_rest.set(e.rest)
            m = e.motion; self.v_ex_spring.set(m.get("mode") == "spring")
            for k, v in self.v_ex_m.items(): v.set(float(m.get(k, 0.0)))
            self.v_ex_preset.set(next((n for n, p in RX.MOTION_PRESETS.items() if all(abs(float(p[k]) - float(m.get(k, 0))) < 1e-6 for k in ("freq", "damp", "inertia", "follow", "sway_amp", "sway_hz")) and p["mode"] == m.get("mode")), CUSTOM))
            sp = RX.sprite_for(e, self.ex_view, self.res.cpp)
            cm = (f"{sp.rgba.shape[1] * self.res.cpp:.0f} x {sp.rgba.shape[0] * self.res.cpp:.0f} cm") if sp is not None else "-"
            self.ex_size_lbl.config(text=f"on the body: about {cm}")
            self.ex_sel_lbl.config(text=f"selected: {e.name}   ({self.ex_view.upper()} view)" if vd else
                                   f"selected: {e.name}   - no {self.ex_view.upper()} sprite yet: cut one in step 1", fg=ACC if vd else WARN)
        finally:
            self._ex_loading = False

    # ================================================================== modes / views / zoom
    @add
    def ex_set_mode(self, m, refresh=True):
        self.ex_mode = m
        for k, b in self.ex_mbtn.items(): self.set_active(b, k == m)
        self.ex_cv.config(cursor={"cut": "crosshair", "sprite": "fleur", "place": "fleur"}[m])
        if refresh: self.ex_redraw()

    @add
    def ex_set_view(self, v, refresh=True):
        self.ex_view = v
        for n, b in self.ex_vbtn.items(): self.set_active(b, n == v)
        for n in VIEWS: self.v_mk_views[n].set(n == v)           # new cuts go to the view you are looking at, nowhere else
        if hasattr(self, "ex_make_btn"): self._ex_update_make_btn()
        if refresh: self.ex_load_vars(); self.ex_redraw()

    @add
    def ex_zoom_by(self, f):
        self.ex_zoom_mult = float(np.clip(self.ex_zoom_mult * f, 0.3, 8)); self.ex_redraw()

    # ================================================================== canvas drawing
    @add
    def ex_redraw(self, *_):
        cv = self.ex_cv; cv.delete("all"); self.ex_photo = None
        hints = {"cut": "STEP 1: load a source, then draw around the part you want (or tick 'whole picture'), then MAKE PART.",
                 "sprite": "STEP 2: drag the pink dot to the point the part should rotate about (tail base, ear root...). Arrow keys nudge (Shift = bigger).",
                 "place": "STEP 3: drag a dot on the body to move that part in this view. Click another dot to select it. Arrow keys nudge (Shift = bigger)."}
        self.ex_hint.config(text=hints[self.ex_mode])
        if self.res is None:
            cv.create_text(320, 200, text="CUT a rig first - extra parts hang from the rig's body parts.", fill=DIM, font=self.F["h"]); return
        try:
            {"cut": self._ex_draw_cut, "sprite": self._ex_draw_sprite, "place": self._ex_draw_place}[self.ex_mode]()
        except Exception as ex:
            self.log(f"extras view failed: {ex!r}", "warn")

    @add
    def _ex_fit(self, w, h, cap=None):
        cv = self.ex_cv; cw, ch = max(300, cv.winfo_width()), max(300, cv.winfo_height())
        z = min((cw - 2 * M) / max(1, w), (ch - 2 * M) / max(1, h))
        if cap: z = min(z, cap)
        return max(0.05, z * self.ex_zoom_mult)

    @add
    def _ex_put(self, im):
        z = self._ex_z
        rs = Image.Resampling.NEAREST if z >= 1 else Image.Resampling.LANCZOS
        self.ex_photo = ImageTk.PhotoImage(im.resize((max(1, int(im.width * z)), max(1, int(im.height * z))), rs))
        self.ex_cv.create_image(M, M, image=self.ex_photo, anchor="nw")

    @add
    def _ex_draw_cut(self):
        cv = self.ex_cv
        self.ex_title.config(text=("source: " + self.ex_src_name) if self.ex_src is not None else "no source yet")
        if self.ex_src is None:
            cv.create_text(max(300, cv.winfo_width()) // 2, 180, text="Load a source on the right:\ncopy one of the rig's parts, or import a picture.", fill=DIM, font=self.F["h"], justify="center"); return
        h, w = self.ex_src.shape[:2]; self._ex_z = self._ex_fit(w, h, cap=24)
        if self._ex_base is None:
            bg = Image.new("RGBA", (w, h), (28, 32, 42, 255)); bg.alpha_composite(Image.fromarray(self.ex_src)); self._ex_base = bg.convert("RGB")
        im = self._ex_base
        if self.ex_closed and not self.v_ex_whole.get():
            if self._ex_dim is None:
                a = np.array(self._ex_base); m = RX.polygon_mask(w, h, self.ex_pts); a[~m] = (a[~m] * 0.28).astype(np.uint8); self._ex_dim = Image.fromarray(a)
            im = self._ex_dim
        self._ex_put(im); self._ex_draw_poly()

    @add
    def _ex_draw_poly(self):
        cv = self.ex_cv; cv.delete("poly")
        if self.v_ex_whole.get():
            h, w = self.ex_src.shape[:2]; z = self._ex_z
            cv.create_rectangle(M, M, M + w * z, M + h * z, outline=GOOD, width=2, dash=(5, 3), tags="poly"); return
        z = self._ex_z; pts = [(M + x * z, M + y * z) for x, y in self.ex_pts]
        if len(pts) >= 2:
            flat = [c for p in pts for c in p] + (list(pts[0]) if self.ex_closed else [])
            cv.create_line(*flat, fill=GOOD if self.ex_closed else ACC, width=2, tags="poly")
        for i, (x, y) in enumerate(pts):
            r = 6 if i == 0 else 3; cv.create_oval(x - r, y - r, x + r, y + r, outline=PINK if i == 0 else ACC, fill=BG0, width=2, tags="poly")

    @add
    def _ex_draw_sprite(self):
        cv = self.ex_cv; e = self._ex_cur()
        if e is None or e.view_data(self.ex_view) is None:
            self.ex_title.config(text="")
            txt = ("Select (or make) an extra part to set its pivot." if e is None else
                   f"'{e.name}' has no {self.ex_view.upper()} sprite yet.\nStep 1: load this view's picture, lasso the part,\nchoose 'add as a VIEW of the selected part'.")
            cv.create_text(max(300, cv.winfo_width()) // 2, 180, text=txt, fill=DIM, font=self.F["h"], justify="center"); return
        vd = e.view_data(self.ex_view); rgba = vd["rgba"]; h, w = rgba.shape[:2]
        self.ex_title.config(text=f"{e.name} - {self.ex_view.upper()} sprite ({w} x {h} px)")
        self._ex_z = self._ex_fit(w, h, cap=24)
        bg = Image.new("RGBA", (w, h), (28, 32, 42, 255)); bg.alpha_composite(Image.fromarray(rgba)); self._ex_put(bg.convert("RGB"))
        z = self._ex_z; px, py = M + vd["pj"][0] * z, M + vd["pj"][1] * z
        cv.create_line(px - 18, py, px + 18, py, fill=PINK, width=1); cv.create_line(px, py - 18, px, py + 18, fill=PINK, width=1)
        cv.create_oval(px - 7, py - 7, px + 7, py + 7, outline=PINK, width=3, fill=BG0)

    @add
    def _ex_draw_place(self):
        cv = self.ex_cv; self.ex_title.config(text=f"{self.ex_view.upper()} view")
        img, meta = G["render_pose"](self.res, self.ex_view, {}, solid=False, shade=True, crop=True, rest=False)
        self._ex_meta = meta; self._ex_z = self._ex_fit(img.width, img.height, cap=12)
        bg = Image.new("RGBA", img.size, (5, 7, 11, 255)); bg.alpha_composite(img); self._ex_put(bg.convert("RGB"))
        z = self._ex_z; piv = dict(meta["pivots_px"]); piv.update(meta.get("extras_px", {}))
        for k, P in meta.get("extras_px", {}).items():
            e = next((x for x in self._ex_list() if x.id == k), None)
            if e is None: continue
            if e.parent in piv:
                Q = piv[e.parent]; cv.create_line(M + Q[0] * z, M + Q[1] * z, M + P[0] * z, M + P[1] * z, fill=LINE2, dash=(3, 3))
            x, y = M + P[0] * z, M + P[1] * z; sel = (k == self.ex_sel)
            cv.create_oval(x - 8, y - 8, x + 8, y + 8, outline=PINK if sel else GOLD, width=3 if sel else 2, fill=BG0)
            cv.create_text(x + 12, y - 10, text=e.name, fill=PINK if sel else GOLD, anchor="w", font=self.F["s"])
        e = self._ex_cur()
        if e is not None and e.view_data(self.ex_view) is None:
            cv.create_text(M + 8, M + 8, text=f"'{e.name}' has no {self.ex_view.upper()} sprite yet - cut one in step 1 and choose 'add as a VIEW of the selected part'.",
                           fill=WARN, anchor="nw", font=self.F["s"])
        elif not meta.get("extras_px"):
            cv.create_text(M + 8, M + 8, text="No extra parts visible in this view yet.", fill=DIM, anchor="nw", font=self.F["s"])

    # ================================================================== sources
    @add
    def ex_set_source(self, rgba, name, kind, parent=None):
        self.ex_src, self.ex_src_name, self.ex_src_kind = np.array(rgba, np.uint8, copy=True), name, kind
        self._ex_base = self._ex_dim = None; self.ex_pts, self.ex_closed = [], False; self.ex_zoom_mult = 1.0
        h, w = self.ex_src.shape[:2]
        if not getattr(self, "_ex_mkmap", None): self._ex_rebuild_parents()
        tr = RX.has_transparency(self.ex_src)
        self.ex_src_lbl.config(text=f"{name}   {w} x {h} px" + ("" if tr else "\nNo transparency - lasso the part, or try REMOVE FLAT BACKGROUND."), fg=MUTED if tr else WARN)
        if parent and getattr(self, "_ex_mkmap", None) and self._ex_label_of(parent) in self._ex_mkmap: self.v_mk_parent.set(self._ex_label_of(parent))
        self.ex_set_mode("cut")

    @add
    def ex_load_part(self):
        if self.res is None: return self.toast("CUT a rig first.", "warn")
        v, k = self.v_ex_srcview.get(), self.v_ex_srcpart.get()
        try: sp = self.res.views[v][k]
        except Exception: return self.toast("That part is not in the rig.", "warn")
        if sp.kind == "disabled" or not (sp.rgba[..., 3] > 0).any(): return self.toast("That part is switched off or empty in this view.", "warn")
        self.ex_set_source(sp.rgba, f"rig part {G['pretty'](k)} ({v})", "rig", parent=G["parent_key"](k))
        self.v_mk_name.set(G["base_of"](k) + "_copy")

    @add
    def ex_load_file(self):
        p = filedialog.askopenfilename(parent=self, title="Import a picture of the part",
                                       filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"), ("All files", "*.*")])
        if not p: return
        try:
            im = Image.open(p); im.seek(0) if getattr(im, "n_frames", 1) > 1 else None
            self.ex_set_source(np.array(im.convert("RGBA")), p.replace("\\", "/").split("/")[-1], "file")
        except Exception as ex: self.toast(f"Could not open that image: {ex}", "err")

    @add
    def ex_remove_bg(self):
        if self.ex_src is None: return self.toast("Load a source first.", "warn")
        try:
            self.ex_src = RX.remove_background(self.ex_src); self._ex_base = self._ex_dim = None
            self.ex_src_lbl.config(text=self.ex_src_lbl.cget("text").split("\n")[0] + "   (background removed)", fg=MUTED); self.ex_redraw()
        except Exception as ex: self.toast(f"Background removal failed: {ex}", "err")

    # ================================================================== lasso
    @add
    def _ex_img_pt(self, e):
        z = self._ex_z; return ((e.x - M) / z, (e.y - M) / z)

    @add
    def ex_press(self, e):
        self.ex_cv.focus_set()
        if self.res is None: return
        if self.ex_mode == "cut":
            if self.ex_src is None or self.v_ex_whole.get(): return
            if self.ex_closed: return
            z = self._ex_z
            if len(self.ex_pts) >= 3 and math.hypot(M + self.ex_pts[0][0] * z - e.x, M + self.ex_pts[0][1] * z - e.y) < 10:
                return self.ex_close()
            self.ex_pts.append(self._ex_img_pt(e)); self.ex_drag = (e.x, e.y); self._ex_draw_poly()
        elif self.ex_mode == "sprite":
            self.ex_drag = True; self._ex_set_pivot(e)
        elif self.ex_mode == "place":
            meta = self._ex_meta
            if not meta: return
            z, best, bd = self._ex_z, None, 22
            for k, P in meta.get("extras_px", {}).items():
                d = math.hypot(M + P[0] * z - e.x, M + P[1] * z - e.y)
                if d < bd: best, bd = k, d
            if best:
                self.ex_sel = best; self._ex_grab = (e.x, e.y); self.ex_load_vars()
                self.ex_lb.selection_clear(0, "end"); self.ex_lb.selection_set([x.id for x in self._ex_list()].index(best)); self.ex_redraw()

    @add
    def ex_motion(self, e):
        if self.ex_mode == "cut":
            if self.ex_drag and not self.ex_closed and not self.v_ex_whole.get() and math.hypot(e.x - self.ex_drag[0], e.y - self.ex_drag[1]) > 6:
                self.ex_pts.append(self._ex_img_pt(e)); self.ex_drag = (e.x, e.y); self._ex_draw_poly()
        elif self.ex_mode == "sprite":
            if self.ex_drag: self._ex_set_pivot(e)
        elif self.ex_mode == "place" and self._ex_grab and self.ex_sel:
            ex_ = self._ex_cur()
            if ex_ is None: return
            z = self._ex_z; d_cm = np.array([(e.x - self._ex_grab[0]) / z, (e.y - self._ex_grab[1]) / z]) * float(self.res.cpp)
            self._ex_grab = (e.x, e.y); self._ex_move(ex_, d_cm); self._ex_soon()

    @add
    def ex_release(self, e):
        was = self.ex_drag or self._ex_grab
        self.ex_drag = None; self._ex_grab = None
        if was and self.ex_mode in ("sprite", "place"): self.ex_commit()

    @add
    def ex_double(self, e):
        if self.ex_mode == "cut" and not self.ex_closed:
            if len(self.ex_pts) >= 2 and math.hypot(self.ex_pts[-1][0] - self.ex_pts[-2][0], self.ex_pts[-1][1] - self.ex_pts[-2][1]) < 2: self.ex_pts.pop()
            self.ex_close()

    @add
    def ex_rclick(self, e):
        if self.ex_mode == "cut": self.ex_undo()

    @add
    def ex_close(self):
        if self.ex_src is None or self.v_ex_whole.get(): return
        if len(self.ex_pts) < 3: return self.toast("Click at least 3 points around the part first.", "warn")
        self.ex_closed = True; self._ex_dim = None; self.ex_redraw()

    @add
    def ex_undo(self):
        if self.ex_closed: self.ex_closed = False; self._ex_dim = None
        elif self.ex_pts: self.ex_pts.pop()
        self.ex_redraw()

    @add
    def ex_clear(self):
        self.ex_pts, self.ex_closed, self._ex_dim = [], False, None; self.ex_redraw()

    # ================================================================== make / edit
    @add
    def ex_make(self):
        if self.res is None: return self.toast("CUT a rig first.", "warn")
        if self.ex_src is None: return self.toast("Load a source first (step 1).", "warn")
        whole = self.v_ex_whole.get()
        if not whole and not self.ex_closed: return self.toast("Draw around the part and close the shape, or tick 'use the whole picture'.", "warn")
        cut, _ = RX.cut_polygon(self.ex_src, None if whole else self.ex_pts)
        if cut is None: return self.toast("Nothing inside that shape - it has no visible pixels.", "warn")
        views = [v for v in VIEWS if self.v_mk_views[v].get()]
        if not views: return self.toast("Tick at least one view.", "warn")
        name = (self.v_mk_name.get() or "part").strip()
        parent = getattr(self, "_ex_mkmap", {}).get(self.v_mk_parent.get(), "head")
        base = 1.0 if self.ex_src_kind == "rig" else RX.auto_base_scale(cut, self.res)
        if self.v_mk_target.get() == "add":
            cur = self._ex_cur()
            if cur is None: return self.toast("Select the part you want to add this view to (in the list on the right).", "warn")
            for v in views: RX.add_view(cur, self.res, v, cut, pivot=self.v_mk_pivot.get(), base_scale=base)
            self.ex_view = views[0]
            for n, b in self.ex_vbtn.items(): self.set_active(b, n == self.ex_view)
            self.ex_set_mode("place", refresh=False); self.extras_refresh(); self.ex_commit()
            return self.toast(f"'{cur.name}' now has its own {', '.join(v.upper() for v in views)} sprite. Drag it into place.", "ok")
        e = RX.make_extra(self.res, name, parent, cut, views=views, pivot=self.v_mk_pivot.get(), base_scale=base,
                          motion=RX.default_motion(self.v_mk_motion.get()), extras=self._ex_list())
        self._ex_list().append(e); self.ex_sel = e.id; self.ex_view = views[0]
        for n, b in self.ex_vbtn.items(): self.set_active(b, n == self.ex_view)
        self.ex_set_mode("place", refresh=False); self.extras_refresh(); self.ex_commit()
        self.toast(f"Made '{e.name}'. Drag it into place, then adjust pivot, size and motion.", "ok")

    @add
    def ex_pick(self, _e=None):
        s = self.ex_lb.curselection(); ex = self._ex_list()
        if s and s[0] < len(ex):
            self.ex_sel = ex[s[0]].id; self._ex_rebuild_parents(); self.ex_load_vars(); self.ex_redraw()

    @add
    def ex_apply(self, commit=False):
        """Push name / layer / visibility into the selected part."""
        e = self._ex_cur()
        if e is None or self._ex_loading: return
        nm = (self.v_ex_name.get() or e.name).strip()
        renamed = nm != e.name; e.name = nm
        vd = e.view_data(self.ex_view)
        if vd is not None:
            rev = {v: k for k, v in RX.LAYER_LABEL.items()}
            vd["layer"] = rev.get(self.v_ex_layer.get(), vd.get("layer", "front")); vd["show"] = bool(self.v_ex_show.get())
        if commit:
            if renamed: self.extras_refresh()
            else: self.ex_redraw()
            self.ex_commit()

    @add
    def ex_slider(self, _v=None):
        e = self._ex_cur()
        if e is None or self._ex_loading: return
        e.rest = float(self.v_ex_rest.get()); vd = e.view_data(self.ex_view)
        if vd is not None: vd["size"] = float(self.v_ex_size.get()); vd.pop("_c", None)
        sp = RX.sprite_for(e, self.ex_view, self.res.cpp)
        if sp is not None: self.ex_size_lbl.config(text=f"on the body: about {sp.rgba.shape[1] * self.res.cpp:.0f} x {sp.rgba.shape[0] * self.res.cpp:.0f} cm")
        self._ex_soon()

    @add
    def ex_parent_changed(self):
        e = self._ex_cur()
        if e is None or self._ex_loading: return
        new = self._ex_pmap.get(self.v_ex_parent.get())
        if not new or new == e.parent: return
        e.parent = new
        for v in e.views: e.views[v]["off"] = np.asarray(RX.default_offset(self.res, v, new), float)
        self.extras_refresh(); self.ex_commit()

    @add
    def ex_preset_changed(self):
        e = self._ex_cur()
        if e is None or self._ex_loading: return
        n = self.v_ex_preset.get()
        if n in RX.MOTION_PRESETS: e.motion = RX.default_motion(n); self.ex_load_vars(); self.ex_commit()

    @add
    def ex_motion_changed(self, *_):
        e = self._ex_cur()
        if e is None or self._ex_loading: return
        e.motion = dict({k: float(v.get()) for k, v in self.v_ex_m.items()}, mode="spring" if self.v_ex_spring.get() else "rigid")
        self._ex_loading = True
        try: self.v_ex_preset.set(next((n for n, p in RX.MOTION_PRESETS.items() if all(abs(float(p[k]) - float(e.motion[k])) < 1e-6 for k in ("freq", "damp", "inertia", "follow", "sway_amp", "sway_hz")) and p["mode"] == e.motion["mode"]), CUSTOM))
        finally: self._ex_loading = False

    @add
    def _ex_set_pivot(self, ev):
        e = self._ex_cur()
        if e is None: return
        vd = e.view_data(self.ex_view); h, w = vd["rgba"].shape[:2]; x, y = self._ex_img_pt(ev)
        vd["pj"] = np.array([float(np.clip(x, 0, w)), float(np.clip(y, 0, h))]); vd.pop("_c", None); self._ex_soon()

    @add
    def ex_pivot_preset(self, name):
        e = self._ex_cur()
        if e is None: return self.toast("Select a part first.", "warn")
        vd = e.views.get(self.ex_view)
        if vd is None: return
        vd["pj"] = np.array(RX.pivot_choices(vd["rgba"])[name], float); vd.pop("_c", None); self.ex_redraw(); self.ex_commit()

    @add
    def _ex_move(self, e, d_cm):
        vd = e.view_data(self.ex_view)
        if vd is None: return
        th = float((self._ex_meta or {}).get("th", {}).get(e.parent, 0.0))
        vd["off"] = np.asarray(vd["off"], float) + np.asarray(G["_rot"](d_cm, -th), float)

    @add
    def ex_flip(self):
        e = self._ex_cur()
        if e is None: return self.toast("Select a part first.", "warn")
        RX.flip_view(e, self.ex_view); self.ex_redraw(); self.ex_commit()

    @add
    def ex_copy_view(self):
        e = self._ex_cur(); vd = e.view_data(self.ex_view) if e else None
        if vd is None: return self.toast("Select a part first.", "warn")
        for v in VIEWS:
            if v == self.ex_view: continue
            off = e.views[v]["off"] if v in e.views else np.asarray(RX.default_offset(self.res, v, e.parent), float)
            lay = e.views[v].get("layer", "front") if v in e.views else vd.get("layer", "front")
            e.set_view(v, vd["rgba"].copy(), vd["pj"].copy(), off, lay, True, bs=vd.get("bs"), size=vd.get("size"))
        self.ex_commit(); self.toast("Copied this view's picture and pivot to the other views (same picture in every view - replace them with real cuts when you can).", "ok")

    @add
    def ex_reset_spot(self):
        e = self._ex_cur()
        if e is None: return self.toast("Select a part first.", "warn")
        vd = e.view_data(self.ex_view)
        if vd is not None: vd["off"] = np.asarray(RX.default_offset(self.res, self.ex_view, e.parent), float)
        self.ex_redraw(); self.ex_commit()

    @add
    def ex_mirror(self):
        e = self._ex_cur()
        if e is None: return self.toast("Select a part first.", "warn")
        if e.twin and any(x.id == e.twin for x in self._ex_list()): return self.toast("That part already has a twin.", "warn")
        n = RX.mirror_extra(e, self._ex_list()); self._ex_list().append(n); self.ex_sel = n.id
        self.extras_refresh(); self.ex_commit(); self.toast(f"Made '{n.name}' on the other side. Check it in each view.", "ok")

    @add
    def ex_dup(self):
        e = self._ex_cur()
        if e is None: return self.toast("Select a part first.", "warn")
        n = RX.duplicate_extra(e, self._ex_list()); self._ex_list().append(n); self.ex_sel = n.id; self.extras_refresh(); self.ex_commit()

    @add
    def ex_delete(self):
        e = self._ex_cur()
        if e is None: return self.toast("Select a part first.", "warn")
        if not messagebox.askyesno("Delete part", f"Delete '{e.name}'?\n(anything attached to it is re-attached to its parent)", parent=self): return
        RX.remove_extra(self.res, e); self.ex_sel = None; self.extras_refresh(); self.ex_commit()

    @add
    def extras_key(self, ev):
        """Arrow keys nudge the pivot (step 2) or the part (step 3).  Returns True when handled."""
        try: foc = self.focus_get()
        except Exception: foc = None
        if (ev.state & 0x4) and ev.keysym.lower() in ("z", "y"):
            if isinstance(foc, (tk.Entry, ttk.Combobox, tk.Spinbox)): return False
            (self.ex_hist_undo if ev.keysym.lower() == "z" else self.ex_hist_redo)(); return True
        if ev.keysym not in ("Left", "Right", "Up", "Down") or self.ex_mode == "cut" or self._ex_cur() is None: return False
        if foc is not self.ex_cv: return False                  # arrows only nudge when you are working IN the canvas
        st = 2.0 if (ev.state & 0x1) else 0.5
        dx = {"Left": -st, "Right": st}.get(ev.keysym, 0.0); dy = {"Up": -st, "Down": st}.get(ev.keysym, 0.0)
        e = self._ex_cur()
        if self.ex_mode == "place": self._ex_move(e, np.array([dx, dy]) * float(self.res.cpp))
        else:
            vd = e.view_data(self.ex_view); h, w = vd["rgba"].shape[:2]
            vd["pj"] = np.array([float(np.clip(vd["pj"][0] + dx, 0, w)), float(np.clip(vd["pj"][1] + dy, 0, h))]); vd.pop("_c", None)
        self.ex_redraw(); self.ex_commit(); return True

    # ================================================================== redraw throttle / commit
    @add
    def _ex_soon(self):
        if self._ex_pending: return
        self._ex_pending = True; self.after(30, self._ex_flush)

    @add
    def _ex_flush(self):
        self._ex_pending = False; self.ex_redraw()

    @add
    def ex_commit(self, push=True):
        """Make the rig sheet / assembled figures / animation see the change, and autosave with the loadout."""
        if self.res is None: return
        try:
            G["render_outputs"](self.res); self.schedule_persist(); self.refresh_view_pages()
        except Exception as ex: self.log(f"extras: could not refresh the rig ({ex!r})", "err")
        if push: self._ex_hist_push()

    # ================================================================== undo / redo of extras edits
    @add
    def _ex_hist_push(self):
        if self.res is None or self._ex_hist_owner is not self.res: return
        snap = RX.snapshot(self._ex_list()); sig = RX.snapshot_sig(snap)
        if self._ex_hist and self._ex_hist[self._ex_hpos][1] == sig: return
        del self._ex_hist[self._ex_hpos + 1:]; self._ex_hist.append((snap, sig)); del self._ex_hist[:-60]; self._ex_hpos = len(self._ex_hist) - 1

    @add
    def _ex_hist_goto(self, i):
        self._ex_hpos = i; self._ex_list()[:] = RX.restore_snapshot(self._ex_hist[i][0])
        if not any(e.id == self.ex_sel for e in self._ex_list()): self.ex_sel = None
        self.extras_refresh(); self.ex_commit(push=False)

    @add
    def ex_hist_undo(self):
        if self.res is None or self._ex_hpos <= 0: return self.toast("Nothing to undo.", "warn")
        self._ex_hist_goto(self._ex_hpos - 1); self.toast("Undone.", "ok")

    @add
    def ex_hist_redo(self):
        if self.res is None or self._ex_hpos >= len(self._ex_hist) - 1: return self.toast("Nothing to redo.", "warn")
        self._ex_hist_goto(self._ex_hpos + 1); self.toast("Redone.", "ok")
