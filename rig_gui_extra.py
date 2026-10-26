"""rig_gui_extra.py - the ALIGN and ANIMATE pages for the HUMAN.RIG window (installed into RigWindow by install())."""
import os, json, math, datetime
import numpy as np
from PIL import Image
import rig_pose as RP
import rig_motion as RM
import rig_extras as RX

VIEWS = ("front", "side", "back")


def install(cls, G, K, tk, ttk, ImageTk, filedialog, messagebox):
    BG0, BG1, BG2, BG3, LINE, LINE2 = K["BG0"], K["BG1"], K["BG2"], K["BG3"], K["LINE"], K["LINE2"]
    TEXT, MUTED, DIM, ACC, PINK, GOLD, GOOD, WARN, BAD = (K[n] for n in ("TEXT", "MUTED", "DIM", "ACC", "PINK", "GOLD", "GOOD", "WARN", "BAD"))
    M = 14
    QCOL = dict(ok=GOOD, near="#d8e04a", off=WARN, gap=BAD)

    def add(fn):
        setattr(cls, fn.__name__, fn); return fn

    # ------------------------------------------------------------------ shared helpers
    @add
    def _al_dict(self):
        if self.res is None: return None
        d = getattr(self.res, "align", None)
        if d is None: d = self._align
        self.res.align = d; self._align = d
        return d

    @add
    def align_state(self):
        return self._align

    @add
    def align_learn(self):
        try:
            if self.model is not None and self.v_trainmodel.get() and self.loadout is not None:
                self.model.train_align(self.loadout.name, self._align or {})
        except Exception as ex: self.log(f"alignment learning skipped ({ex!r})", "warn")

    @add
    def _view_tabs(self, parent, getter, setter):
        r = tk.Frame(parent, bg=BG1); r.pack(fill="x"); btns = {}
        for v in VIEWS:
            b = self.mk_btn(r, v.upper(), lambda v=v: setter(v), "n", pady=4); b.pack(side="left", expand=True, fill="x", padx=1); btns[v] = b
        return btns

    # ================================================================== ALIGN PAGE
    @add
    def build_align(self, page):
        self.al_view, self.al_sel, self.al_zoom_mult, self.al_photo, self.al_meta = "front", None, 1.0, None, None
        self.v_al_mode = tk.StringVar(value="joint"); self.v_al_solid = tk.BooleanVar(value=False); self.v_al_seams = tk.BooleanVar(value=True)
        self._al_pending = False; self._al_drag = None; self._al_zoom = 4.0
        right = tk.Frame(page, bg=BG1, width=310); right.pack(side="right", fill="y", padx=8, pady=8); right.pack_propagate(False)
        left = tk.Frame(page, bg=BG1); left.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        bar = tk.Frame(left, bg=BG1); bar.pack(fill="x", pady=(0, 4))
        self.al_title = tk.Label(bar, text="ALIGN - tie the pieces together", bg=BG1, fg=TEXT, font=self.F["h"]); self.al_title.pack(side="left")
        self.mk_btn(bar, "+", lambda: self.al_zoom_by(1.25), "t", pady=2, padx=8).pack(side="right", padx=2)
        self.mk_btn(bar, "-", lambda: self.al_zoom_by(0.8), "t", pady=2, padx=8).pack(side="right")
        cv = tk.Canvas(left, bg="#05070b", highlightthickness=1, highlightbackground=LINE, cursor="fleur"); cv.pack(fill="both", expand=True); self.al_cv = cv
        cv.bind("<ButtonPress-1>", self.al_press); cv.bind("<B1-Motion>", self.al_motion); cv.bind("<ButtonRelease-1>", self.al_release)
        cv.bind("<Configure>", lambda e: self.after_idle(self.align_refresh))
        self.mk_label(left, "Drag a dot to move that joint (the part and everything hanging from it follows).  Arrow keys nudge the selected part "
                            "(Shift = bigger steps).  Dot colours: green = seam is tight, yellow/orange = slightly off, red = gap.",
                      DIM, wraplength=760, justify="left").pack(anchor="w", pady=(4, 0))
        self.al_vbtn = self._view_tabs(right, None, self.al_set_view)
        p, f = self.mk_panel(right, "WHAT A DRAG DOES", PINK); p.pack(fill="x", pady=(6, 0))
        for val, txt in (("joint", "MOVE JOINT  - shift the part against its parent"), ("pivot", "MOVE PIVOT  - shift the rotation centre only")):
            tk.Radiobutton(f, text=txt, value=val, variable=self.v_al_mode, bg=BG1, fg=TEXT, selectcolor=BG0, activebackground=BG1,
                           activeforeground=ACC, font=self.F["s"], anchor="w", highlightthickness=0).pack(fill="x")
        self.al_sel_lbl = tk.Label(f, text="selected: -", bg=BG1, fg=ACC, font=self.F["cardb"], anchor="w"); self.al_sel_lbl.pack(fill="x", pady=(4, 0))
        g = tk.Frame(f, bg=BG1); g.pack(pady=4)
        for txt, dx, dy, r, c in (("▲", 0, -1, 0, 1), ("◀", -1, 0, 1, 0), ("▶", 1, 0, 1, 2), ("▼", 0, 1, 2, 1)):
            self.mk_btn(g, txt, lambda dx=dx, dy=dy: self.al_nudge(dx * 0.5, dy * 0.5), "n", pady=2, padx=10).grid(row=r, column=c, padx=1, pady=1)
        r = tk.Frame(f, bg=BG1); r.pack(fill="x")
        self.mk_btn(r, "RESET PART", self.al_reset_part, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "RESET VIEW", self.al_reset_view, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")
        p, f = self.mk_panel(right, "AUTOMATIC"); p.pack(fill="x", pady=(6, 0))
        self.mk_btn(f, "⚙ AUTO TIE  (this view)", self.al_auto, "p", pady=6).pack(fill="x", pady=1)
        self.mk_btn(f, "⚙ AUTO TIE  (all views)", lambda: self.al_auto(all_views=True), "n", pady=4).pack(fill="x", pady=1)
        self.mk_label(f, "Centres stacked parts (spine, neck, head, arms, legs) on each other. Your manual moves are kept and refined.", DIM, wraplength=280, justify="left").pack(anchor="w")
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3, 0))
        self.mk_check(r, "Solid colours", self.v_al_solid, self.align_refresh).pack(side="left")
        self.mk_check(r, "Seam dots", self.v_al_seams, self.align_refresh).pack(side="left", padx=6)
        p, f = self.mk_panel(right, "SEAMS  (click to select)"); p.pack(fill="both", expand=True, pady=(6, 0))
        self.al_list = self.mk_list(f, 10); self.al_list.pack(fill="both", expand=True)
        self.al_list.bind("<<ListboxSelect>>", self.al_pick_list); self.al_keys = []
        self.al_set_view("front", refresh=False)

    @add
    def al_set_view(self, v, refresh=True):
        self.al_view = v
        for n, b in self.al_vbtn.items(): self.set_active(b, n == v)
        if refresh: self.align_refresh()

    @add
    def al_zoom_by(self, f):
        self.al_zoom_mult = float(np.clip(self.al_zoom_mult * f, 0.4, 6)); self.align_refresh()

    @add
    def align_refresh(self, *_):
        cv = self.al_cv
        if self.res is None:
            cv.delete("all"); cv.create_text(320, 200, text="CUT a rig first, then tie its pieces together here.", fill=DIM, font=self.F["h"]); return
        al = self._al_dict()
        try:
            img, meta = G["render_pose"](self.res, self.al_view, {}, solid=self.v_al_solid.get(), crop=True, rest=False)
            seams = RP.seam_report(G, self.res, self.al_view)
        except Exception as ex:
            self.log(f"align preview failed: {ex!r}", "warn"); return
        cw, ch = max(300, cv.winfo_width()), max(300, cv.winfo_height())
        z = max(1.0, min((cw - 2 * M) / img.width, (ch - 2 * M) / img.height)) * self.al_zoom_mult; self._al_zoom = z
        self.al_photo = ImageTk.PhotoImage(img.resize((max(1, int(img.width * z)), max(1, int(img.height * z))), Image.Resampling.NEAREST))
        self.al_meta = meta; cv.delete("all"); cv.create_image(M, M, image=self.al_photo, anchor="nw")
        for k, P in meta["pivots_px"].items():
            x, y = M + P[0] * z, M + P[1] * z; q = seams.get(k, {}).get("quality", "ok")
            col = QCOL[q] if self.v_al_seams.get() else MUTED; rr = 5 if k != self.al_sel else 8
            moved = bool((al.get(self.al_view) or {}).get(k))
            cv.create_oval(x - rr, y - rr, x + rr, y + rr, outline=PINK if k == self.al_sel else col, width=3 if k == self.al_sel else 2,
                           fill=(GOLD if moved else BG0), tags=("h:" + k,))
        cv.config(scrollregion=(0, 0, img.width * z + 2 * M, img.height * z + 2 * M))
        self.al_keys = [k for k in G["HIER"] if k in seams]; self.al_list.delete(0, "end")
        for i, k in enumerate(self.al_keys):
            s = seams[k]; dx = "  --  " if s["dx"] is None else f"{s['dx']:+5.1f}px"
            self.al_list.insert("end", f"{G['pretty'](k):10s} {dx}  cover {s['cover']*100:3.0f}%  {s['quality'].upper()}")
            self.al_list.itemconfig(i, fg=QCOL[s["quality"]])
        bad = sum(1 for s in seams.values() if s["quality"] in ("off", "gap"))
        self.al_title.config(text=f"ALIGN - {self.al_view.upper()}   ({bad} seam(s) need attention)" if bad else f"ALIGN - {self.al_view.upper()}   (all seams tight)")
        e = (al.get(self.al_view) or {}).get(self.al_sel) or {}
        self.al_sel_lbl.config(text=f"selected: {G['pretty'](self.al_sel) if self.al_sel else '-'}   moved {tuple(round(x, 1) for x in e.get('off', (0, 0)))} cm   pivot {tuple(round(x, 1) for x in e.get('piv', (0, 0)))} px")

    @add
    def al_pick_list(self, _e=None):
        s = self.al_list.curselection()
        if s and s[0] < len(self.al_keys): self.al_sel = self.al_keys[s[0]]; self.align_refresh()

    @add
    def _al_soon(self):
        if self._al_pending: return
        self._al_pending = True; self.after(30, self._al_flush)

    @add
    def _al_flush(self):
        self._al_pending = False; self.align_refresh()

    @add
    def al_press(self, e):
        if self.al_meta is None: return
        c = self.al_cv; mx, my = c.canvasx(e.x), c.canvasy(e.y); z = self._al_zoom; best, bd = None, 18
        for k, P in self.al_meta["pivots_px"].items():
            d = math.hypot(M + P[0] * z - mx, M + P[1] * z - my)
            if d < bd: best, bd = k, d
        if best: self.al_sel = best; self._al_drag = (mx, my); self.align_refresh()

    @add
    def _al_apply(self, k, dxpx, dypx):
        al = self._al_dict(); e = al.setdefault(self.al_view, {}).setdefault(k, {"off": [0.0, 0.0], "piv": [0.0, 0.0]})
        if self.v_al_mode.get() == "pivot": e["piv"] = [round(e["piv"][0] + dxpx, 3), round(e["piv"][1] + dypx, 3)]
        else: e["off"] = [round(e["off"][0] + dxpx * self.res.cpp, 3), round(e["off"][1] + dypx * self.res.cpp, 3)]

    @add
    def al_motion(self, e):
        if not self._al_drag or not self.al_sel: return
        c = self.al_cv; mx, my = c.canvasx(e.x), c.canvasy(e.y); z = self._al_zoom
        self._al_apply(self.al_sel, (mx - self._al_drag[0]) / z, (my - self._al_drag[1]) / z); self._al_drag = (mx, my); self._al_soon()

    @add
    def al_release(self, e):
        if self._al_drag: self._al_drag = None; self._al_commit()

    @add
    def al_nudge(self, dx, dy):
        if self.res is None or not self.al_sel: return self.toast("Click a joint dot first.", "warn")
        self._al_apply(self.al_sel, dx, dy); self.align_refresh(); self._al_commit(quiet=True)

    @add
    def align_key(self, e):
        if e.keysym not in ("Left", "Right", "Up", "Down") or not self.al_sel: return False
        st = 2.0 if (e.state & 0x1) else 0.5
        self.al_nudge({"Left": -st, "Right": st}.get(e.keysym, 0), {"Up": -st, "Down": st}.get(e.keysym, 0)); return True

    @add
    def _al_commit(self, quiet=False):
        try:
            self.res.align = self._al_dict(); G["render_outputs"](self.res); self.schedule_persist(); self.refresh_view_pages()
            if not quiet: self.toast("Alignment saved with the rig.", "ok")
        except Exception as ex: self.log(f"alignment save failed: {ex!r}", "err")

    @add
    def al_reset_part(self):
        if self.res is None or not self.al_sel: return
        self._al_dict().get(self.al_view, {}).pop(self.al_sel, None); self.align_refresh(); self._al_commit()

    @add
    def al_reset_view(self):
        if self.res is None: return
        self._al_dict().pop(self.al_view, None); self.align_refresh(); self._al_commit()

    @add
    def al_auto(self, all_views=False):
        if self.res is None: return self.toast("CUT a rig first.", "warn")
        self._al_dict(); n = 0
        for v in (VIEWS if all_views else (self.al_view,)):
            ch = RP.auto_tie(G, self.res, v); n += len(ch)
        self.align_refresh(); self._al_commit(quiet=True)
        self.toast(f"AUTO TIE moved {n} joint(s)." if n else "Nothing to tie - the seams already line up.", "ok")

    # ================================================================== ANIMATE PAGE
    @add
    def build_anim(self, page):
        self.an_view, self.an_key = "side", None
        self.an_frames, self.an_imgs, self.an_ghost, self.an_photos = [], [], [], []
        self.an_idx, self.an_playing, self.an_token, self.an_clip = 0, False, 0, None
        self.v_an_loop = tk.BooleanVar(value=True); self.v_an_ghost = tk.BooleanVar(value=True)
        self.v_an_speed = tk.DoubleVar(value=1.0); self.v_an_flex = tk.DoubleVar(value=1.0); self.v_an_bounce = tk.DoubleVar(value=1.0)
        self.v_an_inertia = tk.DoubleVar(value=1.0); self.v_an_jitter = tk.DoubleVar(value=0.0); self.v_an_fps = tk.IntVar(value=24)
        right = tk.Frame(page, bg=BG1, width=360); right.pack(side="right", fill="y", padx=8, pady=8); right.pack_propagate(False)
        left = tk.Frame(page, bg=BG1); left.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        self.an_title = tk.Label(left, text="ANIMATE", bg=BG1, fg=TEXT, font=self.F["h"], anchor="w"); self.an_title.pack(fill="x", pady=(0, 4))
        cv = tk.Canvas(left, bg="#05070b", highlightthickness=1, highlightbackground=LINE); cv.pack(fill="both", expand=True); self.an_cv = cv
        cv.bind("<Configure>", lambda e: self.after_idle(self.an_show))
        tr = tk.Frame(left, bg=BG1); tr.pack(fill="x", pady=(6, 0))
        self.mk_btn(tr, "▶ PLAY", self.an_play, "g", pady=4, padx=12).pack(side="left")
        self.mk_btn(tr, "⏸", self.an_pause, "n", pady=4, padx=10).pack(side="left", padx=3)
        self.mk_btn(tr, "⏹", self.an_stop, "n", pady=4, padx=10).pack(side="left")
        self.an_scrub = tk.Scale(tr, from_=0, to=1, orient="horizontal", showvalue=False, bg=BG1, troughcolor=BG0, highlightthickness=0, bd=0,
                                 activebackground=ACC, command=self.an_scrub_cb); self.an_scrub.pack(side="left", fill="x", expand=True, padx=10)
        self.an_info = tk.Label(tr, text="", bg=BG1, fg=MUTED, font=self.F["mono"], width=22, anchor="e"); self.an_info.pack(side="left")
        self.mk_label(left, "The coloured ghost is the 'muscle' pose; the solid figure is the body following it. The gap between them is the lag: "
                            "hips lead, then spine, then arms, then hands.", DIM, wraplength=760, justify="left").pack(anchor="w", pady=(4, 0))
        self.an_vbtn = self._view_tabs(right, None, self.an_set_view)
        p, f = self.mk_panel(right, "ACTIONS  (what this body can do in this view)", PINK); p.pack(fill="both", expand=True, pady=(6, 0))
        self.an_list = self.mk_list(f, 11); self.an_list.pack(fill="both", expand=True); self.an_list.bind("<<ListboxSelect>>", self.an_pick)
        self.an_desc = tk.Label(f, text="", bg=BG1, fg=MUTED, font=self.F["s"], wraplength=330, justify="left", anchor="nw"); self.an_desc.pack(fill="x", pady=(4, 0))
        p, f = self.mk_panel(right, "SECONDARY MOTION  (lag, bounce, jitter)"); p.pack(fill="x", pady=(6, 0))
        for lab, var, lo, hi, res_ in (("SPEED", self.v_an_speed, 0.25, 2.0, 0.05), ("FLEX / LAG   0 = rigid", self.v_an_flex, 0.0, 2.0, 0.05),
                                       ("BOUNCE (overshoot)", self.v_an_bounce, 0.2, 2.0, 0.05), ("INERTIA (pushed by hips)", self.v_an_inertia, 0.0, 2.0, 0.05),
                                       ("JITTER (tremble)", self.v_an_jitter, 0.0, 3.0, 0.1)):
            self.mk_label(f, lab).pack(anchor="w")
            s = self.mk_scale(f, var, lo, hi, res=res_); s.config(length=320); s.pack(fill="x"); s.bind("<ButtonRelease-1>", lambda e: self.an_build())
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3, 0))
        self.mk_check(r, "Loop", self.v_an_loop).pack(side="left")
        self.mk_check(r, "Ghost", self.v_an_ghost, self.an_rephoto).pack(side="left", padx=6)
        self.mk_label(r, "FPS").pack(side="left", padx=(8, 0))
        tk.Spinbox(r, values=(12, 15, 24, 30), width=3, textvariable=self.v_an_fps, bg=BG0, fg=TEXT, relief="flat", buttonbackground=BG3, font=self.F["mono"],
                   command=self.an_build).pack(side="left", padx=4)
        r = tk.Frame(right, bg=BG1); r.pack(fill="x", pady=(6, 0))
        self.mk_btn(r, "⬇ EXPORT THIS", self.an_export, "g", pady=6).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "⬇ EXPORT ALL (view)", self.an_export_all, "n", pady=6).pack(side="left", expand=True, fill="x")
        self.an_set_view("side", refresh=False)

    @add
    def an_set_view(self, v, refresh=True):
        self.an_view = v
        for n, b in self.an_vbtn.items(): self.set_active(b, n == v)
        if refresh: self.anim_refresh()

    @add
    def anim_refresh(self, *_):
        self.an_stop(quiet=True); self.an_list.delete(0, "end"); self.an_actions = []
        if self.res is None:
            self.an_cv.delete("all"); self.an_cv.create_text(320, 200, text="CUT a rig first, then bring it to life here.", fill=DIM, font=self.F["h"]); return
        self.an_actions = RM.list_actions(self.res, self.an_view)
        first = None
        for i, a in enumerate(self.an_actions):
            icon = "✕" if not a["ok"] else ("✓" if a["quality"] >= 0.85 else "⚠")
            self.an_list.insert("end", f"{icon}  {a['label']:14s} {'' if not a['ok'] else str(int(a['quality']*100)) + '%'}")
            self.an_list.itemconfig(i, fg=DIM if not a["ok"] else (GOOD if a["quality"] >= 0.85 else WARN))
            if first is None and a["ok"] and (self.an_key is None or a["key"] == self.an_key): first = i
        if first is None: first = next((i for i, a in enumerate(self.an_actions) if a["ok"]), None)
        if first is not None:
            self.an_list.selection_clear(0, "end"); self.an_list.selection_set(first); self.an_pick()

    @add
    def an_pick(self, _e=None):
        s = self.an_list.curselection()
        if not s: return
        a = self.an_actions[s[0]]; self.an_desc.config(text=a["note"], fg=MUTED if a["ok"] else DIM)
        if not a["ok"]: return
        self.an_key = a["key"]; self.an_build()

    @add
    def an_build(self, *_):
        if self.res is None or self.an_key is None: return
        self.an_stop(quiet=True); self.an_token += 1; tok = self.an_token
        L = dict(self.res.lengths); fps = int(self.v_an_fps.get() or 24)
        self.an_clip = RM.build_clip(self.an_key, self.an_view, L)
        kw = dict(speed=self.v_an_speed.get(), fps=fps)
        self.an_frames = RM.simulate(self.an_clip, L, flex=self.v_an_flex.get(), bounce=self.v_an_bounce.get(), inertia=self.v_an_inertia.get(), jitter=self.v_an_jitter.get(), **kw)
        self.an_frames = self._an_extras(self.an_frames, self.an_clip, self.v_an_flex.get())
        ghost = self._an_extras(RM.simulate(self.an_clip, L, flex=0.0, **kw), self.an_clip, 0.0); self.an_imgs, self.an_ghost, self.an_photos = [], [], []
        self.an_scrub.config(to=max(1, len(self.an_frames) - 1)); self.config(cursor="watch")
        def chunk():
            if tok != self.an_token: self.config(cursor=""); return
            for _ in range(3):
                i = len(self.an_imgs)
                if i >= len(self.an_frames): break
                fr, gr = self.an_frames[i], ghost[min(i, len(ghost) - 1)]
                self.an_imgs.append(G["render_pose"](self.res, self.an_view, fr["angles"], shade=True, crop=False, rest=False, root=fr["root"])[0])
                self.an_ghost.append(G["render_pose"](self.res, self.an_view, gr["angles"], shade=True, crop=False, rest=False, root=gr["root"])[0])
            self.an_info.config(text=f"rendering {len(self.an_imgs)}/{len(self.an_frames)}")
            if len(self.an_imgs) < len(self.an_frames): self.after(5, chunk)
            else: self.config(cursor=""); self.an_rephoto(); self.an_play()
        chunk()

    @add
    def _an_extras(self, frames, clip, flex):
        """Add the motion of the EXTRA parts (tail, ears, wings ...) to simulated frames."""
        try:
            return RX.simulate_extras(frames, getattr(self.res, "extras", None) or [], self.res, self.an_view,
                                      fps=int(self.v_an_fps.get() or 24), loop=clip.loop, flex=flex, bounce=self.v_an_bounce.get(),
                                      inertia=self.v_an_inertia.get(), speed=self.v_an_speed.get(), clip_dur=clip.dur)
        except Exception as ex:
            self.log(f"extra-part motion skipped ({ex!r})", "warn"); return frames

    @add
    def an_rephoto(self, *_):
        cv = self.an_cv; self.an_photos = []
        if not self.an_imgs: return
        cw, ch = max(300, cv.winfo_width() - 20), max(300, cv.winfo_height() - 20); w, h = self.an_imgs[0].size
        z = max(1.0, min(cw / w, ch / h)); self._an_z = z
        for i, im in enumerate(self.an_imgs):
            if self.v_an_ghost.get():
                g = self.an_ghost[i].copy(); a = np.asarray(g).copy(); a[..., 3] = (a[..., 3] * 0.38).astype(np.uint8); a[..., :3] = (a[..., :3] * 0.5 + np.array([0, 229, 255]) * 0.5).astype(np.uint8)
                im = Image.alpha_composite(Image.fromarray(a), im)
            self.an_photos.append(ImageTk.PhotoImage(im.resize((int(w * z), int(h * z)), Image.Resampling.NEAREST)))
        self.an_show()

    @add
    def an_show(self, *_):
        cv = self.an_cv
        if not self.an_photos: return
        i = min(self.an_idx, len(self.an_photos) - 1); z = getattr(self, "_an_z", 1.0)
        cv.delete("all"); L = self.res.lengths; w, h = self.an_imgs[0].size
        gy = 10 + (0.55 * h + (L["thigh"] + L["shin"]) / self.res.cpp + 3) * z
        cv.create_line(0, gy, cv.winfo_width(), gy, fill=LINE2, dash=(4, 4)); cv.create_image(cv.winfo_width() // 2, 10, image=self.an_photos[i], anchor="n")
        f = self.an_frames[i]
        self.an_info.config(text=f"{self.an_clip.label}  {f['t']:.2f}s  {i+1}/{len(self.an_frames)}")
        self.an_title.config(text=f"ANIMATE - {self.an_view.upper()} - {self.an_clip.label}")

    @add
    def an_play(self):
        if not self.an_photos: return
        if not self.an_playing:
            self.an_playing = True; self._an_tick()

    @add
    def an_pause(self): self.an_playing = False

    @add
    def an_stop(self, quiet=False):
        self.an_playing = False; self.an_idx = 0
        if not quiet and self.an_photos: self.an_show()

    @add
    def an_scrub_cb(self, v):
        if self.an_playing or not self.an_photos: return
        self.an_idx = int(float(v)); self.an_show()

    @add
    def _an_tick(self):
        if not self.an_playing or not self.an_photos: return
        self.an_show(); self.an_scrub.set(self.an_idx)
        self.an_idx += 1
        if self.an_idx >= len(self.an_photos):
            if self.v_an_loop.get(): self.an_idx = 0
            else: self.an_playing = False; self.an_idx = len(self.an_photos) - 1; return
        self.after(int(1000 / max(1, int(self.v_an_fps.get() or 24))), self._an_tick)

    @add
    def _an_folder(self, key):
        lo = self.ensure_loadout(); return lo.path / "animations" / f"{self.an_view}_{key}"

    @add
    def _an_write(self, clip, frames, imgs, key):
        folder = self._an_folder(key)
        meta = dict(action=clip.label, view=self.an_view, loop=clip.loop, speed=self.v_an_speed.get(),
                    flex=self.v_an_flex.get(), bounce=self.v_an_bounce.get(), inertia=self.v_an_inertia.get(), jitter=self.v_an_jitter.get(),
                    note="angles: degrees, clockwise positive, relative to the parent part, ABSOLUTE (rest pose included). root: pelvis offset in cm "
                         "(x right, y down). Drive rig.json pivots with these, or use the PNG/GIF.")
        fr = [dict(t=round(f["t"], 4), root_cm=[round(f["root"][0], 2), round(f["root"][1], 2)], angles={k: round(v, 2) for k, v in f["angles"].items()}) for f in frames]
        return RM.export_animation(imgs, fr, int(self.v_an_fps.get() or 24), str(folder), f"{self.an_view}_{key}", meta)

    @add
    def an_export(self):
        if not self.an_imgs: return self.toast("Pick an action first.", "warn")
        try: d = self._an_write(self.an_clip, self.an_frames, self.an_imgs, self.an_key); self.toast(f"Exported (GIF, animated PNG, sprite sheet, JSON) to {d}", "ok")
        except Exception as ex: self.toast(f"Export failed: {ex}", "err"); self.log(repr(ex), "err")

    @add
    def an_export_all(self):
        if self.res is None: return self.toast("CUT a rig first.", "warn")
        L = dict(self.res.lengths); fps = int(self.v_an_fps.get() or 24); done = 0; self.config(cursor="watch")
        try:
            for a in RM.list_actions(self.res, self.an_view):
                if not a["ok"]: continue
                self.toast(f"Exporting {a['label']}..."); self.update_idletasks()
                clip = RM.build_clip(a["key"], self.an_view, L)
                fr = self._an_extras(RM.simulate(clip, L, flex=self.v_an_flex.get(), bounce=self.v_an_bounce.get(), inertia=self.v_an_inertia.get(), jitter=self.v_an_jitter.get(), speed=self.v_an_speed.get(), fps=fps), clip, self.v_an_flex.get())
                im = [G["render_pose"](self.res, self.an_view, f["angles"], shade=True, crop=False, rest=False, root=f["root"])[0] for f in fr]
                self._an_write(clip, fr, im, a["key"]); done += 1
            self.toast(f"Exported {done} animation(s) to {self.loadout.path / 'animations'}", "ok")
        except Exception as ex: self.toast(f"Export failed: {ex}", "err"); self.log(repr(ex), "err")
        finally: self.config(cursor="")
