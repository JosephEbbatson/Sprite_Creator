"""rig_trim_gui.py - the TRIM page for the HUMAN.RIG window (installed into RigWindow by install()).

Erase leftover pixels from a finished part, and see the effect on the assembled figure in a test pose
(arms up, legs apart) at the same time.  Erased pixels show as a red ghost in the edit view; the RESTORE brush
paints them back.  Trims are remembered per view and part and survive CUTting again (see rig_trim.py).
"""
import math
import numpy as np
from PIL import Image
import rig_trim as TRM

VIEWS = ("front", "side", "back")
M = 14


def install(cls, G, K, tk, ttk, ImageTk, filedialog, messagebox):
    BG0, BG1, BG2, BG3, LINE, LINE2 = K["BG0"], K["BG1"], K["BG2"], K["BG3"], K["LINE"], K["LINE2"]
    TEXT, MUTED, DIM, ACC, PINK, GOLD, GOOD, WARN, BAD = (K[n] for n in ("TEXT", "MUTED", "DIM", "ACC", "PINK", "GOLD", "GOOD", "WARN", "BAD"))
    PART_KEYS = G["PART_KEYS"]
    TOOLS = (("erase", "ERASE BRUSH", "paint over pixels to remove them"),
             ("restore", "RESTORE BRUSH", "paint over red ghost pixels to bring them back"),
             ("lasso_erase", "ERASE INSIDE SHAPE", "click around an area, close the shape: everything inside goes"),
             ("lasso_keep", "KEEP ONLY INSIDE SHAPE", "click around what you want to keep: everything outside goes"))

    def add(fn):
        setattr(cls, fn.__name__, fn); return fn

    # ================================================================== page
    @add
    def build_trim(self, page):
        self.tr_view, self.tr_key = "front", "uarmR"
        self.v_tr_tool, self.v_tr_size, self.v_tr_pose = tk.StringVar(value="erase"), tk.DoubleVar(value=1.5), tk.StringVar(value="arms")
        self.tr_zoom_mult, self.tr_pts, self.tr_closed = 1.0, [], False
        self.tr_undo, self.tr_redo, self._tr_last, self._tr_z = [], [], None, 1.0
        self._tr_photo = self._tr_pphoto = None; self._tr_prev_pending = False; self._tr_keys = []; self._tr_cursor = None

        right_outer, R = self._ex_scrollframe(page, 360)
        right_outer.pack(side="right", fill="y", padx=8, pady=8)
        left = tk.Frame(page, bg=BG1); left.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)

        bar = tk.Frame(left, bg=BG1); bar.pack(fill="x", pady=(0, 4))
        self.tr_title = tk.Label(bar, text="TRIM - erase leftover pixels", bg=BG1, fg=TEXT, font=self.F["h"]); self.tr_title.pack(side="left")
        self.mk_btn(bar, "+", lambda: self.tr_zoom_by(1.25), "t", pady=2, padx=8).pack(side="right", padx=2)
        self.mk_btn(bar, "-", lambda: self.tr_zoom_by(0.8), "t", pady=2, padx=8).pack(side="right")
        vb = tk.Frame(left, bg=BG1); vb.pack(fill="x", pady=(0, 4)); self.tr_vbtn = {}
        for v in VIEWS:
            b = self.mk_btn(vb, v.upper(), lambda v=v: self.tr_set_view(v), "n", pady=3); b.pack(side="left", expand=True, fill="x", padx=1); self.tr_vbtn[v] = b
        cv = tk.Canvas(left, bg="#05070b", highlightthickness=1, highlightbackground=LINE, cursor="crosshair"); cv.pack(fill="both", expand=True); self.tr_cv = cv
        cv.bind("<ButtonPress-1>", self.tr_press); cv.bind("<B1-Motion>", self.tr_motion); cv.bind("<ButtonRelease-1>", self.tr_release)
        cv.bind("<Double-Button-1>", self.tr_double); cv.bind("<Button-3>", self.tr_rclick); cv.bind("<Motion>", self.tr_hover)
        cv.bind("<Configure>", lambda e: self.after_idle(self.tr_redraw))
        self.tr_hint = self.mk_label(left, "", DIM, wraplength=760, justify="left"); self.tr_hint.pack(anchor="w", pady=(4, 0))

        # ---- right: preview
        p, f = self.mk_panel(R, "LIVE PREVIEW  (the whole figure, in a test pose)", PINK); p.pack(fill="x", pady=(0, 6))
        self.tr_pv = tk.Canvas(f, bg="#05070b", highlightthickness=1, highlightbackground=LINE, width=330, height=330); self.tr_pv.pack()
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4, 0)); self.tr_pbtn = {}
        for key, txt in TRM.POSES:
            b = self.mk_btn(r, txt, lambda k=key: self.tr_set_pose(k), "n", pady=3, font=self.F["card"]); b.pack(side="left", expand=True, fill="x", padx=1); self.tr_pbtn[key] = b
        # ---- right: part list
        p, f = self.mk_panel(R, "PART  (pick the one to fix)", GOLD); p.pack(fill="x", pady=(0, 6))
        self.tr_lb = self.mk_list(f, 17); self.tr_lb.pack(fill="x"); self.tr_lb.bind("<<ListboxSelect>>", self.tr_pick)
        # ---- right: tools
        p, f = self.mk_panel(R, "TOOL", ACC); p.pack(fill="x", pady=(0, 6))
        for val, txt, _tip in TOOLS:
            tk.Radiobutton(f, text=txt, value=val, variable=self.v_tr_tool, command=self.tr_tool_changed, bg=BG1, fg=TEXT, selectcolor=BG0,
                           activebackground=BG1, activeforeground=ACC, font=self.F["s"], anchor="w", highlightthickness=0).pack(fill="x")
        self.mk_label(f, "BRUSH SIZE  (pixels)").pack(anchor="w", pady=(4, 0))
        s = self.mk_scale(f, self.v_tr_size, 0.5, 8.0, lambda v: self.tr_hover(None), res=0.5); s.config(length=320); s.pack(fill="x")
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4, 0))
        self.mk_btn(r, "↶ UNDO", self.tr_do_undo, "n", pady=4).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "↷ REDO", self.tr_do_redo, "n", pady=4).pack(side="left", expand=True, fill="x")
        r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(3, 0))
        self.mk_btn(r, "RESET THIS PART", self.tr_reset_part, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x", padx=(0, 2))
        self.mk_btn(r, "RESET EVERYTHING", self.tr_reset_all, "d", pady=3, font=self.F["card"]).pack(side="left", expand=True, fill="x")
        self.tr_tip = self.mk_label(f, "", DIM, wraplength=320, justify="left"); self.tr_tip.pack(anchor="w", pady=(4, 0))
        self.mk_label(f, "Keys: Ctrl+Z undo, Ctrl+Y redo, [ and ] change brush size. Nothing is destroyed: the raw cut is kept and RESTORE brings pixels back.",
                      DIM, wraplength=320, justify="left").pack(anchor="w", pady=(4, 0))
        if hasattr(self, "_ex_wheel_guard"): self._ex_wheel_guard(R, right_outer)
        self.tr_set_view("front", refresh=False); self.tr_set_pose("arms", refresh=False); self.tr_tool_changed()

    # ================================================================== state
    @add
    def _tr_raw(self):
        try: return self.res.views_raw[self.tr_view][self.tr_key]
        except Exception: return None

    @add
    def _tr_ok(self):
        sp = self._tr_raw()
        return sp is not None and getattr(sp, "kind", "") not in TRM.SKIP_KINDS

    @add
    def trim_refresh(self, *_):
        self._tr_fill_list(); self.tr_redraw()

    @add
    def _tr_fill_list(self):
        self.tr_lb.delete(0, "end"); self._tr_keys = []
        if self.res is None: return
        active = set(G["RB"].active_keys(getattr(self.res, "mask", None)))
        for k in PART_KEYS:
            if k not in active: continue
            n = TRM.count(self.res, self.tr_view, k); self._tr_keys.append(k)
            self.tr_lb.insert("end", f"{G['pretty'](k):10s} {str(n) + ' px erased' if n else ''}")
            self.tr_lb.itemconfig(len(self._tr_keys) - 1, fg=GOLD if n else TEXT)
        if self.tr_key in self._tr_keys:
            self.tr_lb.selection_clear(0, "end"); self.tr_lb.selection_set(self._tr_keys.index(self.tr_key))

    @add
    def tr_pick(self, _e=None):
        s = self.tr_lb.curselection()
        if s and s[0] < len(self._tr_keys):
            self.tr_key = self._tr_keys[s[0]]; self.tr_pts, self.tr_closed = [], False; self.tr_redraw()

    @add
    def tr_set_view(self, v, refresh=True):
        self.tr_view = v; self.tr_pts, self.tr_closed = [], False
        for n, b in self.tr_vbtn.items(): self.set_active(b, n == v)
        if refresh: self._tr_fill_list(); self.tr_redraw()

    @add
    def tr_set_pose(self, p, refresh=True):
        self.v_tr_pose.set(p)
        for n, b in self.tr_pbtn.items(): self.set_active(b, n == p)
        if refresh: self._tr_draw_preview()

    @add
    def tr_zoom_by(self, f):
        self.tr_zoom_mult = float(np.clip(self.tr_zoom_mult * f, 0.3, 8)); self.tr_redraw()

    @add
    def tr_tool_changed(self):
        t = self.v_tr_tool.get(); self.tr_pts, self.tr_closed = [], False
        self.tr_tip.config(text=next(tip for v, _n, tip in TOOLS if v == t))
        self.tr_redraw()

    # ================================================================== drawing
    @add
    def tr_redraw(self, *_, fast=False):
        cv = self.tr_cv; cv.delete("all"); self._tr_photo = None
        preview = self._tr_preview_soon if fast else self._tr_draw_preview
        if self.res is None:
            cv.create_text(320, 200, text="CUT a rig first, then fine-tune its parts here.", fill=DIM, font=self.F["h"]); self.tr_pv.delete("all"); return
        if not getattr(self.res, "views_raw", None): return
        if not self._tr_ok():
            cv.create_text(max(300, cv.winfo_width()) // 2, 180, text=f"{G['pretty'](self.tr_key)} has no cut in the {self.tr_view.upper()} view\n(it is missing, switched off, or only a placeholder).", fill=DIM, font=self.F["h"], justify="center")
            self.tr_title.config(text="TRIM"); preview(); return
        raw = self._tr_raw(); h, w = raw.rgba.shape[:2]; offs = TRM.offsets(self.res, self.tr_view, self.tr_key)
        cw, ch = max(300, cv.winfo_width()), max(300, cv.winfo_height())
        self._tr_z = z = max(1.0, min((cw - 2 * M) / w, (ch - 2 * M) / h, 40.0)) * self.tr_zoom_mult
        bg = Image.new("RGBA", (w, h), (28, 32, 42, 255)); bg.alpha_composite(Image.fromarray(raw.rgba)); arr = np.array(bg)
        m = TRM.erased_mask((h, w), raw, offs) & (raw.rgba[..., 3] > 0)
        if m.any(): arr[m, :3] = (arr[m, :3] * 0.35 + np.array([255, 50, 50]) * 0.65).astype(np.uint8)
        im = Image.fromarray(arr).convert("RGB").resize((max(1, int(w * z)), max(1, int(h * z))), Image.Resampling.NEAREST)
        self._tr_photo = ImageTk.PhotoImage(im); cv.create_image(M, M, image=self._tr_photo, anchor="nw")
        px, py = M + raw.pj[0] * z, M + raw.pj[1] * z
        cv.create_line(px - 10, py, px + 10, py, fill=PINK); cv.create_line(px, py - 10, px, py + 10, fill=PINK)
        self._tr_draw_poly()
        n = len(offs); self.tr_title.config(text=f"TRIM - {G['pretty'](self.tr_key)}  ({self.tr_view.upper()})   {n} px erased" if n else f"TRIM - {G['pretty'](self.tr_key)}  ({self.tr_view.upper()})")
        self.tr_hint.config(text="Red = erased (RESTORE brings it back). The pink cross is the part's pivot: the point it rotates about. "
                                 "Use the preview on the right to see what shows when the limb moves.")
        preview()

    @add
    def _tr_draw_poly(self):
        cv = self.tr_cv; cv.delete("poly")
        if not self.v_tr_tool.get().startswith("lasso") or not self.tr_pts: return
        z = self._tr_z; pts = [(M + x * z, M + y * z) for x, y in self.tr_pts]
        if len(pts) >= 2: cv.create_line(*[c for p in pts for c in p], fill=ACC, width=2, tags="poly")
        for i, (x, y) in enumerate(pts):
            r = 6 if i == 0 else 3; cv.create_oval(x - r, y - r, x + r, y + r, outline=PINK if i == 0 else ACC, fill=BG0, width=2, tags="poly")

    @add
    def tr_hover(self, e):
        cv = self.tr_cv; cv.delete("cursor")
        if e is None or self.v_tr_tool.get() not in ("erase", "restore") or not self._tr_ok(): return
        r = max(0.5, float(self.v_tr_size.get())) * self._tr_z
        cv.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, outline=GOOD if self.v_tr_tool.get() == "restore" else BAD, width=2, tags="cursor")

    @add
    def _tr_draw_preview(self):
        pv = self.tr_pv; pv.delete("all"); self._tr_pphoto = None
        if self.res is None: return
        try:
            img, _meta = G["render_pose"](self.res, self.tr_view, TRM.test_pose(self.tr_view, self.v_tr_pose.get()), solid=False, shade=True, crop=True, rest=False)
        except Exception as ex:
            self.log(f"trim preview failed: {ex!r}", "warn"); return
        W, H = 330, 330; z = min((W - 10) / img.width, (H - 10) / img.height, 8.0)
        bg = Image.new("RGBA", img.size, (5, 7, 11, 255)); bg.alpha_composite(img)
        im = bg.convert("RGB").resize((max(1, int(img.width * z)), max(1, int(img.height * z))), Image.Resampling.NEAREST if z >= 1 else Image.Resampling.LANCZOS)
        self._tr_pphoto = ImageTk.PhotoImage(im); pv.create_image(W // 2, H // 2, image=self._tr_pphoto)

    @add
    def _tr_preview_soon(self):
        if self._tr_prev_pending: return
        self._tr_prev_pending = True; self.after(120, self._tr_prev_flush)

    @add
    def _tr_prev_flush(self):
        self._tr_prev_pending = False; self._tr_draw_preview()

    # ================================================================== editing
    @add
    def _tr_pt(self, e):
        z = self._tr_z; return ((e.x - M) / z, (e.y - M) / z)

    @add
    def _tr_push_undo(self):
        self.tr_undo.append((self.tr_view, self.tr_key, set(TRM.offsets(self.res, self.tr_view, self.tr_key)))); self.tr_redo = []
        del self.tr_undo[:-80]

    @add
    def _tr_set(self, offs):
        TRM.set_offsets(self.res, self.tr_view, self.tr_key, offs)

    @add
    def tr_press(self, e):
        self.tr_cv.focus_set()
        if self.res is None or not self._tr_ok(): return
        t = self.v_tr_tool.get(); raw = self._tr_raw()
        if t in ("erase", "restore"):
            self._tr_push_undo(); p = self._tr_pt(e); self._tr_last = p
            m = TRM.brush_mask(raw.rgba.shape[:2], p[0], p[1], float(self.v_tr_size.get()))
            offs = TRM.offsets(self.res, self.tr_view, self.tr_key)
            self._tr_set(TRM.erase(offs, raw, m) if t == "erase" else TRM.restore(offs, raw, m)); self.tr_redraw(fast=True)
        else:
            if self.tr_closed: return
            z = self._tr_z
            if len(self.tr_pts) >= 3 and math.hypot(M + self.tr_pts[0][0] * z - e.x, M + self.tr_pts[0][1] * z - e.y) < 10: return self.tr_close()
            self.tr_pts.append(self._tr_pt(e)); self._tr_draw_poly()

    @add
    def tr_motion(self, e):
        self.tr_hover(e)
        if self._tr_last is None or not self._tr_ok(): return
        t = self.v_tr_tool.get(); raw = self._tr_raw(); p = self._tr_pt(e)
        m = TRM.stroke_mask(raw.rgba.shape[:2], self._tr_last, p, float(self.v_tr_size.get())); self._tr_last = p
        offs = TRM.offsets(self.res, self.tr_view, self.tr_key)
        self._tr_set(TRM.erase(offs, raw, m) if t == "erase" else TRM.restore(offs, raw, m)); self.tr_redraw(fast=True)

    @add
    def tr_release(self, e):
        if self._tr_last is not None:
            self._tr_last = None; self._tr_fill_list(); self.tr_commit(); self.tr_redraw()

    @add
    def tr_double(self, e):
        if self.v_tr_tool.get().startswith("lasso") and not self.tr_closed:
            if len(self.tr_pts) >= 2 and math.hypot(self.tr_pts[-1][0] - self.tr_pts[-2][0], self.tr_pts[-1][1] - self.tr_pts[-2][1]) < 2: self.tr_pts.pop()
            self.tr_close()

    @add
    def tr_rclick(self, e):
        if self.v_tr_tool.get().startswith("lasso") and self.tr_pts: self.tr_pts.pop(); self._tr_draw_poly()

    @add
    def tr_close(self):
        if len(self.tr_pts) < 3: return self.toast("Click at least 3 points around the area first.", "warn")
        raw = self._tr_raw(); m = TRM.polygon_mask(raw.rgba.shape[:2], self.tr_pts); offs = TRM.offsets(self.res, self.tr_view, self.tr_key)
        self._tr_push_undo()
        self._tr_set(TRM.erase(offs, raw, m) if self.v_tr_tool.get() == "lasso_erase" else TRM.keep_only(offs, raw, m))
        self.tr_pts, self.tr_closed = [], False; self._tr_fill_list(); self.tr_commit(); self.tr_redraw()

    @add
    def tr_commit(self):
        """Let the rig sheet / atlas / animation see the change and autosave it with the loadout."""
        try:
            G["render_outputs"](self.res); self.schedule_persist(); self.refresh_view_pages()
        except Exception as ex: self.log(f"trim: could not refresh the rig ({ex!r})", "err")

    @add
    def _tr_jump(self, view, key):
        self.tr_view, self.tr_key = view, key
        for n, b in self.tr_vbtn.items(): self.set_active(b, n == view)

    @add
    def tr_do_undo(self):
        if not self.tr_undo: return self.toast("Nothing to undo.", "warn")
        v, k, offs = self.tr_undo.pop(); self._tr_jump(v, k)
        self.tr_redo.append((v, k, set(TRM.offsets(self.res, v, k)))); TRM.set_offsets(self.res, v, k, offs)
        self._tr_fill_list(); self.tr_commit(); self.tr_redraw()

    @add
    def tr_do_redo(self):
        if not self.tr_redo: return self.toast("Nothing to redo.", "warn")
        v, k, offs = self.tr_redo.pop(); self._tr_jump(v, k)
        self.tr_undo.append((v, k, set(TRM.offsets(self.res, v, k)))); TRM.set_offsets(self.res, v, k, offs)
        self._tr_fill_list(); self.tr_commit(); self.tr_redraw()

    @add
    def tr_reset_part(self):
        if self.res is None or not self._tr_ok(): return
        if not TRM.count(self.res, self.tr_view, self.tr_key): return self.toast("This part has no erased pixels.", "warn")
        self._tr_push_undo(); self._tr_set(set()); self._tr_fill_list(); self.tr_commit(); self.tr_redraw()

    @add
    def tr_reset_all(self):
        if self.res is None or not TRM.count(self.res): return self.toast("Nothing is trimmed yet.", "warn")
        if not messagebox.askyesno("Reset everything", f"Bring back all {TRM.count(self.res)} erased pixels in every part and view?\n(this cannot be undone)", parent=self): return
        for v, parts in list((self.res.trims or {}).items()):
            for k in list(parts): TRM.set_offsets(self.res, v, k, set())
        self.tr_undo, self.tr_redo = [], []; self._tr_fill_list(); self.tr_commit(); self.tr_redraw()

    @add
    def trim_key(self, ev):
        """Ctrl+Z / Ctrl+Y / [ ] on the TRIM page. Returns True when handled."""
        try:
            if isinstance(self.focus_get(), (tk.Entry, ttk.Combobox, tk.Spinbox)): return False
        except Exception: pass
        ctrl = bool(ev.state & 0x4)
        if ctrl and ev.keysym.lower() == "z": self.tr_do_undo(); return True
        if ctrl and ev.keysym.lower() == "y": self.tr_do_redo(); return True
        if ev.keysym == "bracketleft": self.v_tr_size.set(max(0.5, self.v_tr_size.get() - 0.5)); self.tr_hover(None); return True
        if ev.keysym == "bracketright": self.v_tr_size.set(min(8.0, self.v_tr_size.get() + 0.5)); self.tr_hover(None); return True
        return False
