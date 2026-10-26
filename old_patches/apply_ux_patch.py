"""Patch 2 for HUMAN.RIG: hidden-area fill (arm over body), clear workflow coach, save status, model badge,
RIG CHECK and SNAPSHOT.  Run AFTER apply_model_patch.py, with rig_tools.py next to your rig file.
usage:  python apply_ux_patch.py human_rig.py"""
import sys, shutil
E = []
def edit(old, new): E.append((old, new))

# ---- imports / flags -------------------------------------------------------------------------------------------
edit("import organism_model as OM", "import organism_model as OM\nimport rig_tools as RT")
edit("self.model = None; self._psel = None",
     "self.model = None; self._psel = None\n            self.v_fillhidden = tk.BooleanVar(value=True); self.last_saved = None")

# ---- arm-over-body: fill what a front part hides ------------------------------------------------------------------
edit("def render_part(fd, k, pid, A, B, spec, Ltrue, cpp=CM_PER_PX, athr=110, auto_ext=True, ext_scale=1.0):",
     "def render_part(fd, k, pid, A, B, spec, Ltrue, cpp=CM_PER_PX, athr=110, auto_ext=True, ext_scale=1.0, fill=True):")
edit("    sel = own | part_ext_mask(fd, k, x0, y0, x1, y1, auto_ext, ext_scale)",
'''    sel = own | part_ext_mask(fd, k, x0, y0, x1, y1, auto_ext, ext_scale)
    rgbc = fd.rgb[y0:y1, x0:x1]
    if fill:                       # material hidden behind a part drawn in FRONT (arm over torso, near leg over far leg)
        try:
            cap = next(c for c in fd.caps if c[0] == k)
            _o = draw_order(fd.facing); _fr = [PART_KEYS.index(q) for q in _o[_o.index(k)+1:]]
            hid = RT.hidden_mask(fd.lab[y0:y1, x0:x1], fd.alpha[y0:y1, x0:x1] > 127, sel, _fr,
                                 cap[1], cap[2], cap[3], x0, y0, 14*s)
            if hid.any() and hid.sum() <= 0.6*max(1, int(sel.sum())):
                rgbc, _filled = RT.inpaint(rgbc, sel, hid); sel = sel | _filled
        except Exception: pass''')
edit('    im = Image.fromarray(np.dstack([fd.rgb[y0:y1, x0:x1], al])).convert("RGBa")',
     '    im = Image.fromarray(np.dstack([rgbc, al])).convert("RGBa")')
edit('model=None, learn_model=True, owner="",', 'model=None, learn_model=True, owner="", fill_hidden=True,')
edit("cpp, athr, overlap, overlap_scale)", "cpp, athr, overlap, overlap_scale, fill_hidden)")
edit("sp = sprite_from_img(k, smooth_sprite(img, smooth), best, L, cpp, nused, kind)",
     "sp = sprite_from_img(k, (RT.fill_holes(smooth_sprite(img, smooth)) if fill_hidden else smooth_sprite(img, smooth)), best, L, cpp, nused, kind)")
edit('smooth_sprite(c["img"], getattr(self.res, "smooth", 2))', 'RT.fill_holes(smooth_sprite(c["img"], getattr(self.res, "smooth", 2)))')
edit('smooth_sprite(cands[0]["img"], getattr(self.res, "smooth", 2))', 'RT.fill_holes(smooth_sprite(cands[0]["img"], getattr(self.res, "smooth", 2)))')
edit("    res.smooth, res.props = smooth, props",
     '    res.smooth, res.props = smooth, props\n    res.model_tag = (f"{model.organism}/{model.sex}" if model is not None else "")')
edit("sub = f\"{getattr(res,'name','')}  {H:.0f}CM  {cpp:g}CM/PX\"",
     "sub = f\"{getattr(res,'name','')}  {getattr(res,'model_tag','')}  {H:.0f}CM  {cpp:g}CM/PX\"")
edit("owner=lo.name, H=self.get_h(),", "owner=lo.name, fill_hidden=self.v_fillhidden.get(), H=self.get_h(),")
edit('self.mk_check(r, "Learn colours from all frames", self.v_learn).pack(side="left", padx=6)',
     'self.mk_check(r, "Learn colours from all frames", self.v_learn).pack(side="left", padx=6)\n'
     '            r = tk.Frame(f, bg=BG1); r.pack(fill="x", pady=(4,0))\n'
     '            self.mk_check(r, "Fill hidden areas (arm over body)", self.v_fillhidden).pack(side="left")')

# ---- top bar: which model am I in ---------------------------------------------------------------------------------------
edit('self.chip_lo.pack(side="right")',
     'self.chip_lo.pack(side="right")\n'
     '            self.chip_model = tk.Label(top, text="", bg=BG2, fg=TEXT, font=self.F["cardb"], padx=12, pady=6, highlightthickness=2, highlightbackground=LINE)\n'
     '            self.chip_model.pack(side="right", padx=8)')
edit("self.model_lbl.config(text=self.model.summary()", "self.update_model_chip(); self.model_lbl.config(text=self.model.summary()")

# ---- coach bar: what to do next -------------------------------------------------------------------------------------------
edit('body = tk.Frame(self, bg=BG0); body.pack(fill="both", expand=True, padx=10, pady=4)',
'''self.coach = tk.Frame(self, bg=BG2, highlightthickness=1, highlightbackground=ACC); self.coach.pack(fill="x", padx=10, pady=(2,2))
            self.coach_step = tk.Label(self.coach, text="", bg=ACC, fg="#001a20", font=self.F["cardb"], padx=12, pady=8); self.coach_step.pack(side="left")
            self.coach_txt = tk.Label(self.coach, text="", bg=BG2, fg=TEXT, font=self.F["ui"], anchor="w", justify="left", wraplength=1000); self.coach_txt.pack(side="left", fill="x", expand=True, padx=12)
            self.coach_btn = self.mk_btn(self.coach, "", lambda: None, "p", pady=6, padx=16); self.coach_btn.pack(side="right", padx=8, pady=4)
            body = tk.Frame(self, bg=BG0); body.pack(fill="both", expand=True, padx=10, pady=4)''')

# ---- save status chip + tools in the status bar ------------------------------------------------------------------------------
edit('self.pb = tk.Canvas(bar, width=260, height=10, bg=BG0, highlightthickness=0); self.pb.pack(side="right", padx=8)',
     'self.pb = tk.Canvas(bar, width=260, height=10, bg=BG0, highlightthickness=0); self.pb.pack(side="right", padx=8)\n'
     '            self.save_chip = tk.Label(bar, text="", bg=BG1, fg=DIM, font=self.F["cardb"], padx=10); self.save_chip.pack(side="right")')
edit('else: l.config(bg=BG2, fg=DIM, text=f"{i+1}  {STAGE[i]}")',
     'else: l.config(bg=BG2, fg=DIM, text=f"{i+1}  {STAGE[i]}")\n            self.update_coach(); self.refresh_save_chip()')
edit("lo.save(res)", 'lo.save(res); self.last_saved = datetime.datetime.now().strftime("%H:%M:%S")')
edit("lo = self.ensure_loadout(); lo.chosen = self.res.chosen; lo.save(self.res); self.exported = False",
     'lo = self.ensure_loadout(); lo.chosen = self.res.chosen; lo.save(self.res); self.exported = False; self.last_saved = datetime.datetime.now().strftime("%H:%M:%S")')
edit("self.exported = True; self.update_stage()",
     'self.exported = True; self.last_saved = datetime.datetime.now().strftime("%H:%M:%S"); self.update_stage()')

# ---- new buttons: RIG CHECK + SNAPSHOT ------------------------------------------------------------------------------------------
edit('self.mk_btn(a, "⬇   SAVE & EXPORT", self.export, "g", pady=8).pack(fill="x", pady=2)',
     'self.mk_btn(a, "⬇   SAVE & EXPORT", self.export, "g", pady=8).pack(fill="x", pady=2)\n'
     '            rr = tk.Frame(a, bg=BG0); rr.pack(fill="x", pady=2)\n'
     '            self.mk_btn(rr, "✔ RIG CHECK", self.run_check, "n", pady=5).pack(side="left", expand=True, fill="x", padx=(0,2))\n'
     '            self.mk_btn(rr, "⎘ SNAPSHOT", self.snapshot, "n", pady=5).pack(side="left", expand=True, fill="x")')

# ---- methods -----------------------------------------------------------------------------------------------------------------------------
edit("        def refresh_model(self, *_):", '''        def update_model_chip(self):
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
        def refresh_model(self, *_):''')

def main(path):
    src = open(path, encoding="utf-8").read()
    bad = [(i, o[:80]) for i, (o, _) in enumerate(E, 1) if src.count(o) != 1]
    if bad:
        for i, o in bad: print(f"anchor #{i} matched {src.count(E[i-1][0])}x (need exactly 1): {o!r}")
        sys.exit("Nothing written. Was apply_model_patch.py applied first, and is this the right file?")
    shutil.copy2(path, path + ".before_ux.bak")
    spans = sorted(((src.index(o), len(o), n) for o, n in E), reverse=True)
    for i, ln, n in spans: src = src[:i] + n + src[i+ln:]
    open(path, "w", encoding="utf-8").write(src)
    print(f"patched {path} ({len(E)} edits). Backup: {path}.before_ux.bak. Keep rig_tools.py next to it.")

if __name__ == "__main__":
    if len(sys.argv) != 2: sys.exit(__doc__)
    main(sys.argv[1])
