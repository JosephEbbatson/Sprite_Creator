"""rig_sheet_import_gui.py - the IMPORT SHEET button: bring in a finished pixel sprite sheet (see rig_sheet_import.py)."""
import numpy as np
from PIL import Image
import rig_sheet_import as SI


def install(cls, G, K, tk, ttk, ImageTk, filedialog, messagebox):
    BG0, BG1, BG2, TEXT, MUTED, ACC, WARN, GOOD = (K[n] for n in ("BG0", "BG1", "BG2", "TEXT", "MUTED", "ACC", "WARN", "GOOD"))
    PART_KEYS, CENTER, PIV_FRAC, PRIOR = G["PART_KEYS"], G["CENTER"], G["PIV_FRAC"], G["PRIOR"]

    def add(fn): setattr(cls, fn.__name__, fn); return fn

    def joints(k, rgba, cpp):
        """pa (top joint), pb (bottom joint), pj (pivot), pe from the drawn pixels: the joints sit on the sprite's own
        top / bottom edge, at the middle of the pixels drawn there (so a slanted arm is followed)."""
        h, w = rgba.shape[:2]; a = rgba[..., 3] > 0
        def mid(rows):
            xs = np.nonzero(a[rows].any(0))[0] if a[rows].any() else np.array([w // 2])
            return float(xs.mean()) + 0.5
        e = 0.0 if k in CENTER else min(1.5, h * 0.1)
        if k in CENTER:
            pa = np.array([w / 2.0, 0.0]); pb = np.array([w / 2.0, float(h)])
            # the head turns about its neck joint (just above the bottom edge); the other centre parts use the rig's usual pivot
            pj = pa + (0.88 if k == "head" else PIV_FRAC[k]) * (pb - pa); pe = pa.copy()
        else:
            top, bot = slice(0, max(1, int(h * 0.2))), slice(h - max(1, int(h * 0.2)), h)
            pa = np.array([mid(top), e]); pb = np.array([mid(bot), h - e]); pj, pe = pa.copy(), pb.copy()
        return pa, pb, pj, pe

    def anchors(parts, J):
        """Where each child's joint sits inside its parent's picture (the rig attaches a child exactly there)."""
        out = {}
        def put(p, c, xy):
            if p in parts and c in parts: out.setdefault(p, {})[c] = np.asarray(xy, float)
        def wh(k): return parts[k].shape[1], parts[k].shape[0]
        for p, c in (("pelvis", "abdomen"), ("abdomen", "chest"), ("chest", "neck"), ("neck", "head")):
            if p in parts: put(p, c, (wh(p)[0] / 2.0, min(1.5, wh(p)[1] * 0.2)))
        for s in "RL":
            if "chest" in parts:
                cw, ch = wh("chest"); put("chest", "uarm" + s, (1.5 if s == "R" else cw - 1.5, min(3.0, ch * 0.25)))
            if "pelvis" in parts:
                pw, ph = wh("pelvis"); put("pelvis", "thigh" + s, ((0.28 if s == "R" else 0.72) * pw, ph * 0.65))
            for p, c in (("uarm", "farm"), ("farm", "hand"), ("thigh", "shin"), ("shin", "foot")):
                if p + s in J: put(p + s, c + s, J[p + s][1])
        return out

    @add
    def sheet_to_result(self, fig, H, cpp=None, name="imported"):
        """One figure from a sheet (dict parts) -> a finished rig result with the figure as the FRONT view."""
        parts = {k: p for k, p in fig["parts"].items() if p is not None}
        if "chest" not in parts: raise ValueError("the sheet has no chest drawn - nothing to build a rig from")
        if cpp is None:
            col = [parts[k].shape[0] for k in ("head", "neck", "chest", "abdomen", "pelvis", "thighR", "shinR", "footR") if k in parts]
            cpp = float(np.clip(H / max(1, sum(col)), 0.4, 6.0))
        J = {k: joints(k, rgba, cpp) for k, rgba in parts.items()}
        L = {n: p * H for n, p in PRIOR.items()}
        for b in ("uarm", "farm", "thigh", "shin"):       # bone lengths follow the DRAWING, not the average human
            ls = [float(np.linalg.norm(J[b + s][1] - J[b + s][0])) * cpp for s in "RL" if b + s in J]
            if ls: L[b] = float(np.mean(ls))
        anc = anchors(parts, J)
        chosen = {v: {} for v in G["VIEWS"]}
        for k, rgba in parts.items():
            pa, pb, pj, pe = J[k]
            c = dict(score=1.0, fname=name, cid=f"sheet|{k}", flip=False, twin=False, pa=pa, pb=pb, pj=pj, pe=pe, anc=anc.get(k, {}))
            sp = G["sprite_from_img"](k, rgba, c, L, cpp)
            if sp is not None: chosen["front"][k] = sp
        res = G["Sprite"](); res.name = name
        res.chosen, res.cands = chosen, {v: {k: [] for k in PART_KEYS} for v in G["VIEWS"]}
        try: res.palette = G["build_palette"]([s for d in chosen.values() for s in d.values()], 0)
        except Exception: res.palette = []
        res.lengths, res.widths = L, dict(sho=0.23 * H, hip=0.19 * H)
        res.H, res.cpp, res.issues, res.counts = H, cpp, [], dict(front=1, side=0, back=0)
        res.cands_dirty, res.synth, res.smooth, res.props = False, True, 0, dict(G["DEF_PROPS"])
        res.mask, res.scale = G["RB"].normalize_mask(None), G["RB"].normalize_scale(None)
        res.model_tag, res.align, res.extras, res.trims = "", {}, [], {}
        G["finalize_views"](res); G["render_outputs"](res)
        return res

    @add
    def import_sheet(self):
        path = filedialog.askopenfilename(parent=self, title="Pick your finished pixel sprite sheet",
                                          filetypes=[("Pictures", "*.png *.gif *.bmp *.webp"), ("All files", "*.*")])
        if path: self.import_sheet_path(path)

    @add
    def _sheet_dialog(self, figs, path):
        """Ask which figure is the body and which the corpse. Returns (body_i, corpse_i, height) or None."""
        win = tk.Toplevel(self); win.title("IMPORT SHEET"); win.configure(bg=BG0); win.transient(self); win.grab_set()
        tk.Label(win, text="I found these figures on the sheet. Say what each one is.", bg=BG0, fg=TEXT, font=self.F["h"]).pack(padx=14, pady=(12, 6))
        rgba = SI.load_rgb(path); body_i, corpse_i = None, None
        for i, f in enumerate(figs):
            if f.get("n", 0) > 0 and body_i is None: body_i = i
            elif f.get("n", 0) > 0 and corpse_i is None: corpse_i = i
        row = tk.Frame(win, bg=BG0); row.pack(padx=14, pady=4); vars_, imgs = [], []
        for i, f in enumerate(figs):
            box = tk.Frame(row, bg=BG1, highlightthickness=1, highlightbackground=ACC if f.get("n") else MUTED); box.pack(side="left", padx=6)
            if f.get("cells"):
                y0 = min(c["box"][0] for c in f["cells"].values()); x0 = min(c["box"][1] for c in f["cells"].values())
                y1 = max(c["box"][2] for c in f["cells"].values()); x1 = max(c["box"][3] for c in f["cells"].values())
                th = Image.fromarray(rgba[y0:y1, x0:x1]).resize(((x1 - x0) * 2, (y1 - y0) * 2), Image.Resampling.NEAREST)
                im = ImageTk.PhotoImage(th); imgs.append(im); tk.Label(box, image=im, bg=BG1).pack(padx=6, pady=6)
            v = tk.StringVar(value="BODY" if i == body_i else "CORPSE" if i == corpse_i else "IGNORE"); vars_.append(v)
            msg = f"{f.get('n', 0)} of 17 parts drawn" if not f.get("error") else "could not read: " + f["error"]
            tk.Label(box, text=msg, bg=BG1, fg=MUTED, font=self.F["ui"]).pack()
            ttk.Combobox(box, textvariable=v, values=("BODY", "CORPSE", "IGNORE"), state="readonly", width=10).pack(pady=6)
        hr = tk.Frame(win, bg=BG0); hr.pack(pady=6)
        tk.Label(hr, text="HEIGHT OF THE CHARACTER (cm)", bg=BG0, fg=MUTED, font=self.F["ui"]).pack(side="left")
        vh = tk.StringVar(value=str(int(self.get_h()))); tk.Entry(hr, textvariable=vh, width=6).pack(side="left", padx=6)
        out = {}
        def ok():
            ch = [v.get() for v in vars_]
            if "BODY" not in ch: return messagebox.showwarning("IMPORT SHEET", "Mark one figure as BODY.", parent=win)
            try: h = float(vh.get())
            except Exception: h = 172.0
            out["r"] = (ch.index("BODY"), ch.index("CORPSE") if "CORPSE" in ch else None, max(40.0, h)); win.destroy()
        bf = tk.Frame(win, bg=BG0); bf.pack(pady=(4, 12))
        self.mk_btn(bf, "IMPORT", ok, "p", padx=20).pack(side="left", padx=6); self.mk_btn(bf, "CANCEL", win.destroy, "n", padx=14).pack(side="left")
        self.wait_window(win); return out.get("r")

    @add
    def import_sheet_path(self, path, choice=None):
        """choice=(body_index, corpse_index or None, height_cm) skips the dialog (used by tests)."""
        try: figs = SI.read_sheet(path)
        except Exception as ex: return self.toast(f"Could not read that picture ({ex}).", "err")
        if not any(f.get("n", 0) for f in figs):
            return self.toast("No body cells found. The sheet must show each part in a green box with a red border.", "warn")
        choice = choice or self._sheet_dialog(figs, path)
        if not choice: return
        bi, ci, H = choice
        try:
            self.v_h.set(str(int(H)))
            res = self.sheet_to_result(figs[bi], H, name=self.v_name.get())
            if ci is not None:
                cres = {k: p for k, p in figs[ci]["parts"].items() if p is not None}
                res.corpse = {"front": cres}
            self.v_cpp.set(f"{res.cpp:.2f}")
            self.on_cut(res)
        except Exception:
            self.log("sheet import failed:\n" + G["traceback"].format_exc(), "err")
            return self.toast("Import failed - see LOG.", "err")
        n = sum(1 for k in PART_KEYS if k in res.chosen["front"])
        self.toast(f"Imported {n} of 17 parts as the FRONT view" + (f" + {len(res.corpse['front'])} corpse parts" if ci is not None else "")
                   + ". Side and back are guessed from the front until you add real ones.", "ok")
