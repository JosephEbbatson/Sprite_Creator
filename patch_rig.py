"""One-shot patch: adds side-view tags, solid-colour toggle, solid export to human_rig.py"""
from pathlib import Path

p = Path("human_rig.py")
if not p.exists():
    print("Put this next to human_rig.py and run it from that folder.")
    raise SystemExit(1)

src = p.read_text(encoding="utf-8")
orig = src

# --- 1. SOLID_COL ---
if "SOLID_COL" not in src:
    old = """REGION_COL = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
              (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
              (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
              (170, 255, 195), (128, 128, 0)]"""
    new = old + """

# solid colours matching the modular reference (head yellow, torso purple, arms green, legs pink)
SOLID_COL = {
    "head": (240, 220, 60), "neck": (200, 180, 50),
    "chest": (140, 80, 180), "abdomen": (130, 70, 170), "pelvis": (120, 60, 160),
    "uarmR": (60, 180, 100), "uarmL": (60, 180, 100),
    "farmR": (50, 160, 90), "farmL": (50, 160, 90),
    "handR": (40, 140, 80), "handL": (40, 140, 80),
    "thighR": (220, 120, 160), "thighL": (220, 120, 160),
    "shinR": (200, 100, 140), "shinL": (200, 100, 140),
    "footR": (180, 80, 120), "footL": (180, 80, 120),
}"""
    if old not in src:
        raise SystemExit("Could not find REGION_COL – is this the right human_rig.py?")
    src = src.replace(old, new)
    print("+ SOLID_COL")

# --- 2. facing_label + solidify_sprite ---
if "def facing_label" not in src:
    marker = "def front_score(J, V, H, s):"
    idx = src.find(marker)
    end = src.find("\ndef ", idx + 10)
    insert = '''

def facing_label(front):
    """Classify a frame: 'front', 'side', or 'back' from the front_score (0..1)."""
    if front >= 0.55:
        return "front"
    if front <= 0.20:
        return "back"
    return "side"


def solidify_sprite(rgba, key):
    """Replace RGB with the solid modular colour while keeping alpha."""
    col = SOLID_COL.get(key, (180, 180, 180))
    out = rgba.copy()
    m = out[..., 3] > 0
    out[m, 0] = col[0]
    out[m, 1] = col[1]
    out[m, 2] = col[2]
    return out

'''
    src = src[:end] + insert + src[end:]
    print("+ facing_label / solidify_sprite")

# --- 3. fd.facing ---
if "fd.facing" not in src:
    src = src.replace(
        "    fd.front = front_score(J, V, H, s)\n    fd.axes = make_axes(J, s, H)",
        "    fd.front = front_score(J, V, H, s)\n    fd.facing = facing_label(fd.front)\n    fd.axes = make_axes(J, s, H)",
    )
    print("+ fd.facing")

# --- 4. part_score ---
old_ps = '''def part_score(fd, k, L):
    b, sd = base_of(k), side_of(k)
    vis = min(fd.V[j.format(s=sd)] for j in JOINTS[b])
    qv = float(np.clip(vis / 0.55, 0, 1))
    qf = 1.0
    if b in STRETCH:
        A, B = fd.axes[k]
        ratio = np.linalg.norm(B - A) / (L[b] * fd.s)
        qf = float(np.clip(ratio / 0.92, 0.05, 1)) ** 2
    qo = (1 - fd.occ[k]) ** 2
    qfront = fd.front ** 1.5 if k in CENTER else 0.35 + 0.65 * fd.front
    qr = min(1.0, fd.s_orig / 1.2) ** 0.5
    return qv * qf * qo * qfront * qr'''
new_ps = '''def part_score(fd, k, L):
    b, sd = base_of(k), side_of(k)
    vis = min(fd.V[j.format(s=sd)] for j in JOINTS[b])
    qv = float(np.clip(vis / 0.55, 0, 1))
    qf = 1.0
    if b in STRETCH:
        A, B = fd.axes[k]
        ratio = np.linalg.norm(B - A) / (L[b] * fd.s)
        qf = float(np.clip(ratio / 0.92, 0.05, 1)) ** 2
    qo = (1 - fd.occ[k]) ** 2
    facing = getattr(fd, "facing", "front")
    if k in CENTER:
        qfront = fd.front ** 1.5
        if facing == "side":
            qfront *= 0.35
        elif facing == "back":
            qfront *= 0.15
    else:
        qfront = 0.35 + 0.65 * fd.front
        if facing == "side":
            qfront = max(qfront, 0.7)
    qr = min(1.0, fd.s_orig / 1.2) ** 0.5
    return qv * qf * qo * qfront * qr'''
if old_ps in src:
    src = src.replace(old_ps, new_ps)
    print("+ part_score facing-aware")

# --- 5. build_solid_sheet ---
if "def build_solid_sheet" not in src:
    idx = src.find("    return img, atlas, meta")
    exp = src.find("def export(res, folder, name)", idx)
    src = src[:idx] + '''    return img, atlas, meta


def build_solid_sheet(res):
    """Same layout as build_sheet but every part is filled with its modular solid colour."""
    saved = {}
    for k, s in res.sprites.items():
        saved[k] = s.rgba
        s.rgba = solidify_sprite(s.rgba, k)
    try:
        img, atlas, meta = build_sheet(res)
    finally:
        for k, rgba in saved.items():
            res.sprites[k].rgba = rgba
    return img, atlas, meta


''' + src[exp:]
    print("+ build_solid_sheet")

# --- 6. export solid ---
old_ex = '''    res.sheet.save(os.path.join(d, f"{name}_rig.png"))
    res.atlas.save(os.path.join(d, f"{name}_atlas.png"))
    for k, s_ in res.sprites.items():
        Image.fromarray(s_.rgba).save(os.path.join(pd, f"{k}.png"))'''
new_ex = '''    res.sheet.save(os.path.join(d, f"{name}_rig.png"))
    res.atlas.save(os.path.join(d, f"{name}_atlas.png"))
    try:
        solid, _, _ = build_solid_sheet(res)
        solid.save(os.path.join(d, f"{name}_rig_solid.png"))
    except Exception:
        pass
    for k, s_ in res.sprites.items():
        Image.fromarray(s_.rgba).save(os.path.join(pd, f"{k}.png"))
        try:
            Image.fromarray(solidify_sprite(s_.rgba, k)).save(os.path.join(pd, f"{k}_solid.png"))
        except Exception:
            pass'''
if old_ex in src:
    src = src.replace(old_ex, new_ex)
    print("+ export solid outputs")

# --- 7. GUI v_solid ---
if "v_solid" not in src:
    src = src.replace(
        'self.v_zoom, self.v_mode = tk.IntVar(value=6), tk.StringVar(value="Cut regions")',
        'self.v_zoom, self.v_mode = tk.IntVar(value=6), tk.StringVar(value="Cut regions")\n            self.v_solid = tk.BooleanVar(value=False)',
    )
    print("+ v_solid")

# --- 8. checkbox ---
old_cb = '''ttk.Label(bar, text="Zoom").pack(side="left", padx=(12, 0))
            sp = ttk.Spinbox(bar, from_=1, to=16, width=4, textvariable=self.v_zoom, command=self.redraw)
            sp.pack(side="left", padx=6)
            sp.bind("<Return>", lambda e: self.redraw())'''
new_cb = old_cb + '''
            ttk.Checkbutton(bar, text="Solid colours (like reference)", variable=self.v_solid,
                            command=self.redraw).pack(side="left", padx=12)'''
if old_cb in src and "Solid colours" not in src:
    src = src.replace(old_cb, new_cb)
    print("+ solid checkbox")

# --- 9. frame list ---
old_fl = 'self.fl.insert("end", f"{fd.name[:22]:22s} {fd.s_orig:5.2f} px/cm  front {fd.front:.2f}")'
new_fl = 'self.fl.insert("end", f"{fd.name[:18]:18s} {fd.s_orig:5.2f}px/cm {getattr(fd, \'facing\', \'?\'):5s} {fd.front:.2f}")'
if old_fl in src:
    src = src.replace(old_fl, new_fl)
    print("+ frame list facing")

# --- 10. redraw ---
old_rd = '''elif tab == 1 and self.res is not None:
                self.show("sheet", self.res.sheet, fit=True)
                self.info.config(text=f"sheet {self.res.sheet.width} x {self.res.sheet.height} px, "
                                      f"{len(self.res.sprites)}/{len(PART_KEYS)} parts, {len(self.res.palette)} colours")'''
new_rd = '''elif tab == 1 and self.res is not None:
                if self.v_solid.get():
                    if getattr(self.res, "solid_sheet", None) is None:
                        self.res.solid_sheet, _, _ = build_solid_sheet(self.res)
                    sheet_img = self.res.solid_sheet
                else:
                    sheet_img = self.res.sheet
                self.show("sheet", sheet_img, fit=True)
                mode = "solid" if self.v_solid.get() else "skin"
                self.info.config(text=f"sheet {sheet_img.width} x {sheet_img.height} px, "
                                      f"{len(self.res.sprites)}/{len(PART_KEYS)} parts, {mode}")'''
if old_rd in src:
    src = src.replace(old_rd, new_rd)
    print("+ redraw solid")

if src == orig:
    print("Nothing changed – already patched?")
else:
    p.write_text(src, encoding="utf-8")
    print("Done. human_rig.py updated.")