"""rig_game_export.py - turn a finished HUMAN.RIG loadout into a LIFEX-100 game package.

The game (Godot) draws the player from a "rig": ONE sprite sheet plus, for every body part, a rectangle on
that sheet, a pivot, and the point on its parent where it hangs (a "socket").  HUMAN.RIG already knows all of
this, so this module only translates.  It writes into  <loadout>/game/ :

    sheet.png   every part of the FRONT view packed onto one transparent sheet
    rig.json    the 17 parts: rect, pivot, parent, socket, draw order   (read by Player/CustomRig.gd)
    preview.png the assembled figure, for the character picker

Why FRONT: the game's own player is a front-view paper doll that it mirrors for left/right facing.

How the numbers are made (the whole trick):
  * rig_pose.layout() gives the neutral position of every part's pivot (cm) with all manual alignment applied.
  * game pivot of a part      = that part's own pivot inside its sprite (px, rounded)
  * game socket on its parent = parent pivot + (child pivot position - parent pivot position) in game pixels
  * everything is in game pixels (2 cm each, see GAME_CM_PER_PX): nothing is resized unless the loadout's own
    cm-per-pixel is not 2
  The game then sets   child bone position = socket - parent pivot   and draws the child with its pivot on the
  bone, which is exactly the same attachment HUMAN.RIG used when it drew the assembled figure.

Left/right: in the FRONT view HUMAN.RIG's "R" parts are on the image LEFT; the game calls the image-left parts
"l ...".  So creator R -> game "l", creator L -> game "r".

Run on its own (needs the same packages as HUMAN.RIG):   python rig_game_export.py loadouts/<name>
Normally it is called for you by SAVE & EXPORT.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import rig_pose as RP
import rig_trim as TRM

GAME_FORMAT = "lifex100-rig"
GAME_VERSION = 1
VIEW = "front"
# The game's own figure (Human_M, about 90 px for a 172 cm person) is drawn at 1 px = 2 cm, which is also
# HUMAN.RIG's default (CM_PER_PX).  So a default loadout is exported as it is.  If you changed "cm per px" in
# the creator (say 1 for finer detail) the parts are resized here (nearest neighbour, pixels stay crisp) so the
# character stands as tall in the game as a default one would.
GAME_CM_PER_PX = 2.0

# creator base part -> the game's part name (the game names come from Life/Player/Player.tres)
BASE_NAME = {"pelvis": "hip", "abdomen": "mid", "chest": "chest", "neck": "neck", "head": "head",
             "uarm": "arm upper", "farm": "arm lower", "hand": "hand",
             "thigh": "leg upper", "shin": "leg lower", "foot": "foot"}
SIDE_NAME = {"R": "l", "L": "r"}          # image-left is the game's "l" (see module docstring)


def game_name(G, key):
    """Creator key ('uarmR') -> game part name ('l arm upper')."""
    base = G["base_of"](key)
    if key in G["CENTER"]:
        return BASE_NAME[base]
    return SIDE_NAME[key[-1]] + " " + BASE_NAME[base]


def _scaled(arr, factor):
    """Nearest-neighbour resize of an RGBA array by `factor` (1.0 returns it unchanged)."""
    if abs(factor - 1.0) < 1e-9:
        return arr
    img = Image.fromarray(np.ascontiguousarray(arr))
    size = (max(1, int(round(img.width * factor))), max(1, int(round(img.height * factor))))
    return np.array(img.resize(size, Image.Resampling.NEAREST))


def _pack(sprites):
    """Shelf-pack {key: rgba array} onto one transparent sheet with 1 px between parts.
    Returns (PIL image, {key: (x, y, w, h)})."""
    items = sorted(sprites.items(), key=lambda kv: (-kv[1].shape[0], kv[0]))
    area = sum((a.shape[0] + 2) * (a.shape[1] + 2) for _, a in items)
    widest = max(a.shape[1] for _, a in items) + 2
    width = max(widest, 64, int(math.ceil(math.sqrt(area) * 1.3)))
    x = y = 1
    row_h = 0
    rects = {}
    for key, arr in items:
        h, w = arr.shape[:2]
        if x + w + 1 > width:
            x, y, row_h = 1, y + row_h + 1, 0
        rects[key] = (x, y, w, h)
        x += w + 1
        row_h = max(row_h, h)
    sheet = Image.new("RGBA", (width, y + row_h + 1), (0, 0, 0, 0))
    for key, arr in items:
        sheet.paste(Image.fromarray(np.ascontiguousarray(arr)), rects[key][:2])
    return sheet, rects


def export_game_package(G, res, out_dir, name, view=VIEW):
    """Write the game package for `res` (a finished HUMAN.RIG result) into out_dir.
    G is the globals() of human_rig.py (same convention as rig_pose).  Returns a small report dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = res.views[view]
    keys = [k for k in G["HIER"] if k in S]                  # parents always before children
    cpp = float(res.cpp)
    factor = cpp / GAME_CM_PER_PX                             # HUMAN.RIG pixels -> game pixels
    warnings = []

    # 1. neutral layout: pivot positions in cm, no pose angles, manual alignment included
    th, pos, pj = RP.layout(G, res, view, {}, (0.0, 0.0))

    # 2. the pixels of every part (trims and paint included) and the packed sheet
    pixels = {}
    for k in keys:
        sp = S[k]
        pixels[k] = _scaled(TRM.final_rgba(res, view, k, sp), factor)
        kind = getattr(sp, "kind", "")
        if kind in ("placeholder", "borrowed", "synth"):
            warnings.append(f"{game_name(G, k)}: made up by HUMAN.RIG ({kind}), not cut from your pictures")
    sheet, rects = _pack(pixels)
    sheet.save(out_dir / "sheet.png")

    # 3. pivots (whole pixels) and sockets
    pivot = {k: [int(round(float(pj[k][0]) * factor)), int(round(float(pj[k][1]) * factor))] for k in keys}
    # Round every pivot's position on the figure ONCE, in absolute terms, and build each socket from the rounded
    # numbers.  Chained links then add up exactly (no drift down shoulder -> elbow -> wrist): every part ends up
    # within half a pixel of where HUMAN.RIG put it.
    world = {k: np.round(np.asarray(pos[k], float) / GAME_CM_PER_PX).astype(int) for k in keys}
    sockets = {k: {} for k in keys}
    parts = []
    order = list(G["draw_order"](view, getattr(res, "mask", None)))
    for k in keys:
        parent = G["parent_key"](k)
        entry = {"name": game_name(G, k), "rect": list(rects[k]), "pivot": pivot[k],
                 "parent": "", "parent_socket": "", "z": order.index(k) + 1 if k in order else 1}
        if parent is not None:
            delta = world[k] - world[parent]
            sock = [int(pivot[parent][0] + delta[0]), int(pivot[parent][1] + delta[1])]
            socket_name = game_name(G, k)                   # one socket per child, named after the child
            sockets[parent][socket_name] = sock
            entry["parent"] = game_name(G, parent)
            entry["parent_socket"] = socket_name
        parts.append(entry)
    by_key = dict(zip(keys, parts))
    for k in keys:
        by_key[k]["sockets"] = sockets[k]

    names = [p["name"] for p in parts]
    if len(set(names)) != len(names):
        raise ValueError("duplicate game part names: " + ", ".join(sorted(n for n in names if names.count(n) > 1)))

    rig = {"format": GAME_FORMAT, "version": GAME_VERSION, "name": name, "source": "HUMAN.RIG",
           "view": view, "height_cm": float(res.H), "cm_per_px": GAME_CM_PER_PX, "sheet": "sheet.png",
           "root_part": game_name(G, "pelvis"), "parts": parts}
    (out_dir / "rig.json").write_text(json.dumps(rig, indent=2), encoding="utf-8")

    # 4. a picture for the character picker (best effort - never blocks the export)
    try:
        img, _meta = RP.render_pose(G, res, view)
        img.save(out_dir / "preview.png")
    except Exception as exc:                                # pragma: no cover
        warnings.append("preview.png not written: " + str(exc))

    return {"dir": str(out_dir), "parts": len(parts), "sheet": list(sheet.size), "warnings": warnings}


def main(argv):
    """python rig_game_export.py loadouts/<name>   -> writes loadouts/<name>/game/"""
    if len(argv) != 2:
        print(__doc__)
        return 2
    import human_rig as H                                    # needs tkinter + the HUMAN.RIG packages
    path = Path(argv[1])
    lo = H.Loadout(path.name)
    lo.path = path
    if not lo.load():
        print("No meta.json in", path)
        return 1
    res = lo.to_result()
    info = export_game_package(vars(H), res, path / "game", lo.name)
    print(f"Wrote {info['dir']}  ({info['parts']} parts, sheet {info['sheet'][0]}x{info['sheet'][1]})")
    for w in info["warnings"]:
        print("  note:", w)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
