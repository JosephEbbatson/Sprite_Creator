"""Test for rig_game_export.py.   Run:  python test_game_export.py     (ends with ALL PASS)

Builds a small made-up character on disk (17 plain coloured parts with joints), lets HUMAN.RIG's own loader
rebuild it, exports the game package, then checks that
  * the names are the game's 17 part names and left/right land on the right image side,
  * every part sits within 1 px of where HUMAN.RIG itself puts it (the game's attachment maths replayed here),
  * the sheet holds every part, and the rects do not overlap,
  * rig.json has what Player/CustomRig.gd needs.
No window and no tkinter needed.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import human_rig as H
import rig_pose as RP
import rig_game_export as RGE

fails = []


def check(cond, what):
    if not cond:
        fails.append(what)
        print("FAIL:", what)


# key: (w, h, pivot (x, y) = the joint where it hangs from its parent, colour)
SIZES = {
    "head": (12, 14, (6, 13), (230, 190, 150)), "neck": (6, 6, (3, 5), (220, 180, 140)),
    "chest": (24, 16, (12, 15), (60, 90, 200)), "abdomen": (20, 8, (10, 7), (70, 100, 190)),
    "pelvis": (22, 10, (11, 5), (50, 50, 80)),
    "uarm": (6, 16, (3, 2), (200, 60, 60)), "farm": (5, 16, (2, 2), (230, 190, 150)),
    "hand": (5, 6, (2, 1), (230, 190, 150)),
    "thigh": (9, 20, (4, 2), (40, 40, 60)), "shin": (7, 20, (3, 2), (45, 45, 65)), "foot": (9, 5, (3, 1), (90, 60, 40)),
}
# where each CHILD joint sits inside its PARENT sprite (px): the "anchors" HUMAN.RIG records
ANCHORS = {
    "pelvis": {"abdomen": (11, 2), "thighR": (6, 4), "thighL": (16, 4)},
    "abdomen": {"chest": (10, 1)},
    "chest": {"neck": (12, 2), "uarmR": (3, 3), "uarmL": (21, 3)},
    "neck": {"head": (3, 1)},
    "uarmR": {"farmR": (3, 15)}, "uarmL": {"farmL": (3, 15)},
    "farmR": {"handR": (2, 15)}, "farmL": {"handL": (2, 15)},
    "thighR": {"shinR": (4, 19)}, "thighL": {"shinL": (4, 19)},
    "shinR": {"footR": (3, 19)}, "shinL": {"footL": (3, 19)},
}


def make_loadout(root: Path, cpp: float = 2.0) -> "H.Loadout":
    lo = H.Loadout("testfig")
    lo.path = root / "testfig"
    lo.ensure_dirs()
    views = {"front": {}}
    for key in H.PART_KEYS:
        w, h, piv, col = SIZES[H.base_of(key)]
        arr = np.zeros((h, w, 4), np.uint8)
        arr[..., :3] = col
        arr[..., 3] = 255
        Image.fromarray(arr).save(lo.path / "parts" / "front" / f"{key}.png")
        views["front"][key] = dict(pivot_a=list(piv), pivot_b=[w / 2, h], pivot_j=list(piv), pivot_e=[w / 2, h - 1],
                                   length_cm=float(h * 2), score=0.95, source="test", cid="",
                                   anchors={c: list(p) for c, p in ANCHORS.get(key, {}).items()})
    lo.meta["views"] = views
    lo.meta["height_cm"] = 172.0
    lo.meta["cm_per_px"] = cpp
    lo.write_meta()
    return lo


def run(cpp):
    print("--- loadout at %.1f cm per pixel ---" % cpp)
    with tempfile.TemporaryDirectory() as tmp:
        lo = make_loadout(Path(tmp), cpp)
        check(lo.load(), "loadout loads")
        res = lo.to_result()
        out = Path(tmp) / "testfig" / "game"
        info = RGE.export_game_package(vars(H), res, out, "testfig")

        rig = json.loads((out / "rig.json").read_text())
        sheet = Image.open(out / "sheet.png")
        check(rig["format"] == "lifex100-rig" and rig["version"] == 1, "format and version")
        check(rig["root_part"] == "hip", "root is hip")
        check(rig["cm_per_px"] == 2.0, "game scale is 2 cm per pixel")
        check(res.cpp == cpp, "loadout really is %.1f cm/px" % cpp)
        check(len(rig["parts"]) == 17, "17 parts, got %d" % len(rig["parts"]))
        names = {p["name"] for p in rig["parts"]}
        want = {"head", "neck", "chest", "mid", "hip"} | {s + " " + n for s in "lr" for n in (
            "arm upper", "arm lower", "hand", "leg upper", "leg lower", "foot")}
        check(names == want, "game part names, missing %s extra %s" % (sorted(want - names), sorted(names - want)))
        check(not info["warnings"], "no warnings, got %s" % info["warnings"])
        check((out / "preview.png").exists(), "preview.png written")

        by = {p["name"]: p for p in rig["parts"]}
        # the order lists parents before children (the game builds top-down)
        seen = set()
        for p in rig["parts"]:
            check(p["parent"] == "" or p["parent"] in seen, "parent before child: " + p["name"])
            seen.add(p["name"])
        check([p["name"] for p in rig["parts"] if p["parent"] == ""] == ["hip"], "only hip has no parent")
        for p in rig["parts"]:
            if p["parent"]:
                check(p["parent_socket"] in by[p["parent"]]["sockets"], "socket exists: " + p["name"])

        # rects: inside the sheet, no overlap, pixels match
        rects = [tuple(p["rect"]) for p in rig["parts"]]
        for (x, y, w, h), p in zip(rects, rig["parts"]):
            check(x >= 0 and y >= 0 and x + w <= sheet.width and y + h <= sheet.height, "rect inside sheet: " + p["name"])
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                check(a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1],
                      "rects do not overlap: %s %s" % (rig["parts"][i]["name"], rig["parts"][j]["name"]))

        # left/right: game "l ..." is on the image LEFT of "r ..."
        fk = {}
        for p in rig["parts"]:
            if p["parent"] == "":
                fk[p["name"]] = np.zeros(2)
            else:
                par = by[p["parent"]]
                fk[p["name"]] = fk[p["parent"]] + np.array(par["sockets"][p["parent_socket"]], float) - np.array(par["pivot"], float)
        for n in ("arm upper", "leg upper", "foot"):
            check(fk["l " + n][0] < fk["r " + n][0], "'l %s' is image-left of 'r %s'" % (n, n))

        # same place as HUMAN.RIG puts it (replaying the game's maths: bone = socket - parent pivot, sprite = bone - pivot)
        th, pos, pj = RP.layout(vars(H), res, "front", {}, (0.0, 0.0))
        worst = 0.0
        for key in H.HIER:
            gname = RGE.game_name(vars(H), key)
            tool_pivot = np.asarray(pos[key], float) / RGE.GAME_CM_PER_PX
            game_pivot = fk[gname]
            err = float(np.abs(tool_pivot - game_pivot).max())
            worst = max(worst, err)
            check(err <= 0.5 + 1e-6, "%s pivot within half a pixel of HUMAN.RIG (off by %.2f)" % (gname, err))
            f = res.cpp / RGE.GAME_CM_PER_PX
            check(by[gname]["pivot"] == [int(round(pj[key][0] * f)), int(round(pj[key][1] * f))], "pivot of " + gname)
            w0, h0 = SIZES[H.base_of(key)][:2]
            check(by[gname]["rect"][2:] == [int(round(w0 * f)), int(round(h0 * f))], "part size scaled to game pixels: " + gname)
        print("worst pivot error: %.2f px" % worst)



def main():
    for cpp in (2.0, 4.0, 1.0):      # as is, shrunk-in-creator (resized up here), finer (resized down here)
        run(cpp)
    if fails:
        print("%d FAILED" % len(fails))
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
