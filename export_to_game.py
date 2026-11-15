"""export_to_game.py - turn sprites you have ALREADY saved in HUMAN.RIG into LIFEX-100 game packages.

No window, no tkinter, no re-cutting: it reads the loadout HUMAN.RIG saved (meta.json + parts/) and writes
loadouts/<name>/game/ containing  sheet.png  rig.json  preview.png  (the files the game's Player/CustomRig.gd reads).

Put this file next to human_rig.py (together with rig_game_export.py) and run it from that folder:

    python export_to_game.py                      every loadout in ./loadouts
    python export_to_game.py hero                 one loadout, by name
    python export_to_game.py D:\\sprites\\hero      one loadout, by folder
    python export_to_game.py --game "C:\\...\\lifex-100"        also copy the result into the game project
    python export_to_game.py hero --zip           also write hero_game.zip (unzip it inside Tools/SpriteCreator)

On Windows you can also double-click Export_To_Game.bat, or drag a loadout folder onto it.

--game takes the folder that holds project.godot (the outer folder is fine too).  The package lands in
<game>/Tools/SpriteCreator/loadouts/<name>/game/, which is where the game looks, so the sprite shows up on the
start screen.  Running it again for the same name replaces the earlier copy.
"""
import argparse
import sys
import traceback
import zipfile
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GAME_FILES = ("sheet.png", "rig.json", "preview.png")      # preview.png is optional (picker thumbnail)
REQUIRED = ("sheet.png", "rig.json")


def find_loadouts(names):
    """Return a list of loadout folders (each holds meta.json).  No names -> every one in ./loadouts."""
    root = HERE / "loadouts"
    if not names:
        if not root.is_dir():
            raise SystemExit("No loadouts folder at %s - save a sprite in HUMAN.RIG first, "
                             "or pass the sprite's folder." % root)
        found = sorted(d for d in root.iterdir() if d.is_dir() and (d / "meta.json").exists())
        if not found:
            raise SystemExit("No saved sprites found in %s" % root)
        return found
    out = []
    for n in names:
        for cand in (Path(n), root / n):
            if (cand / "meta.json").exists():
                out.append(cand.resolve())
                break
        else:
            raise SystemExit("'%s' is not a saved sprite (no meta.json in it, and none at %s)" % (n, root / n))
    return out


def export_one(path, H, RGE):
    """Export loadout folder `path`.  Returns (name, info dict from rig_game_export)."""
    lo = H.Loadout(path.name)
    lo.path = path
    if not lo.load():
        raise ValueError("could not read meta.json")
    try:
        res = lo.to_result()
    except Exception as exc:
        raise ValueError("this sprite has not been cut yet (press CUT in HUMAN.RIG, then SAVE & EXPORT): %s" % exc)
    if "front" not in res.views or not res.views["front"]:
        raise ValueError("this sprite has no FRONT view - the game needs one")
    info = RGE.export_game_package(vars(H), res, path / "game", lo.name)
    made_up = [w for w in info["warnings"] if "made up by HUMAN.RIG" in w]
    if len(made_up) >= info["parts"]:                          # nothing real was cut: do not leave a junk package behind
        shutil.rmtree(path / "game", ignore_errors=True)
        raise ValueError("none of its parts have been cut from pictures yet - open it in HUMAN.RIG, press CUT, "
                         "then SAVE & EXPORT")
    return lo.name, info


def find_game_project(given):
    """The folder holding project.godot: `given` itself, or the one folder inside it that does."""
    proj = Path(given).expanduser()
    if (proj / "project.godot").exists():
        return proj
    if proj.is_dir():
        inner = [d for d in proj.iterdir() if d.is_dir() and (d / "project.godot").exists()]
        if len(inner) == 1:
            return inner[0]
    raise ValueError("no project.godot in %s - give the folder that holds the game's project.godot" % proj)


def install(game_dir, name, game_project):
    """Copy <game_dir>/{sheet.png,rig.json,preview.png} into the game project.  Returns the destination folder."""
    proj = find_game_project(game_project)
    dest = proj / "Tools" / "SpriteCreator" / "loadouts" / name / "game"
    dest.mkdir(parents=True, exist_ok=True)
    for f in GAME_FILES:
        src = game_dir / f
        if src.exists():
            shutil.copy2(src, dest / f)
        elif f in REQUIRED:
            raise ValueError("%s was not written" % f)
    return dest


def make_zip(game_dir, name):
    """<loadout>/<name>_game.zip, laid out so unzipping it inside Tools/SpriteCreator puts the files where the game looks."""
    zpath = game_dir.parent / (name + "_game.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in GAME_FILES:
            if (game_dir / f).exists():
                z.write(game_dir / f, "loadouts/%s/game/%s" % (name, f))
    return zpath


def main(argv):
    ap = argparse.ArgumentParser(description="Export saved HUMAN.RIG sprites as LIFEX-100 game packages.")
    ap.add_argument("sprites", nargs="*", help="loadout names or folders (default: every loadout in ./loadouts)")
    ap.add_argument("--game", metavar="PROJECT", help="copy the result into this LIFEX-100 project folder")
    ap.add_argument("--zip", action="store_true", help="also write <name>_game.zip next to the loadout")
    args = ap.parse_args(argv)

    if args.game:                                             # fail early, before any work
        try:
            find_game_project(args.game)
        except ValueError as exc:
            print("Error:", exc)
            return 2
    try:
        import human_rig as H                                 # headless: tkinter is only imported when the GUI starts
        import rig_game_export as RGE
    except ImportError as exc:
        print("Missing piece: %s\nRun this from the folder holding human_rig.py and rig_game_export.py, "
              "with its packages installed (pip install numpy pillow scipy)." % exc)
        return 2

    failed = 0
    for path in find_loadouts(args.sprites):
        print("\n== %s" % path.name)
        try:
            name, info = export_one(path, H, RGE)
            print("  wrote %s  (%d parts, sheet %dx%d)" % (info["dir"], info["parts"], *info["sheet"]))
            for w in info["warnings"]:
                print("  note:", w)
            game_dir = Path(info["dir"])
            if args.zip:
                print("  zip:", make_zip(game_dir, name))
            if args.game:
                print("  installed into:", install(game_dir, name, args.game))
        except Exception as exc:
            failed += 1
            print("  FAILED:", exc)
            if not isinstance(exc, ValueError):
                traceback.print_exc()
    if failed:
        print("\n%d sprite(s) failed." % failed)
        return 1
    if not args.game:
        print("\nDone. To use in the game, copy each loadouts/<name>/game folder to\n"
              "  <game project>/Tools/SpriteCreator/loadouts/<name>/game\n"
              "or run again with  --game \"<path to the game project>\".")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
