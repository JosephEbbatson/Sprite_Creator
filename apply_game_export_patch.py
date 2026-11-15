"""One-shot patch: makes SAVE & EXPORT in human_rig.py also write the LIFEX-100 game package (loadouts/<name>/game/).

Put this next to human_rig.py (with rig_game_export.py) and run it from that folder:   python apply_game_export_patch.py

Safe to run twice (it notices it has already been applied).  It keeps a copy of your file as human_rig.py.bak_before_game_export,
and writes nothing at all if the code it expects is not there - in that case the standalone export_to_game.py still works.
"""
from pathlib import Path

p = Path("human_rig.py")
if not p.exists() or not Path("rig_game_export.py").exists():
    print("Put this next to human_rig.py AND rig_game_export.py, and run it from that folder.")
    raise SystemExit(1)

src = p.read_text(encoding="utf-8")
if "RGE.export_game_package" in src:
    print("Already patched - nothing to do.")
    raise SystemExit(0)

# (label, required?, old text, new text)
EDITS = [
    ("import", True,
     "import rig_sheet_import_gui as SIG\n",
     "import rig_sheet_import_gui as SIG\nimport rig_game_export as RGE\n"),
    ("loadouts folder beside the file", False,
     'LOADOUTS_DIR = Path("loadouts")\n',
     'LOADOUTS_DIR = Path(__file__).resolve().parent / "loadouts"   # beside this file, whatever folder it was started from (the game finds characters here)\n'),
    ("export hook", True,
     '            self.toast(f"Exported to {lo.path.resolve()}  (rig.png, atlas.png, rig.json, assembled_*.png, parts/)", "ok")\n',
     '            game_note = ""\n'
     '            try:                                           # the LIFEX-100 game package: game/sheet.png + game/rig.json\n'
     '                gi = RGE.export_game_package(globals(), self.res, lo.path/"game", lo.name)\n'
     '                game_note = "  + game/ (ready to pick in LIFEX-100)"\n'
     '                for w in gi["warnings"]: self.log("game export: " + w, "warn")\n'
     '            except Exception:\n'
     '                self.log("game export failed (the normal export above is fine):\\n" + traceback.format_exc(), "err")\n'
     '                game_note = "  (game/ package FAILED - see LOG)"\n'
     '            self.toast(f"Exported to {lo.path.resolve()}  (rig.png, atlas.png, rig.json, assembled_*.png, parts/){game_note}", "ok")\n'),
]

new = src
missing = []
for label, required, old, repl in EDITS:
    if new.count(old) == 1:
        new = new.replace(old, repl)
    else:
        print("%s: the expected code was %s in your human_rig.py" % (label, "not found" if new.count(old) == 0 else "found more than once"))
        if required:
            missing.append(label)
if missing:
    print("\nNothing was changed (your human_rig.py differs from the one this patch was written for: %s)." % ", ".join(missing))
    print("You can still use  python export_to_game.py  to export saved sprites.")
    raise SystemExit(1)

compile(new, "human_rig.py", "exec")                         # refuse to write a file that no longer parses
Path("human_rig.py.bak_before_game_export").write_text(src, encoding="utf-8")
p.write_text(new, encoding="utf-8")
print("Patched human_rig.py (backup: human_rig.py.bak_before_game_export).")
print("SAVE & EXPORT now also writes loadouts/<name>/game/  (sheet.png + rig.json).")
