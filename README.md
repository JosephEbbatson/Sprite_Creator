# HUMAN.RIG v3

Rig cutter + paint editor + part picker.  
Turns reference photos (or a GIF) of a character into a fully posed, animatable sprite sheet with pivots, bones, and per-part PNGs.

**Pipeline**

1. **ANALYSE** – MediaPipe skeleton + pixel labels for every body part  
2. **EDIT** – paint regions, drag joints, force view, disable frames (edits survive re-analysis)  
3. **CUT** – extract every part for FRONT / SIDE / BACK  
4. **PICK** – choose the best candidate per part/view (locks survive re-cuts)  
5. **EXPORT** – `rig.png`, `atlas.png`, `rig.json`, assembled figures, per-part PNGs  

Side-view sprites always face **RIGHT**. In side view: **R = near limb**, **L = far limb**.  
Default scale: **1 px = 2 cm**.

---

## What is included

```
human_rig.py          main application + GUI + cutter
organism_model.py     permanent self-improving body model
rig_body.py           part mask + local scale (add/remove limbs, bulk)
rig_gui_extra.py      ALIGN page + ANIMATE page
rig_pose.py           pose renderer, seam report, auto-tie
rig_motion.py         preset actions + secondary motion
rig_tools.py          hidden-area fill, hole repair, rig health check
rig_extras.py         EXTRA PARTS: cut / copy / mirror / motion (no GUI)
rig_extras_gui.py     the EXTRAS page
rig_trim.py           TRIM: erase / restore single pixels of finished parts (no GUI)
rig_trim_gui.py       the TRIM page
requirements.txt      Python packages
README.md             this file
```

---

## Install (Windows / macOS / Linux)

### 1. Python

You need **Python 3.10 – 3.12** (3.11 recommended).  
Check with:

```bash
python --version
# or
python3 --version
```

### 2. Create a folder and put the files in it

Unzip the package so you have a folder that looks like:

```
human_rig/
  human_rig.py
  organism_model.py
  rig_gui_extra.py
  rig_motion.py
  rig_pose.py
  rig_tools.py
  requirements.txt
  README.md
```

### 3. Virtual environment (strongly recommended)

```bash
cd human_rig
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install the packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Notes**

- `mediapipe` downloads a pose model the first time you run ANALYSE (~30 MB). It is cached in `~/.humanrig/`.
- `rembg` is optional but recommended (clean background removal). The first run also downloads a model.
- On Linux you may also need:  
  `sudo apt install python3-tk`  
  (or the equivalent for your distro) so the GUI works.
- If MediaPipe or rembg fail to install, the program still runs with a fallback detector and a simple alpha threshold. You can still drag joints and paint by hand.

### 5. Run

```bash
python human_rig.py
```

A dark window titled **HUMAN.RIG** should open.

---

## First-time walkthrough

1. **Load sources**  
   Click the folder / “+ sources” button and select one or more images (or a short GIF).  
   Best results: character standing, plain background, front + side + back views if possible.

2. **ANALYSE**  
   The program detects the skeleton and labels every pixel.  
   Failed frames appear in a list so you can rescue them with the joint tool.

3. **EDIT**  
   - Switch to the editor page.  
   - Paint any mis-labelled regions (brush / erase / fill / extend).  
   - Drag the yellow joint dots if the pose detector was wrong.  
   - Force a frame to FRONT / SIDE / BACK if the automatic facing is wrong.  
   - Disable bad frames.

4. **CUT**  
   Press CUT. Parts are extracted for all three views.  
   Missing views are synthesised or borrowed when possible.

5. **PICK**  
   Review the candidate cards and lock the ones you like.  
   Locks survive future re-cuts.

6. **ALIGN** (optional but recommended)  
   Drag the coloured dots so the pieces sit tightly on each other.  
   “AUTO TIE” does most of the work.

7. **ANIMATE**  
   Choose an action (idle, walk, run, jump \ldots).  
   Adjust flex / bounce / inertia if you want more or less secondary motion.  
   Export GIF / APNG / sprite sheet + JSON.

8. **EXPORT**  
   Writes everything into a `loadouts/<name>/` folder:

   - `rig.png` / `rig_x4.png` – sheet with exploded parts + assembled figures  
   - `atlas.png` – clean transparent atlas  
   - `assembled_front.png`, `assembled_side.png`, `assembled_back.png`  
   - `rig.json` – pivots, hierarchy, lengths, alignment  
   - `parts/` – individual PNGs  
   - `animations/` – if you exported any clips  

Loadouts live next to the scripts in a folder called `loadouts/`.  
Body models (the permanent learner) live in `~/.humanrig/models/`.

---

## Useful folders after first run

| Location | What it is |
|----------|------------|
| `./loadouts/` | All your character loadouts |
| `~/.humanrig/` | MediaPipe model + permanent body models |
| `./loadouts/<name>/edits/` | Your paint strokes & joint overrides (survive re-analysis) |
| `./loadouts/<name>/animations/` | Exported GIFs / APNGs / sheets |

---

## Optional: run without MediaPipe / rembg

If those packages are a problem:

```bash
pip install numpy Pillow scipy
python human_rig.py
```

You will get the template pose detector. Use the **JOINTS** tool in the editor to place the yellow dots, then paint the regions. Everything else still works.

---

## BODY panel – add / remove parts & scale

A **BODY** panel sits in the left sidebar.

### Turn limbs off

Checkboxes for Right arm, Left arm, Right leg, Left leg, Head + neck, Torso.
Uncheck a limb, then press **CUT / UPDATE RIG**. That limb is no longer cut, posed, or animated.
Turning an upper arm off also removes the forearm and hand. Pelvis is always kept (root).

### Scale regions

Sliders (0.55 – 1.55) for Neck, Chest, Abdomen / gut, Upper arms, Thighs.
Scale changes both **length** and **thickness** of that region on the next CUT.

**RESET BODY** restores full human + normal proportions.

Settings are saved inside the loadout (`meta.json` → `mask` + `scale`) and survive re-opening.

## TRIM page - fine-tune finished parts (white crescents under raised arms, hips ...)

Some cut parts carry material that is hidden while the limb hangs down but shows when it rotates (the upper-arm sleeve
under a raised arm, the underwear edge beside the hip). TRIM lets you delete those pixels by hand **after** the cut.

- Pick FRONT / SIDE / BACK and a part. Tools: **ERASE BRUSH**, **RESTORE BRUSH**, **ERASE INSIDE SHAPE**, **KEEP ONLY INSIDE SHAPE**
  (lasso: click points, click the first dot or double-click to close, right-click undoes a point).
- The **live preview** shows the whole figure in NEUTRAL, ARMS UP or LEGS APART so you see exactly what the trim fixes.
- Ctrl+Z / Ctrl+Y undo and redo, `[` and `]` change brush size, RESET THIS PART / RESET EVERYTHING bring pixels back.
- Trims are **non-destructive**: stored in `meta.json` -> `trims` as offsets from each part's pivot, so they survive
  re-cutting. `parts/` holds the trimmed (final) PNGs, `parts_raw/` the untouched cuts. Tip: the CUT page's
  *Overlap SIZE %* slider is the first lever if a crescent is large.

## EXTRAS page - add parts (tails, ears, horns, wings, extra arms ...)

The base rig is always the same 17 human parts. **Extras** are new parts you add on top, so one body can become many
variants. They live in their own layer: the learned body model and the 17 base parts are never touched.
Needs a CUT rig first.

1. **CUT** - load a source: copy one of the rig's own parts (FRONT/SIDE/BACK + part), or **IMPORT IMAGE...** (any picture).
   Click around the bit you want with the lasso (or drag freehand; click the first dot or double-click to close; right-click
   undoes a point). Tick *No lasso* for a plain copy. **REMOVE FLAT BACKGROUND** helps with cut-outs on a plain colour.
   Pick what to attach it to, which views it applies to, the pivot, and a motion preset, then **MAKE PART**.
2. **PIVOT** - drag the pink dot to where the part rotates (base of a tail, root of an ear). Or use the pivot presets.
3. **PLACE** - drag the part on the body. Positions, pivot, size, layer and visibility are stored **per view**.

**Every view is its own part.** A part made for the front does not show in side or back until you make one for them:
cut in that view, choose *add to the selected part* instead of *new part*, and press the button (or COPY TO ALL VIEWS for
a quick start). The list shows `[F S B]` for the views a part has. Ctrl+Z / Ctrl+Y undo extras edits; arrow keys nudge only
while the canvas has focus, and the mouse wheel over sliders scrolls the panel instead of changing them.

Per part you can set: size, rest rotation, parent (any base part, or another extra - so a tail can be a chain),
layer (behind / in front of its parent, or of the whole body), flip, **MIRROR TWIN** (left/right pair), duplicate.

**Motion** - presets (rigid, ear, tail, hair/cape, wing, antenna, dangling) or your own: stiffness, damping, inertia
(shoved by the hips), how much it follows its parent, and an idle sway. Extras move on the ANIMATE page together with the
body (and use its FLEX / BOUNCE / INERTIA sliders) and their angles are included in the exported animation JSON.

Saved inside the loadout (`meta.json` -> `extras`, pictures in `extras/src/`). On SAVE & EXPORT the game-ready sprites are
written to `extras/<id>_<view>.png` and described in `rig.json` under `"extras"` (parent, pivot_px, offset_cm per view,
layer, motion). The assembled figures include the extras at rest.

### What still comes later

- Named bulk paint fields (pec / bicep / calf)
- Extra spine / neck vertebrae
- Non-human default masks (dog, bird, plant ...)
- Extras in the exploded RIG SHEET / atlas (they are exported as separate PNGs for now)

## Troubleshooting

- **Safety net (v3.1):** saves are written atomically and the previous copy is kept as `meta.json.bak` / `rig.json.bak`; a damaged
  file is recovered from it automatically. Any error inside the app is shown in LOG and written to `humanrig_errors.log`
  (send me that file if something breaks). Missing packages give a plain message instead of a traceback.

| Problem | Fix |
|---------|-----|
| “No module named mediapipe” | `pip install mediapipe` (or use the fallback detector) |
| GUI does not open / “no display” | On a headless server you need a virtual display or run on a machine with a desktop |
| Black / empty sprites after CUT | Make sure the character is cut out (alpha) or turn on rembg / paint the background away |
| Joints look wrong | Switch to JOINTS tool, drag the yellow dots, then CUT again |
| Side limbs swapped | Use the FLIP FACING button on that frame |
| “tkinter not found” (Linux) | `sudo apt install python3-tk` |

---

## License / credit

This is your system. The scripts are exactly the ones you provided.  
Enjoy the rigs.
