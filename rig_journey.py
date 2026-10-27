"""rig_journey.py - GUIDED mode: a short, ordered path through the program instead of ten tabs.

GUIDED (default)  - the tab bar shows only the main steps in order, numbered:
                    1 CHECK  2 REVIEW  3 CLEAN UP  4 ADD PARTS  5 ANIMATE
                    Every other page (assembled view, pose, pick parts, align, log) lives under MORE.
                    A strip under the page says what this step is for and has BACK / NEXT buttons that also do the
                    obvious action (NEXT on step 1 = CUT, NEXT on the last step = SAVE & EXPORT).
ALL TABS          - the old layout with every tab, for people who know the program.

The choice is remembered in humanrig_prefs.json next to the program.
"""
import json
from pathlib import Path

PREFS = Path(__file__).resolve().parent / "humanrig_prefs.json"

# (page key, short name, what it is for, optional?)
JOURNEY = (
    ("editor", "CHECK",
     "Each colour is a body part. Click through your pictures and paint over anything that is wrong "
     "(an arm coloured as torso, a missing hand). When it looks right, press CUT.", False),
    ("sheet", "REVIEW",
     "Your character cut into parts. Look for parts that look wrong or missing. Something off? "
     "Open MORE > PICK PARTS to choose a better cut, or go BACK and fix the colours.", False),
    ("trim", "CLEAN UP",
     "Optional. Test with ARMS UP / LEGS APART: if you see stray bits (white crescents under the arms, "
     "edges at the hips) rub them out with the brush. You can skip this.", True),
    ("extras", "ADD PARTS",
     "Optional. Give your character a tail, ears, wings, a belly... Cut it from the rig or import a picture. "
     "Remember each view (front / side / back) needs its own version.", True),
    ("anim", "ANIMATE",
     "Watch your character walk, run or wave. When you are happy, press SAVE & EXPORT to write the game-ready files.", False),
)
JKEYS = tuple(j[0] for j in JOURNEY)
MORE = (("asm", "ASSEMBLED  -  the whole figure"), ("pose", "POSE  -  pose it by hand"),
        ("guide", "PICK PARTS  -  choose between cuts"), ("align", "ALIGN  -  fine-tune joints"),
        ("log", "LOG / HELP"))


def load_prefs():
    try: return json.loads(PREFS.read_text())
    except Exception: return {}


def save_prefs(d):
    try:
        tmp = PREFS.with_name(PREFS.name + ".tmp"); tmp.write_text(json.dumps(d, indent=2)); tmp.replace(PREFS)
    except Exception: pass


def install(cls, G, K, tk, ttk, ImageTk, filedialog, messagebox):
    BG0, BG1, BG2, LINE, TEXT, MUTED, DIM = (K[n] for n in ("BG0", "BG1", "BG2", "LINE", "TEXT", "MUTED", "DIM"))
    ACC, GOLD, GOOD, WARN = (K[n] for n in ("ACC", "GOLD", "GOOD", "WARN"))

    def add(fn): setattr(cls, fn.__name__, fn); return fn

    @add
    def build_journey(self, main):
        self.j_guided = bool(load_prefs().get("guided", True)); self.j_i = 0; self.j_last = "editor"
        self._tab_titles = {k: t.cget("text") for k, t in self.tabs.items()}
        self._tab_order = list(self.tabs)
        self.jfoot = tk.Frame(main, bg=BG2, highlightthickness=1, highlightbackground=ACC)
        self.jfoot.pack(side="bottom", fill="x", pady=(4, 0), before=self.stack)
        self.j_back = self.mk_btn(self.jfoot, "◀ BACK", self.j_prev, "n", pady=6, padx=14); self.j_back.pack(side="left", padx=8, pady=6)
        self.j_next = self.mk_btn(self.jfoot, "NEXT ▶", self.j_go_next, "p", pady=6, padx=18); self.j_next.pack(side="right", padx=8, pady=6)
        self.j_step = tk.Label(self.jfoot, text="", bg=ACC, fg="#001a20", font=self.F["cardb"], padx=10, pady=6); self.j_step.pack(side="left")
        self.j_txt = tk.Label(self.jfoot, text="", bg=BG2, fg=TEXT, font=self.F["ui"], anchor="w", justify="left", wraplength=760)
        self.j_txt.pack(side="left", fill="x", expand=True, padx=12)
        self.j_more = tk.Label(self.tabbar, text="MORE ▾", bg=BG2, fg=MUTED, font=self.F["cardb"], padx=16, pady=7, cursor="hand2",
                               highlightthickness=1, highlightbackground=LINE)
        self.j_more.bind("<Button-1>", self.j_more_menu)
        self.j_mode = tk.Label(self.tabbar, text="", bg=BG0, fg=MUTED, font=self.F["ui"], padx=10, pady=7, cursor="hand2")
        self.j_mode.bind("<Button-1>", lambda e: self.j_toggle())
        self.journey_layout()

    @add
    def journey_layout(self):
        for t in self.tabs.values(): t.pack_forget()
        self.j_more.pack_forget(); self.j_mode.pack_forget()
        if self.j_guided:
            for i, k in enumerate(JKEYS):
                name = JOURNEY[i][1] + (" ·" if JOURNEY[i][3] else "")
                self.tabs[k].config(text=f"{i + 1}  {name}"); self.tabs[k].pack(side="left", padx=(0, 2))
            self.j_more.pack(side="left", padx=(8, 0))
            self.j_mode.config(text="◉ GUIDED   (click for ALL TABS)")
            self.jfoot.pack(side="bottom", fill="x", pady=(4, 0), before=self.stack)
            self.stage_fr.pack_forget()          # the numbered tabs replace the old 5-box tracker (two numberings confuse)
        else:
            self.stage_fr.pack(side="left", padx=20)
            for k in self._tab_order:
                self.tabs[k].config(text=self._tab_titles[k]); self.tabs[k].pack(side="left", padx=(0, 2))
            self.j_mode.config(text="○ ALL TABS   (click for GUIDED)")
            self.jfoot.pack_forget()
        self.j_mode.pack(side="right")
        self.journey_update(getattr(self, "cur_page", "editor"))

    @add
    def j_toggle(self):
        self.j_guided = not self.j_guided
        d = load_prefs(); d["guided"] = self.j_guided; save_prefs(d)
        self.journey_layout()

    @add
    def j_more_menu(self, e):
        m = tk.Menu(self, tearoff=0, bg=BG2, fg=TEXT, activebackground=ACC, activeforeground="#001a20", bd=0)
        for k, label in MORE: m.add_command(label=label, command=lambda k=k: self.show_page(k))
        try: m.tk_popup(e.x_root, e.y_root)
        finally: m.grab_release()

    @add
    def j_state(self):
        """(next label, next action, enabled, hint) for the current journey step."""
        key = JKEYS[self.j_i]; has_res = self.res is not None
        if key == "editor":
            if not self.frames and not has_res:
                if getattr(self, "sources", None): return "ANALYSE ▶", self.analyse, True, "Press ANALYSE to find the skeleton and the body parts."
                return "NEXT ▶", None, False, "Create a character (NEW, top left) and add your pictures first."
            if not has_res or self.dirty: return "CUT ▶", self.cut, True, "Happy with the colours? Press CUT to build the rig."
            return "NEXT ▶", lambda: self.show_page("sheet"), True, ""
        if not has_res: return "NEXT ▶", None, False, "Go back to step 1 and CUT first."
        if key == "anim": return "SAVE & EXPORT ▶", self.export, True, ""
        return ("SKIP ▶" if JOURNEY[self.j_i][3] else "NEXT ▶"), lambda: self.show_page(JKEYS[self.j_i + 1]), True, ""

    @add
    def journey_update(self, key=None):
        if not hasattr(self, "jfoot"): return
        try:
            key = key or getattr(self, "cur_page", "editor")
            if not self.j_guided:
                return
            more = key in dict(MORE)
            self.j_more.config(bg=BG1 if more else BG2, fg=ACC if more else MUTED)
            if key in JKEYS: self.j_i = JKEYS.index(key); self.j_last = key
            if more:
                title = dict(MORE)[key].split("  -  ")[0]
                self.j_step.config(text=f"MORE  {title}")
                self.j_txt.config(text="An extra page. Nothing here is required - go back to the guide when you are done.")
                self.j_back.config(text="◀ BACK TO GUIDE"); self.j_back.unbind("<Button-1>")
                self.j_back.bind("<Button-1>", lambda e: self.show_page(self.j_last)); self.j_back.pack(side="left", padx=8, pady=6)
                self.j_next.pack_forget(); return
            i = self.j_i; label, _cmd, ok, hint = self.j_state()
            self.j_step.config(text=f"STEP {i + 1} OF {len(JOURNEY)}  {JOURNEY[i][1]}" + ("  (optional)" if JOURNEY[i][3] else ""))
            self.j_txt.config(text=(hint + "\n" if hint else "") + JOURNEY[i][2])
            self.j_back.config(text="◀ BACK"); self.j_back.unbind("<Button-1>"); self.j_back.bind("<Button-1>", lambda e: self.j_prev())
            self.j_back.pack(side="left", padx=8, pady=6) if i > 0 else self.j_back.pack_forget()
            self.j_next.pack(side="right", padx=8, pady=6); self.j_next.config(text=label)
            self.j_next.config(fg=("#001a20" if ok else DIM))
        except Exception: pass

    @add
    def j_prev(self):
        if self.j_i > 0: self.show_page(JKEYS[self.j_i - 1])

    @add
    def j_go_next(self, *_):
        label, cmd, ok, hint = self.j_state()
        if cmd is None or not ok: return self.toast(hint or "Not ready yet.", "warn")
        cmd()
