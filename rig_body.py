"""rig_body.py - part mask + local scale / bulk for HUMAN.RIG.

Lets a loadout disable limbs (one-armed, one-legged, headless ...) and apply
small continuous scales so a character can be bulkier, skinnier, or longer-necked.
The full human hierarchy stays the canonical maximum; everything else filters it.
"""
from __future__ import annotations
import copy
from typing import Dict, Iterable, List, Optional

# Canonical full set (must match human_rig.PART_KEYS order)
CENTER = ["head", "neck", "chest", "abdomen", "pelvis"]
PART_KEYS = CENTER + [
    "uarmR", "uarmL", "farmR", "farmL", "handR", "handL",
    "thighR", "thighL", "shinR", "shinL", "footR", "footL",
]
PARENT = {
    "pelvis": None, "abdomen": "pelvis", "chest": "abdomen", "neck": "chest", "head": "neck",
    "uarm": "chest", "farm": "uarm", "hand": "farm",
    "thigh": "pelvis", "shin": "thigh", "foot": "shin",
}
HIER = [
    "pelvis", "abdomen", "chest", "neck", "head",
    "uarmR", "uarmL", "farmR", "farmL", "handR", "handL",
    "thighR", "thighL", "shinR", "shinL", "footR", "footL",
]
POSE_ORDER = {
    "front": [
        "pelvis", "abdomen", "chest",
        "thighR", "thighL", "shinR", "shinL", "footR", "footL",
        "uarmR", "uarmL", "farmR", "farmL", "handR", "handL",
        "neck", "head",
    ],
    "side": [
        "uarmL", "farmL", "handL", "thighL", "shinL", "footL",
        "pelvis", "abdomen", "chest",
        "thighR", "shinR", "footR", "uarmR", "farmR", "handR",
        "neck", "head",
    ],
}
POSE_ORDER["back"] = list(POSE_ORDER["front"])

# Groups used by the UI and by cascade logic
LIMB_GROUPS = {
    "armR": ["uarmR", "farmR", "handR"],
    "armL": ["uarmL", "farmL", "handL"],
    "legR": ["thighR", "shinR", "footR"],
    "legL": ["thighL", "shinL", "footL"],
    "head": ["head", "neck"],
    "torso": ["chest", "abdomen", "pelvis"],
}
GROUP_LABELS = {
    "armR": "Right arm",
    "armL": "Left arm",
    "legR": "Right leg",
    "legL": "Left leg",
    "head": "Head + neck",
    "torso": "Torso",
}

# Bases that can be scaled (length + thickness)
SCALE_BASES = ["head", "neck", "chest", "abdomen", "pelvis",
               "uarm", "farm", "hand", "thigh", "shin", "foot"]
SCALE_LABELS = {
    "head": "Head", "neck": "Neck", "chest": "Chest", "abdomen": "Abdomen / gut",
    "pelvis": "Pelvis", "uarm": "Upper arms", "farm": "Forearms", "hand": "Hands",
    "thigh": "Thighs", "shin": "Shins", "foot": "Feet",
}
# Allowed scale range (keeps cuts and animation stable)
SCALE_MIN, SCALE_MAX = 0.55, 1.55
DEFAULT_SCALE = 1.0


def base_of(k: str) -> str:
    return k[:-1] if k[-1] in "LR" and k not in CENTER else k


def side_of(k: str) -> str:
    return k[-1] if k not in CENTER else ""


def parent_key(k: str) -> Optional[str]:
    p = PARENT[base_of(k)]
    return None if p is None else (p if p in CENTER else p + side_of(k))


def children_of(k: str, keys: Iterable[str] = PART_KEYS) -> List[str]:
    return [c for c in keys if parent_key(c) == k]


def default_mask() -> Dict[str, bool]:
    return {k: True for k in PART_KEYS}


def default_scale() -> Dict[str, float]:
    return {b: DEFAULT_SCALE for b in SCALE_BASES}


def cascade_mask(mask: Dict[str, bool]) -> Dict[str, bool]:
    """If a parent is off, force all its descendants off. Pelvis is always kept."""
    out = dict(mask)
    out["pelvis"] = True
    # walk hierarchy top-down
    for k in HIER:
        if k == "pelvis":
            continue
        p = parent_key(k)
        if p is not None and not out.get(p, True):
            out[k] = False
    return out


def set_group(mask: Dict[str, bool], group: str, on: bool) -> Dict[str, bool]:
    """Toggle a whole limb / region. Returns a new cascaded mask."""
    out = dict(mask)
    for k in LIMB_GROUPS.get(group, []):
        out[k] = bool(on)
    return cascade_mask(out)


def group_state(mask: Dict[str, bool], group: str) -> bool:
    """True if every key in the group is currently on."""
    keys = LIMB_GROUPS.get(group, [])
    return all(mask.get(k, True) for k in keys) if keys else True


def active_keys(mask: Optional[Dict[str, bool]] = None) -> List[str]:
    m = cascade_mask(mask or default_mask())
    return [k for k in PART_KEYS if m.get(k, True)]


def active_hier(mask: Optional[Dict[str, bool]] = None) -> List[str]:
    m = cascade_mask(mask or default_mask())
    return [k for k in HIER if m.get(k, True)]


def active_order(view: str, mask: Optional[Dict[str, bool]] = None) -> List[str]:
    m = cascade_mask(mask or default_mask())
    return [k for k in POSE_ORDER[view] if m.get(k, True)]


def clamp_scale(v: float) -> float:
    return float(max(SCALE_MIN, min(SCALE_MAX, v)))


def normalize_scale(raw: Optional[Dict] = None) -> Dict[str, float]:
    out = default_scale()
    if not raw:
        return out
    for b, v in raw.items():
        if b in out:
            try:
                out[b] = clamp_scale(float(v))
            except (TypeError, ValueError):
                pass
    return out


def normalize_mask(raw: Optional[Dict] = None) -> Dict[str, bool]:
    out = default_mask()
    if not raw:
        return cascade_mask(out)
    for k, v in raw.items():
        if k in out:
            out[k] = bool(v)
    return cascade_mask(out)


def scale_of(scale: Dict[str, float], key: str) -> float:
    """Length + radius multiplier for a part key."""
    return float(scale.get(base_of(key), DEFAULT_SCALE))


def apply_scale_to_lengths(lengths: Dict[str, float], scale: Dict[str, float]) -> Dict[str, float]:
    """Return a copy of lengths with scale applied to matching bases."""
    out = dict(lengths)
    for b, s in scale.items():
        if b in out and abs(s - 1.0) > 1e-6:
            out[b] = float(out[b]) * s
    # spine is derived from torso pieces in some places; keep consistent
    if "spine" in out:
        # approximate: spine length dominated by chest+abdomen
        factor = (scale.get("chest", 1.0) + scale.get("abdomen", 1.0)) / 2.0
        out["spine"] = float(out["spine"]) * factor
    return out


def body_summary(mask: Dict[str, bool], scale: Dict[str, float]) -> str:
    """Short human-readable status for the UI."""
    off = [GROUP_LABELS[g] for g in LIMB_GROUPS if not group_state(mask, g)]
    scaled = [f"{SCALE_LABELS[b]}×{scale[b]:.2f}" for b in SCALE_BASES
              if abs(scale.get(b, 1.0) - 1.0) > 0.02]
    bits = []
    if off:
        bits.append("off: " + ", ".join(off))
    if scaled:
        bits.append("scale: " + ", ".join(scaled))
    return "  |  ".join(bits) if bits else "full body, normal proportions"


def ensure_on_result(res, mask=None, scale=None):
    """Attach normalised mask + scale to a result object."""
    res.mask = normalize_mask(mask if mask is not None else getattr(res, "mask", None))
    res.scale = normalize_scale(scale if scale is not None else getattr(res, "scale", None))
    return res
