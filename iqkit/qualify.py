"""A colour qualifier for skin, in the spirit of DaVinci Resolve's HSL qualifier.

Inside each detected face the skin is selected by colour range instead of by fixed patches:
hue (an angle in CIELAB), chroma (colourfulness, the saturation analogue) and L* (lightness).

By default the ranges are seeded per face, like using the eyedropper: the seed colour is the
trimmed median colour of the face's core region (eyes, brows, mouth and nostrils already left out),
and the qualifier keeps pixels whose hue and chroma are close to it, with a wide lightness range
so that shaded and lit skin are both included. Pass `manual` ranges to set any of the three
yourself, as with the qualifier's sliders.

Limits: hair, beards and shadows of similar hue can still pass, and very dark skin has low chroma
so its hue is less reliable. Look at the mask (it can be saved) before trusting a result.
"""
from __future__ import annotations

import numpy as np

from . import skintone


def hue_difference(a, b):
    """Signed difference a - b in degrees, wrapped to [-180, 180)."""
    return (np.asarray(a, float) - b + 180.0) % 360.0 - 180.0


def seed_colour(lab, trim_dark=0.15, trim_light=0.10):
    """Median Lab of seed pixels after dropping the darkest and lightest few percent
    (hair, shadow and shine), so a little contamination cannot move the seed."""
    lab = np.asarray(lab, float).reshape(-1, 3)
    if len(lab) >= 10:
        order = np.argsort(lab[:, 0])
        lo, hi = int(trim_dark * len(lab)), int(np.ceil((1 - trim_light) * len(lab)))
        lab = lab[order[lo : max(hi, lo + 1)]]
    return np.median(lab, axis=0)


def qualify(lab, hue, chroma, region, core, hue_tol=18.0, chroma_tol=(7.0, 0.5), luma_tol=28.0,
            manual=None, clean=1, min_seed=20):
    """Select skin pixels inside each face region.

    lab, hue, chroma : per-pixel CIELAB, hue angle (degrees) and chroma of the picture
    region, core     : label images from faces.face_region_labels (face number per pixel)
    hue_tol          : keep pixels within this many degrees of the seed hue (hue varies by +/-15 deg across one lit face)
    chroma_tol       : (absolute, fraction): keep chroma within seed +/- (absolute + fraction * seed chroma)
    luma_tol         : keep L* within seed +/- this
    manual           : optional {"hue": (lo, hi), "chroma": (lo, hi), "luma": (lo, hi)}; a given range
                       replaces the seeded one for that channel
    clean            : passes of closing (fills small holes such as highlights) then opening (removes specks), 3x3; 0 = off
    Returns (mask, seeds) where seeds maps face number to {"lab", "hue", "chroma", "n_seed"}.
    """
    manual = manual or {}
    mask = np.zeros(region.shape, bool)
    seeds = {}
    for i in np.unique(region[region > 0]):
        in_face = region == i
        seed_pix = core == i
        if seed_pix.sum() < min_seed:
            seed_pix = in_face
        seed = seed_colour(lab[seed_pix])
        s_chroma, s_hue = (float(v) for v in skintone.chroma_hue(seed))
        seeds[int(i)] = {"lab": seed, "hue": s_hue, "chroma": s_chroma, "n_seed": int(seed_pix.sum())}

        if "hue" in manual:
            lo, hi = manual["hue"]
            h = hue[in_face]
            hue_ok = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)  # wraps through 0 degrees
        else:
            hue_ok = np.abs(hue_difference(hue[in_face], s_hue)) <= hue_tol
        if "chroma" in manual:
            chroma_ok = (chroma[in_face] >= manual["chroma"][0]) & (chroma[in_face] <= manual["chroma"][1])
        else:
            chroma_ok = np.abs(chroma[in_face] - s_chroma) <= chroma_tol[0] + chroma_tol[1] * s_chroma
        if "luma" in manual:
            luma_ok = (lab[in_face, 0] >= manual["luma"][0]) & (lab[in_face, 0] <= manual["luma"][1])
        else:
            luma_ok = np.abs(lab[in_face, 0] - seed[0]) <= luma_tol
        mask[in_face] = hue_ok & chroma_ok & luma_ok

    if clean and mask.any():
        try:
            from scipy import ndimage
        except ImportError:  # pragma: no cover
            return mask, seeds
        box = np.ones((3, 3), bool)
        closed = ndimage.binary_closing(mask, structure=box, iterations=clean) & (region > 0)
        cleaned = ndimage.binary_opening(closed, structure=box, iterations=clean)
        if cleaned.any():
            mask = cleaned
    return mask, seeds


def mask_outline(mask):
    """Boundary pixels of a boolean mask (one pixel wide), for drawing the selection on a picture."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        return mask.copy()
    return mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), bool))
