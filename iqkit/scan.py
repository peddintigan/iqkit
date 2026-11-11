"""Find which Monk Skin Tone level skin in an image is closest to, linked to Zone System zones.

Pipeline
--------
1. Optional gray-card white balance (linear light, sRGB primaries assumed).
2. Luma -> Zone System zone for every pixel, using the camera tone curve (tone.py).
3. Select pixels: zones you choose, clipped/crushed pixels excluded, then either
   a colour gate (hue/chroma in CIELAB) or a mask you painted yourself.
4. Nearest MST level for every selected pixel, per zone, and per connected region.

Honest limits
-------------
* The colour gate is a heuristic. Wood, cardboard, sand and warm walls can pass it, and
  very dark or heavily shaded skin can fail it. Always look at the overlay image, and use
  a mask (--mask) when you need to be sure what is being measured.
* Per-pixel MST level mixes skin tone with lighting: the same face spans several zones
  from highlight to shadow, so its pixels spread across several MST levels. The region
  median, taken from well-lit zones, is the more stable estimate.
* The result depends on the lighting and the camera's rendering, not just the person.
"""
from __future__ import annotations

import numpy as np

from . import qualify as qualifymod
from . import skintone, tone

N_LEVELS = len(skintone.MST_LEVELS)


def block_reduce(arr, k):
    """Average k x k blocks (works for HxW and HxWxC arrays)."""
    if k <= 1:
        return arr
    h, w = arr.shape[:2]
    h2, w2 = h // k * k, w // k * k
    a = arr[:h2, :w2].reshape(h2 // k, k, w2 // k, k, *arr.shape[2:])
    return a.mean(axis=(1, 3))


def white_balance(img, roi):
    """Scale R and B (in linear light) so the gray card inside `roi` (x,y,w,h) is neutral.

    Green is left alone, so the card's brightness is unchanged.
    """
    x, y, w, h = roi
    lin = skintone.srgb_to_linear(img[..., :3])
    card = lin[y : y + h, x : x + w].reshape(-1, 3).mean(axis=0)
    if np.any(card <= 0):
        raise ValueError("white-balance ROI is black; pick the gray card")
    gains = card[1] / card
    return skintone.linear_to_srgb(lin * gains)


def classify(lab, chunk=200_000):
    """Nearest MST level (1..10) and its dE76 for an (N,3) array of Lab values."""
    lab = np.asarray(lab, float).reshape(-1, 3)
    levels = np.empty(len(lab), dtype=np.uint8)
    des = np.empty(len(lab))
    for s in range(0, len(lab), chunk):
        d = np.linalg.norm(lab[s : s + chunk, None, :] - skintone.MST_LAB[None, :, :], axis=-1)
        i = d.argmin(axis=1)
        levels[s : s + chunk] = skintone.MST_LEVELS[i]
        des[s : s + chunk] = d[np.arange(len(i)), i]
    return levels, des


def scan(
    img,
    curve=None,
    gray_value=None,
    zones=None,
    hue=(35.0, 95.0),
    chroma=(3.0, 45.0),
    wb_roi=None,
    mask=None,
    labels=None,
    part_labels=None,
    qualifier=None,
    max_pixels=1_000_000,
    min_area_frac=0.002,
    n_regions=5,
):
    """Scan an encoded-sRGB image (0..1) and classify skin pixels and regions to MST levels.

    zones      : (lo, hi) zone range to analyse, e.g. (4, 7); None = every zone except clipped/crushed
    hue, chroma: colour gate in CIELAB, ignored when `mask` is given
    mask       : boolean array the size of `img`; True = analyse this pixel as skin (replaces the colour gate)
    labels     : integer array the size of `img`, 0 = ignore, i > 0 = skin candidates of face/subject i
                 (see faces.face_labels). Unlike `mask`, the colour gate is still applied inside them,
                 and each label becomes one reported region.
    part_labels: label image from faces.face_part_labels (4*face + part code). Each patch is then also
                 measured on its own and reported in out["part_report"], so forehead and cheeks can be
                 compared; it does not change which pixels feed the per-face result.
    qualifier  : needs `labels`. Dict with "core" (label image of seed regions, from faces.face_region_labels)
                 and optional "hue_tol", "chroma_tol", "luma_tol", "manual", "clean" (see qualify.qualify). The
                 per-face colour qualifier then replaces the global hue/chroma gate.
    Returns a dict; see the keys at the end of the function.
    """
    img = np.clip(np.asarray(img, float), 0.0, 1.0)[..., :3]
    curve = curve or tone.ToneCurve.standard("srgb")
    h, w = img.shape[:2]
    k = max(1, int(np.ceil(np.sqrt(h * w / max_pixels))))
    original_small = block_reduce(img, k)  # untouched colours, for display (before any white balance)
    if wb_roi is not None:
        img = white_balance(img, wb_roi)
    small = block_reduce(img, k)
    sh, sw = small.shape[:2]

    zr = tone.zone_map(tone.luma(small), curve, gray_value)
    lab = skintone.rgb_to_lab(small)
    c, hu = skintone.chroma_hue(lab)

    valid = ~zr.clipped & ~zr.crushed
    if zones is not None:
        valid &= (zr.zone_float >= zones[0] - 0.5) & (zr.zone_float < zones[1] + 0.5)
    hue_ok = (hu >= hue[0]) & (hu <= hue[1])
    chroma_ok = (c >= chroma[0]) & (c <= chroma[1])
    colour = hue_ok & chroma_ok
    small_labels = None
    colour_names = ("hue", "chroma")
    qualifier_seeds = None
    if qualifier is not None and labels is None:
        raise ValueError("qualifier needs labels (faces.face_region_labels)")
    if mask is not None and labels is not None:
        raise ValueError("give either mask or labels, not both")
    if mask is not None:
        m = block_reduce(np.asarray(mask, float), k) > 0.5
        if m.shape != (sh, sw):
            raise ValueError("mask size does not match the image")
        sel = valid & m
    elif labels is not None:
        labels = np.asarray(labels)
        if labels.shape[:2] != (h, w):
            raise ValueError("labels size does not match the image")
        small_labels = labels[k // 2 :: k, k // 2 :: k][:sh, :sw]
        if qualifier is not None:
            small_core = np.asarray(qualifier["core"])[k // 2 :: k, k // 2 :: k][:sh, :sw]
            qmask, qseeds = qualifymod.qualify(
                lab, hu, c, small_labels, small_core,
                hue_tol=qualifier.get("hue_tol", 18.0),
                chroma_tol=qualifier.get("chroma_tol", (7.0, 0.5)),
                luma_tol=qualifier.get("luma_tol", 28.0),
                manual=qualifier.get("manual"),
                clean=qualifier.get("clean", 1),
            )
            colour, hue_ok, chroma_ok = qmask, qmask, np.ones_like(qmask)
            colour_names = ("qualifier", "chroma")
            qualifier_seeds = qseeds
        sel = valid & colour & (small_labels > 0)
    else:
        sel = valid & colour

    levels_img = np.zeros((sh, sw), np.uint8)
    n_sel = int(sel.sum())
    out = {
        "scale": k,
        "work_shape": (sh, sw),
        "selected": n_sel,
        "selected_fraction": n_sel / (sh * sw),
        "level_share": np.zeros(N_LEVELS),
        "zone_level_counts": np.zeros((tone.N_ZONES, N_LEVELS), int),
        "regions": [],
    }
    if n_sel:
        lv, _ = classify(lab[sel])
        levels_img[sel] = lv
        out["level_share"] = np.bincount(lv - 1, minlength=N_LEVELS) / n_sel
        np.add.at(out["zone_level_counts"], (zr.zone_int[sel], lv - 1), 1)
        out["regions"] = _regions(sel, lab, zr, levels_img, k, min_area_frac, n_regions, small_labels)

    if part_labels is not None:
        pl = np.asarray(part_labels)
        if pl.shape[:2] != (h, w):
            raise ValueError("part_labels size does not match the image")
        small_parts = pl[k // 2 :: k, k // 2 :: k][:sh, :sw]
        out["part_report"] = _part_report(small_parts, valid & colour, lab, zr)
    if small_labels is not None:
        zone_ok = np.ones_like(valid)
        if zones is not None:
            zone_ok = (zr.zone_float >= zones[0] - 0.5) & (zr.zone_float < zones[1] + 0.5)
        out["face_report"] = _face_report(small_labels, zr, zone_ok, hue_ok, chroma_ok, hu, c, colour_names)
    if qualifier_seeds is not None:
        out["qualifier_seeds"] = qualifier_seeds
    out["overlay"] = _overlay(small, sel, levels_img)
    out["original_small"] = (np.clip(original_small, 0, 1) * 255 + 0.5).astype(np.uint8)
    out["zone_int"] = zr.zone_int
    out["selected_mask"] = sel
    return out


def _part_report(small_parts, usable, lab, zr):
    """Measure every labelled patch on its own: median colour of its usable pixels and nearest MST level."""
    rows = []
    for v in np.unique(small_parts[small_parts > 0]):
        pix = small_parts == v
        use = pix & usable
        ys, xs = np.nonzero(pix)
        row = {"face": int(v) // 4, "code": int(v) % 4, "pixels": int(pix.sum()), "used": int(use.sum()), "median_lab": None,
               "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)}
        if row["used"]:
            med = np.median(lab[use], axis=0)
            level, de, _ = skintone.nearest_mst(med)
            chroma, hue = skintone.chroma_hue(med)
            row.update(
                median_lab=med,
                nearest_mst=level,
                nearest_dE=de,
                chroma=float(chroma),
                hue=float(hue),
                median_zone=float(np.median(zr.zone_float[use])),
            )
        rows.append(row)
    return rows


def _face_report(labels, zr, zone_ok, hue_ok, chroma_ok, hue, chroma, colour_names=("hue", "chroma")):
    """For each face label: how many patch pixels there were, how many were used, and what removed the rest.

    Pixels are attributed to the first filter that rejects them, in this order:
    exposure (clipped/crushed), zone range, hue gate, chroma gate.
    """
    report = []
    for i in np.unique(labels[labels > 0]):
        pix = labels == i
        bad = pix & (zr.clipped | zr.crushed)
        rest = pix & ~bad
        no_zone = rest & ~zone_ok
        rest = rest & zone_ok
        no_hue = rest & ~hue_ok
        rest = rest & hue_ok
        no_chroma = rest & ~chroma_ok
        used = rest & chroma_ok
        report.append(
            {
                "label": int(i),
                "patch_pixels": int(pix.sum()),
                "used": int(used.sum()),
                "excluded": {
                    "exposure": int(bad.sum()),
                    "zone": int(no_zone.sum()),
                    colour_names[0]: int(no_hue.sum()),
                    colour_names[1]: int(no_chroma.sum()),
                },
                "median_hue": float(np.median(hue[pix])),
                "median_chroma": float(np.median(chroma[pix])),
                "median_zone": float(np.median(zr.zone_float[pix])),
            }
        )
    return report


def _regions(sel, lab, zr, levels_img, k, min_area_frac, n_regions, given_labels=None):
    if given_labels is not None:
        # one region per subject label, all of them (n_regions only limits the colour-gate search);
        # no connected-component search, no minimum area
        labels = np.where(sel, given_labels, 0).astype(int)
        sizes = np.bincount(labels.ravel())
        order = [i for i in np.argsort(sizes[1:])[::-1] + 1 if sizes[i] > 0]  # every face, however many
    else:
        try:
            from scipy import ndimage
        except ImportError:  # pragma: no cover
            labels, n, opened = sel.astype(int), 1, sel
        else:
            opened = ndimage.binary_opening(sel, structure=np.ones((3, 3), bool))
            if not opened.any():
                opened = sel
            labels, n = ndimage.label(opened)
        min_px = max(1, int(min_area_frac * sel.size))
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        order = [i for i in np.argsort(sizes[1:])[::-1] + 1 if sizes[i] >= min_px][:n_regions]
    regions = []
    for i in order:
        idx = labels == i
        med = np.median(lab[idx], axis=0)
        level, de, _ = skintone.nearest_mst(med)
        counts = np.bincount(levels_img[idx].astype(int) - 1, minlength=N_LEVELS) / idx.sum()
        rows, cols = np.nonzero(idx)
        top = np.argsort(counts)[::-1][:2]
        regions.append(
            {
                "label": int(i),
                "pixels": int(idx.sum()),
                "bbox": (int(cols.min() * k), int(rows.min() * k), int((cols.max() - cols.min() + 1) * k),
                         int((rows.max() - rows.min() + 1) * k)),  # x, y, w, h in original pixels
                "median_lab": med,
                "nearest_mst": level,
                "nearest_dE": de,
                "median_zone": float(np.median(zr.zone_float[idx])),
                "top_levels": [(int(t + 1), float(counts[t])) for t in top if counts[t] > 0],
            }
        )
    return regions


def _overlay(small, sel, levels_img):
    """Dimmed gray picture with selected pixels painted in their MST swatch colour."""
    g = (tone.luma(small) * 0.45 * 255).astype(np.uint8)
    out = np.stack([g, g, g], axis=-1)
    if sel.any():
        out[sel] = skintone.MST_RGB[levels_img[sel].astype(int) - 1].astype(np.uint8)
    return out


def isolate_colour(img, mask, feather=0.0):
    """Keep colour only where `mask` is set; everything else becomes black and white.

    img     : encoded RGB picture, float 0..1 (or uint8)
    mask    : boolean or 0..1 array, at the picture's size or smaller (it is scaled up smoothly)
    feather : Gaussian softening of the mask edge, in the mask's own pixels (0 = hard edge)
    The grays are the Rec.709 luma of the encoded values, so brightness is unchanged.
    Returns uint8 RGB at the picture's size.
    """
    from PIL import Image

    a = np.asarray(img)
    a = a.astype(float) / 255.0 if a.dtype.kind in "ui" else a.astype(float)
    a = np.clip(a, 0.0, 1.0)[..., :3]
    h, w = a.shape[:2]
    m = np.asarray(mask, float)
    if feather > 0:
        try:
            from scipy import ndimage

            m = ndimage.gaussian_filter(m, feather)
        except ImportError:  # pragma: no cover
            pass
    if m.shape != (h, w):
        m = np.asarray(Image.fromarray(m.astype(np.float32)).resize((w, h), Image.BILINEAR), float)
    m = np.clip(m, 0.0, 1.0)[..., None]
    gray = tone.luma(a)[..., None]
    out = m * a + (1.0 - m) * gray
    return (out * 255 + 0.5).astype(np.uint8)


def add_legend(overlay, level_share=None):
    """Return the overlay with a side panel of the ten MST reference orbs.

    Each orb is a flat disc of the published sRGB value (not Google's official orb images,
    which contain many varied colours). If `level_share` is given, the share of selected
    pixels for each level is printed beside its orb and the dominant level is ringed.
    """
    from PIL import Image, ImageDraw, ImageFont

    h = overlay.shape[0]
    slot = h / N_LEVELS
    d = max(8, int(0.62 * slot))
    fs = max(10, int(0.30 * slot))
    pad = max(4, int(0.18 * slot))
    pw = int(2.4 * slot) if slot >= 30 else int(2.6 * slot) + 20
    ss = 4  # draw 4x larger, then shrink, for smooth circle edges
    big = Image.new("RGB", (pw * ss, h * ss), (28, 28, 28))
    dr = ImageDraw.Draw(big)
    top = int(np.argmax(level_share)) if level_share is not None and np.max(level_share) > 0 else None
    for i in range(N_LEVELS):
        cy = (i + 0.5) * slot
        x0, y0 = pad * ss, (cy - d / 2) * ss
        box = [x0, y0, x0 + d * ss, y0 + d * ss]
        dr.ellipse(box, fill=tuple(int(v) for v in skintone.MST_RGB[i]))
        if i == top:
            dr.ellipse(box, outline=(255, 255, 255), width=max(2, ss * 2))
    panel = big.resize((pw, h), Image.LANCZOS)
    dr = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:  # older Pillow: fixed-size bitmap font
        font = ImageFont.load_default()
    for i in range(N_LEVELS):
        cy = (i + 0.5) * slot
        label = str(i + 1)
        if level_share is not None and level_share[i] >= 0.005:
            label += f"  {100 * level_share[i]:.0f}%"
        xt = pad + d + max(4, pad // 2)
        try:
            dr.text((xt, cy), label, fill=(225, 225, 225), font=font, anchor="lm")
        except ValueError:
            dr.text((xt, cy - fs // 2), label, fill=(225, 225, 225), font=font)
    return np.hstack([overlay, np.asarray(panel)])
