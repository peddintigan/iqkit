"""Compare the forehead and the two cheeks of each face, measured independently.

Why: a single combined number hides problems that affect only one patch. Facial hair darkens
the cheeks, a fringe or hat darkens the forehead, forehead shine lifts it, and uneven lighting
makes the left and right cheeks differ. Looking at the patches side by side shows which
patches agree and which one to distrust.

The thresholds are heuristics, not standards: patches "agree" when their nearest MST levels
are at most `agree_levels` apart and the forehead-vs-cheeks CIEDE2000 is at most `agree_de`.
The explanations are possibilities to check on the image, not diagnoses.
"""
from __future__ import annotations

import numpy as np

from . import faces as facemod
from . import skintone

NAMES = ("forehead", "left_cheek", "right_cheek")


def summarise_faces(part_report, agree_de=5.0, agree_levels=1, big_dl=5.0):
    """Group scan()'s part_report by face and compare the patches.

    Returns {face number: {"parts": {name: row}, "verdict": "agree"|"differ"|"incomplete"|"no data",
    "levels": [...], "de_forehead_cheeks": float|None, "dl_forehead_cheeks": float|None,
    "de_left_right": float|None, "notes": [str, ...]}}.
    """
    out = {}
    for row in part_report:
        name = facemod.CODE_PARTS[row["code"]]
        out.setdefault(row["face"], {"parts": {}})["parts"][name] = row
    for face, info in out.items():
        parts = info["parts"]
        got = {n: r for n, r in parts.items() if r["median_lab"] is not None}
        notes = [f"no usable pixels in the {n.replace('_', ' ')}" for n in NAMES if n in parts and n not in got]
        levels = [got[n]["nearest_mst"] for n in NAMES if n in got]
        info["levels"] = levels
        cheeks = [got[n]["median_lab"] for n in ("left_cheek", "right_cheek") if n in got]
        de_fc = dl_fc = de_lr = None
        if "forehead" in got and cheeks:
            cheek_lab = np.mean(cheeks, axis=0)
            fl = got["forehead"]["median_lab"]
            de_fc = float(skintone.delta_e_2000(fl, cheek_lab))
            dl_fc = float(fl[0] - cheek_lab[0])
            if dl_fc >= big_dl:
                notes.append(
                    f"forehead is {dl_fc:.0f} L* lighter than the cheeks: the cheeks may be darkened by facial hair or "
                    "shadow, or the forehead may have shine"
                )
            elif dl_fc <= -big_dl:
                notes.append(
                    f"forehead is {-dl_fc:.0f} L* darker than the cheeks: it may include hair, a fringe, a hat or shadow"
                )
        if len(cheeks) == 2:
            de_lr = float(skintone.delta_e_2000(cheeks[0], cheeks[1]))
            if de_lr > agree_de:
                notes.append(
                    f"left and right cheeks differ (dE2000 {de_lr:.1f}): uneven lighting, or facial hair on one side"
                )
        info.update(de_forehead_cheeks=de_fc, dl_forehead_cheeks=dl_fc, de_left_right=de_lr, notes=notes)
        if not levels:
            info["verdict"] = "no data"
        elif len(levels) < len(parts):
            info["verdict"] = "incomplete"
        else:
            spread_ok = max(levels) - min(levels) <= agree_levels
            de_ok = all(d is None or d <= agree_de for d in (de_fc, de_lr))
            info["verdict"] = "agree" if spread_ok and de_ok else "differ"
        if len(levels) >= 2 and max(levels) - min(levels) > agree_levels and info["verdict"] == "incomplete":
            info["verdict"] = "differ"
    return out


def format_comparison(summaries):
    """Plain-text table, one block per face."""
    lines = []
    for face in sorted(summaries):
        info = summaries[face]
        lines.append(f"  face {face}: {info['verdict']}" + (f"  (MST levels {' / '.join(map(str, info['levels']))})" if info["levels"] else ""))
        lines.append(f"    {'patch':<12} {'used/total':>11} {'L*':>5} {'chroma':>7} {'hue':>5} {'zone':>5}  MST (dE76)")
        for n in NAMES:
            r = info["parts"].get(n)
            if r is None:
                continue
            if r["median_lab"] is None:
                lines.append(f"    {n.replace('_', ' '):<12} {r['used']:>5}/{r['pixels']:<5} {'no usable pixels':>35}")
                continue
            lines.append(
                f"    {n.replace('_', ' '):<12} {r['used']:>5}/{r['pixels']:<5} {r['median_lab'][0]:>5.1f} {r['chroma']:>7.1f} "
                f"{r['hue']:>5.0f} {r['median_zone']:>5.1f}  MST {r['nearest_mst']} ({r['nearest_dE']:.1f})"
            )
        extra = []
        if info["de_forehead_cheeks"] is not None:
            extra.append(f"forehead vs cheeks dE2000 {info['de_forehead_cheeks']:.1f}, dL* {info['dl_forehead_cheeks']:+.1f}")
        if info["de_left_right"] is not None:
            extra.append(f"left vs right cheek dE2000 {info['de_left_right']:.1f}")
        if extra:
            lines.append("    " + "; ".join(extra))
        for note in info["notes"]:
            lines.append(f"    note: {note}")
    return "\n".join(lines)


def comparison_image(original, faces, summaries, scale=1):
    """Picture comparing every face's patches: the actual pixels, the measured colour and the matched MST swatch.

    `original` is the uint8 RGB picture (working resolution) and `scale` its downsample factor
    relative to the original image, as in faces.draw_faces.
    """
    from PIL import Image, ImageDraw, ImageFont

    def font(size):
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # older Pillow
            return ImageFont.load_default()

    f_big, f_small = font(16), font(12)
    crop, blk, pad = 96, 44, 12
    cell_w = crop + blk + 2 * pad
    row_h = crop + 62
    width = 3 * cell_w + pad
    height = max(1, len(faces)) * (row_h + 34) + pad
    im = Image.new("RGB", (width, height), (28, 28, 28))
    dr = ImageDraw.Draw(im)
    pic = np.asarray(original)
    H, W = pic.shape[:2]
    for i, face in enumerate(faces, 1):
        info = summaries.get(i)
        y0 = pad + (i - 1) * (row_h + 34)
        head = f"Face {i}" + (f": patches {info['verdict']}" if info else "")
        dr.text((pad, y0), head, fill=(255, 255, 255), font=f_big)
        for j, name in enumerate(NAMES):
            x0, y = pad + j * cell_w, y0 + 26
            dr.text((x0, y), name.replace("_", " "), fill=(200, 200, 200), font=f_small)
            row = info["parts"].get(name) if info else None
            if row is not None and row.get("bbox"):  # the area that was really measured, in picture pixels
                xa, ya, xb, yb = row["bbox"]
                xb, yb = min(xb, W), min(yb, H)
            else:
                poly = np.asarray(face.patches[name]) / scale
                xa, ya = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
                xb, yb = np.minimum(np.ceil(poly.max(axis=0)).astype(int), [W, H])
            if xb > xa and yb > ya:
                patch = Image.fromarray(pic[ya:yb, xa:xb]).resize((crop, crop), Image.NEAREST)
                im.paste(patch, (x0, y + 16))
            else:
                dr.rectangle([x0, y + 16, x0 + crop, y + 16 + crop], outline=(120, 120, 120))
            if row is not None and row["median_lab"] is not None:
                lab = row["median_lab"]
                rgb = np.clip(_lab_to_srgb8(lab), 0, 255)
                dr.rectangle([x0 + crop + pad, y + 16, x0 + crop + pad + blk, y + 16 + blk], fill=tuple(int(v) for v in rgb))
                sw = skintone.MST_RGB[row["nearest_mst"] - 1]
                dr.rectangle([x0 + crop + pad, y + 16 + blk + 6, x0 + crop + pad + blk, y + 16 + 2 * blk + 6], fill=tuple(int(v) for v in sw))
                dr.text((x0, y + 16 + crop + 4), f"MST {row['nearest_mst']}  dE {row['nearest_dE']:.1f}", fill=(255, 255, 255), font=f_small)
                dr.text((x0, y + 16 + crop + 20), f"L* {lab[0]:.0f}  zone {row['median_zone']:.1f}", fill=(200, 200, 200), font=f_small)
            else:
                dr.text((x0, y + 16 + crop + 4), "no usable pixels", fill=(255, 140, 140), font=f_small)
    return np.asarray(im)


def _lab_to_srgb8(lab):
    """CIELAB (D65) to 8-bit sRGB, for drawing the measured colour block."""
    L, a, b = (float(v) for v in lab)
    fy = (L + 16) / 116
    fx, fz = fy + a / 500, fy - b / 200
    eps, kappa = 216 / 24389, 24389 / 27
    f3 = lambda t: t**3 if t**3 > eps else (116 * t - 16) / kappa
    xyz = np.array([f3(fx), (L / kappa if L <= 8 else fy**3), f3(fz)]) * skintone.D65_WHITE
    lin = np.linalg.solve(skintone._M_SRGB_TO_XYZ, xyz)
    return skintone.linear_to_srgb(lin) * 255 + 0.5
