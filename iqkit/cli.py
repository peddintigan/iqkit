"""Command line interface.

  python -m iqkit.cli table   --curve rec709 [--gray-ire 45]
  python -m iqkit.cli table   --csv steps.csv [--max-value 255]
  python -m iqkit.cli zonemap photo.png --curve srgb [--gray-ire 46] --out zones.png
  python -m iqkit.cli mst-table --curve rec709 [--gray-ire 45]
  python -m iqkit.cli mst-roi photo.png --roi 120,80,60,60 [--expect 6] [--curve srgb]
  python -m iqkit.cli mst-scan photo.png --curve srgb [--zones 4-7] [--wb x,y,w,h] [--mask m.png] --out overlay.png
  python -m iqkit.cli mst-scan photo.png --curve srgb --faces [--face-method auto|haar|yunet --face-model m.onnx] --out overlay.png

steps.csv has a header `stops,value`: patch exposure in stops relative to the 18% gray patch,
and the measured mean encoded value for that patch.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from . import faces as facemod
from . import scan as scanmod
from . import skintone, tone


def _load_curve(args):
    if args.csv:
        stops, vals = [], []
        with open(args.csv, newline="") as f:
            for row in csv.DictReader(f):
                stops.append(float(row["stops"]))
                vals.append(float(row["value"]) / args.max_value)
        return tone.recover_curve(stops, vals, name=f"recovered:{args.csv}")
    return tone.ToneCurve.standard(args.curve or "rec709")


def _gray(args):
    if args.gray_ire is not None:
        return args.gray_ire / 100.0
    return args.gray


def _print_table(curve, gray):
    print(f"curve: {curve.name}   clip at {curve.clip_stops} stops, floor at {curve.floor_stops} stops")
    print(f"{'zone':>4} {'stops':>6} {'lin':>8} {'IRE':>7} {'8b legal':>9} {'10b legal':>10}  status")
    for r in tone.zone_table(curve, gray):
        print(
            f"{r['zone']:>4} {r['stops_from_gray']:>+6d} {r['linear_rel']:>8.4f} {r['ire']:>7.1f} "
            f"{r['code8_legal']:>9.0f} {r['code10_legal']:>10.0f}  {r['status']}"
        )


def _print_mst_table(curve, gray):
    print(f"curve: {curve.name}   (MST level placement, nominal exposure unless --gray is given)")
    print(f"{'MST':>3} {'L*':>6} {'stops':>6} {'zone':>5} {'IRE':>6} {'EV to V':>8}  status")
    for r in skintone.mst_placement(curve, gray):
        print(
            f"{r['level']:>3} {r['L']:>6.1f} {r['stops_from_gray']:>+6.2f} {r['zone']:>5.2f} "
            f"{r['ire']:>6.1f} {r['ev_to_middle_gray']:>+8.2f}  {r['status']}"
        )
    print("EV to V = exposure change that would put that level at middle gray (zone V).")


def _print_mst_roi(args, curve, gray):
    roi = tuple(int(v) for v in args.roi.split(","))
    if len(roi) != 4:
        raise SystemExit("--roi needs four integers: x,y,w,h")
    img = _load_image(args.image)
    r = skintone.roi_stats(img, roi, curve, gray)
    L, a, b = r["lab"]
    print(f"mean RGB (0..1): {r['rgb'][0]:.3f} {r['rgb'][1]:.3f} {r['rgb'][2]:.3f}")
    print(f"Lab (sRGB, D65): L*={L:.1f} a*={a:.1f} b*={b:.1f}   chroma={r['chroma']:.1f} hue={r['hue_deg']:.0f} deg")
    print(f"nearest MST level: {r['nearest_mst']} (dE76 {r['nearest_dE']:.1f})")
    print(f"luma {100 * r['luma']:.1f} IRE   zone {r['zone']:.2f}   clipped={r['clipped']}   crushed={r['crushed']}")
    if args.expect:
        c = skintone.compare_to_level(r["lab"], args.expect)
        print(
            f"vs MST{args.expect}: dE2000={c['dE00']:.1f}  dL*={c['dL']:+.1f}  dC*={c['dC']:+.1f}  dh={c['dh_deg']:+.0f} deg"
        )


def _print_face_report(res, found):
    """One line per detected face: result or the reason it has none."""
    by_label = {r["label"]: r for r in res["regions"]}
    print("\nper face (patch pixels used / total, and what removed the rest)")
    missing = 0
    for rep in res["face_report"]:
        ex = ", ".join(f"{k} {v}" for k, v in rep["excluded"].items() if v)
        line = f"  face {rep['label']}: {rep['used']}/{rep['patch_pixels']} used"
        if ex:
            line += f"  (removed by: {ex})"
        r = by_label.get(rep["label"])
        if r:
            line += f"  -> MST {r['nearest_mst']}"
        print(line)
        if not r:
            missing += 1
            cause = max(rep["excluded"], key=rep["excluded"].get)
            hint = {
                "hue": f"its patches have median hue {rep['median_hue']:.0f} deg; try --hue {max(0, rep['median_hue'] - 25):.0f},{rep['median_hue'] + 25:.0f}",
                "chroma": f"its patches have median chroma {rep['median_chroma']:.1f}; try --chroma {max(0, rep['median_chroma'] - 8):.0f},{rep['median_chroma'] + 15:.0f}",
                "zone": f"its patches sit near zone {rep['median_zone']:.1f}; widen or drop --zones",
                "exposure": "its patches are clipped or crushed; check exposure",
            }[cause]
            print(f"      no result: {hint}")
    for i in range(len(found) - len(res["face_report"])):
        missing += 1  # a face whose patches fell outside the image or were fully overwritten
    if missing:
        print(f"  {missing} of {len(found)} detected faces have no MST value (see lines above).")


def _pair(text, cast=float):
    parts = [cast(v) for v in text.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"expected two comma-separated numbers, got {text!r}")
    return tuple(parts)


def _zone_range(text):
    if text is None:
        return None
    lo, _, hi = text.partition("-")
    return float(lo), float(hi or lo)


def _run_mst_scan(args, curve, gray):
    img = _load_image(args.image)
    wb = tuple(int(v) for v in args.wb.split(",")) if args.wb else None
    mask = None
    if args.mask:
        from PIL import Image

        mask = np.asarray(Image.open(args.mask).convert("L")) > 127
    found, labels = [], None
    parts = {"all": facemod.PARTS, "cheeks": ("left_cheek", "right_cheek"), "forehead": ("forehead",)}[args.face_parts]
    if args.faces:
        if mask is not None:
            raise SystemExit("use either --faces or --mask, not both")
        try:
            method, _ = facemod.resolve_method(args.face_method, args.face_model)
            found = facemod.detect_faces(img, args.face_method, args.face_model, args.min_face)
        except (ValueError, RuntimeError, ImportError) as exc:
            raise SystemExit(str(exc))
        print(f"faces found: {len(found)} (detector: {method})")
        if method == "haar":
            print("note: the Haar detector is weak (turned, tilted, shadowed or partly hidden faces are often missed). "
                  "Check the overlay for missing faces; for better results install the YuNet model: " + facemod.MODEL_URL)
        if not found:
            print("no faces detected; try --min-face 0.02, a clearer photo, the YuNet detector, or use --mask")
            return
        labels = facemod.face_labels(img.shape, found, parts)
    res = scanmod.scan(
        img,
        curve,
        gray,
        zones=_zone_range(args.zones),
        hue=_pair(args.hue or ("10,100" if args.faces else "35,95")),
        chroma=_pair(args.chroma or ("2,60" if args.faces else "3,45")),
        wb_roi=wb,
        mask=mask,
        labels=labels,
        max_pixels=args.max_pixels,
        n_regions=args.regions,
    )
    sh, sw = res["work_shape"]
    print(f"analysed at {sw}x{sh} (downsample x{res['scale']}); skin pixels: {res['selected']} "
          f"({100 * res['selected_fraction']:.1f}% of image)")
    if not res["selected"]:
        print("no pixels matched; relax --hue/--chroma/--zones or supply --mask")
        if args.faces:
            _print_face_report(res, found)
        return

    print("\nnearest MST level, all selected pixels")
    for lvl, share in enumerate(res["level_share"], 1):
        if share >= 0.005:
            print(f"  MST {lvl:>2}  {100 * share:5.1f}%  " + "#" * int(round(share * 40)))

    print("\nby zone (share of selected pixels, dominant MST level)")
    zc = res["zone_level_counts"]
    for z in range(tone.N_ZONES):
        n = zc[z].sum()
        if n:
            print(f"  zone {z:>2}  {100 * n / res['selected']:5.1f}%  dominant MST {int(zc[z].argmax()) + 1}")

    print("\nregions (largest first); median colour of each region, matched to its nearest level")
    for i, r in enumerate(res["regions"], 1):
        L, a, b = r["median_lab"]
        tops = ", ".join(f"MST {lv} {100 * sh_:.0f}%" for lv, sh_ in r["top_levels"])
        who = f"face {r['label']}" if args.faces else f"#{i}"
        print(
            f"  {who}: bbox x,y,w,h={r['bbox']}  median zone {r['median_zone']:.1f}  "
            f"Lab {L:.0f},{a:.0f},{b:.0f}  -> MST {r['nearest_mst']} (dE76 {r['nearest_dE']:.1f});  pixels: {tops}"
        )
    if args.faces:
        _print_face_report(res, found)
    print("\nCheck the overlay: colour gating is a heuristic and can select non-skin or miss shaded skin.")
    if args.out:
        from PIL import Image

        style = args.overlay if args.overlay != "auto" else ("original" if args.faces else "paint")
        pic = res["original_small"] if style == "original" else res["overlay"]
        if found:
            levels = {r["label"]: r["nearest_mst"] for r in res["regions"]}
            pic = facemod.draw_faces(pic, found, res["scale"], parts, levels)
        if not args.no_legend:
            pic = scanmod.add_legend(pic, res["level_share"])
        Image.fromarray(pic).save(args.out)
        print(f"wrote {args.out}")


def _load_image(path):
    from PIL import Image

    im = Image.open(path)
    arr = np.asarray(im)
    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    return arr.astype(float) / scale


def main(argv=None):
    p = argparse.ArgumentParser(prog="iqkit")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("table", "zonemap", "mst-table", "mst-roi", "mst-scan"):
        sp = sub.add_parser(name)
        if name in ("zonemap", "mst-roi", "mst-scan"):
            sp.add_argument("image")
        if name == "mst-scan":
            sp.add_argument("--zones", help="zone range to analyse, e.g. 4-7 or 5 (default: all but clipped/crushed)")
            sp.add_argument("--hue", help="skin hue gate in degrees, min,max (default 35,95; with --faces 10,100)")
            sp.add_argument("--chroma", help="skin chroma gate, min,max (default 3,45; with --faces 2,60)")
            sp.add_argument("--wb", help="gray-card region x,y,w,h used to white-balance the image first")
            sp.add_argument("--mask", help="image file where white = skin; replaces the colour gate")
            sp.add_argument("--regions", type=int, default=5, help="how many skin regions to report")
            sp.add_argument("--max-pixels", type=int, default=1_000_000, help="downsample above this size")
            sp.add_argument("--out", help="write an overlay PNG (skin painted in MST swatch colours)")
            sp.add_argument("--no-legend", action="store_true", help="leave the MST orb panel off the overlay")
            sp.add_argument("--faces", action="store_true", help="find faces automatically and measure skin patches inside them")
            sp.add_argument("--face-method", choices=("auto", "haar", "yunet"), default="auto",
                            help="auto = YuNet if its model file is found, otherwise Haar (weaker)")
            sp.add_argument("--face-model", help="path to the YuNet .onnx file (or set IQKIT_YUNET_MODEL, or put it in ./models/)")
            sp.add_argument("--overlay", choices=("auto", "original", "paint"), default="auto",
                            help="original = your photo in its own colours; paint = skin repainted in MST swatch colours "
                                 "(auto: original with --faces, paint otherwise)")
            sp.add_argument("--min-face", type=float, default=0.04, help="smallest face as a fraction of the shorter image side")
            sp.add_argument("--face-parts", choices=("all", "cheeks", "forehead"), default="all",
                            help="which patches to measure inside each face (the forehead patch often catches hair and brows)")
        if name == "zonemap":
            sp.add_argument("--out", help="write false-colour PNG here")
        if name == "mst-roi":
            sp.add_argument("--roi", required=True, help="x,y,w,h in pixels (e.g. a cheek or forehead patch)")
            sp.add_argument("--expect", type=int, choices=range(1, 11), help="MST level the region should match")
        sp.add_argument("--curve", choices=sorted(tone.STANDARD_CURVES))
        sp.add_argument("--csv", help="step-chart measurements to recover the curve from")
        sp.add_argument("--max-value", type=float, default=1.0, help="value of white in the CSV (e.g. 255)")
        sp.add_argument("--gray", type=float, help="measured level of the 18%% gray card, 0..1")
        sp.add_argument("--gray-ire", type=float, help="measured level of the 18%% gray card in IRE")
    args = p.parse_args(argv)

    curve = _load_curve(args)
    gray = _gray(args)
    if args.cmd == "table":
        _print_table(curve, gray)
        return
    if args.cmd == "mst-table":
        _print_mst_table(curve, gray)
        return
    if args.cmd == "mst-roi":
        _print_mst_roi(args, curve, gray)
        return
    if args.cmd == "mst-scan":
        _run_mst_scan(args, curve, gray)
        return

    img = _load_image(args.image)
    res = tone.zone_map(tone.luma(img), curve, gray)
    hist = tone.zone_histogram(res.zone_int)
    print("zone  share")
    for z, h in enumerate(hist):
        print(f"{z:>4}  {100 * h:5.1f}%  " + "#" * int(round(h * 50)))
    print(f"clipped: {100 * res.clipped.mean():.1f}%   crushed: {100 * res.crushed.mean():.1f}%")
    if args.out:
        from PIL import Image

        Image.fromarray(tone.false_color(res)).save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
