"""Automatic skin regions from face detection (OpenCV), to replace a hand-painted mask.

Each detected face becomes a `Face` with an outline and three small skin patches
(forehead, left cheek, right cheek) that avoid eyes, nose, mouth and most hair. scan.py then
filters those patches by zone and colour and reports one MST result per face.

Detectors
---------
yunet  OpenCV's YuNet neural detector (cv2.FaceDetectorYN). Handles several faces, tilt and
       partial occlusion far better than Haar, and returns five landmarks (eyes, nose,
       mouth corners) so the patches follow the face's tilt. It needs a model file (about
       230 KB, MIT licence) that you download once, see MODEL_URL below.
       NOTE: the YuNet path has not been run against the real model in the environment where
       this code was written (no network there); its parsing and geometry are unit-tested
       with synthetic data. Check the first overlay you get.
haar   OpenCV's bundled Haar cascades. No download, but weak: it mostly finds upright,
       frontal, well-lit faces. To help with tilted heads it also searches rotated copies of
       the image. It misses faces that are turned, shadowed or partly hidden, so treat a
       Haar result as "at least these faces", never "all faces".
auto   (default) YuNet if a model file is found, otherwise Haar.

Measure the detector before trusting it: detection rate per lighting condition and per skin
tone is part of the evaluation, not an assumption.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/" + MODEL_NAME
PARTS = ("forehead", "left_cheek", "right_cheek")

# Patches in a box-shaped face frame: fractions of the box (x0, y0, w, h).
BOX_PATCHES = {
    "forehead": (0.30, 0.08, 0.40, 0.16),
    "left_cheek": (0.12, 0.52, 0.24, 0.22),
    "right_cheek": (0.64, 0.52, 0.24, 0.22),
}
# Patches in a landmark face frame: origin at the midpoint of the eyes, x along the eye line,
# y perpendicular (down the face), unit = distance between the eyes (D). (cx, cy, half_w, half_h)
LANDMARK_PATCHES = {
    "forehead": (0.0, -0.75, 0.35, 0.15),
    "left_cheek": (-0.75, 0.80, 0.20, 0.16),
    "right_cheek": (0.75, 0.80, 0.20, 0.16),
}


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("face detection needs OpenCV: python3 -m pip install 'opencv-python-headless<5'") from exc
    return cv2


@dataclass
class Face:
    box: tuple  # (x, y, w, h) bounding box of the outline, original pixels
    outline: np.ndarray  # 4x2 polygon of the face box (rotated if the face is tilted)
    patches: dict  # part name -> 4x2 polygon
    score: float | None = None
    landmarks: np.ndarray | None = None  # 5x2: eye, eye, nose, mouth corner, mouth corner
    angle: float = 0.0  # rotation of the image copy the face was found in (Haar only)
    source: str = ""
    extra: dict = field(default_factory=dict)
    frame: tuple | None = None  # (origin, u, v, D): eye-line midpoint, unit vectors along/down the face, eye distance


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _bbox(poly):
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def patches_from_box(box, transform=None):
    """Outline and patch polygons for an upright face box, optionally mapped by a 2x3 affine."""
    x, y, w, h = box
    outline = _rect(x, y, x + w, y + h)
    patches = {n: _rect(x + fx * w, y + fy * h, x + (fx + fw) * w, y + (fy + fh) * h) for n, (fx, fy, fw, fh) in BOX_PATCHES.items()}
    if transform is not None:
        T = np.asarray(transform, float)
        apply = lambda p: p @ T[:, :2].T + T[:, 2]
        outline = apply(outline)
        patches = {n: apply(p) for n, p in patches.items()}
    return outline, patches


def patches_from_landmarks(lm):
    """Outline and patch polygons from five landmarks, following the face's tilt.

    Landmarks are the two eyes, nose tip and two mouth corners (YuNet's order). Eyes are
    re-ordered by x so left/right always means image-left/image-right.
    """
    lm = np.asarray(lm, float).reshape(5, 2)
    e1, e2 = sorted([lm[0], lm[1]], key=lambda p: p[0])
    centre = (e1 + e2) / 2
    D = float(np.linalg.norm(e2 - e1))
    if D < 1e-6:
        raise ValueError("degenerate landmarks: both eyes at the same point")
    u = (e2 - e1) / D
    v = np.array([-u[1], u[0]])  # for an upright face u=(1,0) and v=(0,1), i.e. down the image

    def poly(cx, cy, hw, hh):
        c = centre + (cx * u + cy * v) * D
        return np.array([c + (-hw * u - hh * v) * D, c + (hw * u - hh * v) * D, c + (hw * u + hh * v) * D, c + (-hw * u + hh * v) * D])

    patches = {n: poly(*spec) for n, spec in LANDMARK_PATCHES.items()}
    outline = poly(0.0, 0.45, 1.15, 1.75)  # generous face extent for drawing, not used for measuring
    return outline, patches


def frame_from_landmarks(lm):
    """Face frame from five landmarks: origin at the midpoint of the eyes, u along the eye line,
    v down the face, D = distance between the eyes. All face geometry below is in units of D."""
    lm = np.asarray(lm, float).reshape(5, 2)
    e1, e2 = sorted([lm[0], lm[1]], key=lambda p: p[0])
    D = float(np.linalg.norm(e2 - e1))
    if D < 1e-6:
        raise ValueError("degenerate landmarks: both eyes at the same point")
    u = (e2 - e1) / D
    return (e1 + e2) / 2, u, np.array([-u[1], u[0]]), D


def frame_from_box(box, transform=None):
    """Approximate face frame for a detector box: eyes about 38% down the box, D = width / 2.2."""
    x, y, w, h = box
    o = np.array([x + w / 2, y + 0.38 * h], float)
    u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    if transform is not None:
        T = np.asarray(transform, float)
        o = T[:, :2] @ o + T[:, 2]
        u, v = T[:, :2] @ u, T[:, :2] @ v
    return o, u, v, w / 2.2


def face_frame(face):
    """The face's frame, derived from its outline polygon if it was not stored."""
    if face.frame is not None:
        return face.frame
    p = np.asarray(face.outline, float)
    wv, hv = p[1] - p[0], p[3] - p[0]
    w, h = np.linalg.norm(wv), np.linalg.norm(hv)
    return p[0] + 0.5 * wv + 0.38 * hv, wv / w, hv / h, w / 2.2


def make_face(outline, patches, **kw):
    return Face(box=_bbox(outline), outline=outline, patches=patches, **kw)


PART_CODES = {"forehead": 1, "left_cheek": 2, "right_cheek": 3}
CODE_PARTS = {v: k for k, v in PART_CODES.items()}


def face_part_labels(shape, faces, parts=PARTS):
    """Integer label image: 0 = background, 4*i + code = patch `code` of face i (largest face = 1).

    Codes: 1 forehead, 2 left cheek, 3 right cheek (see PART_CODES). Where patches overlap, each
    contested pixel goes to the patch whose centre is nearest (relative to patch size), so
    neighbouring faces cannot erase each other.
    """
    from PIL import Image, ImageDraw

    h, w = shape[:2]
    labels = np.zeros((h, w), np.int32)
    best = np.full((h, w), np.inf, np.float32)
    for i, f in enumerate(faces, 1):
        for name in parts:
            poly = np.asarray(f.patches[name], float)
            x0, y0 = np.maximum(np.floor(poly.min(axis=0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(poly.max(axis=0)).astype(int) + 1, [w, h])
            if x1 <= x0 or y1 <= y0:
                continue
            m = Image.new("L", (int(x1 - x0), int(y1 - y0)), 0)
            ImageDraw.Draw(m).polygon([(px - x0, py - y0) for px, py in poly], fill=1)
            ys, xs = np.nonzero(np.asarray(m))
            if not len(ys):
                continue
            ys, xs = ys + y0, xs + x0
            c = poly.mean(axis=0)
            r2 = max(1.0, 0.25 * len(ys))  # (half the patch side)^2
            d = ((xs - c[0]) ** 2 + (ys - c[1]) ** 2) / r2
            win = d < best[ys, xs]
            labels[ys[win], xs[win]] = 4 * i + PART_CODES[name]
            best[ys[win], xs[win]] = d[win]
    return labels


def face_labels(shape, faces, parts=PARTS):
    """Integer label image: 0 = background, i = skin patches of face i (largest face = 1)."""
    pl = face_part_labels(shape, faces, parts)
    return np.where(pl > 0, pl // 4, 0).astype(np.int32)


# --------------------------------------------------------------------------- #
# Skin regions for the colour qualifier (lengths in units of D, the eye distance)
# --------------------------------------------------------------------------- #
SKIN_ELLIPSE = (0.0, 0.40, 0.95, 1.35)  # cx, cy, a, b: forehead to chin, cheek to cheek, slightly shrunk
CORE_ELLIPSE = (0.0, 0.45, 0.55, 0.75)  # where the qualifier takes its seed colour
FOREHEAD_Y = -0.30  # frame y above this is "forehead"
CHEEK_Y, CHEEK_X = 0.35, 0.25  # frame y below CHEEK_Y and |x| above CHEEK_X are the cheeks


def exclusion_ellipses(face):
    """Eyes with brows, mouth and nose tip as (cx, cy, a, b) in face-frame units. Uses the detected
    landmarks when there are any, otherwise typical positions."""
    o, u, v, D = face_frame(face)
    if face.landmarks is not None:
        lm = np.asarray(face.landmarks, float).reshape(5, 2)
        rel = lambda p: np.array([((p - o) @ u) / D, ((p - o) @ v) / D])
        e1, e2, nose = rel(lm[0]), rel(lm[1]), rel(lm[2])
        m1, m2 = rel(lm[3]), rel(lm[4])
        mouth = (m1 + m2) / 2
        mw = float(abs(m1[0] - m2[0]))
        eyes = [e1, e2]
    else:
        eyes = [np.array([-0.5, 0.0]), np.array([0.5, 0.0])]
        nose, mouth, mw = np.array([0.0, 0.95]), np.array([0.0, 1.45]), 0.8
    out = [(e[0], e[1] - 0.12, 0.42, 0.34) for e in eyes]  # eye and brow
    out.append((mouth[0], mouth[1], max(0.55, 0.7 * mw + 0.1), 0.34))  # lips and teeth
    out.append((nose[0], nose[1], 0.28, 0.16))  # nostrils
    return out


def face_region_labels(shape, faces):
    """Label images for the colour qualifier.

    Returns (region, core, parts), all int32 of the picture's shape:
      region  face number i inside that face's skin ellipse, minus eyes, brows, mouth and nostrils
      core    face number i inside the smaller ellipse used to pick the seed colour
      parts   4*i + code (forehead 1, left cheek 2, right cheek 3) for the qualifier's measurement areas
    Overlaps between faces go to the face whose ellipse centre is nearer (relative to its size).
    """
    h, w = shape[:2]
    region = np.zeros((h, w), np.int32)
    core = np.zeros((h, w), np.int32)
    parts = np.zeros((h, w), np.int32)
    best = np.full((h, w), np.inf, np.float32)
    cx, cy, a, b = SKIN_ELLIPSE
    for i, f in enumerate(faces, 1):
        o, u, v, D = face_frame(f)
        reach = D * max(abs(cx) + a, abs(cy) + b) * 1.05
        x0, x1 = int(max(o[0] - reach, 0)), int(min(o[0] + reach + 1, w))
        y0, y1 = int(max(o[1] - reach, 0)), int(min(o[1] + reach + 1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx, dy = xs - o[0], ys - o[1]
        X, Y = (dx * u[0] + dy * u[1]) / D, (dx * v[0] + dy * v[1]) / D
        dist = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2
        keep = dist <= 1.0
        for ex, ey, ea, eb in exclusion_ellipses(f):
            keep &= ((X - ex) / ea) ** 2 + ((Y - ey) / eb) ** 2 > 1.0
        win = keep & (dist < best[y0:y1, x0:x1])
        best[y0:y1, x0:x1][win] = dist[win]
        region[y0:y1, x0:x1][win] = i
        ccx, ccy, ca, cb = CORE_ELLIPSE
        in_core = win & (((X - ccx) / ca) ** 2 + ((Y - ccy) / cb) ** 2 <= 1.0)
        core[y0:y1, x0:x1][in_core] = i
        sub = parts[y0:y1, x0:x1]
        sub[win & (Y < FOREHEAD_Y)] = 4 * i + PART_CODES["forehead"]
        sub[win & (Y > CHEEK_Y) & (X < -CHEEK_X)] = 4 * i + PART_CODES["left_cheek"]
        sub[win & (Y > CHEEK_Y) & (X > CHEEK_X)] = 4 * i + PART_CODES["right_cheek"]
    return region, core, parts


def face_ellipse_mask(shape, faces):
    """Boolean picture-size mask of each face's whole skin ellipse, eyes, mouth and nostrils included."""
    h, w = shape[:2]
    mask = np.zeros((h, w), bool)
    cx, cy, a, b = SKIN_ELLIPSE
    for f in faces:
        o, u, v, D = face_frame(f)
        reach = D * max(abs(cx) + a, abs(cy) + b) * 1.05
        x0, x1 = int(max(o[0] - reach, 0)), int(min(o[0] + reach + 1, w))
        y0, y1 = int(max(o[1] - reach, 0)), int(min(o[1] + reach + 1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx, dy = xs - o[0], ys - o[1]
        X, Y = (dx * u[0] + dy * u[1]) / D, (dx * v[0] + dy * v[1]) / D
        mask[y0:y1, x0:x1] |= ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1.0
    return mask


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def haar_available():
    """True if this OpenCV build has the Haar cascade detector and its bundled cascade files.

    OpenCV 5 moved CascadeClassifier into the optional contrib modules, so the standard
    opencv-python(-headless) 5.x wheels may not have it. OpenCV 4.x does.
    """
    cv2 = _require_cv2()
    return hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data")


HAAR_HELP = (
    "This OpenCV build has no Haar face detector (OpenCV 5 moved it to the optional contrib modules). "
    "Fix: python3 -m pip install 'opencv-python-headless<5'   "
    "or use the neural detector: download the YuNet model and pass --face-model (see faces.MODEL_URL)."
)


def find_yunet_model(model_path=None):
    """Path of the YuNet model file, or None. Looks at the given path, $IQKIT_YUNET_MODEL, then ./models and ."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        model_path,
        os.environ.get("IQKIT_YUNET_MODEL"),
        os.path.join(os.getcwd(), "models", MODEL_NAME),
        os.path.join(os.getcwd(), MODEL_NAME),
        os.path.join(here, "..", "models", MODEL_NAME),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def resolve_method(method="auto", model_path=None):
    """Turn 'auto' into 'yunet' or 'haar', and return (method, model file or None)."""
    model = find_yunet_model(model_path)
    if method == "auto":
        return ("yunet", model) if model else ("haar", None)
    if method == "yunet":
        if not model:
            raise ValueError(
                f"YuNet model file not found. Download {MODEL_URL} (for example into ./models/) "
                "and pass --face-model, or set IQKIT_YUNET_MODEL."
            )
        return "yunet", model
    if method == "haar":
        return "haar", None
    raise ValueError(f"unknown method {method!r}; use 'auto', 'haar' or 'yunet'")


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def merge_faces(faces, iou_threshold=0.25):
    """Greedy non-maximum suppression. Prefers higher score, then the most upright, then the largest.

    Each surviving face gets extra["votes"]: how many input detections (itself included)
    were merged into it. Agreement between detectors and rotations is evidence a face is real.
    """
    order = sorted(faces, key=lambda f: (f.score or 0.0, -abs(f.angle), f.box[2] * f.box[3]), reverse=True)
    kept, votes = [], []
    for f in order:
        for i, k in enumerate(kept):
            if _iou(f.box, k.box) >= iou_threshold:
                votes[i] += 1
                break
        else:
            kept.append(f)
            votes.append(1)
    for k, v in zip(kept, votes):
        k.extra["votes"] = v
    return sorted(kept, key=lambda f: f.box[2] * f.box[3], reverse=True)


def faces_from_yunet_rows(rows, scale=1.0, min_px=0):
    """Convert YuNet's N x 15 array [x, y, w, h, 5 x (x, y), score] into Face objects.

    `scale` maps coordinates from the detection image back to the original (original = value / scale).
    """
    out = []
    if rows is None:
        return out
    for r in np.asarray(rows, float).reshape(-1, 15):
        w, h = r[2] / scale, r[3] / scale
        if min(w, h) < min_px:
            continue
        lm = r[4:14].reshape(5, 2) / scale
        try:
            outline, patches = patches_from_landmarks(lm)
        except ValueError:
            continue
        out.append(make_face(outline, patches, score=float(r[14]), landmarks=lm, source="yunet", frame=frame_from_landmarks(lm)))
    return out


def _to_uint8(img):
    a = np.asarray(img)
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8) if a.dtype.kind == "f" else a.astype(np.uint8)


def _detect_yunet(u8, model, min_size_frac, scales, score_threshold):
    cv2 = _require_cv2()
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("this OpenCV build has no FaceDetectorYN; install 'opencv-python-headless<5'")
    h, w = u8.shape[:2]
    found = []
    for side in scales:  # YuNet sees faces of roughly 10-300 px, so look at more than one size
        s = min(1.0, side / max(h, w))
        small = cv2.resize(u8, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else u8
        sh, sw = small.shape[:2]
        det = cv2.FaceDetectorYN.create(model, "", (sw, sh), score_threshold, 0.3, 5000)
        det.setInputSize((sw, sh))
        _, rows = det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        found += faces_from_yunet_rows(rows, s, int(min_size_frac * min(h, w)))
    return merge_faces(found)


def _detect_haar(u8, min_size_frac, rotations, cascades, scale_factor, min_neighbors, equalize, min_votes):
    cv2 = _require_cv2()
    if not haar_available():
        raise RuntimeError(HAAR_HELP)
    h, w = u8.shape[:2]
    gray0 = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    min_px = max(20, int(min_size_frac * min(h, w)))
    classifiers = [cv2.CascadeClassifier(cv2.data.haarcascades + f"haarcascade_{n}.xml") for n in cascades]
    found = []
    for angle in rotations:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        g = gray0 if angle == 0 else cv2.warpAffine(gray0, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=int(gray0.mean()))
        if equalize:
            g = cv2.equalizeHist(g)  # helps detection only; skin is measured on the original image
        Minv = cv2.invertAffineTransform(M)
        for c in classifiers:
            for box in c.detectMultiScale(g, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_px, min_px)):
                outline, patches = patches_from_box(tuple(int(v) for v in box), None if angle == 0 else Minv)
                found.append(make_face(outline, patches, angle=float(angle), source="haar",
                                       frame=frame_from_box(tuple(int(v) for v in box), None if angle == 0 else Minv)))
    # a real face is usually hit by several cascades/rotations; a single stray hit is often a false positive
    return [f for f in merge_faces(found) if f.extra.get("votes", 1) >= min_votes]


def detect_faces(img, method="auto", model_path=None, min_size_frac=0.04, max_side=1600, equalize=True,
                 rotations=(0, -15, 15, -30, 30), cascades=("frontalface_default", "frontalface_alt2"),
                 scale_factor=1.1, min_neighbors=4, min_votes=2, score_threshold=0.7, yunet_scales=(640, 1280)):
    """Detect faces in an encoded image (float 0..1 or uint8, HxWx3, RGB order).

    Returns a list of Face objects, largest first, in original pixel coordinates.
    Haar only: a face must be hit `min_votes` times (different cascades or rotations) to be kept;
    lower it to 1 for more recall and more false positives.
    Use resolve_method() first if you need to know which detector 'auto' will use.
    """
    method, model = resolve_method(method, model_path)
    _require_cv2()
    u8 = _to_uint8(img)[..., :3]
    h, w = u8.shape[:2]
    if method == "yunet":
        return _detect_yunet(u8, model, min_size_frac, yunet_scales, score_threshold)
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        import cv2

        u8s = cv2.resize(u8, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    else:
        u8s = u8
    faces = _detect_haar(u8s, min_size_frac, rotations, cascades, scale_factor, min_neighbors, equalize, min_votes)
    if s < 1:
        for f in faces:
            f.outline = f.outline / s
            f.patches = {n: p / s for n, p in f.patches.items()}
            f.box = _bbox(f.outline)
            if f.frame is not None:
                o, u, v, D = f.frame
                f.frame = (o / s, u, v, D / s)
    return faces


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_faces(img, faces, scale=1, parts=PARTS, levels=None, show_patches=True):
    """Draw face outlines (white, numbered) and skin patches (cyan) on a copy of `img` (uint8 RGB).

    `scale` is the downsample factor of `img` relative to the original image.
    `levels` maps face number (1-based) to an MST level; if given, a chip with that level's
    swatch colour and the text 'MST n' is drawn under the face.
    """
    from PIL import Image, ImageDraw, ImageFont

    from . import skintone

    im = Image.fromarray(np.asarray(img).copy())
    dr = ImageDraw.Draw(im)
    for i, f in enumerate(faces, 1):
        outline = [tuple(p / scale) for p in f.outline]
        dr.line(outline + [outline[0]], fill=(255, 255, 255), width=2)
        dr.text((outline[0][0] + 3, outline[0][1] + 2), str(i), fill=(255, 255, 255))
        for name in parts if show_patches else ():
            poly = [tuple(p / scale) for p in f.patches[name]]
            dr.line(poly + [poly[0]], fill=(0, 220, 255), width=2)
        if levels and i in levels:
            xs = [p[0] for p in outline]
            ys = [p[1] for p in outline]
            r = int(np.clip(0.07 * (max(ys) - min(ys)), 9, 26))
            cx, cy = (min(xs) + max(xs)) / 2 - r * 2, max(ys) + r + 5
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=tuple(int(v) for v in skintone.MST_RGB[levels[i] - 1]), outline=(255, 255, 255), width=2)
            try:
                font = ImageFont.load_default(size=max(12, int(r * 1.3)))
            except TypeError:  # older Pillow
                font = ImageFont.load_default()
            dr.text((cx + r + 6, cy), f"MST {levels[i]}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0), anchor="lm")
    return np.asarray(im)
