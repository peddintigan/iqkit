"""Monk Skin Tone (MST) reference data and skin-tone tests.

Reference data
--------------
The MST scale was developed by Dr. Ellis Monk with Google and is published at
https://skintone.google under CC BY 4.0. The CIELAB and sRGB values below are the ones
tabulated in Cheng (2024), J. Imaging Sci. Technol. 68(5), Table I. The sRGB values are
the CIELAB values converted with a D65 white point.

Things worth knowing before you use it
--------------------------------------
* The scale is defined *digitally*. A screen or an unmanaged print of these sRGB values
  is not a validated reference (Cheng 2024 shows printed copies can be off by 10+ dE).
* It is not strictly monotonic in lightness: MST3 (L*=93.1) is lighter than MST2 (92.3),
  and MST1-3 differ mostly in chroma.
* It describes a bin, not a measurement of a person. Use it to organise a test,
  not to label people.
"""
from __future__ import annotations

import numpy as np

from . import tone

# level, (L*, a*, b*), (R, G, B) 8-bit sRGB
MST_DATA = [
    (1, (94.2, 1.5, 5.4), (246, 237, 228)),
    (2, (92.3, 2.1, 7.3), (243, 231, 219)),
    (3, (93.1, 0.2, 14.2), (247, 234, 208)),
    (4, (87.6, 0.5, 17.7), (234, 218, 186)),
    (5, (77.9, 3.5, 23.1), (215, 189, 150)),
    (6, (55.1, 7.8, 26.7), (160, 126, 86)),
    (7, (42.5, 12.3, 20.5), (130, 92, 67)),
    (8, (30.7, 11.7, 13.3), (96, 65, 52)),
    (9, (21.1, 2.7, 6.0), (58, 49, 42)),
    (10, (14.6, 1.5, 3.5), (41, 36, 32)),
]
MST_LEVELS = np.array([d[0] for d in MST_DATA])
MST_LAB = np.array([d[1] for d in MST_DATA], dtype=float)
MST_RGB = np.array([d[2] for d in MST_DATA], dtype=float)

D65_WHITE = np.array([0.95047, 1.0, 1.08883])
_M_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)


# --------------------------------------------------------------------------- #
# Colour conversions (sRGB encoded 0..1  ->  CIELAB, D65)
# --------------------------------------------------------------------------- #
def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin):
    lin = np.clip(np.asarray(lin, dtype=float), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def rgb_to_lab(rgb):
    """Encoded sRGB (0..1, last axis = RGB) to CIELAB with a D65 white point."""
    xyz = srgb_to_linear(rgb) @ _M_SRGB_TO_XYZ.T / D65_WHITE
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lstar_to_y(lstar):
    """Relative luminance Y (white = 1) from CIE L*."""
    lstar = np.asarray(lstar, dtype=float)
    return np.where(lstar > 8, ((lstar + 16) / 116) ** 3, lstar / (24389 / 27))


def chroma_hue(lab):
    lab = np.asarray(lab, dtype=float)
    return np.hypot(lab[..., 1], lab[..., 2]), np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360


# --------------------------------------------------------------------------- #
# Colour differences
# --------------------------------------------------------------------------- #
def delta_e_ab(lab1, lab2):
    """CIE 1976 colour difference."""
    return np.linalg.norm(np.asarray(lab1, float) - np.asarray(lab2, float), axis=-1)


def delta_e_2000(lab1, lab2):
    """CIEDE2000 (Sharma, Wu, Dalal 2005)."""
    lab1, lab2 = np.asarray(lab1, float), np.asarray(lab2, float)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    Cbar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25.0**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp, dCp = L2 - L1, C2p - C1p
    zero = C1p * C2p == 0
    diff = h2p - h1p
    dhp = np.where(zero, 0.0, np.where(np.abs(diff) <= 180, diff, np.where(diff > 180, diff - 360, diff + 360)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)

    Lbar, Cbarp = (L1 + L2) / 2, (C1p + C2p) / 2
    hsum = h1p + h2p
    hbar = np.where(
        zero,
        hsum,
        np.where(np.abs(h1p - h2p) <= 180, hsum / 2, np.where(hsum < 360, (hsum + 360) / 2, (hsum - 360) / 2)),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hbar - 30))
        + 0.24 * np.cos(np.radians(2 * hbar))
        + 0.32 * np.cos(np.radians(3 * hbar + 6))
        - 0.20 * np.cos(np.radians(4 * hbar - 63))
    )
    dtheta = 30 * np.exp(-(((hbar - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbarp**7 / (Cbarp**7 + 25.0**7))
    Sl = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbarp, 1 + 0.015 * Cbarp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


# --------------------------------------------------------------------------- #
# Classification and comparison
# --------------------------------------------------------------------------- #
def nearest_mst(lab):
    """Closest MST level by CIE76 dE (the 'ideal observer' match of Cheng 2024).

    Returns (level, dE, dE to all ten levels)."""
    d = delta_e_ab(np.asarray(lab, float)[None, :], MST_LAB)
    i = int(np.argmin(d))
    return int(MST_LEVELS[i]), float(d[i]), d


def compare_to_level(lab, level):
    """How a measured colour differs from a given MST level: dE00, dL*, dC*, dh (degrees)."""
    ref = MST_LAB[level - 1]
    lab = np.asarray(lab, float)
    c, h = chroma_hue(lab)
    cr, hr = chroma_hue(ref)
    dh = (h - hr + 180) % 360 - 180
    return {
        "level": level,
        "dE00": float(delta_e_2000(lab, ref)),
        "dL": float(lab[0] - ref[0]),
        "dC": float(c - cr),
        "dh_deg": float(dh),
    }


# --------------------------------------------------------------------------- #
# Where each MST level should land for a given camera tone curve
# --------------------------------------------------------------------------- #
def mst_placement(curve, gray_value=None):
    """Expected exposure placement of each MST level on a camera tone curve.

    Assumes each swatch is a diffuse reflector whose luminance Y (from its L*, white = 1)
    is lit like the 18% gray card, i.e. stops from middle gray = log2(Y / 0.18).
    That is a modelling assumption: real skin has specular and subsurface behaviour.
    `gray_value` (0..1) is where the 18% card actually reads, as in tone.zone_table.
    """
    off = tone.exposure_offset(curve, gray_value)
    rows = []
    for lvl, lab in zip(MST_LEVELS, MST_LAB):
        y = float(lstar_to_y(lab[0]))
        stops = float(np.log2(y / tone.MIDDLE_GRAY))
        st = stops + off
        sig = float(curve.signal_at(st))
        if curve.clip_stops is not None and st >= curve.clip_stops:
            status = "clipped"
        elif curve.floor_stops is not None and st <= curve.floor_stops:
            status = "below floor"
        else:
            status = "ok"
        rows.append(
            {
                "level": int(lvl),
                "L": float(lab[0]),
                "Y": y,
                "stops_from_gray": stops,
                "zone": stops + tone.ZONE_V,
                "signal": sig,
                "ire": 100 * sig,
                "ev_to_middle_gray": -stops,
                "status": status,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Measuring a region of an image
# --------------------------------------------------------------------------- #
def roi_stats(img, roi, curve=None, gray_value=None):
    """Measure a rectangular region (x, y, w, h) of an encoded-sRGB image (0..1).

    The mean is taken over encoded values, and colour is computed assuming sRGB primaries
    and a D65 white point. If `curve` is given, the region's luma is also placed on the
    Zone System scale.
    """
    x, y, w, h = roi
    patch = np.asarray(img, float)[y : y + h, x : x + w, :3]
    if patch.size == 0:
        raise ValueError("ROI is empty or outside the image")
    rgb = patch.reshape(-1, 3).mean(axis=0)
    lab = rgb_to_lab(rgb)
    c, hue = chroma_hue(lab)
    level, de, _ = nearest_mst(lab)
    out = {"rgb": rgb, "lab": lab, "chroma": float(c), "hue_deg": float(hue), "nearest_mst": level, "nearest_dE": de}
    if curve is not None:
        zr = tone.zone_map(np.array([tone.luma(rgb)]), curve, gray_value)
        out.update(
            {
                "luma": float(tone.luma(rgb)),
                "ire": 100 * float(tone.luma(rgb)),
                "zone": float(zr.zone_float[0]),
                "clipped": bool(zr.clipped[0]),
                "crushed": bool(zr.crushed[0]),
            }
        )
    return out
