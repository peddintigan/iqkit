"""Tone-curve recovery and Zone System mapping.

Conventions
-----------
* Exposure is expressed in *stops relative to 18% middle gray* (0 = middle gray).
* Zone V is middle gray, and each zone is one stop: zone = stops + 5 (zones 0..10).
* "signal" is the encoded value normalised to 0..1 (full range). IRE here is simply
  100 * signal. Legal-range code values are reported separately because waveform
  scales differ between cameras and monitors.
* A ToneCurve maps scene exposure (stops) to encoded signal. Everything else
  (IRE tables, zone maps) is derived from that one object.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIDDLE_GRAY = 0.18
ZONE_V = 5
N_ZONES = 11  # zones 0..X
REC709_LUMA = np.array([0.2126, 0.7152, 0.0722])


# --------------------------------------------------------------------------- #
# Standard transfer functions: scene-linear (0..1+) -> encoded signal
# --------------------------------------------------------------------------- #
def oetf_rec709(lin):
    lin = np.asarray(lin, dtype=float)
    return np.where(lin < 0.018, 4.5 * lin, 1.099 * np.power(np.maximum(lin, 0), 0.45) - 0.099)


def oetf_srgb(lin):
    lin = np.asarray(lin, dtype=float)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(np.maximum(lin, 0), 1 / 2.4) - 0.055)


def oetf_gamma22(lin):
    return np.power(np.maximum(np.asarray(lin, dtype=float), 0), 1 / 2.2)


def oetf_gamma24(lin):
    return np.power(np.maximum(np.asarray(lin, dtype=float), 0), 1 / 2.4)


STANDARD_CURVES = {
    "rec709": oetf_rec709,
    "srgb": oetf_srgb,
    "gamma22": oetf_gamma22,
    "gamma24": oetf_gamma24,
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _isotonic(y):
    """Pool-adjacent-violators: closest non-decreasing sequence to y (equal weights)."""
    vals, wts = [], []
    for v in np.asarray(y, dtype=float):
        vals.append(v)
        wts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    return np.repeat(vals, wts)


def _pchip_resample(x, y, n=512):
    """Shape-preserving dense resample (falls back to the raw points without scipy)."""
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:  # pragma: no cover
        return x, y
    xs = np.linspace(x[0], x[-1], n)
    return xs, PchipInterpolator(x, y)(xs)


# --------------------------------------------------------------------------- #
# Tone curve
# --------------------------------------------------------------------------- #
@dataclass
class ToneCurve:
    stops: np.ndarray  # exposure, stops relative to middle gray, increasing
    signal: np.ndarray  # encoded value 0..1, strictly increasing
    name: str = "curve"
    clip_stops: float | None = None  # exposure at/above which the signal saturates
    floor_stops: float | None = None  # exposure at/below which the signal is lost in black

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_samples(cls, stops, signal, name="curve", clip_stops=None, floor_stops=None):
        stops = np.asarray(stops, dtype=float)
        signal = np.asarray(signal, dtype=float)
        order = np.argsort(stops)
        s, v = stops[order], _isotonic(signal[order])
        xs, ys = _pchip_resample(s, v)
        ys = np.clip(ys, 0.0, 1.0) + 1e-9 * np.arange(len(ys))  # strictly increasing -> invertible
        return cls(xs, ys, name, clip_stops, floor_stops)

    @classmethod
    def from_function(cls, fn, name="curve", lo=-14.0, hi=8.0, n=441):
        stops = np.linspace(lo, hi, n)
        sig = np.clip(fn(MIDDLE_GRAY * 2.0**stops), 0.0, 1.0)
        hit = np.nonzero(sig >= 1.0 - 1e-9)[0]
        clip = None
        if hit.size:
            i = int(hit[0])
            stops, sig, clip = stops[: i + 1], sig[: i + 1], float(stops[i])
        sig = sig + 1e-9 * np.arange(len(sig))
        return cls(stops, sig, name, clip, None)

    @classmethod
    def standard(cls, name):
        if name not in STANDARD_CURVES:
            raise ValueError(f"unknown curve {name!r}; choose from {sorted(STANDARD_CURVES)}")
        return cls.from_function(STANDARD_CURVES[name], name=name)

    # -- evaluation --------------------------------------------------------- #
    def signal_at(self, stops):
        """Encoded signal for an exposure in stops from middle gray (clamped to the curve range)."""
        return np.interp(stops, self.stops, self.signal)

    def stops_at(self, signal):
        """Inverse: exposure in stops from middle gray for an encoded signal."""
        return np.interp(signal, self.signal, self.stops)

    def ire_at(self, stops):
        return 100.0 * self.signal_at(stops)


def recover_curve(stops, values, clip_level=0.985, black_level=0.002, name="recovered"):
    """Recover a tone curve from step-chart patch measurements.

    stops   : known patch exposure in stops relative to the 18% gray patch
              (e.g. a 21-step wedge with 0.5-stop steps -> -5, -4.5, ... , +5 around gray)
    values  : measured mean encoded value per patch, normalised to 0..1
    Patches at/above `clip_level` are treated as clipped: only the lowest one is kept, as the
    curve's saturation point (its exposure is recorded as `clip_stops`), and the rest are
    excluded because they carry no information. Patches at/below `black_level` are collapsed to the
    single highest one (recorded as `floor_stops`). The remaining points are made monotonic
    (isotonic regression) so measurement noise cannot produce a non-invertible curve.
    """
    stops = np.asarray(stops, dtype=float)
    values = np.asarray(values, dtype=float)
    if stops.shape != values.shape or stops.ndim != 1:
        raise ValueError("stops and values must be 1-D arrays of equal length")
    order = np.argsort(stops)
    stops, values = stops[order], values[order]

    clipped = values >= clip_level
    clip_stops = None
    if clipped.any():
        first = int(np.nonzero(clipped)[0].min())  # sorted, so this is the lowest clipped exposure
        clip_stops = float(stops[first])
        clipped[first] = False  # keep it as the curve's saturation anchor; drop the rest
    stops, values = stops[~clipped], values[~clipped]

    black = values <= black_level
    floor_stops = None
    if black.any():
        floor_stops = float(stops[black].max())
        keep = ~black
        keep[np.nonzero(black)[0].max()] = True  # keep the top black patch as the curve's start
        stops, values = stops[keep], values[keep]

    if len(stops) < 4:
        raise ValueError("need at least 4 usable (unclipped, non-black) patches to recover a curve")
    return ToneCurve.from_samples(stops, values, name, clip_stops, floor_stops)


# --------------------------------------------------------------------------- #
# Zone System mapping
# --------------------------------------------------------------------------- #
def exposure_offset(curve, gray_value=None):
    """Exposure shift (stops) implied by where the 18% gray card actually reads.

    gray_value=None assumes nominal exposure (middle gray sits at curve stops = 0).
    Pass the measured card level (0..1, i.e. IRE/100) to account for over/under exposure.
    """
    return 0.0 if gray_value is None else float(curve.stops_at(gray_value))


def zone_table(curve, gray_value=None):
    """IRE / code values for zones 0..X given the camera curve and where gray reads."""
    off = exposure_offset(curve, gray_value)
    rows = []
    for z in range(N_ZONES):
        st = z - ZONE_V + off  # position on the curve's exposure axis
        sig = float(curve.signal_at(st))
        if curve.clip_stops is not None and st >= curve.clip_stops:
            status = "clipped"
        elif curve.floor_stops is not None and st <= curve.floor_stops:
            status = "below floor"
        else:
            status = "ok"
        rows.append(
            {
                "zone": z,
                "stops_from_gray": z - ZONE_V,
                "linear_rel": MIDDLE_GRAY * 2.0 ** (z - ZONE_V),
                "signal": sig,
                "ire": 100.0 * sig,
                "code8_legal": 16 + 219 * sig,
                "code10_legal": 4 * (16 + 219 * sig),
                "status": status,
            }
        )
    return rows


def luma(img, weights=REC709_LUMA):
    """Y' from encoded RGB (0..1). Grayscale input is returned unchanged.

    This is luma (weighted *encoded* values), not scene luminance.
    """
    img = np.asarray(img, dtype=float)
    if img.ndim == 2:
        return img
    return np.tensordot(img[..., :3], weights, axes=([-1], [0]))


@dataclass
class ZoneResult:
    zone_float: np.ndarray  # continuous zone value (5.0 = middle gray)
    zone_int: np.ndarray  # nearest zone, clipped to 0..10
    clipped: np.ndarray  # signal at the curve's saturation point: true zone is >= reported
    crushed: np.ndarray  # signal at the curve's black point: true zone is <= reported


def zone_map(signal, curve, gray_value=None):
    """Map encoded values (0..1) to Zone System zones using the camera's tone curve."""
    signal = np.asarray(signal, dtype=float)
    off = exposure_offset(curve, gray_value)
    zf = curve.stops_at(signal) - off + ZONE_V
    zi = np.clip(np.rint(zf), 0, N_ZONES - 1).astype(np.uint8)
    return ZoneResult(
        zf,
        zi,
        clipped=signal >= curve.signal[-1] - 1e-6 if curve.clip_stops is not None else np.zeros(signal.shape, bool),
        crushed=signal <= curve.signal[0] + 1e-6,
    )


def zone_histogram(zone_int):
    """Fraction of pixels in each zone 0..X."""
    counts = np.bincount(np.asarray(zone_int).ravel(), minlength=N_ZONES).astype(float)
    return counts / max(counts.sum(), 1.0)


ZONE_COLORS = np.array(
    [
        (75, 0, 130),  # 0 purple
        (0, 0, 200),  # I
        (0, 100, 230),  # II
        (0, 170, 200),  # III
        (110, 110, 110),  # IV
        (60, 180, 60),  # V  middle gray (green)
        (170, 170, 170),  # VI
        (230, 220, 60),  # VII
        (250, 150, 30),  # VIII
        (240, 60, 30),  # IX
        (255, 0, 0),  # X
    ],
    dtype=np.uint8,
)


def false_color(result):
    """RGB uint8 false-colour image; clipped pixels are magenta, crushed pixels black."""
    out = ZONE_COLORS[result.zone_int].copy()
    out[result.crushed] = (0, 0, 0)
    out[result.clipped] = (255, 0, 255)
    return out
