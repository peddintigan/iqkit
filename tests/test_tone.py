import numpy as np

from iqkit import tone


def test_rec709_and_srgb_middle_gray_ire():
    # 18% gray should land near 41 IRE on Rec.709 and near 46 IRE on sRGB.
    r709 = tone.zone_table(tone.ToneCurve.standard("rec709"))[5]
    srgb = tone.zone_table(tone.ToneCurve.standard("srgb"))[5]
    assert abs(r709["ire"] - 40.9) < 0.3
    assert abs(srgb["ire"] - 46.1) < 0.3


def test_curve_roundtrip():
    c = tone.ToneCurve.standard("srgb")
    s = np.linspace(-6, 2, 33)
    assert np.allclose(c.stops_at(c.signal_at(s)), s, atol=0.02)


def test_rec709_clips_about_2p5_stops_over_gray():
    c = tone.ToneCurve.standard("rec709")
    assert 2.4 < c.clip_stops < 2.6
    table = tone.zone_table(c)
    assert table[7]["status"] == "ok" and table[8]["status"] == "clipped"


def test_recover_curve_from_noisy_synthetic_chart():
    rng = np.random.default_rng(0)
    stops = np.arange(-7, 3.01, 0.5)
    true = tone.oetf_srgb(tone.MIDDLE_GRAY * 2.0**stops)
    measured = np.clip(true + rng.normal(0, 0.003, stops.shape), 0, 1)
    rec = tone.recover_curve(stops, measured)
    ref = tone.ToneCurve.standard("srgb")
    probe = np.linspace(-6, 2, 40)
    assert np.max(np.abs(rec.signal_at(probe) - ref.signal_at(probe))) < 0.02
    assert rec.clip_stops is not None and 2.0 <= rec.clip_stops <= 3.0


def test_recover_curve_flags_black_floor():
    stops = np.arange(-8, 2.01, 0.5)
    true = tone.oetf_srgb(tone.MIDDLE_GRAY * 2.0**stops)
    measured = np.clip(true - 0.05, 0, 1)  # black offset swallows the darkest patches
    rec = tone.recover_curve(stops, measured)
    assert rec.floor_stops is not None


def test_zone_map_recovers_zone_centres():
    c = tone.ToneCurve.standard("srgb")
    zones = np.arange(0, 8)  # zone VII = +2 stops, still below sRGB clip
    sig = tone.oetf_srgb(tone.MIDDLE_GRAY * 2.0 ** (zones - 5))
    res = tone.zone_map(sig, c)
    assert np.array_equal(res.zone_int, zones)
    assert not res.clipped.any()


def test_gray_card_reading_shifts_zones():
    c = tone.ToneCurve.standard("srgb")
    over = float(c.signal_at(+1.0))  # card reads one stop hot
    res = tone.zone_map(np.array([over]), c, gray_value=over)
    assert abs(res.zone_float[0] - 5.0) < 0.02
    # ...and the table shows highlights clipping one zone sooner
    nominal = [r["status"] for r in tone.zone_table(c)]
    shifted = [r["status"] for r in tone.zone_table(c, gray_value=over)]
    assert shifted.count("clipped") == nominal.count("clipped") + 1


def test_clipped_pixels_are_flagged():
    c = tone.ToneCurve.standard("rec709")
    res = tone.zone_map(np.array([0.2, 1.0]), c)
    assert list(res.clipped) == [False, True]


def test_luma_and_histogram():
    img = np.ones((4, 4, 3)) * 0.5
    assert np.allclose(tone.luma(img), 0.5)
    h = tone.zone_histogram(np.array([[5, 5], [6, 5]]))
    assert abs(h[5] - 0.75) < 1e-9 and abs(h.sum() - 1.0) < 1e-9
