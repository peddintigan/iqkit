import numpy as np

from iqkit import skintone as st
from iqkit import tone


def test_srgb_values_reproduce_published_lab():
    # The sRGB triplets are the published Lab values converted with a D65 white point.
    lab = st.rgb_to_lab(st.MST_RGB / 255.0)
    assert np.max(np.abs(lab - st.MST_LAB)) < 0.1


def test_delta_e_2000_matches_sharma_reference_pairs():
    pairs = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ]
    for a, b, expected in pairs:
        assert abs(float(st.delta_e_2000(a, b)) - expected) < 1e-3


def test_each_reference_classifies_as_itself():
    for lvl, lab in zip(st.MST_LEVELS, st.MST_LAB):
        level, de, _ = st.nearest_mst(lab)
        assert level == lvl and de < 1e-9


def test_documented_lightness_non_monotonicity():
    # Cheng (2024): MST3 is lighter than MST2 even though the scale is numbered by darkness.
    assert st.MST_LAB[2, 0] > st.MST_LAB[1, 0]


def test_placement_rec709_nominal_exposure():
    rows = st.mst_placement(tone.ToneCurve.standard("rec709"))
    assert abs(rows[0]["zone"] - 7.25) < 0.05  # lightest level sits high, just under clip
    assert abs(rows[9]["zone"] - 1.71) < 0.05  # darkest level sits low in the shadows
    assert all(r["status"] == "ok" for r in rows)
    assert 5 < rows[5]["zone"] < 6  # MST6 is near middle gray


def test_placement_shifts_with_gray_card_reading():
    curve = tone.ToneCurve.standard("rec709")
    hot = float(curve.signal_at(+1.0))  # card reads one stop hot
    rows = st.mst_placement(curve, gray_value=hot)
    assert rows[0]["status"] == "clipped"  # lightest level now clips
    assert st.mst_placement(curve)[0]["status"] == "ok"


def test_compare_to_level_is_zero_for_the_reference():
    res = st.compare_to_level(st.MST_LAB[5], 6)
    assert res["dE00"] < 1e-9 and abs(res["dL"]) < 1e-9 and abs(res["dh_deg"]) < 1e-9


def test_roi_stats_on_synthetic_image():
    img = np.zeros((40, 40, 3))
    img[10:30, 10:30] = st.MST_RGB[6] / 255.0  # an MST7 coloured square
    out = st.roi_stats(img, (12, 12, 10, 10), curve=tone.ToneCurve.standard("srgb"))
    assert out["nearest_mst"] == 7 and out["nearest_dE"] < 0.1
    assert not out["clipped"] and not out["crushed"]
    empty = st.roi_stats(img, (0, 0, 5, 5))
    assert empty["nearest_mst"] in (9, 10)  # black is closest to the darkest levels
