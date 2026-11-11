import numpy as np

from iqkit import compare, faces, scan, skintone


def _scene(forehead, left, right, box=(60, 40, 200, 200), size=(300, 320)):
    """A neutral gray picture with one face whose three patches are painted MST colours (None = leave gray)."""
    img = np.full((size[0], size[1], 3), 0.45)
    face = faces.make_face(*faces.patches_from_box(box))
    pl = faces.face_part_labels(img.shape, [face])
    for code, level in ((1, forehead), (2, left), (3, right)):
        if level is not None:
            img[pl == 4 + code] = skintone.MST_RGB[level - 1] / 255.0
    return img, face, pl


def _summary(forehead, left, right, **kw):
    img, face, pl = _scene(forehead, left, right)
    res = scan.scan(img, labels=faces.face_labels(img.shape, [face]), part_labels=pl, **kw)
    return res, compare.summarise_faces(res["part_report"])[1], img, face


def test_part_labels_encode_face_and_patch():
    img, face, pl = _scene(5, 5, 5)
    assert set(np.unique(pl)) == {0, 5, 6, 7}  # face 1: forehead 5, left cheek 6, right cheek 7
    ys, xs = np.nonzero(pl == 6)
    ys2, xs2 = np.nonzero(pl == 7)
    yf, _ = np.nonzero(pl == 5)
    assert xs.mean() < xs2.mean() and yf.mean() < ys.mean()  # left cheek is left of the right cheek, forehead is higher
    assert np.array_equal(np.where(pl > 0, pl // 4, 0), faces.face_labels(img.shape, [face]))


def test_matching_patches_agree():
    res, info, *_ = _summary(6, 6, 6)
    assert info["verdict"] == "agree" and info["levels"] == [6, 6, 6] and info["notes"] == []
    assert info["de_forehead_cheeks"] < 1 and info["de_left_right"] < 1


def test_darker_cheeks_are_flagged_as_possible_facial_hair():
    res, info, *_ = _summary(4, 8, 8)  # light forehead, dark cheeks
    rows = {r["code"]: r for r in res["part_report"]}
    assert rows[1]["nearest_mst"] == 4 and rows[2]["nearest_mst"] == 8 and rows[3]["nearest_mst"] == 8
    assert info["verdict"] == "differ" and info["dl_forehead_cheeks"] > 5 and info["de_forehead_cheeks"] > 10
    assert any("facial hair" in n for n in info["notes"])


def test_darker_forehead_is_flagged_as_possible_hair():
    res, info, *_ = _summary(8, 5, 5)
    assert info["verdict"] == "differ" and info["dl_forehead_cheeks"] < -5
    assert any("fringe" in n or "hair" in n for n in info["notes"])


def test_left_and_right_cheeks_that_differ_point_to_lighting_or_one_sided_hair():
    res, info, *_ = _summary(5, 3, 7)
    assert info["de_left_right"] > 5 and info["verdict"] == "differ"
    assert any("left and right cheeks differ" in n for n in info["notes"])


def test_a_patch_without_usable_pixels_makes_the_comparison_incomplete():
    res, info, *_ = _summary(None, 6, 6)  # forehead stays neutral gray: fails the skin colour gate
    assert info["verdict"] == "incomplete"
    assert any("forehead" in n and "no usable pixels" in n for n in info["notes"])
    assert info["parts"]["forehead"]["median_lab"] is None


def test_part_measurement_does_not_change_the_combined_result():
    img, face, pl = _scene(4, 8, 8)
    labels = faces.face_labels(img.shape, [face])
    with_parts = scan.scan(img, labels=labels, part_labels=pl)
    without = scan.scan(img, labels=labels)
    assert with_parts["selected"] == without["selected"]
    assert [r["nearest_mst"] for r in with_parts["regions"]] == [r["nearest_mst"] for r in without["regions"]]


def test_part_labels_size_must_match_the_image():
    img = np.full((50, 50, 3), 0.4)
    try:
        scan.scan(img, part_labels=np.zeros((40, 40), np.int32))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_text_table_lists_every_patch_and_the_notes():
    res, info, *_ = _summary(4, 8, 8)
    text = compare.format_comparison({1: info})
    for word in ("forehead", "left cheek", "right cheek", "MST 4", "MST 8", "differ", "facial hair"):
        assert word in text


def test_comparison_image_shows_pixels_measured_colour_and_matched_swatch():
    res, info, img, face = _summary(4, 8, 8)
    pic = compare.comparison_image((img * 255).round().astype(np.uint8), [face], {1: info})
    assert pic.ndim == 3 and pic.shape[2] == 3 and pic.shape[0] > 100
    for level in (4, 8):  # the matched swatch colours are drawn
        assert (np.all(pic == skintone.MST_RGB[level - 1].astype(np.uint8), axis=-1)).sum() > 100
    blank = compare.comparison_image((img * 255).round().astype(np.uint8), [face], {})
    assert blank.shape[1] == pic.shape[1]  # a face without measurements still draws


def test_lab_to_srgb_round_trip():
    for rgb in skintone.MST_RGB:
        back = compare._lab_to_srgb8(skintone.rgb_to_lab(rgb / 255.0))
        assert np.max(np.abs(back - rgb)) < 1.5
