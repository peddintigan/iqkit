import numpy as np

from iqkit import compare, faces, qualify, scan, skintone

LM = np.array([[120, 120], [180, 120], [150, 155], [127, 190], [173, 190]], float)  # eyes, nose, mouth corners (D = 60)


def _face(lm=LM):
    (f,) = faces.faces_from_yunet_rows(np.concatenate([[100, 60, 100, 130], np.asarray(lm).ravel(), [0.95]])[None, :])
    return f


def _coords(shape, face):
    o, u, v, D = faces.face_frame(face)
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    dx, dy = xs - o[0], ys - o[1]
    return (dx * u[0] + dy * u[1]) / D, (dx * v[0] + dy * v[1]) / D


def _render(beard=True, hair=True, shape=(300, 320)):
    """A synthetic face in MST 6 skin with dark hair on top, white eyes, red lips and an optional dark beard."""
    face = _face()
    X, Y = _coords(shape, face)
    img = np.full((*shape, 3), 0.45)
    skin = (((X / 1.05) ** 2 + ((Y - 0.40) / 1.45) ** 2) <= 1)
    img[skin] = skintone.MST_RGB[5] / 255.0
    if hair:
        img[skin & (Y < -0.55)] = skintone.MST_RGB[9] / 255.0  # a low fringe
    if beard:
        img[skin & (Y > 0.75) & (np.abs(X) > 0.30)] = skintone.MST_RGB[8] / 255.0
    for sx in (-0.5, 0.5):  # eyes
        img[((X - sx) ** 2 + Y**2) <= 0.18**2] = 0.92
    img[(((X / 0.45) ** 2 + ((Y - 1.45) / 0.18) ** 2) <= 1)] = (0.62, 0.18, 0.2)  # lips
    return img, face, skin, X, Y


def _scan(img, face, **qkw):
    region, core, parts = faces.face_region_labels(img.shape, [face])
    qualifier = {"core": core, **qkw}
    return scan.scan(img, labels=region, part_labels=parts, qualifier=qualifier), region, parts


def test_face_frame_from_landmarks_and_from_a_box():
    o, u, v, D = faces.face_frame(_face())
    assert np.allclose(o, [150, 120]) and np.allclose(u, [1, 0]) and np.allclose(v, [0, 1]) and abs(D - 60) < 1e-9
    box_face = faces.make_face(*faces.patches_from_box((100, 50, 110, 140)))
    o2, u2, v2, D2 = faces.face_frame(box_face)
    assert np.allclose(o2, [155, 50 + 0.38 * 140]) and abs(D2 - 50) < 1e-9


def test_region_leaves_out_eyes_mouth_and_nostrils_and_stays_inside_the_face():
    img, face, skin, X, Y = _render()
    region, core, parts = faces.face_region_labels(img.shape, [face])
    assert region[120, 120] == 0 and region[120, 180] == 0  # at the eye landmarks
    assert region[190, 150] == 0  # at the mouth
    assert region[155, 150] == 0  # at the nose tip (nostrils)
    assert region[150, 108] == 1 and region[150, 192] == 1  # cheeks are inside
    assert region[20, 20] == 0  # far from the face
    assert (core[core > 0] == 1).all() and (core > 0).sum() < (region > 0).sum()


def test_parts_split_the_region_into_forehead_and_left_and_right_cheek():
    img, face, *_ = _render()
    region, core, parts = faces.face_region_labels(img.shape, [face])
    assert set(np.unique(parts)) == {0, 5, 6, 7}  # face 1: forehead 5, left cheek 6, right cheek 7
    fy, _ = np.nonzero(parts == 5)
    ly, lx = np.nonzero(parts == 6)
    ry, rx = np.nonzero(parts == 7)
    assert fy.mean() < 120 < ly.mean() and lx.mean() < 150 < rx.mean()


def test_qualifier_keeps_skin_and_drops_hair_beard_and_lips():
    img, face, skin, X, Y = _render()
    res, region, parts = _scan(img, face)
    sel = res["selected_mask"]
    beard = skin & (Y > 0.80) & (np.abs(X) > 0.35) & (region > 0)
    hair = skin & (Y < -0.62) & (region > 0)
    clean_skin = skin & (region > 0) & (Y > -0.45) & (Y < 0.70)
    assert beard.sum() > 200 and hair.sum() > 100
    assert sel[beard].mean() < 0.02 and sel[hair].mean() < 0.02  # dark hair and beard are left out
    assert sel[clean_skin].mean() > 0.95  # the skin itself is kept
    assert [r["nearest_mst"] for r in res["regions"]] == [6]


def test_without_the_qualifier_the_beard_would_darken_the_face_result():
    img, face, *_ = _render()
    region, core, parts = faces.face_region_labels(img.shape, [face])
    plain = scan.scan(img, labels=region, hue=(0, 360), chroma=(0, 100))  # no qualifier: everything in the region counts
    qualified, *_ = _scan(img, face)
    spread = lambda r: r["top_levels"][0][1]
    assert qualified["regions"][0]["nearest_mst"] == 6
    assert spread(qualified["regions"][0]) > spread(plain["regions"][0])  # the qualified selection is more uniform


def test_forehead_and_cheeks_agree_once_the_qualifier_has_removed_the_beard():
    img, face, *_ = _render()
    res, *_ = _scan(img, face)
    info = compare.summarise_faces(res["part_report"])[1]
    assert info["verdict"] == "agree" and info["levels"] == [6, 6, 6]
    assert all(r.get("bbox") for r in res["part_report"])


def test_seed_report_and_manual_ranges():
    img, face, skin, X, Y = _render()
    res, region, parts = _scan(img, face)
    seed = res["qualifier_seeds"][1]
    assert abs(seed["lab"][0] - skintone.MST_LAB[5][0]) < 3 and abs(seed["hue"] - 73.7) < 3
    everything, *_ = _scan(img, face, manual={"hue": (0, 360), "chroma": (0, 120), "luma": (0, 100)}, clean=0)
    assert everything["selected"] > res["selected"]  # wide manual ranges also take the beard and hair
    nothing, *_ = _scan(img, face, manual={"luma": (0, 5)})
    assert nothing["selected"] == 0
    rep = nothing["face_report"][0]
    assert rep["used"] == 0 and rep["excluded"]["qualifier"] > 0 and "hue" not in rep["excluded"]


def test_hue_ranges_can_wrap_through_zero_degrees():
    lab = np.zeros((1, 6, 3))
    lab[..., 0] = 50
    hue = np.array([[350.0, 355, 5, 15, 40, 200]])
    chroma = np.full((1, 6), 20.0)
    region = np.ones((1, 6), int)
    manual = {"hue": (340, 20), "chroma": (0, 100), "luma": (0, 100)}
    mask, _ = qualify.qualify(lab, hue, chroma, region, region, manual=manual, clean=0)
    assert mask.tolist() == [[True, True, True, True, False, False]]
    assert abs(float(qualify.hue_difference(350, 10)) + 20) < 1e-9


def test_seed_colour_ignores_a_little_dark_contamination():
    lab = np.tile([60.0, 12.0, 20.0], (100, 1))
    lab[:15] = [10.0, 0.0, 0.0]  # 15% dark hair pixels
    assert abs(qualify.seed_colour(lab)[0] - 60) < 1e-6


def test_mask_outline_is_one_pixel_wide():
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    edge = qualify.mask_outline(mask)
    assert edge.sum() == 36 and not edge[8:12, 8:12].any()


def test_qualifier_requires_labels():
    try:
        scan.scan(np.full((40, 40, 3), 0.4), qualifier={"core": np.zeros((40, 40), int)})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_overlapping_faces_both_keep_a_region():
    shape = (260, 400)
    a = _face()
    b = _face(LM + [60, 0])  # a second face overlapping the first
    region, core, parts = faces.face_region_labels(shape, [a, b])
    assert {1, 2} <= set(np.unique(region))


def test_isolate_colour_keeps_colour_inside_and_gray_outside():
    rng = np.random.default_rng(0)
    img = rng.uniform(0.1, 0.9, (40, 60, 3))
    mask = np.zeros((40, 60), bool)
    mask[10:30, 20:40] = True
    out = scan.isolate_colour(img, mask)
    assert out.shape == img.shape and out.dtype == np.uint8
    expected = (img * 255 + 0.5).astype(np.uint8)
    assert np.array_equal(out[mask], expected[mask])  # colour untouched inside
    outside = out[~mask].astype(int)
    assert (outside.max(axis=1) - outside.min(axis=1) <= 1).all()  # R = G = B outside (within rounding)
    luma = (img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722) * 255
    assert np.abs(out[..., 1][~mask] - luma[~mask]).max() <= 1  # and as bright as the original luma


def test_isolate_colour_feathers_the_edge_and_accepts_a_smaller_mask():
    img = np.zeros((40, 40, 3))
    img[..., 0] = 0.9  # saturated red
    mask = np.zeros((20, 20), bool)  # half-resolution mask
    mask[5:15, 5:15] = True
    hard = scan.isolate_colour(img, mask, feather=0)
    soft = scan.isolate_colour(img, mask, feather=2)
    sat = lambda a: a.astype(int).max(axis=2) - a.astype(int).min(axis=2)
    assert hard.shape == img.shape == soft.shape
    assert sat(hard)[20, 20] > 200 and sat(hard)[2, 2] == 0  # inside is red, outside is gray
    edge = sat(soft)[20, 8]  # near the old boundary: partly coloured
    assert 5 < edge < sat(soft)[20, 20] and sat(soft)[20, 20] > 150
    assert scan.isolate_colour((img * 255).astype(np.uint8), mask).shape == img.shape  # uint8 input works


def test_face_ellipse_mask_covers_eyes_and_mouth_that_the_region_leaves_out():
    img, face, *_ = _render()
    whole = faces.face_ellipse_mask(img.shape, [face])
    region, core, parts = faces.face_region_labels(img.shape, [face])
    assert whole[120, 120] and whole[190, 150] and not region[120, 120] and not region[190, 150]
    assert whole.sum() > (region > 0).sum() and not whole[20, 20]
