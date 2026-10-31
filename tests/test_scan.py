import numpy as np

from iqkit import scan, skintone, tone


def _scene():
    """Neutral gray scene (300 x 600) with three 100 x 100 patches in MST 3, 6 and 9 colours."""
    img = np.full((300, 600, 3), 0.45)
    for i, level in enumerate((3, 6, 9)):
        x = 60 + i * 180
        img[100:200, x : x + 100] = skintone.MST_RGB[level - 1] / 255.0
    return img


def test_finds_each_patch_as_its_own_mst_level():
    res = scan.scan(_scene())
    regions = sorted(res["regions"], key=lambda r: r["bbox"][0])
    assert [r["nearest_mst"] for r in regions] == [3, 6, 9]
    assert all(r["nearest_dE"] < 0.5 for r in regions)
    assert abs(res["selected_fraction"] - 3 * 100 * 100 / (300 * 600)) < 0.01  # gray background excluded


def test_zone_selection_picks_only_the_requested_tones():
    # With the sRGB curve the three patches sit near zones 7.2, 5.3 and 2.5.
    res = scan.scan(_scene(), zones=(2, 3))
    assert [r["nearest_mst"] for r in res["regions"]] == [9]
    res = scan.scan(_scene(), zones=(5, 5))
    assert [r["nearest_mst"] for r in res["regions"]] == [6]


def test_gray_card_white_balance_removes_a_colour_cast():
    scene = _scene()
    lin = skintone.srgb_to_linear(scene) * np.array([1.06, 1.0, 0.88])  # mild warm cast, no clipping
    cast = skintone.linear_to_srgb(lin)
    # Known limitation: under warm light the neutral background passes the skin colour gate.
    assert scan.scan(cast)["selected_fraction"] > 0.9
    # A gray-card white balance removes the cast and the three patches come back.
    fixed = scan.scan(cast, wb_roi=(500, 10, 50, 50))
    regions = sorted(fixed["regions"], key=lambda r: r["bbox"][0])
    assert [r["nearest_mst"] for r in regions] == [3, 6, 9]
    assert all(r["nearest_dE"] < 0.5 for r in regions)
    assert fixed["selected_fraction"] < 0.2


def test_user_mask_overrides_the_colour_gate():
    scene = _scene()
    mask = np.zeros(scene.shape[:2], bool)
    mask[10:60, 10:60] = True  # a neutral gray patch the colour gate would reject
    assert scan.scan(scene)["selected"] > 0
    res = scan.scan(scene, mask=mask)
    assert res["selected"] == 50 * 50


def test_shaded_face_spreads_across_zones_and_levels():
    img = np.full((200, 200, 3), 0.45)
    ramp = np.linspace(1.0, 0.08, 160)  # lighting falling off across the face
    face_lin = skintone.srgb_to_linear(skintone.MST_RGB[5] / 255.0)[None, :] * ramp[:, None]
    img[20:180, 20:180] = skintone.linear_to_srgb(face_lin)[None, :, :]
    res = scan.scan(img)
    assert len(res["regions"]) == 1
    assert (res["zone_level_counts"].sum(axis=1) > 0).sum() >= 3  # spans several zones
    assert (res["level_share"] > 0.02).sum() >= 3  # and several MST levels


def test_large_images_are_downsampled():
    big = np.full((2000, 3000, 3), 0.45)
    res = scan.scan(big, max_pixels=500_000)
    assert res["scale"] >= 3 and res["work_shape"][0] * res["work_shape"][1] <= 500_000


def test_legend_adds_ten_reference_orbs_beside_the_overlay():
    res = scan.scan(_scene())
    pic = scan.add_legend(res["overlay"], res["level_share"])
    h, w = res["overlay"].shape[:2]
    assert pic.shape[0] == h and pic.shape[1] > w
    panel = pic[:, w:]
    for rgb in skintone.MST_RGB:  # every reference colour is drawn, at full value inside its orb
        assert (np.all(panel == rgb.astype(np.uint8), axis=-1)).sum() > 50
    assert np.array_equal(pic[:, :w], res["overlay"])  # the original overlay is untouched


def _six_faces_image():
    """Six faces side by side; the last two are small, so a size-ranked cap of 5 would drop one."""
    from iqkit import faces

    img = np.full((200, 700, 3), 0.45)
    boxes = [(10 + i * 110, 40, 100 - 8 * i, 100 - 8 * i) for i in range(6)]
    for i, (x, y, w, h) in enumerate(boxes):
        img[y : y + h, x : x + w] = skintone.MST_RGB[[2, 4, 5, 6, 7, 8][i] - 1] / 255.0
    fs = [faces.make_face(*faces.patches_from_box(b)) for b in boxes]
    return img, faces.face_labels(img.shape, fs), fs


def test_every_labelled_face_is_reported_even_beyond_the_default_region_cap():
    img, labels, fs = _six_faces_image()
    res = scan.scan(img, labels=labels)  # default n_regions is 5
    assert sorted(r["label"] for r in res["regions"]) == [1, 2, 3, 4, 5, 6]
    assert [rep["label"] for rep in res["face_report"]] == [1, 2, 3, 4, 5, 6]
    assert all(rep["used"] > 0 for rep in res["face_report"])


def test_face_report_names_the_filter_that_removed_a_faces_pixels():
    # a warm, dark skin colour whose hue sits below the default gate of 35 degrees
    rgb = np.array([100, 58, 56]) / 255.0
    lab = skintone.rgb_to_lab(rgb)
    assert skintone.chroma_hue(lab)[1] < 30
    from iqkit import faces

    img = np.full((200, 200, 3), 0.45)
    img[20:180, 20:180] = rgb
    labels = faces.face_labels(img.shape, [faces.make_face(*faces.patches_from_box((20, 20, 160, 160)))])
    narrow = scan.scan(img, labels=labels, hue=(35, 95))
    rep = narrow["face_report"][0]
    assert rep["used"] == 0 and rep["excluded"]["hue"] > 0 and narrow["regions"] == []
    assert rep["median_hue"] < 30
    wide = scan.scan(img, labels=labels, hue=(10, 100), chroma=(2, 60))
    assert wide["face_report"][0]["used"] > 0 and len(wide["regions"]) == 1


def test_zone_range_is_reported_as_the_reason_for_a_dark_face():
    from iqkit import faces

    img = np.full((200, 200, 3), 0.45)
    img[20:180, 20:180] = skintone.MST_RGB[7] / 255.0  # MST 8 sits near zone 3
    labels = faces.face_labels(img.shape, [faces.make_face(*faces.patches_from_box((20, 20, 160, 160)))])
    res = scan.scan(img, labels=labels, zones=(5, 6))
    rep = res["face_report"][0]
    assert rep["used"] == 0 and rep["excluded"]["zone"] > 0 and 2.5 < rep["median_zone"] < 4
