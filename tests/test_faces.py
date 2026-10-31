import os

import numpy as np

from iqkit import faces, scan, skintone


def _skip(msg):
    try:
        import pytest
    except ImportError:  # plain runner without pytest: treat as skipped
        return True
    pytest.skip(msg)


def _upright_landmarks():
    # eyes 60 px apart; nose and mouth below (YuNet order: eye, eye, nose, mouth corner, mouth corner)
    return np.array([[100, 100], [160, 100], [130, 150], [110, 185], [150, 185]], float)


def _rotate(points, centre, degrees):
    a = np.radians(degrees)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (points - centre) @ R.T + centre


def test_box_patches_sit_inside_the_box_away_from_the_middle_band():
    outline, patches = faces.patches_from_box((100, 50, 200, 200))
    for poly in patches.values():
        assert poly[:, 0].min() >= 100 and poly[:, 0].max() <= 300 and poly[:, 1].min() >= 50 and poly[:, 1].max() <= 250
    assert patches["forehead"][:, 1].max() < 50 + 0.30 * 200  # above the eye line
    assert patches["left_cheek"][:, 0].max() < 100 + 0.40 * 200  # left of the nose
    assert patches["right_cheek"][:, 0].min() > 100 + 0.60 * 200  # right of the nose


def test_landmark_patches_for_an_upright_face():
    outline, patches = faces.patches_from_landmarks(_upright_landmarks())
    eye_y = 100
    assert patches["forehead"][:, 1].max() < eye_y  # forehead above the eyes
    assert patches["left_cheek"][:, 1].min() > eye_y + 30  # cheeks well below the eyes
    assert patches["left_cheek"][:, 0].max() < 130 < patches["right_cheek"][:, 0].min()  # either side of the nose
    assert patches["left_cheek"][:, 1].max() < 185  # and above the mouth line


def test_landmark_patches_follow_a_tilted_face_and_ignore_eye_order():
    lm = _upright_landmarks()
    centre = lm[:2].mean(axis=0)
    base = faces.patches_from_landmarks(lm)[1]
    tilted_lm = _rotate(lm, centre, 25)
    tilted = faces.patches_from_landmarks(tilted_lm)[1]
    for name in base:  # every patch is the upright patch rotated by the same angle
        assert np.allclose(tilted[name], _rotate(base[name], centre, 25), atol=1e-6)
    swapped = tilted_lm.copy()
    swapped[[0, 1]] = swapped[[1, 0]]  # detector reported the eyes in the other order
    for name, poly in faces.patches_from_landmarks(swapped)[1].items():
        assert np.allclose(poly, tilted[name], atol=1e-6)


def test_degenerate_landmarks_are_rejected_and_skipped():
    lm = _upright_landmarks()
    lm[1] = lm[0]
    try:
        faces.patches_from_landmarks(lm)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    row = np.concatenate([[90, 60, 90, 120], lm.ravel(), [0.9]])
    assert faces.faces_from_yunet_rows(row[None, :]) == []


def test_yunet_rows_become_faces_with_scaled_coordinates():
    lm = _upright_landmarks()
    row = np.concatenate([[90, 60, 90, 120], lm.ravel() / 2.0, [0.93]])  # detected on a half-size copy
    (f,) = faces.faces_from_yunet_rows(row[None, :], scale=0.5)
    assert abs(f.score - 0.93) < 1e-9 and np.allclose(f.landmarks, lm)
    assert faces.faces_from_yunet_rows(row[None, :], scale=0.5, min_px=1000) == []  # too small to keep
    assert faces.faces_from_yunet_rows(None) == []


def test_merge_keeps_one_of_overlapping_detections_and_both_of_separate_faces():
    mk = lambda box, **kw: faces.make_face(*faces.patches_from_box(box), **kw)
    a, b = mk((100, 100, 100, 100), score=0.9), mk((105, 102, 98, 100), score=0.6)
    c = mk((400, 100, 100, 100), score=0.8)
    kept = faces.merge_faces([a, b, c])
    assert len(kept) == 2 and kept[0].score in (0.9, 0.8)
    assert any(abs(k.score - 0.9) < 1e-9 for k in kept)  # higher score wins the overlap
    assert sorted(k.extra["votes"] for k in kept) == [1, 2]  # the merged pair counts as two votes
    upright, tilted = mk((100, 100, 100, 100), angle=0.0), mk((100, 100, 100, 100), angle=30.0)
    assert faces.merge_faces([tilted, upright])[0].angle == 0.0  # same face: prefer the upright detection


def test_face_labels_number_faces_and_ignore_everything_else():
    mk = lambda box: faces.make_face(*faces.patches_from_box(box))
    fs = [mk((50, 50, 100, 100)), mk((300, 80, 120, 120))]
    labels = faces.face_labels((300, 600, 3), fs)
    assert set(np.unique(labels)) == {0, 1, 2}
    assert labels[0:40, :].sum() == 0
    cheeks = faces.face_labels((300, 600, 3), fs[:1], parts=("left_cheek", "right_cheek"))
    assert 0 < (cheeks > 0).sum() < (labels == 1).sum()


def test_tilted_patch_polygons_are_rasterised_in_place():
    lm = _rotate(_upright_landmarks(), _upright_landmarks()[:2].mean(axis=0), 30) + [50, 40]
    f = faces.make_face(*faces.patches_from_landmarks(lm))
    labels = faces.face_labels((300, 300), [f])
    ys, xs = np.nonzero(labels)
    assert len(ys) > 200
    right_cheek = faces.face_labels((300, 300), [f], parts=("right_cheek",))
    cy, cx = np.nonzero(right_cheek)
    centre = f.patches["right_cheek"].mean(axis=0)
    assert abs(cx.mean() - centre[0]) < 1.5 and abs(cy.mean() - centre[1]) < 1.5


def test_scan_with_labels_reports_one_region_per_face():
    img = np.full((300, 600, 3), 0.45)
    boxes = [(40, 60, 160, 160), (330, 60, 160, 160)]
    for (x, y, w, h), level in zip(boxes, (3, 8)):
        img[y : y + h, x : x + w] = skintone.MST_RGB[level - 1] / 255.0
    fs = [faces.make_face(*faces.patches_from_box(b)) for b in boxes]
    res = scan.scan(img, labels=faces.face_labels(img.shape, fs))
    assert sorted((r["label"], r["nearest_mst"]) for r in res["regions"]) == [(1, 3), (2, 8)]
    assert res["selected"] > 0 and res["selected_fraction"] < 0.15
    assert res["original_small"].shape == img.shape and res["original_small"].dtype == np.uint8


def test_scan_labels_still_apply_the_colour_gate_inside_patches():
    img = np.full((200, 200, 3), 0.45)  # neutral gray everywhere: nothing passes the gate
    f = faces.make_face(*faces.patches_from_box((20, 20, 160, 160)))
    assert scan.scan(img, labels=faces.face_labels(img.shape, [f]))["selected"] == 0


def test_mask_and_labels_together_are_rejected():
    img = np.full((50, 50, 3), 0.4)
    try:
        scan.scan(img, mask=np.ones((50, 50), bool), labels=np.ones((50, 50), int))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_draw_faces_keeps_the_picture_and_adds_chips():
    img = np.full((200, 300, 3), 90, np.uint8)
    f = faces.make_face(*faces.patches_from_box((60, 30, 120, 120)))
    plain = faces.draw_faces(img, [f])
    chipped = faces.draw_faces(img, [f], levels={1: 6})
    assert plain.shape == img.shape and not np.array_equal(plain, img)
    assert np.array_equal(img, np.full((200, 300, 3), 90, np.uint8))  # input untouched
    assert (np.all(chipped == skintone.MST_RGB[5].astype(np.uint8), axis=-1)).sum() > 30  # MST 6 swatch drawn


def test_model_finder_uses_explicit_path_and_environment(tmp=None):
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        model = os.path.join(d, faces.MODEL_NAME)
        open(model, "wb").close()
        assert faces.find_yunet_model(model) == model
        old = os.environ.get("IQKIT_YUNET_MODEL")
        os.environ["IQKIT_YUNET_MODEL"] = model
        try:
            assert faces.find_yunet_model() == model
            assert faces.resolve_method("auto") == ("yunet", model)
        finally:
            if old is None:
                os.environ.pop("IQKIT_YUNET_MODEL")
            else:
                os.environ["IQKIT_YUNET_MODEL"] = old
    assert faces.find_yunet_model(os.path.join("nope", "missing.onnx")) in (None, faces.find_yunet_model())


def test_resolve_method_errors_are_clear():
    for bad in ("nope",):
        try:
            faces.resolve_method(bad)
        except ValueError as exc:
            assert "unknown method" in str(exc)
        else:
            raise AssertionError("expected ValueError")
    if faces.find_yunet_model() is None:
        try:
            faces.resolve_method("yunet")
        except ValueError as exc:
            assert "faces.MODEL_URL" in str(exc) or "github.com/opencv/opencv_zoo" in str(exc)
        else:
            raise AssertionError("expected ValueError when the model file is missing")
        assert faces.resolve_method("auto") == ("haar", None)


def test_missing_haar_gives_a_clear_error_message():
    try:
        import cv2
    except ImportError:
        return _skip("opencv not installed")
    saved = getattr(cv2, "CascadeClassifier", None)
    if saved is not None:
        del cv2.CascadeClassifier  # simulate OpenCV 5 builds
    try:
        try:
            faces.detect_faces(np.full((64, 64, 3), 0.5), method="haar")
        except RuntimeError as exc:
            assert "opencv-python-headless<5" in str(exc) and "YuNet" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
    finally:
        if saved is not None:
            cv2.CascadeClassifier = saved


def test_haar_on_a_flat_picture_finds_nothing():
    try:
        import cv2  # noqa: F401
    except ImportError:
        return _skip("opencv not installed")
    if not faces.haar_available():
        return _skip("this OpenCV build has no Haar detector; install opencv-python-headless<5")
    assert faces.detect_faces(np.full((120, 160, 3), 0.5), method="haar") == []


def _astronaut():
    from skimage import data

    return data.astronaut()


def test_haar_finds_the_face_in_a_public_domain_test_photo():
    try:
        import cv2  # noqa: F401
        _astronaut()
    except ImportError:
        return _skip("needs opencv and scikit-image (for the sample photo)")
    if not faces.haar_available():
        return _skip("this OpenCV build has no Haar detector; install opencv-python-headless<5")
    found = faces.detect_faces(_astronaut(), method="haar")
    assert len(found) == 1
    x, y, w, h = found[0].box
    assert 150 < x + w / 2 < 300 and 40 < y + h / 2 < 170


def test_haar_still_finds_a_tilted_face():
    try:
        import cv2
        img = _astronaut()
    except ImportError:
        return _skip("needs opencv and scikit-image (for the sample photo)")
    if not faces.haar_available():
        return _skip("this OpenCV build has no Haar detector; install opencv-python-headless<5")
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 20, 1.0)  # head tilted by 20 degrees
    tilted = cv2.warpAffine(img, M, (w, h), borderValue=(90, 90, 90))
    found = faces.detect_faces(tilted, method="haar")
    assert len(found) >= 1
    cx, cy = M @ np.array([177 + 47, 66 + 47, 1.0])  # where the original face centre ended up
    assert any(abs(f.box[0] + f.box[2] / 2 - cx) < 40 and abs(f.box[1] + f.box[3] / 2 - cy) < 40 for f in found)


def test_overlapping_patches_go_to_the_nearer_face():
    mk = lambda box: faces.make_face(*faces.patches_from_box(box))
    a, b = mk((0, 0, 100, 100)), mk((40, 0, 100, 100))  # b's left cheek lies over a's right cheek
    labels = faces.face_labels((100, 160), [a, b], parts=("left_cheek", "right_cheek"))
    assert (labels == 1).sum() > 0 and (labels == 2).sum() > 0  # neither face is erased
    contested = faces.face_labels((100, 160), [a], parts=("right_cheek",)) > 0
    contested &= faces.face_labels((100, 160), [b], parts=("left_cheek",)) > 0
    assert contested.sum() > 0  # the two patches really do overlap in this setup
    cols = np.nonzero(contested.any(axis=0))[0]
    mid = cols.mean()  # skip the columns at the midpoint, where the two centres are equally near
    left_half, right_half = cols[cols < mid - 1], cols[cols > mid + 1]
    row = contested.any(axis=1).nonzero()[0][0]
    # contested columns nearer b's left-cheek centre go to face 2, those nearer a's right-cheek centre to face 1
    assert (labels[row, left_half] == 2).all() and (labels[row, right_half] == 1).all()
