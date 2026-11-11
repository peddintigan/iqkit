import os
import struct
import tempfile

import numpy as np
from PIL import Image

from iqkit import cli


def _s15(x):
    return struct.pack(">i", int(round(x * 65536)))


def _xyz_tag(x, y, z):
    return b"XYZ \0\0\0\0" + _s15(x) + _s15(y) + _s15(z)


def _text_tag(sig_type, text):
    if sig_type == "desc":
        raw = text.encode("ascii") + b"\0"
        return b"desc\0\0\0\0" + struct.pack(">I", len(raw)) + raw + b"\0" * 4 + b"\0" * 4 + b"\0\0" + b"\0" + b"\0" * 67
    return b"text\0\0\0\0" + text.encode("ascii") + b"\0"


def display_p3_profile():
    """A minimal ICC v2 matrix/TRC profile with Display P3 primaries (D50-adapted) and the sRGB curve."""
    x = np.linspace(0, 1, 1024)
    srgb = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    curve = b"curv\0\0\0\0" + struct.pack(">I", 1024) + b"".join(struct.pack(">H", int(round(v * 65535))) for v in srgb)
    tags = {
        "desc": _text_tag("desc", "Display P3 test profile"),
        "cprt": _text_tag("text", "public domain test profile"),
        "wtpt": _xyz_tag(0.9642, 1.0, 0.8249),
        "rXYZ": _xyz_tag(0.51512, 0.24120, -0.00105),
        "gXYZ": _xyz_tag(0.29198, 0.69225, 0.04189),
        "bXYZ": _xyz_tag(0.15710, 0.06657, 0.78407),
        "rTRC": curve,
        "gTRC": curve,
        "bTRC": curve,
    }
    pad = lambda b: b + b"\0" * (-len(b) % 4)
    table, body, offset = b"", b"", 128 + 4 + 12 * len(tags)
    for sig, data in tags.items():
        data = pad(data)
        table += sig.encode() + struct.pack(">II", offset + len(body), len(data) if sig not in ("rTRC", "gTRC", "bTRC") else len(data))
        body += data
    size = 128 + 4 + len(table) + len(body)
    header = struct.pack(">I4sI4s4s4s", size, b"\0\0\0\0", 0x02100000, b"mntr", b"RGB ", b"XYZ ")
    header += struct.pack(">6H", 2026, 1, 1, 0, 0, 0) + b"acsp" + b"\0" * 4 + b"\0" * 4 + b"\0" * 4 + b"\0" * 4 + b"\0" * 8
    header += struct.pack(">I", 0) + _s15(0.9642) + _s15(1.0) + _s15(0.8249) + b"\0" * 4
    header = header.ljust(128, b"\0")
    return header + struct.pack(">I", len(tags)) + table + body


def _expected_srgb_from_p3(rgb8):
    c = np.asarray(rgb8, float) / 255
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    p3_to_xyz = np.array([[0.4865709, 0.2656677, 0.1982173], [0.2289746, 0.6917385, 0.0792869], [0.0, 0.0451134, 1.0439444]])
    xyz_to_srgb = np.linalg.inv(np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]))
    s = np.clip(xyz_to_srgb @ (p3_to_xyz @ lin), 0, 1)
    enc = np.where(s <= 0.0031308, 12.92 * s, 1.055 * s ** (1 / 2.4) - 0.055)
    return enc * 255


def _save(img, path, **kw):
    Image.fromarray(img).save(path, **kw)


def test_exif_rotation_is_applied():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "phone.jpg")
        exif = Image.Exif()
        exif[274] = 6  # camera was held in portrait: rotate 90 degrees when displaying
        _save(np.full((20, 40, 3), 128, np.uint8), path, exif=exif)
        assert cli._load_image(path).shape[:2] == (40, 20)


def test_a_display_p3_picture_is_converted_to_srgb():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p3.png")
        skin = (200, 120, 100)  # a warm, fairly saturated tone in P3 numbers
        img = np.zeros((8, 8, 3), np.uint8)
        img[:] = skin
        img[0, 0] = (128, 128, 128)  # a neutral gray pixel
        _save(img, path, icc_profile=display_p3_profile())
        out = cli._load_image(path) * 255
        expected = _expected_srgb_from_p3(skin)
        assert np.abs(out[4, 4] - expected).max() < 2.5  # converted, as a colour-managed viewer would
        assert np.abs(out[4, 4] - np.array(skin)).max() > 5  # and not left as raw P3 numbers
        assert np.abs(out[0, 0] - 128).max() < 2  # neutrals stay neutral
        assert out[4, 4, 0] > skin[0]  # the P3 numbers read as more saturated in sRGB; unconverted they look washed out


def test_a_picture_with_no_or_a_broken_profile_is_left_alone():
    with tempfile.TemporaryDirectory() as d:
        plain, broken = os.path.join(d, "plain.png"), os.path.join(d, "broken.png")
        img = np.full((6, 6, 3), (200, 120, 100), np.uint8)
        _save(img, plain)
        _save(img, broken, icc_profile=b"this is not an icc profile")
        for path in (plain, broken):
            assert np.abs(cli._load_image(path) * 255 - np.array([200, 120, 100])).max() < 0.5


def test_saved_pictures_carry_an_srgb_profile_and_reload_unchanged():
    import io

    from PIL import ImageCms

    img = np.full((10, 10, 3), (200, 120, 100), np.uint8)
    with tempfile.TemporaryDirectory() as d:
        for name, tol in (("out.png", 0.5), ("out.jpg", 6)):  # JPEG is lossy, PNG is exact
            path = os.path.join(d, name)
            cli._save_picture(img, path)
            icc = Image.open(path).info.get("icc_profile")
            assert icc, "no profile embedded"
            assert "srgb" in ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).lower()
            assert np.abs(cli._load_image(path) * 255 - np.array([200, 120, 100])).max() < tol  # no extra conversion
