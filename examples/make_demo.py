"""Create a synthetic test scene (no photos of people, no third-party images).

Four flat patches painted with Monk Skin Tone swatch colours on a neutral gray background;
the second patch has lighting falling off from left to right, to show why zones matter.

    python3 examples/make_demo.py
    iqkit mst-scan examples/demo_scene.png --curve srgb --out docs/demo_overlay.png
"""
import os

import numpy as np
from PIL import Image

from iqkit import skintone

HERE = os.path.dirname(os.path.abspath(__file__))
img = np.full((300, 700, 3), 0.40)


def patch(level, x, y, w, h, falloff=None):
    lin = skintone.srgb_to_linear(skintone.MST_RGB[level - 1] / 255.0)
    block = np.tile(lin, (h, w, 1))
    if falloff:
        block = block * np.linspace(1.0, falloff, w)[None, :, None]
    img[y : y + h, x : x + w] = skintone.linear_to_srgb(block)


patch(4, 30, 60, 130, 180)
patch(6, 200, 60, 150, 180, falloff=0.25)  # lit from the left
patch(8, 390, 60, 130, 180)
patch(10, 560, 60, 110, 180)

out = os.path.join(HERE, "demo_scene.png")
Image.fromarray((img * 255).round().astype(np.uint8)).save(out)
print("wrote", out)
