# Changelog

## 0.2.0

### Added
- **Face detection** (`--faces`): YuNet neural detector (model downloaded separately, see `models/README.md`) with a
  Haar fallback; detections are merged, tilted faces are searched for, and single stray hits are dropped.
- **Skin colour qualifier** (`--skin qualifier`, the default with `--faces`): skin is selected inside each face by
  hue, chroma and lightness range, seeded from the face's own colour, with eyes, brows, mouth and nostrils cut out.
  Manual ranges (`--q-hue`, `--q-chroma`, `--q-luma`) and a mask export (`--mask-out`).
- **Forehead and cheek comparison** (`--compare-out`): the forehead, left cheek and right cheek are measured
  independently, with a heuristic agreement check and notes on possible causes of a mismatch.
- **Per-face report** that says how many pixels each face used and why a face had no result.
- **Colour isolation overlay** (`--overlay isolate`, `--isolate-region`, `--feather`, `--side-by-side`): colour only
  on the selected skin or face, everything else black and white, no marks.
- MST orb legend panel on overlay pictures.
- `examples/make_demo.py` (synthetic scene), `CHANGELOG.md`, GitHub Actions test workflow.

### Changed
- Pictures are read as sRGB: embedded colour profiles (such as Display P3) are converted and the camera's EXIF
  rotation is applied. Saved pictures carry an sRGB profile.
- Every detected face is reported (the earlier cap of five regions is gone).
- Overlapping skin areas of neighbouring faces are shared by nearest centre instead of overwritten.

### Fixed
- Faces with warm, dark skin were dropped by a hue gate that was too narrow; face mode now uses wider defaults.

## 0.1.0
- Tone curves (Rec.709, sRGB, gamma, recovery from step-chart measurements), Zone System tables and zone maps.
- Monk Skin Tone reference data, CIELAB, CIE76 and CIEDE2000, expected placement of each level on a camera curve,
  region measurement.
