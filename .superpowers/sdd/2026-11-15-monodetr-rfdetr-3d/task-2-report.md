# Task 2 report: mono3d_data.py

Built `mono3d_data.py`: `Mono3DDetrDataset`, `collate_mono3d`, `horizontal_flip`, `ProposalCache`. Tests in `tests/test_mono3d_data.py` (6 tests).
No approach-A files modified.

TDD: RED = `ModuleNotFoundError: No module named 'mono3d_data'` (1 error). GREEN = `.venv/bin/python -m unittest tests.test_mono3d_data -v` -> 6 tests OK. Full suite `unittest discover -s tests -q` -> 67 tests OK (61 existing + 6 new).

Details:
- Constructor: `(image_dir, label_dir, calib_dir, feature_size, image_size=384, flip_prob=0.0, split_file=None, bins=12)`. Images with zero valid labels are KEPT (empty targets: labels[0] long, boxes2d[0,4], etc.); collate handles them (tested).
- `mean_dims`: dict class -> [h,w,l] over the split; `classes` sorted, `class_ids`.
- Extra sample/collate keys: `stem`, `flip`, `orig_size` ([2] tensor (W,H) in collate [B,2]).
- Image scaled to [0,1], resized to image_size square. p2 = rescale_p2 of original.

## Concern: flip convention differs from brief wording
Brief said "ry -> -ry / negate alpha". Geometrically, KITTI heading is (cos ry, -sin ry) in x-z; mirroring x gives ry' = pi - ry, and with x' = -x, alpha' = pi - alpha (not -alpha). Using -ry would turn a car facing away into one facing the camera. I implemented pi - alpha, tested against alpha_from_ry(pi-ry, -x, z), and flip-twice identity. Also P2 translation term p2[3] is negated (besides c_x -> W - c_x) so a mirrored 3D point projects to the mirrored pixel (tested). Flip is applied to depth maps by tensor flip (minor floor/ceil asymmetry at box edges).
