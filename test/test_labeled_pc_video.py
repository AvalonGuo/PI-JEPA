"""
Quick sanity check for semantic-label colorized point cloud video export.
Simulates the data flow of gen_demonstration_expert.py (adroit single collector):
  - full_point_cloud_arrays_sub:  list of (N_i, 6) xyz+rgb point clouds
  - full_point_cloud_label_arrays_sub: list of (N_i,) int32 semantic labels
Verifies save_pointcloud_video(..., labels=...) produces a video without error,
and that the recorded frames actually differ from the RGB-based rendering.
"""
import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from visualizer.visualizer.pointcloud import Visualizer, DEFAULT_LABEL_COLORS


def make_frames(n_frames=6, n_pts=200):
    """Create labeled point clouds: few clusters shaped like cubes/spheres."""
    rng = np.random.RandomState(0)
    frames_pc = []
    frames_lbl = []
    for fi in range(n_frames):
        # cluster centers for each semantic label
        centers = {
            0: np.array([0.0, 0.0, 0.0]),
            1: np.array([0.5, 0.0, 0.0]),
            2: np.array([0.0, 0.5, 0.0]),
            3: np.array([0.0, 0.0, 0.5]),
            4: np.array([0.4, 0.4, 0.0]),
        }
        pts = []
        lbls = []
        for lbl, c in centers.items():
            n = n_pts // len(centers)
            offset = np.array([fi * 0.02, 0.0, 0.0])
            pts.append(rng.randn(n, 3) * 0.05 + c + offset)
            lbls.append(np.full(n, lbl, dtype=np.int32))
        pts = np.concatenate(pts, axis=0)
        lbls = np.concatenate(lbls, axis=0)
        # rgb channels (would come from the camera render in the real pipeline)
        rgb = np.tile(np.array([[255, 255, 255]], dtype=np.float32), (len(pts), 1))
        frames_pc.append(np.concatenate([pts, rgb], axis=1).astype(np.float32))
        frames_lbl.append(lbls)
    return frames_pc, frames_lbl


def test_labels_to_colors():
    """Deterministic unit test of the semantic-label -> RGB mapping."""
    vis = Visualizer()
    labels = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    colors = vis._labels_to_colors(labels)  # (8, 3) float in [0, 1]
    assert colors.shape == (8, 3)
    # known labels must match DEFAULT_LABEL_COLORS exactly
    expected = np.array([DEFAULT_LABEL_COLORS[i] for i in range(6)], dtype=np.float64) / 255.0
    np.testing.assert_allclose(colors[:6], expected, atol=1e-9)
    # unknown labels fall back to tab20 (deterministic): idx 6,7 of _TAB20_RGB
    assert tuple(np.round(colors[6] * 255).astype(int)) == (227, 119, 194)
    assert tuple(np.round(colors[7] * 255).astype(int)) == (127, 127, 127)
    # helper handles a python list as well
    colors2 = vis._labels_to_colors([1, 2, 3])
    assert colors2.shape == (3, 3)
    print("test_labels_to_colors OK")


def test_video_colorization():
    """End-to-end: label colorized video differs from rgb video and contains
    the actual per-label colors as dominant hues."""
    import imageio

    out_dir = os.path.join(ROOT, 'test', '_labeled_video_out')
    os.makedirs(out_dir, exist_ok=True)

    # very few frames, few points and large markers -> large solid-color regions
    frames_pc, frames_lbl = make_frames(n_frames=3, n_pts=250)

    rgb_video = os.path.join(out_dir, 'rgb_video.mp4')
    Visualizer().save_pointcloud_video(frames_pc, rgb_video, fps=5, width=320, height=240,
                                       point_size=20)
    lbl_video = os.path.join(out_dir, 'label_video.mp4')
    Visualizer().save_pointcloud_video(frames_pc, lbl_video, fps=5, width=320, height=240,
                                       point_size=20, labels=frames_lbl)

    assert os.path.getsize(rgb_video) > 0 and os.path.getsize(lbl_video) > 0

    rgb_frames = imageio.v3.imread(rgb_video)
    lbl_frames = imageio.v3.imread(lbl_video)
    assert rgb_frames.shape == lbl_frames.shape and rgb_frames.ndim == 4

    # the two videos must differ (label colors override white rgb)
    diff = np.abs(rgb_frames.astype(np.int32) - lbl_frames.astype(np.int32)).mean()
    print(f"mean pixel diff between rgb/label videos: {diff:.3f}")
    assert diff > 5.0, "label video does not differ enough from rgb video"

    # count how many label-palette colors have a solid presence in the frames.
    # account for matplotlib's 0-255 -> 0-1 float render and png encoding:
    # the dominant color of each cluster should be very close to the palette color.
    labels_expected = {
        1: (255, 128, 0),   # hand
        2: (255, 0, 0),     # hammer
        3: (0, 0, 255),     # nail_board
        4: (255, 255, 0),   # nail
    }
    # flatten all frames and build a color histogram
    all_px = lbl_frames.reshape(-1, 3).astype(np.int32)
    unique, counts = np.unique(all_px, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_colors = unique[order][:200]  # most common colors across the video
    top_counts = counts[order][:200]

    # for each expected semantic color, check that a near-pure version exists
    matched = 0
    for lbl, (r, g, b) in labels_expected.items():
        # allow small tolerance for antialiasing / compression
        dist = np.abs(top_colors - np.array([r, g, b])).sum(axis=1)
        best = int(dist.min())
        if best <= 40 and int(top_counts[int(dist.argmin())]) > 0:
            matched += 1
        print(f"label {lbl} color {DEFAULT_LABEL_COLORS[lbl]} min abs dist={best}")
    assert matched >= 3, f"expected at least 3 semantic colors clearly present, matched={matched}"
    print(f"matched {matched} semantic label colors in video")


def main():
    test_labels_to_colors()
    test_video_colorization()
    print("All label-colorization checks passed.")


if __name__ == '__main__':
    main()