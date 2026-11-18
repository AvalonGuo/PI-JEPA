#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-frame point-cloud PNG export for Adroit expert episodes.

Reference chain (same env / expert ckpt / rollout loop):
    OVERWRITE=1 bash scripts/gen_demonstration_adroit.sh hammer 1 1 42 0 /hy-tmp/3DP/data 256
    -> third_party/VRL3/src/gen_demonstration_expert.py

Three deliberate differences vs that collector:
  1) matplotlib axis limits are shrunk (--fit/--zoom/--center/--half-span/--lim).
     The collector fits ONE cube around the min/max of every frame, so the scene
     objects only fill ~20% x ~20% of the image. Measured on hammer ep0 (14 kept
     frames, share of the 640x480 image covered by the projected point cloud):
         --fit bbox --zoom 1     half_span 0.283  20% x 20%  (collector behaviour)
         --fit pct  --zoom 2.0   half_span 0.107  54% x 51%  <- default, no clipping
         --fit pct  --zoom 2.5   half_span 0.086  68% x 63%  (~2% points clipped)
         --fit pct  --zoom 3.0   half_span 0.072  82% x 75%  (~18% points clipped)
  2) only --extra-steps-after-success frames are kept after success (default 1).
     Kept frame count = success_step + extra_steps (same formula in both sources).
  3) one PNG per frame instead of an mp4.
  4) tick handling is configurable with --ticks; the default is "lines", i.e. the tick
     marks are drawn but the numbers are not.
  5) the pngs are slide-ready: --bg transparent (default) drops the white figure and
     the grey 3D walls, --tight 1 stretches the 3D box over the canvas and --trim 1
     crops the remaining uniform border (--trim-pad px kept), so the point cloud
     fills most of the image instead of ~53% x ~66%

Sources:
  live : run the VRL3 expert in Adroit/MuJoCo (needs a GPU, ~1 min/episode).
  zarr : re-render an existing adroit_<task>_expert.zarr offline (no simulation),
         cutting each episode at meta/success_steps + --extra-steps-after-success.

Pollution branch (--poll 1, default) -- one extra png per frame, same frames:
  The blob is forged with the training-chain helper
  ``jepa.forge.forge_voxel_rows`` inside the task workspace
  (``voxel.workspace.resolve_workspace``, adroit_hammer -> frozen table box,
  grid 0.02, 35x27x17 cells), exactly like ``PI-JEPA/train_pijepa.py``
  (CleanPollDataset: ``sample_512(concat(raw[f], blob))``). The forged rows keep
  the training colour in the tensor and are painted --poll-rgb (pure red) only
  in the png, so the image shows pollution without changing the data semantics.
  --poll-points dense  keep every raw row + the blob (same density as frame_*.png)
  --poll-points 512    exactly the poll-branch tensor (FPS, zero pad dropped)
  --poll-shape virus   re-lay the forged rows out as a virus particle (solid ball
                       core + jittered capsid surface + spike stalks + spike heads)
                       centred on the centroid of the real forged rows (default);
                       the png fades the markers from --poll-rgb in the middle to a
                       pale red at the spike tips (--poll-virus-fade) and shrinks
                       them towards the rim (--poll-virus-core-scale /
                       --poll-virus-edge-scale); the row count is unchanged, so the
                       amount of pollution still equals the training chain
  --poll-shape cloud   the raw scatter forge_voxel_rows produces
  --poll-fit clean     same axis cube AND zoom as frame_KKKK.png (default: the poll
                       png is the clean png plus the red particle, pixel comparable)
  --poll-fit workspace|bbox|self|shared  workspace box | episode min/max |
                       per-frame cube | the clean cube at --poll-zoom

Outputs (under --out):
  adroit_<task>_ep{NNNN}/frame_{KKKK}.png       point cloud, zoomed axes
  adroit_<task>_ep{NNNN}/meta.json              limits / success_step / poll summary
  adroit_<task>_ep{NNNN}/frame_{KKKK}_rgb.png   only with --save-img 1 (live)
  adroit_<task>_ep{NNNN}_poll/                  poll frames only (--poll 1), never mixed
    frame_{KKKK}_poll.png                       same frame + red pollution (virus)
    meta.json                                   poll config / workspace / axis limits / counts
  --poll-out <root> moves the *_poll directory to another root (default: --out)

Image output (for slides):
  --bg white           opaque white figure + the light grey 3D walls (old look)
  --bg transparent     alpha 0 figure, walls removed, grid + ticks kept (default)
  --bg plain           alpha 0 figure, no grid / ticks / walls (bare point cloud)
  --tight 1            the 3D box fills the canvas; --tight-zoom adds zoom (1.08)
  --trim 1             crop the uniform border, --trim-pad px kept (6); the crop box
                       is shared per canvas size so the clean and the poll frames of
                       one episode keep identical pixel dimensions
"""
import argparse
import json
import os
import pathlib
import re
import shutil
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[1]
_VRL3_SRC = _REPO / "third_party" / "VRL3" / "src"
# insert order -> final sys.path head is VRL3/src (as scripts/gen_demonstration_adroit.sh does)
for _p in (str(_REPO), str(_REPO / "3D-Diffusion-Policy"), str(_VRL3_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_OUT = "/hy-tmp/3DP/vis/pc_frames"
DEFAULT_DATA_ROOT = "/hy-tmp/3DP/data"


# --------------------------------------------------------------------------- args
def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("live", "zarr"), default="live")

    # live (mirrors gen_demonstration_adroit.sh arguments)
    p.add_argument("--task", default="hammer")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--pc-resolution", type=int, default=256, choices=(84, 128, 256))
    p.add_argument("--success-goal-threshold", type=float, default=10.0)
    p.add_argument("--extra-steps-after-success", type=int, default=1,
                   help="frames kept after success (1 = the single step the task asks for; "
                        "0 = no truncation, run to the env horizon)")
    p.add_argument("--min-episode-steps", type=int, default=100)
    p.add_argument("--expert-ckpt", default=None,
                   help="default <repo>/third_party/VRL3/ckpts/vrl3_<task>.pt")
    p.add_argument("--no-point-crop", action="store_true",
                   help="drop --use_point_crop (the collector always passes it)")

    # zarr
    p.add_argument("--zarr", default=None,
                   help="adroit_<task>_expert.zarr (default %s/adroit_<task>_expert.zarr)"
                        % DEFAULT_DATA_ROOT)
    p.add_argument("--episodes", default="0", help="comma separated episode indices")
    p.add_argument("--pc-key", default="raw_point_cloud",
                   choices=("raw_point_cloud", "point_cloud"))

    # rendering
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--frames", default="",
                   help="comma separated frame indices to render, e.g. 7 or 2,5,7-9 "
                        "(default: every frame); the png keeps its real index, so "
                        "--frames 7 still writes frame_0007.png")
    p.add_argument("--fit", choices=("pct", "bbox"), default="pct",
                   help="pct = per-axis percentile box (robust to far-away points), "
                        "bbox = full min/max box (the collector's behaviour at --zoom 1)")
    p.add_argument("--pct", type=float, nargs=2, default=(2.0, 98.0))
    p.add_argument("--zoom", type=float, default=2.0,
                   help="axis half-extent divided by this factor (1 = collector behaviour, "
                        "2.0 = default, 2.5 tighter, 3.0 starts clipping the cloud)")
    p.add_argument("--center", default=None, help="x,y,z centre in the plotted frame")
    p.add_argument("--half-span", type=float, default=None,
                   help="explicit axis half-extent; needs --center")
    p.add_argument("--lim", default=None, help="x0,x1,y0,y1,z0,z1 (exact axis limits)")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--point-size", type=float, default=3.0)
    p.add_argument("--elev", type=float, default=30.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--ticks", choices=("off", "lines", "labels", "on"), default="lines",
                   help="lines = tick marks without the numbers (default), "
                        "off = no tick marks and no numbers, "
                        "labels = numbers without the tick marks, "
                        "on = matplotlib default")
    p.add_argument("--max-points", type=int, default=0,
                   help=">0: deterministic subsample of each frame (0 = keep all)")
    p.add_argument("--save-img", type=int, default=0,
                   help="live only: also dump the camera colour render per frame")
    p.add_argument("--bg", choices=("white", "transparent", "plain"), default="transparent",
                   help="slide background: white = opaque white figure with the 3D panes "
                        "(old look), transparent = alpha 0 figure, pane fill removed so "
                        "only grid + ticks + points remain (default, drops onto any slide "
                        "colour), plain = alpha 0 figure without grid/ticks/panes (bare "
                        "point cloud)")
    p.add_argument("--tight", type=int, choices=(0, 1), default=1,
                   help="stretch the 3D box over the whole figure (default 1) instead of "
                        "keeping matplotlib's inner margin")
    p.add_argument("--tight-zoom", type=float, default=1.08,
                   help="extra zoom of the 3D box while --tight 1 (1.0 = no extra zoom, "
                        "values above ~1.15 start clipping the tick marks at the edge)")
    p.add_argument("--trim", type=int, choices=(0, 1), default=1,
                   help="crop the uniform border of the pngs (default 1); frames of the "
                        "same canvas size share one crop box, so an episode stays pixel "
                        "aligned in a slide deck")
    p.add_argument("--trim-pad", type=int, default=6,
                   help="background pixels kept around the content by --trim 1 (default 6)")
    p.add_argument("--overwrite", type=int, default=0,
                   help="replace an existing output episode directory")

    # pollution branch (mirrors the training-chain pollution synthesis)
    p.add_argument("--poll", type=int, default=1,
                   help="1 = also write frame_{KKKK}_poll.png (clean points + red forger "
                        "blob), 0 = clean frames only")
    p.add_argument("--poll-points", choices=("dense", "512"), default="dense",
                   help="dense = every raw row + blob (same density as frame_{KKKK}.png, "
                        "default), 512 = exactly the poll-branch tensor "
                        "sample_512(concat(raw, blob)) with the zero padding dropped")
    p.add_argument("--poll-num-points", type=int, default=512,
                   help="num_points of the 512 branch (= data/point_cloud.shape[1])")
    p.add_argument("--poll-voxel-min", type=int, default=10,
                   help="POLL_VOXEL_MIN of the training synthesis (blob size lower bound)")
    p.add_argument("--poll-voxel-max", type=int, default=50,
                   help="POLL_VOXEL_MAX of the training synthesis (blob size upper bound)")
    p.add_argument("--poll-mode", choices=("bbox", "occupied"), default="bbox",
                   help="jepa.forge.forge_voxel_rows anchor mode (training default: bbox)")
    p.add_argument("--poll-seed", type=int, default=0,
                   help="np.random.default_rng(seed), one blob per frame in frame order")
    p.add_argument("--poll-rgb", default="255,0,0",
                   help="png colour of the pollution rows (default pure red)")
    p.add_argument("--poll-forge-rgb", default=None,
                   help="colour WRITTEN into the forged rows; default = the workspace colour "
                        "(OBSTACLE_RGB), i.e. identical to the training pollution")
    p.add_argument("--poll-point-size", type=float, default=12.0,
                   help="scatter size of the pollution rows (clean rows use --point-size); "
                        "with --poll-shape virus the base size is "
                        "--poll-virus-point-size")
    p.add_argument("--poll-shape", choices=("virus", "cloud"), default="virus",
                   help="virus = re-lay the forged rows out as a virus particle "
                        "(solid ball + spike stalks + spike heads, same row count, "
                        "centred on the real forged centroid, clamped into the "
                        "plotted cube), cloud = the raw scatter of forge_voxel_rows")
    p.add_argument("--poll-virus-radius", type=float, default=0.0,
                   help="virus body radius in metres (0 = auto: %g x the plotted cube "
                        "side, see VIRUS_RADIUS_FRAC)" % VIRUS_RADIUS_FRAC)
    p.add_argument("--poll-virus-spikes", type=int, default=0,
                   help="number of spikes (0 = auto from the row count, 4..10)")
    p.add_argument("--poll-virus-point-size", type=float, default=0.0,
                   help="base scatter size of the virus rows (0 = auto: %g x "
                        "--poll-point-size); the body rows scale by "
                        "--poll-virus-core-scale, the spike heads by "
                        "--poll-virus-edge-scale" % VIRUS_POINT_FACTOR)
    p.add_argument("--poll-virus-fade", type=float, default=0.62,
                   help="how far the rim is blended towards white: 0 = all rows "
                        "--poll-rgb, 1 = the spike heads are white (default 0.62, "
                        "red centre -> pale red spikes)")
    p.add_argument("--poll-virus-core-scale", type=float, default=1.25,
                   help="marker scale of the particle centre (1.0 = base size)")
    p.add_argument("--poll-virus-edge-scale", type=float, default=0.58,
                   help="marker scale at the spike heads (big centre, small spikes)")
    p.add_argument("--poll-clamp", type=int, choices=(0, 1), default=1,
                   help="shift the virus particle so that it stays inside the plotted cube")
    p.add_argument("--poll-on-top", type=int, choices=(0, 1), default=1,
                   help="draw the pollution rows on top of the cloud (zorder) instead of "
                        "letting mplot3d depth-sort them, so they are never hidden")
    p.add_argument("--poll-fit", choices=("clean", "workspace", "bbox", "self", "shared"),
                   default="clean",
                   help="poll axis cube: clean = exactly the cube and zoom of "
                        "frame_KKKK.png (default), workspace = the box the pollution is "
                        "drawn from, bbox = episode min/max of clean+blobs, self = one "
                        "cube per frame (view jumps), shared = the clean cube at "
                        "--poll-zoom")
    p.add_argument("--poll-zoom", type=float, default=None,
                   help="zoom of the poll cube (default = --zoom)")
    p.add_argument("--poll-panel", choices=("single", "pair"), default="single",
                   help="single = one 3d panel, pair = clean cube | poll cube side by side")
    p.add_argument("--poll-out", default=None,
                   help="root of the <ep>_poll directory (default = --out), so the poll frames "
                        "never sit next to the clean ones")
    p.add_argument("--workspace", choices=("auto", "hammer", "explicit"), default="auto",
                   help="voxel.workspace.resolve_workspace mode for the pollution box")
    p.add_argument("--workspace-lo", default=None,
                   help="x,y,z workspace origin (needs --workspace explicit)")
    p.add_argument("--workspace-hi", default=None,
                   help="x,y,z workspace top corner (needs --workspace explicit)")
    p.add_argument("--workspace-grid", type=float, default=0.02,
                   help="voxel size in metres of the pollution grid (training: 0.02)")
    p.add_argument("--fps-device", choices=("cpu", "cuda", "auto"), default="cpu",
                   help="device of the FPS in the 512 branch (cpu matches the training runs; "
                        "CPU and CUDA kernels select different rows)")
    return p


def _floats(text, n, name):
    vals = [float(t) for t in str(text).split(",")]
    if len(vals) != n:
        raise SystemExit("[frames] --%s needs %d comma separated numbers, got %r"
                         % (name, n, text))
    return vals


# ------------------------------------------------------------------- axis limits
def fit_limits(frames_xyz, args):
    """-> (limits, info) with limits = ((x0, x1), (y0, y1), (z0, z1)).

    One cube for every frame of the episode so the view does not jitter.
    Precedence: --lim > (--center + --half-span) > (--fit, --zoom).
    """
    if args.lim:
        v = _floats(args.lim, 6, "lim")
        lim = ((v[0], v[1]), (v[2], v[3]), (v[4], v[5]))
        return lim, {"mode": "explicit_lim", "lim": v}

    if args.center or args.half_span is not None:
        if not args.center or args.half_span is None:
            raise SystemExit("[frames] --center and --half-span must be given together")
        c = np.asarray(_floats(args.center, 3, "center"), dtype=np.float64)
        s = float(args.half_span)
        if s <= 0:
            raise SystemExit("[frames] --half-span must be > 0")
        lim = tuple((float(c[i] - s), float(c[i] + s)) for i in range(3))
        return lim, {"mode": "center_half_span", "center": c.tolist(), "half_span": s}

    allp = np.concatenate(frames_xyz, axis=0)
    if args.fit == "bbox":
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        src = "bbox"
    else:
        lo = np.percentile(allp, args.pct[0], axis=0)
        hi = np.percentile(allp, args.pct[1], axis=0)
        src = "pct%.1f_%.1f" % (args.pct[0], args.pct[1])
    center = (lo + hi) / 2.0
    span = max(float((hi - lo).max()) / 2.0, 1e-6) / max(float(args.zoom), 1e-6)
    lim = tuple((float(center[i] - span), float(center[i] + span)) for i in range(3))
    return lim, {"mode": "auto", "fit": src, "zoom": float(args.zoom),
                 "center": [round(float(v), 6) for v in center],
                 "half_span": round(span, 6),
                 "raw_lo": [round(float(v), 6) for v in lo],
                 "raw_hi": [round(float(v), 6) for v in hi]}


def points_in_limits(frames_xyz, lim):
    """Fraction of points inside the limit box - used for the run log only."""
    lo = np.array([lim[i][0] for i in range(3)])
    hi = np.array([lim[i][1] for i in range(3)])
    n_in = n_all = 0
    for xyz in frames_xyz:
        n_all += len(xyz)
        n_in += int(np.all((xyz >= lo) & (xyz <= hi), axis=1).sum())
    return (n_in / max(n_all, 1)), n_all


# --------------------------------------------------------------------- rendering
def _subsample(pc, max_points, rng):
    if max_points <= 0 or len(pc) <= max_points:
        return pc
    idx = rng.choice(len(pc), size=max_points, replace=False)
    idx.sort()
    return pc[idx]


def _apply_ticks(ax, mode):
    """mode: off = no tick marks and no tick labels, lines = tick marks only (default),
    labels = tick labels only, on = matplotlib default."""
    from matplotlib.ticker import NullFormatter

    if mode == "on":
        return
    if mode == "off":
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        return
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        if mode == "lines":
            # NullFormatter keeps the locator (and therefore the tick marks) but
            # renders empty strings instead of the numbers
            axis.set_major_formatter(NullFormatter())
        else:  # labels only: mplot3d draws the marks from Axis3D._axinfo
            axis.set_tick_params(which="major", length=0, width=0, pad=6)
            info = getattr(axis, "_axinfo", None)
            if info is not None and "tick" in info:
                info["tick"]["inward_factor"] = 0.0
                info["tick"]["outward_factor"] = 0.0
                info["tick"]["linewidth"] = {True: 0.0, False: 0.0}


def render_frames(frames, out_dir, lim, args, ids=None):
    """One PNG per frame via matplotlib Agg (no mp4, no headless browser)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
    from termcolor import cprint

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    paths, n_pts = [], []
    for i in (range(len(frames)) if ids is None else list(ids)):
        pc = _subsample(np.asarray(frames[i], dtype=np.float32), args.max_points, rng)
        n_pts.append(int(len(pc)))
        fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi),
                         dpi=args.dpi)
        ax = fig.add_subplot(111, projection="3d")
        if pc.shape[1] >= 6:
            rgb = np.clip(pc[:, 3:6] / 255.0, 0.0, 1.0)
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=rgb, s=args.point_size)
        else:
            ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=args.point_size)
        ax.set_xlim(lim[0])
        ax.set_ylim(lim[1])
        ax.set_zlim(lim[2])
        ax.view_init(elev=args.elev, azim=args.azim)
        _apply_ticks(ax, args.ticks)
        _style_bg(fig, ax, args)
        _tight_layout(fig, args)
        path = os.path.join(out_dir, "frame_%04d.png" % i)
        _save_fig(fig, path, args)
        plt.close(fig)
        paths.append(path)
    cprint("[frames] %d png -> %s (ticks=%s)" % (len(paths), out_dir, args.ticks), "green")
    return paths, n_pts


def _prepare_output(out_dir, overwrite):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise SystemExit("[frames] %s already exists, pass --overwrite 1 to replace it"
                             % out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def frame_ids(args, n):
    """Frame indices to render: every frame, or the subset given by --frames."""
    spec = (args.frames or "").strip()
    if not spec:
        return list(range(n))
    ids = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        head, _, tail = tok.partition("-")
        try:
            if tail:                         # a range such as 3-6
                ids += list(range(int(head), int(tail) + 1))
            else:
                ids.append(int(head))
        except ValueError:
            raise SystemExit("[frames] --frames: cannot read %r (expected 7, 2,5 or 3-6)"
                             % tok)
    if not ids:
        raise SystemExit("[frames] --frames %r selects no frame" % spec)
    bad = [i for i in ids if i < 0 or i >= n]
    if bad:
        raise SystemExit("[frames] --frames %s out of range (the episode has %d frames)"
                         % (bad, n))
    return sorted(set(ids))


# ------------------------------------------------------------------ png styling
_FRAME_RE = re.compile(r"^frame_\d{4}(_poll)?\.png$")


def _style_bg(fig, ax, args):
    """Slide-ready background: opaque white, transparent figure or bare cloud."""
    if args.bg == "white":
        return
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane = getattr(axis, "pane", None)
        if pane is not None:            # drop the light grey 3D walls
            pane.set_alpha(0.0)
            pane.set_edgecolor((0.0, 0.0, 0.0, 0.0))
    if args.bg == "plain":              # bare point cloud: no grid, no ticks
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


def _tight_layout(fig, args):
    """Stretch the 3D box over the figure so the png has little dead space."""
    if not args.tight:
        return
    if len(fig.axes) == 1:              # single panel: fill the canvas
        fig.axes[0].set_position([0.0, 0.0, 1.0, 1.0])
    else:                               # pair panel: keep the two subplots side by side
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0)
    for ax in fig.axes:
        try:
            ax.set_box_aspect(None, zoom=float(args.tight_zoom))
        except TypeError:               # mpl without the zoom kwarg
            pass


def _save_fig(fig, path, args):
    fig.savefig(path, format="png", dpi=args.dpi,
                facecolor="white" if args.bg == "white" else "none",
                edgecolor="none", transparent=args.bg != "white")


def _content_mask(path):
    """Pixels that carry anything (alpha > 0, or not white for an opaque png)."""
    from PIL import Image
    with Image.open(path) as im:
        a = np.array(im.convert("RGBA"))     # copy: the handle is closed below
    if int(a[..., 3].min()) < 250:           # transparent margins
        return a[..., 3] > 8
    return (a[..., :3] < 250).any(axis=2)


def trim_outputs(dirs, args):
    """Crop the uniform border of every frame png (one box per canvas size).

    The box is the union of all content boxes of the same canvas size, so the
    clean frames and the poll frames keep identical sizes and the episode does not
    jitter when the images are dropped into a slide deck one after another.
    """
    if not args.trim:
        return None
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if _FRAME_RE.match(f)]
    if not files:
        return None
    from PIL import Image
    groups = {}
    for p in files:
        with Image.open(p) as im:
            groups.setdefault(im.size, []).append(p)
    pad = int(max(0, args.trim_pad))
    infos = []
    for (w, h), paths in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        x0 = y0 = 10 ** 9
        x1 = y1 = -1
        for p in paths:
            ys, xs = np.where(_content_mask(p))
            if not len(xs):
                continue
            x0, y0 = min(x0, int(xs.min())), min(y0, int(ys.min()))
            x1, y1 = max(x1, int(xs.max())), max(y1, int(ys.max()))
        if x1 < 0:
            continue
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + 1 + pad), min(h, y1 + 1 + pad)
        for p in paths:
            with Image.open(p) as im:
                im.load()                       # read before touching the file again
                cropped = im.crop((x0, y0, x1, y1))
            cropped.save(p, format="png")
        infos.append({"canvas": [int(w), int(h)], "box": [x0, y0, x1, y1],
                      "size": [int(x1 - x0), int(y1 - y0)], "n": len(paths),
                      "dirs": sorted({os.path.basename(os.path.dirname(p)) for p in paths})})
    if not infos:
        return None
    head = infos[0]
    return {"bg": args.bg, "tight": bool(args.tight), "tight_zoom": float(args.tight_zoom),
            "trim": True, "pad": pad, "box": head["box"], "size": head["size"],
            "groups": infos}


# --------------------------------------------------------------- pollution branch
def _forge_imports():
    """Lazy import of the training-chain pollution helpers (free when --poll 0)."""
    try:
        from diffusion_policy_3d.jepa.forge import (
            forge_voxel_rows, occupied_cells_from_raws, occupied_cells_from_zarr)
        from diffusion_policy_3d.voxel.workspace import resolve_workspace
    except ImportError as e:
        raise SystemExit("[frames] --poll 1 needs 3D-Diffusion-Policy on PYTHONPATH: %r" % (e,))
    return forge_voxel_rows, occupied_cells_from_raws, occupied_cells_from_zarr, resolve_workspace


def _poll_rgb01(args):
    return tuple(v / 255.0 for v in _floats(args.poll_rgb, 3, "poll-rgb"))


def resolve_forge_workspace(args, task=None, zarr_path=None):
    """The workspace box the training pollution is drawn from.

    ``resolve_workspace`` resolution: adroit_hammer hits the frozen table (the same
    box as voxelizer.WORKSPACE_LO/HI), other tasks are derived from their own zarr.
    """
    resolve_workspace = _forge_imports()[3]
    task_key = None
    if args.workspace == "auto" and task:
        task_key = str(task)
        if not task_key.startswith("adroit_"):
            task_key = "adroit_" + task_key
    lo = _floats(args.workspace_lo, 3, "workspace-lo") if args.workspace_lo else None
    hi = _floats(args.workspace_hi, 3, "workspace-hi") if args.workspace_hi else None
    if args.workspace == "explicit" and (lo is None or hi is None):
        raise SystemExit("[frames] --workspace explicit needs --workspace-lo and --workspace-hi")
    try:
        return resolve_workspace(task=task_key, zarr=zarr_path, mode=args.workspace,
                                 lo=lo, hi=hi, grid=float(args.workspace_grid))
    except (ValueError, KeyError, FileNotFoundError) as e:
        raise SystemExit("[frames] cannot resolve the pollution workspace: %s\n"
                         "         pass --workspace explicit --workspace-lo x,y,z "
                         "--workspace-hi x,y,z, or run with --poll 0" % (e,))


def fps_with_idx(pts, num_points=512, device="cpu"):
    """``pc_sample.sample_512`` that also reports the selected row indices.

    Same ops / same pinned device as sample_512, so the rows match bit-for-bit;
    the indices are what lets us flag the forged rows without colour heuristics.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] <= int(num_points):
        return pts, np.arange(pts.shape[0], dtype=np.int64), "pad"
    import pytorch3d.ops
    import torch

    dev = torch.device(device if device != "auto" else
                       ("cuda:0" if torch.cuda.is_available() else "cpu"))
    xyz = torch.from_numpy(pts[:, :3].copy()).float().unsqueeze(0).to(dev)
    k = torch.tensor([int(num_points)], dtype=torch.long, device=dev)
    _, inds = pytorch3d.ops.sample_farthest_points(points=xyz, K=k)
    idx = inds[0].cpu().numpy().astype(np.int64)
    return pts[idx], idx, "fps"


def forge_blobs(frames, ws, occ_cells, args):
    """One blob per frame from a single rng stream (training draws per frame too)."""
    forge_voxel_rows = _forge_imports()[0]
    rgb = None
    if args.poll_forge_rgb:
        rgb = _floats(args.poll_forge_rgb, 3, "poll-forge-rgb")
    rng = np.random.default_rng(int(args.poll_seed))
    return [forge_voxel_rows(rng, p_min=int(args.poll_voxel_min),
                             p_max=int(args.poll_voxel_max), ws=ws, rgb=rgb,
                             mode=args.poll_mode, occ_cells=occ_cells)
            for _ in frames]


def _fib_dirs(n):
    """``n`` roughly uniform unit vectors (Fibonacci lattice)."""
    n = max(int(n), 1)
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)


VIRUS_REACH = 1.75        # spike heads, in units of the capsid radius
VIRUS_RADIUS_FRAC = 0.085  # default capsid radius as a fraction of the cube side
VIRUS_POINT_FACTOR = 18.8  # default marker area, in units of --poll-point-size
VIRUS_LAYERS = ("body", "stalk", "head")


def virus_layout(n, spikes=0):
    """Row budget of the virus shape -> {'body', 'stalk', 'head', 'spikes'}.

    The body gets the majority of the rows (that is the "big middle"), the spikes
    only a few: small blobs spend relatively more on the spikes so the silhouette
    still reads as a spiky particle.
    """
    n = int(max(0, n))
    if spikes <= 0:                      # auto: more rows -> more spikes
        spikes = int(np.clip(round(n / 5.0), 4, 10))
    small = n < 20
    n_head = int(min(spikes, max(1, round((0.30 if small else 0.20) * n)))) if n > 3 else 0
    n_stalk = int(min(spikes, max(0, round((0.15 if small else 0.12) * n)))) if n > 3 else 0
    return {"body": max(0, n - n_head - n_stalk), "stalk": n_stalk, "head": n_head,
            "spikes": int(max(n_head, n_stalk))}


def virus_points(n, center, radius, rng, spikes=0):
    """``n`` positions of a virus-like particle + one layer id per row.

    Layer ids: 0 = body (uniform inside the capsid ball, so the middle reads
    solid), 1 = spike stalk, 2 = spike head (small knob at the tip of a spike).
    The row budget (``virus_layout``) goes mostly to the body, the spikes only get
    a few rows, which is what makes the particle look like "big in the middle,
    spiky outside".

    Geometry only (visualisation): the row count equals the forged count of the
    training chain and the particle is centred on the centroid of the real forged
    rows, so the amount and the location of the pollution are unchanged.
    """
    n = int(n)
    if n <= 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int8)
    center = np.asarray(center, np.float64).reshape(3)
    radius = max(float(radius), 1e-9)
    if n <= 3:                           # too few rows for a body + spikes
        r = radius * 0.55 * rng.random(n) ** (1.0 / 3.0)
        return (center + _fib_dirs(n) * r[:, None]).astype(np.float32), np.zeros(n, np.int8)
    plan = virus_layout(n, spikes)
    n_body, n_stalk, n_head = plan["body"], plan["stalk"], plan["head"]
    dirs = _fib_dirs(max(plan["spikes"], 1))
    parts, layers = [], []
    if n_body > 0:                       # solid capsid ball (uniform in volume)
        r = radius * rng.random(n_body) ** (1.0 / 3.0)
        parts.append(center + _fib_dirs(n_body) * r[:, None])
        layers.append(np.zeros(n_body, np.int8))
    if n_stalk > 0:                      # stalks: staggered lengths inside the tips
        idx = np.arange(n_stalk)
        r_s = radius * (1.22 + 0.14 * (idx % 3))
        parts.append(center + dirs[idx % len(dirs)] * r_s[:, None])
        layers.append(np.full(n_stalk, 1, np.int8))
    if n_head > 0:                       # heads: small knobs, slight jitter only
        d = dirs[:n_head]
        tang = rng.normal(scale=0.05 * radius, size=(n_head, 3))
        tang -= (tang * d).sum(axis=1, keepdims=True) * d
        jit = rng.normal(scale=0.03 * radius, size=(n_head, 3))
        parts.append(center + d * (radius * (VIRUS_REACH - 0.05)) + tang + jit)
        layers.append(np.full(n_head, 2, np.int8))
    pts = np.concatenate(parts, axis=0)[:n]
    lay = np.concatenate(layers, axis=0)[:n]
    return pts.astype(np.float32), lay


def virus_point_size(args):
    """Marker area of the virus rows: fixed, so the particle keeps the same look."""
    if args.poll_virus_point_size > 0:
        return float(args.poll_virus_point_size)
    return VIRUS_POINT_FACTOR * float(args.poll_point_size)


def virus_blob(blob, lim, args, rng):
    """Replace the forged scatter by a virus particle of the same row count.

    The particle sits on the centroid of the real forged rows (shifted inside
    ``lim`` when --poll-clamp 1) and keeps the forged colour columns, so the
    tensor semantics of the pollution rows are untouched.
    """
    blob = np.asarray(blob, dtype=np.float32)
    if blob.ndim != 2 or not len(blob):
        return blob, None
    xyz = blob[:, :3].astype(np.float64)
    center = xyz.mean(axis=0)
    ext = np.array([float(lim[i][1] - lim[i][0]) for i in range(3)])
    radius = float(args.poll_virus_radius) or VIRUS_RADIUS_FRAC * float(ext.min())
    spikes = int(args.poll_virus_spikes)
    reach = radius * (VIRUS_REACH + 0.08)          # clamp margin: tip + jitter
    clamped = False
    if args.poll_clamp:
        lo = np.array([lim[i][0] for i in range(3)]) + reach
        hi = np.array([lim[i][1] for i in range(3)]) - reach
        new = np.clip(center, lo, hi)
        clamped = bool(np.any(np.abs(new - center) > 1e-9))
        center = new
    pts, lay = virus_points(len(blob), center, radius, rng, spikes=spikes)
    if blob.shape[1] > 3:                # keep the forged colour columns
        rgb = np.repeat(blob[:, 3:6].mean(axis=0, keepdims=True), len(pts), axis=0)
        pts = np.concatenate([pts, rgb.astype(np.float32)], axis=1)
    counts = {VIRUS_LAYERS[k]: int((lay == k).sum()) for k in range(len(VIRUS_LAYERS))}
    return pts.astype(np.float32), {"radius": round(radius, 6),
                                    "spikes": counts.get("head", 0),
                                    "layers": counts,
                                    "center": [round(float(v), 6) for v in center],
                                    "centroid_forged": [round(float(v), 6) for v in xyz.mean(axis=0)],
                                    "clamped": clamped}


def _occupied_cells(args, ws, frames, zarr_path, cache):
    """Cells used by --poll-mode occupied (training uses the whole zarr)."""
    if cache.get("cells") is not None:
        return cache["cells"]
    _, occupied_cells_from_raws, occupied_cells_from_zarr, _ = _forge_imports()
    if zarr_path and os.path.exists(zarr_path):
        cells, dims = occupied_cells_from_zarr(zarr_path, ws)
        src = os.path.basename(str(zarr_path))
    else:
        cells, dims = occupied_cells_from_raws(frames, ws)
        src = "episode frames"
    cache.update({"cells": cells, "dims": dims, "source": src})
    return cells


def compose_poll_frame(raw, blob, args):
    """clean rows + forged blob -> (pc (M,6), forged (M,) bool, info dict).

    dense: every raw row is kept, so the png has the same density as frame_KKKK.png.
    512  : exactly the poll-branch tensor (FPS over raw+blob, zero pad dropped);
           forged rows are located by row index (idx >= n_raw), not by colour.
    """
    raw = np.asarray(raw, dtype=np.float32)
    blob = np.asarray(blob, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] not in (3, 6):
        raise SystemExit("[frames] unexpected raw point cloud shape %r" % (raw.shape,))
    blob = blob.reshape(-1, blob.shape[-1])
    if blob.size and blob.shape[1] != raw.shape[1]:
        blob = blob[:, :raw.shape[1]]
    n_raw, n_blob = int(len(raw)), int(len(blob))
    cat = np.concatenate([raw, blob], axis=0) if n_blob else raw.copy()

    if args.poll_points == "dense":
        forged = np.zeros(len(cat), dtype=bool)
        forged[n_raw:] = True
        return cat, forged, {"branch": "dense", "n_raw": n_raw, "n_blob": n_blob,
                             "n_rendered": int(len(cat)), "n_real_rendered": n_raw,
                             "n_forged_rendered": int(forged.sum())}

    pc, idx, branch = fps_with_idx(cat, int(args.poll_num_points), args.fps_device)
    forged = idx >= n_raw
    keep = np.any(pc != 0.0, axis=1)          # drop the zero padding rows
    return pc[keep], forged[keep], {
        "branch": branch, "n_raw": n_raw, "n_blob": n_blob,
        "n_rendered": int(keep.sum()),
        "n_real_rendered": int((keep & ~forged).sum()),
        "n_forged_rendered": int((keep & forged).sum()),
        "n_pad_dropped": int((~keep).sum())}


def _cube_from_points(pts, zoom):
    """Symmetric cube around the min/max box of ``pts`` (bbox, not percentiles)."""
    pts = np.asarray(pts, dtype=np.float64)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2.0
    span = max(float((hi - lo).max()) / 2.0, 1e-6) / max(float(zoom), 1e-6)
    lim = tuple((float(center[i] - span), float(center[i] + span)) for i in range(3))
    return lim, center, span


def poll_limits(clean_xyz, blob_xyz, clean_lim, ws, args):
    """-> (fixed_lim | None, per_frame_lims | None, info).

    A percentile fit is useless here: ~370 blob rows are drowned by ~54k clean rows
    (measured on hammer ep0: pct2/98 of clean+blobs gives half_span 0.181 vs 0.179
    for clean alone, and 9/14 blobs sit outside that cube). Hence bbox/workspace.

    clean (default): exactly the cube and the zoom of frame_KKKK.png, so the poll png
                     is the clean png plus the red particle (pixel comparable).
    workspace       : one cube = the workspace box the pollution is drawn from.
    bbox    : one cube = min/max of clean+blobs over the episode.
    self    : one cube per frame (min/max of that frame's clean+blob); the view jumps.
    shared  : the clean cube, zoomed by --poll-zoom.
    """
    if args.poll_fit == "clean":
        return clean_lim, None, {"mode": "clean", "zoom": float(args.zoom),
                                 "note": "identical cube and zoom as frame_KKKK.png"}
    zoom = float(args.poll_zoom if args.poll_zoom is not None else args.zoom)
    if args.poll_fit == "shared":
        return clean_lim, None, {"mode": "shared", "zoom": zoom,
                                 "note": "clean cube reused; blobs outside it are clipped"}
    if args.poll_fit == "workspace":
        lo, hi = np.asarray(ws.lo, np.float64), np.asarray(ws.hi, np.float64)
        center = (lo + hi) / 2.0
        span = max(float((hi - lo).max()) / 2.0, 1e-6) / max(zoom, 1e-6)
        lim = tuple((float(center[i] - span), float(center[i] + span)) for i in range(3))
        return lim, None, {"mode": "workspace", "zoom": zoom,
                           "center": [round(float(v), 6) for v in center],
                           "half_span": round(span, 6)}
    if args.poll_fit == "bbox":
        parts = [p for p in list(clean_xyz) + list(blob_xyz) if len(p)]
        lim, center, span = _cube_from_points(np.concatenate(parts, axis=0), zoom)
        return lim, None, {"mode": "bbox", "zoom": zoom,
                           "center": [round(float(v), 6) for v in center],
                           "half_span": round(span, 6)}
    lims, spans = [], []
    for c, b in zip(clean_xyz, blob_xyz):
        parts = [p for p in (c, b) if len(p)]
        lim, _center, span = _cube_from_points(np.concatenate(parts, axis=0), zoom)
        lims.append(lim)
        spans.append(round(span, 6))
    return None, lims, {"mode": "self", "zoom": zoom, "half_span_per_frame": spans}


def _plot_points(ax, pc, lim, args, forged=None, forged_size=None, virus=None):
    """Real rows in their own colour, forged rows in --poll-rgb (plotted last).

    ``virus=(center, radius)`` switches the forged rows to the virus look: the
    marker fades from the deep red of --poll-rgb in the middle to a pale red at
    the spike tips, and shrinks towards the rim (big centre, small spikes).
    """
    xyz = np.asarray(pc[:, :3])
    real = np.ones(len(xyz), dtype=bool) if forged is None else ~forged
    if real.any():
        if pc.shape[1] >= 6:
            rgb = np.clip(np.asarray(pc[real, 3:6], np.float32) / 255.0, 0.0, 1.0)
            ax.scatter(xyz[real, 0], xyz[real, 1], xyz[real, 2], c=rgb,
                       s=args.point_size, zorder=2)
        else:
            ax.scatter(xyz[real, 0], xyz[real, 1], xyz[real, 2],
                       s=args.point_size, zorder=2)
    if forged is not None and forged.any():
        f = xyz[forged]
        s = args.poll_point_size if forged_size is None else float(forged_size)
        c = _poll_rgb01(args)
        if virus is not None:
            cen = np.asarray(virus[0], np.float64).reshape(3)
            rad = max(float(virus[1]), 1e-9)
            t = (np.linalg.norm(f - cen, axis=1) / rad - 0.55) / (VIRUS_REACH - 0.55)
            t = np.clip(t, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)                   # smoothstep
            sc = (float(args.poll_virus_core_scale)
                  + (float(args.poll_virus_edge_scale) - float(args.poll_virus_core_scale)) * t)
            s = float(s) * sc
            base = np.asarray(c, np.float64)
            fade = float(np.clip(args.poll_virus_fade, 0.0, 1.0))
            c = base[None, :] + (1.0 - base[None, :]) * (fade * t)[:, None]
        ax.scatter(f[:, 0], f[:, 1], f[:, 2], c=c, s=s,
                   edgecolors="none", depthshade=False, zorder=10)
    ax.set_xlim(lim[0])
    ax.set_ylim(lim[1])
    ax.set_zlim(lim[2])
    ax.view_init(elev=args.elev, azim=args.azim)
    _apply_ticks(ax, args.ticks)


def render_poll_frames(items, out_dir, lim_fixed, lims_self, lim_clean, args,
                       virus_geoms=None, ids=None):
    """frame_{KKKK}_poll.png = the same clean frame + the red pollution rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers '3d' projection)
    from termcolor import cprint

    os.makedirs(out_dir, exist_ok=True)
    pair = args.poll_panel == "pair"
    paths, n_forged = [], []

    def _add_ax(fig, *a):
        ax = fig.add_subplot(*a, projection="3d")
        if args.poll_on_top:            # zorder instead of depth sort -> always visible
            ax.computed_zorder = False
        return ax

    for i in (range(len(items)) if ids is None else list(ids)):
        pc, forged, _info = items[i]
        lim_poll = lims_self[i] if lims_self is not None else lim_fixed
        vg = virus_geoms[i] if virus_geoms is not None else None
        n_f = int(forged.sum())
        n_forged.append(n_f)
        s_f = args.poll_point_size
        if args.poll_shape == "virus":             # fixed size: only the density varies
            s_f = virus_point_size(args)
        if pair:
            fig = plt.figure(figsize=(2.0 * args.width / args.dpi, args.height / args.dpi),
                             dpi=args.dpi)
            ax_l = _add_ax(fig, 1, 2, 1)
            ax_r = _add_ax(fig, 1, 2, 2)
            _plot_points(ax_l, pc[~forged], lim_clean, args)
            _plot_points(ax_r, pc, lim_poll, args, forged=forged, forged_size=s_f,
                         virus=vg)
            ax_l.set_title("clean view", fontsize=8)
            ax_r.set_title("poll view (red = forged)", fontsize=8)
        else:
            fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi),
                             dpi=args.dpi)
            ax = _add_ax(fig, 111)
            _plot_points(ax, pc, lim_poll, args, forged=forged, forged_size=s_f,
                         virus=vg)
        for ax in fig.axes:
            _style_bg(fig, ax, args)
        _tight_layout(fig, args)
        path = os.path.join(out_dir, "frame_%04d_poll.png" % i)
        _save_fig(fig, path, args)
        plt.close(fig)
        paths.append(path)
    cprint("[frames] %d poll png -> %s (fit=%s, forged %d..%d rows/frame)"
           % (len(paths), out_dir, args.poll_fit,
              min(n_forged) if n_forged else 0, max(n_forged) if n_forged else 0), "green")
    return paths, n_forged


def render_poll(frames, out_dir, clean_lim, args, task=None, zarr_path=None, cache=None,
                ids=None):
    """Forge + compose + render the poll branch of one episode; returns its meta."""
    from termcolor import cprint

    cache = {} if cache is None else cache
    ws = resolve_forge_workspace(args, task=task, zarr_path=zarr_path)
    occ = None
    if args.poll_mode == "occupied":
        occ = _occupied_cells(args, ws, frames, zarr_path, cache)
    blobs = forge_blobs(frames, ws, occ, args)
    clean_xyz = [np.asarray(f[:, :3]) for f in frames]
    raw_xyz = [np.asarray(b[:, :3]).reshape(-1, 3) for b in blobs]
    shape_info = None
    if args.poll_shape == "virus":
        if args.poll_fit in ("clean", "shared"):
            view_lims = [clean_lim] * len(frames)
        else:                            # clamp into the cube this fit will use
            lim_fx, lims_sf, _linfo = poll_limits(clean_xyz, raw_xyz, clean_lim, ws, args)
            view_lims = lims_sf if lims_sf is not None else [lim_fx] * len(frames)
        rng = np.random.default_rng(int(args.poll_seed) + 9176)
        shaped, shape_info = [], []
        for b, vlim in zip(blobs, view_lims):
            vb, vinfo = virus_blob(b, vlim, args, rng)
            shaped.append(vb)
            shape_info.append(vinfo)
        blobs = shaped
        radii = [v["radius"] for v in shape_info if v]
        spikes = [v["spikes"] for v in shape_info if v]
        n_clamp = sum(1 for v in shape_info if v and v["clamped"])
        cprint("[frames] virus particle (rows unchanged): radius=%s m, spikes=%s, "
               "centred on the forged centroid, %d/%d frames clamped into the cube"
               % ("%.4f" % radii[0] if radii else "-", spikes[0] if spikes else "-",
                  n_clamp, len(frames)), "cyan")
    items = [compose_poll_frame(f, b, args) for f, b in zip(frames, blobs)]
    blob_xyz = [np.asarray(b[:, :3]).reshape(-1, 3) for b in blobs]
    lim_fixed, lims_self, linfo = poll_limits(clean_xyz, blob_xyz, clean_lim, ws, args)
    if args.poll_fit in ("clean", "shared"):
        lo = np.array([clean_lim[i][0] for i in range(3)])
        hi = np.array([clean_lim[i][1] for i in range(3)])
        n_out = sum(1 for b in blob_xyz if len(b) and not np.all((b >= lo) & (b <= hi)))
        if n_out:
            frac, n_all = points_in_limits(blob_xyz, clean_lim)
            cprint("[frames] WARN --poll-fit %s: %d/%d frames have pollution rows outside "
                   "the plotted cube (%.1f%% of %d rows inside); mplot3d does not clip "
                   "scatter, so those rows are drawn outside the axes box just like the "
                   "clean frames' out-of-cube points"
                   % (args.poll_fit, n_out, len(frames), 100.0 * frac, n_all), "yellow")
    paths, n_forged = render_poll_frames(
        items, out_dir, lim_fixed, lims_self, clean_lim, args,
        virus_geoms=(None if shape_info is None
                     else [(v["center"], v["radius"]) if v else None for v in shape_info]),
        ids=ids)
    ws_dict = ws.to_dict()
    return {
        "dir": out_dir,
        "points": args.poll_points, "num_points": int(args.poll_num_points),
        "voxel": [int(args.poll_voxel_min), int(args.poll_voxel_max)],
        "mode": args.poll_mode, "seed": int(args.poll_seed),
        "rgb": _floats(args.poll_rgb, 3, "poll-rgb"),
        "forge_rgb": (_floats(args.poll_forge_rgb, 3, "poll-forge-rgb")
                      if args.poll_forge_rgb else "workspace(OBSTACLE_RGB)"),
        "point_size": args.poll_point_size, "panel": args.poll_panel,
        "shape": args.poll_shape,
        "virus": (None if shape_info is None else {
            "radius": shape_info[0]["radius"] if shape_info[0] else None,
            "radius_auto": not bool(args.poll_virus_radius),
            "spikes": shape_info[0]["spikes"] if shape_info[0] else None,
            "spikes_per_frame": [v["spikes"] if v else None for v in shape_info],
            "point_size": virus_point_size(args),
            "core_scale": float(args.poll_virus_core_scale),
            "edge_scale": float(args.poll_virus_edge_scale),
            "color_center": _floats(args.poll_rgb, 3, "poll-rgb"),
            "fade": float(args.poll_virus_fade),
            "on_top": bool(args.poll_on_top),
            "clamp": bool(args.poll_clamp),
            "clamped_frames": [i for i, v in enumerate(shape_info) if v and v["clamped"]],
            "seed": int(args.poll_seed) + 9176,
            "rows_unchanged": True}),
        "fps_device": args.fps_device if args.poll_points == "512" else None,
        "workspace": ws_dict, "workspace_cells": int(ws.n_cells()),
        "occupied_cells": (int(len(cache["cells"])) if cache.get("cells") is not None else None),
        "occupied_source": cache.get("source"),
        "limits": linfo,
        "axis_limits": ([list(v) for v in lim_fixed] if lim_fixed is not None
                        else [list(v) for v in lims_self[0]]),
        "per_frame": [dict(info, n_poll_xyz=int(len(b)),
                           virus=(None if shape_info is None else shape_info[i]))
                      for i, (info, b) in enumerate(zip([it[2] for it in items], blob_xyz))],
        "forged_rendered": n_forged,
        "forged_all": [int(np.asarray(it[1]).sum()) for it in items],
        "render_ids": list(range(len(frames))) if ids is None else list(ids),
        "files": [os.path.basename(p) for p in paths]}


# ------------------------------------------------------------------- live source
def make_env(args, seed):
    """Same env stack as gen_demonstration_expert.create_env (Adroit branch)."""
    from termcolor import cprint
    from adroit import AdroitEnv
    from diffusion_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit

    env = AdroitEnv(env_name=args.task + "-v0", test_image=False, num_repeats=2,
                    num_frames=1, env_feature_type="pixels", device="cuda",
                    reward_rescale=True)
    env = MujocoPointcloudWrapperAdroit(env=env, env_name="adroit_" + args.task,
                                        use_point_crop=(not args.no_point_crop),
                                        img_size=args.pc_resolution)
    # fix HammerEnvV0.np_random (nail_board Z) through the same set_seed chain
    try:
        env.env._env.set_seed(seed)
    except Exception as e:
        cprint("[frames] WARN set_seed failed: %r" % (e,), "yellow")
    return env


def load_expert(ckpt_path):
    import torch
    from termcolor import cprint

    agent = torch.load(str(ckpt_path), map_location="cpu")["agent"]
    agent = agent.to("cuda")
    cprint("[frames] expert: %s" % ckpt_path, "cyan")
    return agent


def rollout_episode(env, agent, args, episode_idx):
    """Collect full point clouds, stopping 1 step after success.

    Kept frame count = success_step + extra_steps (success_step is set to the frame
    index that follows the successful action, same convention as the collector).
    """
    import torch
    import utils
    from termcolor import cprint

    time_step = env.reset()
    frames, imgs = [], []
    total_reward, n_goal, step_count, success_step = 0.0, 0.0, 0, -1
    while (not time_step.last()) or step_count < args.min_episode_steps:
        if (args.extra_steps_after_success > 0 and success_step >= 0 and
                step_count - success_step >= args.extra_steps_after_success):
            cprint("[frames] ep%d: success@%d, stopped %d step(s) later, %d frames kept"
                   % (episode_idx, success_step, step_count - success_step, step_count),
                   "green")
            break
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs=time_step.observation, step=0, eval_mode=True,
                               obs_sensor=time_step.observation_sensor)
            full_pc, _ = env.get_full_point_cloud()
            frames.append(np.asarray(full_pc, dtype=np.float32))
            if args.save_img:
                cam = env.pc_generator.cam_names[0]
                img_cap, _ = env.pc_generator.captureImage(cam, capture_depth=True)
                imgs.append(np.ascontiguousarray(img_cap))
        time_step = env.step(action)
        n_goal += time_step.n_goal_achieved
        total_reward += time_step.reward
        step_count += 1
        if success_step < 0 and n_goal >= args.success_goal_threshold:
            success_step = step_count

    cprint("[frames] ep%d: frames=%d success_step=%d n_goal=%.1f reward=%.2f"
           % (episode_idx, len(frames), success_step, n_goal, total_reward),
           "green" if success_step >= 0 else "yellow")
    return {"episode": episode_idx, "frames": frames, "imgs": imgs,
            "success_step": int(success_step), "n_goal": float(n_goal),
            "reward": float(total_reward), "steps": int(step_count)}


# ------------------------------------------------------------------- zarr source
def _task_from_zarr(path):
    name = os.path.basename(os.path.normpath(str(path)))
    if name.startswith("adroit_") and name.endswith("_expert.zarr"):
        return name[len("adroit_"):-len("_expert.zarr")]
    return "task"


def load_episodes_from_zarr(path, args):
    """Read stored full point clouds and truncate each episode like the live loop."""
    import zarr
    from termcolor import cprint

    root = zarr.open(str(path), mode="r")
    ends = np.asarray(root["meta"]["episode_ends"][:]).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    success = None
    if "success_steps" in root["meta"]:
        success = np.asarray(root["meta"]["success_steps"][:]).astype(np.int64)

    key = args.pc_key
    if key not in root["data"]:
        cprint("[frames] WARN data/%s missing, falling back to point_cloud" % key, "yellow")
        key = "point_cloud"
    raw_len = None
    if key == "raw_point_cloud" and "raw_point_cloud_len" in root["data"]:
        raw_len = np.asarray(root["data"]["raw_point_cloud_len"][:]).astype(np.int64)

    episodes = [int(e) for e in str(args.episodes).split(",") if str(e).strip() != ""]
    out = []
    for ep in episodes:
        if ep < 0 or ep >= len(ends):
            raise SystemExit("[frames] episode %d out of range (zarr has %d)"
                             % (ep, len(ends)))
        s, e = int(starts[ep]), int(ends[ep])
        frames = []
        for i in range(s, e):
            arr = np.asarray(root["data"][key][i], dtype=np.float32)
            if arr.ndim == 1:
                if raw_len is not None:
                    arr = arr.reshape(int(raw_len[i]), -1)
                else:
                    arr = arr.reshape(-1, 6 if arr.size % 6 == 0 else 3)
            frames.append(arr)
        ss = int(success[ep]) if success is not None else -1
        if args.extra_steps_after_success > 0 and ss >= 0:
            n_keep = min(len(frames), ss + args.extra_steps_after_success)
            cprint("[frames] ep%d: success@%d, keep %d/%d frames (pc=%s)"
                   % (ep, ss, n_keep, len(frames), key), "green")
            frames = frames[:n_keep]
        else:
            cprint("[frames] ep%d: keep %d/%d frames (pc=%s, success_step=%d)"
                   % (ep, len(frames), len(frames), key, ss), "yellow")
        out.append({"episode": ep, "frames": frames, "imgs": [], "success_step": ss,
                    "n_goal": None, "reward": None, "steps": len(frames)})
    return out, _task_from_zarr(path)


# ------------------------------------------------------------------------- main
def main():
    args = build_parser().parse_args()
    from termcolor import cprint

    zarr_path = None
    if args.source == "zarr":
        zarr_path = args.zarr or os.path.join(DEFAULT_DATA_ROOT,
                                              "adroit_%s_expert.zarr" % args.task)
        if not os.path.exists(zarr_path):
            raise SystemExit("[frames] zarr not found: %s" % zarr_path)
        episodes, task = load_episodes_from_zarr(zarr_path, args)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        import torch  # noqa: F401  (imported after CUDA_VISIBLE_DEVICES is set)
        ckpt = args.expert_ckpt or str(_REPO / "third_party" / "VRL3" / "ckpts"
                                       / ("vrl3_%s.pt" % args.task))
        if not os.path.exists(ckpt):
            raise SystemExit("[frames] expert ckpt not found: %s" % ckpt)
        agent = load_expert(ckpt)
        episodes, task = [], args.task
        for ep in range(args.num_episodes):
            env = make_env(args, args.seed + ep)
            result = rollout_episode(env, agent, args, ep)
            if result["n_goal"] >= args.success_goal_threshold:
                episodes.append(result)
            else:
                cprint("[frames] ep%d discarded (n_goal=%.1f < %.1f)"
                       % (ep, result["n_goal"], args.success_goal_threshold), "red")
            try:
                env.close()
            except Exception:
                pass
        if not episodes:
            raise SystemExit("[frames] no successful episode, nothing to render")

    occ_cache = {}
    for result in episodes:
        ep = int(result["episode"])
        frames = result["frames"]
        if not frames:
            cprint("[frames] ep%d has no frames, skip" % ep, "red")
            continue
        out_dir = os.path.join(args.out, "adroit_%s_ep%04d" % (task, ep))
        poll_dir = os.path.join(args.poll_out or args.out,
                                "adroit_%s_ep%04d_poll" % (task, ep))
        _prepare_output(out_dir, args.overwrite)
        if args.poll:
            _prepare_output(poll_dir, args.overwrite)
        xyz = [f[:, :3] for f in frames]
        lim, info = fit_limits(xyz, args)
        frac, n_all = points_in_limits(xyz, lim)
        cprint("[frames] ep%d limits x=%.3f..%.3f y=%.3f..%.3f z=%.3f..%.3f "
               "(%.1f%% of %d points inside)"
               % (ep, lim[0][0], lim[0][1], lim[1][0], lim[1][1], lim[2][0], lim[2][1],
                  100.0 * frac, n_all), "cyan")
        # --frames only trims the render list: the cube fit, the forge rng stream and
        # the virus layout still run over the whole episode, so a subset renders
        # exactly like the same frame of the full run
        ids = frame_ids(args, len(frames))
        if len(ids) != len(frames):
            cprint("[frames] rendering %d/%d frame(s): %s"
                   % (len(ids), len(frames), ids), "cyan")
        paths, n_pts = render_frames(frames, out_dir, lim, args, ids=ids)
        poll_meta = None
        if args.poll:
            try:
                poll_meta = render_poll(frames, poll_dir, lim, args, task=task,
                                        zarr_path=(zarr_path if args.source == "zarr" else None),
                                        cache=occ_cache, ids=ids)
            except SystemExit:
                raise
            except Exception as e:  # keep the clean frames usable if the forge fails
                cprint("[frames] WARN poll branch failed: %r" % (e,), "yellow")
        # one shared crop box for the clean and the poll frames of this episode
        img_info = trim_outputs([out_dir] + ([poll_dir] if poll_meta is not None else []), args)
        if img_info is not None:
            cprint("[frames] trim -> %dx%d px (bg=%s, pad=%d)"
                   % (img_info["size"][0], img_info["size"][1], img_info["bg"],
                      img_info["pad"]), "green")
        if poll_meta is not None:
            poll_doc = {"source": args.source, "task": task, "episode": ep,
                        "clean_dir": out_dir,
                        "frame_ids": ids,
                        "clean_frames": [os.path.basename(p) for p in paths],
                        "extra_steps_after_success": int(args.extra_steps_after_success),
                        "view": {"elev": args.elev, "azim": args.azim, "ticks": args.ticks},
                        "fig": {"width": args.width, "height": args.height,
                                "dpi": args.dpi, "point_size": args.point_size},
                        "image": img_info,
                        "poll": poll_meta}
            poll_meta_path = os.path.join(poll_dir, "meta.json")
            with open(poll_meta_path, "w") as f:
                json.dump(poll_doc, f, indent=2)
            cprint("[frames] poll meta -> %s" % poll_meta_path, "green")
        if args.save_img:
            import imageio
            for i, img in enumerate(result["imgs"]):
                if i not in ids:
                    continue
                imageio.imwrite(os.path.join(out_dir, "frame_%04d_rgb.png" % i),
                                np.asarray(img))
        meta = {"source": args.source, "task": task, "episode": ep,
                "success_step": int(result["success_step"]),
                "extra_steps_after_success": int(args.extra_steps_after_success),
                "n_goal": result["n_goal"], "reward": result["reward"],
                "frame_ids": ids,
                "n_frames": len(paths), "n_points_per_frame": n_pts,
                "limits": info, "axis_limits": [list(v) for v in lim],
                "points_inside": round(float(frac), 4),
                "view": {"elev": args.elev, "azim": args.azim, "ticks": args.ticks},
                "fig": {"width": args.width, "height": args.height, "dpi": args.dpi,
                        "point_size": args.point_size},
                "image": img_info,
                "pc_resolution": None if args.source == "zarr" else args.pc_resolution,
                "seed": None if args.source == "zarr" else args.seed,
                "poll": poll_meta,
                "frames": [os.path.basename(p) for p in paths]}
        meta_path = os.path.join(out_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        cprint("[frames] meta -> %s" % meta_path, "green")
    cprint("[frames] done, %d episode(s) under %s" % (len(episodes), args.out), "green")


if __name__ == "__main__":
    main()
