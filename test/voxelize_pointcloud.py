#!/usr/bin/env python
"""
Voxelize point clouds from the Adroit hammer full-point-cloud zarr and
visualize the result on a headless server.

Voxel workspace
---------------
The voxel grid is aligned with the *adroit_hammer environment workspace*:
the fixed 3D bounding box that the top camera (ENV_POINT_CLOUD_CONFIG in
`diffusion_policy_3d/gym_util/mjpc_wrapper.py`) actually observes. By default
this box is computed from the union of *all* frames of the dataset
(lo/hi of every raw point cloud), which is the empirically occupied workspace
of the environment.  It can be overridden with --workspace.

Voxelization (two interchangeable implementations)
---------------------------------------------------
* numpy : pure-numpy grid binning.  Per voxel it keeps the centroid, the
          occupancy count and the average RGB colour.  Grid is aligned to
          the workspace origin (lo).
* open3d: `open3d.geometry.PointCloud.voxel_down_sample(voxel_size)` (0.19.0
          is installed).  Grid origin is the point cloud min, so voxel
          centers may differ from the numpy grid by < voxel_size.

Visualization (verified to work on the current headless server)
---------------------------------------------------------------
* matplotlib (Agg backend) -> PNG.  Side-by-side original / voxelized view.
  No display needed.
* plotly -> interactive standalone HTML (pio.to_html, plotly.js embedded).
  No Chrome / kaleido needed; open the file in a local browser to rotate.
* open3d rendering is NOT usable here (GLFW/GLEW fail without a display),
  it is only used for the voxel_down_sample computation.

Outputs (default: /hy-tmp/3DP/voxel_out/)
------------------------------------------
* voxel_result_frame<F>-<F+N-1>_kernel_<method>.npz  : voxel centers / counts / avg RGB
* pointcloud_voxelized_frame<F>-<F+N-1>_kernel_<method>.png : matplotlib view
* pointcloud_voxelized_frame<F>-<F+N-1>.html         : plotly interactive view
* voxel_summary_frame<F>-<F+N-1>.json                : workspace + per-method stats

Examples
--------
python test/voxelize_pointcloud.py                                    # frame 0, both methods
python test/voxelize_pointcloud.py --frame 50 --voxel_size 0.025
python test/voxelize_pointcloud.py --frames 100 --frame 0            # aggregate ep0 (denser)
python test/voxelize_pointcloud.py --frames 100 --workspace_mode agg # tight workspace
python test/voxelize_pointcloud.py --which fps512 --method open3d
python test/voxelize_pointcloud.py --workspace -0.3 -0.1 -0.1 0.5 0.5 0.3
python test/voxelize_pointcloud.py --target_voxels 20000              # auto voxel size
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import zarr
from termcolor import cprint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The package lives in 3D-Diffusion-Policy/: bootstrap it so the documented
# "python test/voxelize_pointcloud.py" works from the repo root (the voxelizer /
# workspace implementations are imported from the package, see voxelize_numpy).
_DP3_DIR = os.path.join(REPO_ROOT, '3D-Diffusion-Policy')
for _p in (_DP3_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_ZARR_PATH = '/hy-tmp/3DP/data/adroit_hammer_expert_fullpc.zarr'
DEFAULT_OUT_DIR = '/hy-tmp/3DP/voxel_out'

# adroit_hammer env point-cloud config reference:
#   diffusion_policy_3d/gym_util/mjpc_wrapper.py::ENV_POINT_CLOUD_CONFIG
#     cam 'top' + ADROIT_PC_TRANSFORM(45deg) + offset [0,0,1] + z>=-0.099 crop.
# The empirical occupied workspace of the whole dataset (all 1000 frames):
WORKSPACE_DEFAULT = {
    'lo': np.array([-0.24570936, -0.07013131, -0.09785966]),
    'hi': np.array([0.44392079, 0.45474687, 0.22984123]),
}


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def _open_data(zarr_path):
    root = zarr.open(zarr_path, mode='r')
    if 'data' not in root.group_keys():
        raise FileNotFoundError(
            '{} is not a valid dataset (no "data" group). '.format(zarr_path)
            + 'Use the real zarr: {}'.format(DEFAULT_ZARR_PATH))
    return root['data']


def load_frame_pc(zarr_path, frame, which='raw'):
    """Return the (N, 6) xyz+rgb point cloud of a single frame."""
    d = _open_data(zarr_path)
    if which == 'raw':
        n = int(np.asarray(d['raw_point_cloud_len'][frame]))
        pc = np.asarray(d['raw_point_cloud'][frame]).reshape(n, 6)
    else:
        pc = np.asarray(d['point_cloud'][frame]).copy()  # (512, 6) fps
        # drop zero-padding rows (fixed 512-fps arrays are zero padded)
        pad = np.all(np.abs(pc[:, :3]) < 1e-6, axis=1)
        pc = pc[~pad]
    return pc.astype(np.float32)


def compute_workspace(zarr_path):
    """Union bounding box of every raw point cloud in the dataset.

    Thin wrapper over ``diffusion_policy_3d.voxel.workspace.compute_workspace_from_zarr``
    (single source of truth; the historical hammer constants were produced by
    exactly this convention).
    """
    from diffusion_policy_3d.voxel.workspace import compute_workspace_from_zarr
    return compute_workspace_from_zarr(zarr_path)


# ---------------------------------------------------------------------------
# voxelization
# ---------------------------------------------------------------------------
def voxelize_numpy(xyz, rgb, lo, hi, voxel_size):
    """Grid-binning voxelization aligned to the workspace origin (lo).

    Delegates to the single source of truth
    ``diffusion_policy_3d.voxel.voxelizer.voxelize`` — same grid convention
    (``floor((p - lo) / grid)`` with ``dims = ceil((hi - lo) / grid)``; ``cap``
    only scales ``occ`` and never truncates the template) — and maps the canonical
    ``VoxelField`` back onto this script's dict contract.

    The historical inline implementation was machine-checked against the package
    (occupied cells / per-cell counts / centers / mean RGB all identical,
    max|Δ| = 0 on hammer + door) by ``test/verify_voxelize_equiv.py``.

    NOTE on bookkeeping keys (unconsumed by the plots, semantics differ slightly):
      * ``n_input``          finite input rows (package view)
      * ``n_inside``         rows inside bbox+grid (the old ``n_input``)
      * ``n_dropped_outside`` package view = not inside bbox+grid
    """
    from diffusion_policy_3d.voxel.voxelizer import voxelize as _pkg_voxelize
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = None if rgb is None else np.asarray(rgb, dtype=np.float32)
    pc = xyz if (rgb is None or rgb.shape[1] != 3) else \
        np.concatenate([xyz, rgb], axis=1)
    fld = _pkg_voxelize(pc, lo=lo, hi=hi, grid_size=voxel_size)
    if fld.n_cells == 0:
        return None
    centers = (fld.cell_idx.astype(np.float64) + 0.5) * float(voxel_size) + \
        np.asarray(lo, dtype=np.float64)
    if rgb is not None and rgb.shape[1] == 3:
        avg_rgb = (np.add.reduceat(fld.template[:, 3:6], fld.starts, axis=0) /
                   fld.counts[:, None].astype(np.float64)).astype(np.float32)
    else:
        avg_rgb = None
    return {
        'method': 'numpy',
        'centers': centers.astype(np.float32),
        'counts': fld.counts,
        'avg_rgb': avg_rgb,
        'grid_index': fld.cell_idx,
        'voxel_size': voxel_size,
        'workspace_lo': np.asarray(lo, dtype=np.float64),
        'workspace_hi': np.asarray(hi, dtype=np.float64),
        'n_input': int(fld.n_input),
        'n_inside': int(np.asarray(fld.inside).sum()),
        'n_dropped_outside': int(fld.n_dropped_outside),
    }


def voxelize_open3d(xyz, rgb, voxel_size):
    """Voxelization via open3d voxel_down_sample (keeps averaged colours)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if rgb is not None and rgb.shape[1] == 3:
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(rgb.astype(np.float64) / 255.0, 0.0, 1.0))
    ds = pcd.voxel_down_sample(voxel_size)
    centers = np.asarray(ds.points)
    colors = np.asarray(ds.colors)
    return {
        'method': 'open3d',
        'centers': centers.astype(np.float32),
        'counts': None,  # open3d does not expose occupancy counts
        'avg_rgb': (colors * 255.0).astype(np.float32) if len(colors) else None,
        'grid_index': None,
        'voxel_size': voxel_size,
        'workspace_lo': None, 'workspace_hi': None,
        'n_input': len(xyz), 'n_dropped_outside': 0,
    }


# ---------------------------------------------------------------------------
# visualization (headless-friendly)
# ---------------------------------------------------------------------------
def _shared_axis_limits(ax, lo, hi):
    span = max((hi - lo).max(), 1e-6)
    center = (lo + hi) / 2.0
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((hi - lo) / (hi - lo).max())
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')


def save_matplotlib_png(orig_pc, vox, out_png, lo, hi, frame, title_suffix=''):
    """Side-by-side original vs voxelized 3D scatter PNG (Agg, headless)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(13, 6.2), dpi=110)

    # left: original point cloud (RGB colour)
    ax1 = fig.add_subplot(121, projection='3d')
    rgb1 = np.clip(orig_pc[:, 3:6] / 255.0, 0.0, 1.0) \
        if orig_pc.shape[1] >= 6 else None
    ax1.scatter(orig_pc[:, 0], orig_pc[:, 1], orig_pc[:, 2],
                s=1.2, c=rgb1, alpha=0.9)
    _shared_axis_limits(ax1, lo, hi)
    ax1.set_title('original ({} pts)'.format(len(orig_pc)))

    # right: voxelized (marker size ~ sqrt(occupancy))
    ax2 = fig.add_subplot(122, projection='3d')
    c = vox['centers']
    rgb2 = np.clip(vox['avg_rgb'] / 255.0, 0.0, 1.0) \
        if vox['avg_rgb'] is not None else None
    counts = vox['counts'] if vox['counts'] is not None else np.ones(len(c))
    s = 6.0 + 60.0 * np.sqrt(counts / float(counts.max()))
    ax2.scatter(c[:, 0], c[:, 1], c[:, 2], s=s, c=rgb2, alpha=0.95)
    _shared_axis_limits(ax2, lo, hi)
    ax2.set_title('voxelized {} ({} voxels){}'.format(
        vox['method'], len(c), title_suffix))

    fig.suptitle('Adroit hammer  frame {}  voxel_size={}  '
                 'lo=[{:.3f} {:.3f} {:.3f}] hi=[{:.3f} {:.3f} {:.3f}]'.format(
                     frame, vox['voxel_size'], *lo, *hi))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)
    cprint('PNG saved to {}'.format(out_png), 'green')


def save_plotly_html(orig_pc, vox_results, out_html, lo, hi, frame,
                     orig_note=''):
    """Interactive standalone HTML (plotly.js embedded, no Chrome needed)."""
    import plotly.graph_objs as go
    import plotly.io as pio

    def rgb_str(rgb):
        return 'rgb({},{},{})'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def trace_rgb(pc, name, size):
        return go.Scatter3d(
            x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], mode='markers', name=name,
            marker=dict(size=size,
                        color=[rgb_str(r) for r in pc[:, 3:6]]))

    traces = [trace_rgb(orig_pc,
                        'original ({} pts){}'.format(len(orig_pc), orig_note),
                        2)]
    for method, vox in vox_results.items():
        c = vox['centers']
        counts = vox['counts'] if vox['counts'] is not None else np.ones(len(c))
        colors = [rgb_str(rgb) for rgb in vox['avg_rgb']] \
            if vox['avg_rgb'] is not None else 'grey'
        traces.append(go.Scatter3d(
            x=c[:, 0], y=c[:, 1], z=c[:, 2], mode='markers',
            name='voxelized {} ({} voxels)'.format(method, len(c)),
            marker=dict(
                size=2 + 12.0 * np.sqrt(counts / float(counts.max())),
                color=colors, opacity=0.95)))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Adroit hammer  frame {}  voxel_size={}'.format(
            frame, list(vox_results.values())[0]['voxel_size']),
        scene=dict(
            aspectmode='cube',
            xaxis=dict(range=[lo[0], hi[0]], showbackground=False),
            yaxis=dict(range=[lo[1], hi[1]], showbackground=False),
            zaxis=dict(range=[lo[2], hi[2]], showbackground=False),
            bgcolor='white'),
        legend=dict(x=0, y=1))
    os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
    with open(out_html, 'w') as f:
        f.write(pio.to_html(fig, full_html=True, include_plotlyjs=True))
    cprint('HTML saved to {}'.format(out_html), 'green')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--zarr_path', default=DEFAULT_ZARR_PATH,
                   help='path to the full-pc zarr (default: %(default)s)')
    p.add_argument('--frame', type=int, default=0,
                   help='frame index to voxelize (default: %(default)s)')
    p.add_argument('--episode', type=int, default=None,
                   help='if set, use the first frame of this 0-based episode '
                        '(overrides --frame)')
    p.add_argument('--frame_in_episode', type=int, default=0,
                   help='frame offset inside the episode (default: %(default)s)')
    p.add_argument('--which', choices=['raw', 'fps512'], default='raw',
                   help='which point cloud to voxelize: "raw" full point cloud '
                        'or "fps512" FPS-downsampled one (default: %(default)s)')
    p.add_argument('--voxel_size', type=float, default=0.02,
                   help='voxel edge length in metres (default: %(default)s)')
    p.add_argument('--frames', type=int, default=1,
                   help='aggregate N consecutive frames (starting at the '
                        'selected frame) before voxelizing; multi-frame '
                        'aggregation fills the voxel grid much denser '
                        '(default: %(default)s = single frame)')
    p.add_argument('--target_voxels', type=int, default=None,
                   help='if set, auto-compute voxel_size so the workspace '
                        'splits into ~this many voxels')
    p.add_argument('--workspace', nargs=6, type=float, default=None,
                   metavar=('lo_x', 'lo_y', 'lo_z', 'hi_x', 'hi_y', 'hi_z'),
                   help='fixed workspace bbox; overrides --workspace_mode')
    p.add_argument('--workspace_mode', choices=['dataset', 'agg'], default='dataset',
                   help='how to choose the voxel workspace when --workspace is '
                        'not given: "dataset" = full-dataset bbox (the '
                        'adroit_hammer environment workspace, default), '
                        '"agg" = bbox of the aggregated frames only (more '
                        'compact, denser occupancy)')
    p.add_argument('--method', choices=['numpy', 'open3d', 'both'],
                   default='both',
                   help='voxelization implementation(s) (default: %(default)s)')
    p.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                   help='output directory (default: %(default)s)')
    p.add_argument('--skip_png', action='store_true',
                   help='do not render matplotlib PNG')
    p.add_argument('--skip_html', action='store_true',
                   help='do not write plotly HTML')
    return p.parse_args()


def main():
    args = parse_args()

    # ---- episode / frame resolution ---------------------------------------
    if args.episode is not None:
        root = zarr.open(args.zarr_path, mode='r')
        ee = np.asarray(root['meta']['episode_ends'])
        start = int(ee[args.episode - 1]) if args.episode > 0 else 0
        frame = start + args.frame_in_episode
    else:
        frame = args.frame


    # ---- load frames (aggregated if --frames > 1) ---------------------------
    max_frames = int(len(_open_data(args.zarr_path)['raw_point_cloud_len']))
    n_frames = min(args.frames, max_frames - frame)
    if n_frames < 1:
        raise IndexError('frame {} out of range [0, {})'.format(frame, max_frames))
    pcs = [load_frame_pc(args.zarr_path, f, which=args.which)
           for f in range(frame, frame + n_frames)]
    pc = np.concatenate(pcs, axis=0).astype(np.float32)
    xyz = pc[:, :3].astype(np.float64)
    rgb = pc[:, 3:6].astype(np.float64)
    if n_frames > 1:
        cprint('aggregated frames {}-{} [{}]: {} points, '
               'bbox lo={} hi={}'.format(frame, frame + n_frames - 1,
                                         args.which, len(pc),
                                         xyz.min(0), xyz.max(0)), 'green')
    else:
        cprint('frame {} [{}]: {} points, bbox lo={} hi={}'.format(
            frame, args.which, len(pc), xyz.min(0), xyz.max(0)), 'green')
    # display cloud: subsample huge aggregates for a readable plot
    if len(pc) > 5000:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(pc), 5000, replace=False)
        disp_pc = pc[sel]
        disp_note = ' (subsample 5000 of {} pts)'.format(len(pc))
    else:
        disp_pc = pc
        disp_note = ''

    # ---- workspace ----------------------------------------------------------
    if args.workspace is not None:
        lo = np.asarray(args.workspace[:3])
        hi = np.asarray(args.workspace[3:])
        ws_src = 'cli --workspace'
    elif args.workspace_mode == 'agg':
        lo = xyz.min(axis=0)
        hi = xyz.max(axis=0)
        ws_src = 'aggregated-frames bbox (frames {}-{})'.format(
            frame, frame + n_frames - 1)
    else:
        t0 = time.time()
        lo, hi = compute_workspace(args.zarr_path)
        cprint('computed full-dataset workspace in {:.2f}s'.format(
            time.time() - t0), 'cyan')
        ws_src = 'full-dataset bbox ({} frames)'.format(
            len(_open_data(args.zarr_path)['raw_point_cloud_len']))
    cprint('workspace ({}): lo=[{:.4f} {:.4f} {:.4f}] '
           'hi=[{:.4f} {:.4f} {:.4f}] extent=[{:.3f} {:.3f} {:.3f}]'.format(
               ws_src, *lo, *hi, *(hi - lo)), 'cyan')

    # ---- voxel size ---------------------------------------------------------
    voxel_size = args.voxel_size
    if args.target_voxels is not None:
        volume = float(np.prod(hi - lo))
        voxel_size = (volume / args.target_voxels) ** (1.0 / 3.0)
        cprint('auto voxel_size={:.4f} m for ~{} voxels'.format(
            voxel_size, args.target_voxels), 'cyan')
    grid_dims = np.ceil((hi - lo) / voxel_size).astype(int)
    n_grid_voxels = int(grid_dims.prod())


    # ---- voxelize -----------------------------------------------------------
    methods = ['numpy', 'open3d'] if args.method == 'both' else [args.method]
    vox_results = {}
    os.makedirs(args.out_dir, exist_ok=True)
    frame_tag = '{}-{}'.format(frame, frame + n_frames - 1)
    for method in methods:
        t0 = time.time()
        vox = voxelize_numpy(xyz, rgb, lo, hi, voxel_size) \
            if method == 'numpy' else voxelize_open3d(xyz, rgb, voxel_size)
        dt = time.time() - t0
        if vox is None:
            cprint('{} voxelization produced no voxels'.format(method), 'red')
            continue
        vox_results[method] = vox
        stats = '{} voxelization: {} voxels in {:.3f}s'.format(
            method, len(vox['centers']), dt)
        if vox['counts'] is not None:
            stats += ' | occupancy max={} mean={:.1f}'.format(
                vox['counts'].max(), vox['counts'].mean())
        stats += ' | grid filled {:.1%} of {}'.format(
            len(vox['centers']) / n_grid_voxels, n_grid_voxels)
        cprint(stats, 'green')

        # save npz
        out_npz = os.path.join(
            args.out_dir,
            'voxel_result_frame{}_kernel_{}.npz'.format(frame_tag, method))
        np.savez_compressed(
            out_npz,
            centers=vox['centers'],
            counts=vox['counts'] if vox['counts'] is not None
            else np.array([]),
            avg_rgb=vox['avg_rgb'] if vox['avg_rgb'] is not None
            else np.array([]),
            voxel_size=np.array([voxel_size]),
            workspace_lo=lo, workspace_hi=hi, frame=np.array([frame]))
        cprint('NPZ saved to {}'.format(out_npz), 'green')



    # ---- visualize ----------------------------------------------------------
    if not args.skip_png:
        for method, vox in vox_results.items():
            out_png = os.path.join(
                args.out_dir,
                'pointcloud_voxelized_frame{}_kernel_{}.png'.format(
                    frame_tag, method))
            save_matplotlib_png(disp_pc, vox, out_png, lo, hi, frame,
                                title_suffix=disp_note)

    if not args.skip_html and vox_results:
        out_html = os.path.join(
            args.out_dir,
            'pointcloud_voxelized_frame{}.html'.format(frame_tag))
        save_plotly_html(disp_pc, vox_results, out_html, lo, hi, frame,
                         orig_note=disp_note)

    # ---- summary json --------------------------------------------------------
    summary = {
        'zarr_path': args.zarr_path, 'frame': frame,
        'n_frames': n_frames, 'which': args.which,
        'workspace': {'lo': lo.tolist(), 'hi': hi.tolist(), 'source': ws_src},
        'voxel_size': voxel_size,
        'grid_dims': grid_dims.tolist(),
        'n_grid_voxels': n_grid_voxels,
        'n_original_points': int(len(pc)),
        'methods': {m: {'n_voxels': int(len(v['centers'])),
                        'occupancy_rate': float(len(v['centers']) / n_grid_voxels),
                        'n_dropped_outside': int(v['n_dropped_outside'])}
                    for m, v in vox_results.items()},
    }
    out_json = os.path.join(
        args.out_dir, 'voxel_summary_frame{}.json'.format(frame_tag))
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    cprint('summary saved to {}'.format(out_json), 'green')
    cprint('done.', 'green')


if __name__ == '__main__':
    main()
