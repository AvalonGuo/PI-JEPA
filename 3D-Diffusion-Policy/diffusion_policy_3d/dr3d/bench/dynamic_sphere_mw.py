#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: metaworld dynamic distractor -- a static kinematic sphere teleported
(possibly floating) at every decision step.

Same semantics as adroit's dynamic.DynamicDistractor (episode-level radius, decision-level
position/height, zero contact, point cloud / image only), different driver and visibility test:

  * Driver: adroit uses a mocap body (data.mocap_pos); metaworld MUST use a static body +
    model.body_pos. Reason (hard constraint): sawyer_xyz_env.set_xyz_action reads the whole
    data.mocap_pos and calls set_mocap_pos('mocap', ...), so a second mocap body makes every
    step() raise "could not broadcast (2,3) into (3,)"; a freejoint would shift qpos[9:12].
    A static body keeps nq/nv/nmocap untouched -- the only non-invasive option.

  * Visibility: corner2 is an oblique camera (adroit's top-down formula does not apply) and the
    crop z upper bound is 100 (crop alone lets the sphere float metres up). All 8 corners of the
    sphere AABB (x±r, y±r, h±r) must pass both:
        (1) crop affine: cloud = w @ A + c inside [min_bound, max_bound];
        (2) corner2 frustum: cam = (w-cam_pos) @ Rcam with z_cam<0 and pixel
            u,v in [0, img-1] (f = (img/2)/tan(fovy/2), u = c + f*x_cam/(-z_cam),
            c = (img-1)/2 -- same pixel convention as the green_cells_metaworld mask).
    Only a fully visible sphere counts, so it always reaches the render / point cloud uncropped.
    Placements are scanned per radius r over (candidate cells x height grid), sampled
    deterministically.

  * Optional data-driven constraints (omitted = legacy behaviour, assembly baseline unchanged):
      - free=dict(occ,lo,gs): the sphere AABB must not intersect the 3D occupancy
        back-projected from the zarr depth (arm + task object, 0.03 voxels, dilated by 1)
        => the sphere never sits inside the arm / on the object (~17-24% did before);
      - screen_mask=(img,img) bool: task-object screen footprint built by
        green_cells_metaworld from the per-task spec (stick-push: one disc per stick end +
        the thermos; disassemble: nut ring + wrench handle + static peg); the sphere's screen
        disc must not cover it => the sphere never hides the manipulated object;
      - vis_frac=(n_cells,n_heights,n_radii) float: fraction of data frames whose sphere-centre
        pixel is unoccluded; keep only placements >= min_vis_frac (default 0.5)
        => the sphere does not stay behind the arm (~13% rendered no points without it).
      All three come from dr3d/bench/green_cells_metaworld.dyn_volumes(task) (data only, no render).

Reproducibility: reuses dynamic.py's mix(seed, ep, channel) RNG.
"""
import pathlib
import sys

import numpy as np

_BENCH = pathlib.Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from dynamic import R_SET, unwrap_sim   # noqa: E402  radius set + sim unwrap
from memlog import once                  # noqa: E402  logging helpers

DEFAULT_NAME = "dr_sphere"
_CAM = "corner2"
_H_MAX = 0.80        # max scan height (m)
_H_STEP = 0.02


def find_pc_constants(env):
    """-> (pc_transform, min_bound, max_bound, image_size); walks the wrapper chain."""
    e = env
    for _ in range(12):
        if all(hasattr(e, a) for a in ("pc_transform", "min_bound", "max_bound")):
            return (np.asarray(e.pc_transform, float),
                    np.asarray(e.min_bound, float),
                    np.asarray(e.max_bound, float),
                    int(getattr(e, "image_size", 128)))
        if not hasattr(e, "env"):
            break
        e = e.env
    raise AttributeError("find_pc_constants: pc_transform/min_bound/max_bound not found")


class DynamicSphereMW:
    """Fixed radius per episode; teleports the static kinematic sphere to a visible spot
    (floating allowed) at every decision step."""

    def __init__(self, sim, cells, pc_transform, min_bound, max_bound,
                 image_size=128, name=DEFAULT_NAME, r_set=R_SET, cam=_CAM,
                 verbose=True, free=None, screen_mask=None,
                 screen_margin_px=4.0):
        self.sim = sim
        self.cells = np.asarray(cells, float)
        if len(self.cells) == 0:
            raise ValueError("DynamicSphereMW: empty candidate-cell list")
        self.r_set = tuple(float(r) for r in r_set)
        self.name = name
        # --- optional data-driven constraints (omitted = legacy behaviour) ------------------
        #   free        : dict(occ=(nx,ny,nz) bool dilated, lo=(3,), gs=float)
        #                 -> the sphere AABB must not hit the 3D occupancy (arm / object)
        #   screen_mask : (img,img) bool = task-object screen footprint
        #                 -> the sphere's screen disc must not cover it
        #   vis_frac    : (n_cells,n_heights,n_radii) float carried by free
        #                 -> per-placement fraction of data frames with an unoccluded centre
        self.free = None
        self.vis = None
        if free is not None:
            self.free = dict(occ=np.asarray(free["occ"], bool),
                             lo=np.asarray(free["lo"], float),
                             gs=float(free["gs"]))
            vf = free.get("vis_frac")
            hts = free.get("heights")
            rds = free.get("radii")
            if vf is not None and hts is not None and rds is not None:
                vf = np.asarray(vf, np.float32)
                hts = np.asarray(hts, float)
                rds = np.asarray(rds, float)
                cells = free.get("cells")
                ok = (cells is not None and vf.ndim == 3 and
                      vf.shape == (len(self.cells), len(hts), len(rds)) and
                      np.allclose(np.asarray(cells, float), self.cells, atol=1e-3))
                if not ok:
                    raise ValueError(
                        "DynamicSphereMW: free['cells']/vis_frac do not match self.cells "
                        "(the cell set must equal green_cells_metaworld.cells_for(task,"
                        "'dynsphere'))")
                self.vis = dict(frac=vf, heights=hts, radii=rds,
                                reset=(None if free.get("vis_reset") is None else
                                       np.asarray(free["vis_reset"], bool)),
                                min_frac=float(free.get("min_vis_frac", 0.0)))
        self.obj_dist = None
        if screen_mask is not None:
            from scipy import ndimage
            self.obj_dist = ndimage.distance_transform_edt(
                ~np.asarray(screen_mask, bool)).astype(np.float32)
        self.screen_margin_px = float(screen_margin_px)
        self.diag = dict(n_aabb=0, n_drop_free=0, n_drop_screen=0, n_final=0)
        self.body_id = int(sim.model.body_name2id(name))
        if int(sim.model.body_mocapid[self.body_id]) >= 0:
            raise ValueError(f"{name} must be a static body (not mocap); "
                             f"this suite cannot take another mocap body")
        # all geoms under the body subtree (dr_<name> embeds <body name='object'><geom/></body>)
        ids = {self.body_id}
        changed = True
        while changed:
            changed = False
            for b in range(sim.model.nbody):
                if b not in ids and int(sim.model.body_parentid[b]) in ids:
                    ids.add(b)
                    changed = True
        self.geom_ids = [g for g in range(sim.model.ngeom)
                         if int(sim.model.geom_bodyid[g]) in ids]
        if not self.geom_ids:
            raise ValueError(f"{name}: no geom under the body subtree")
        self.geom_id = int(self.geom_ids[0])
        # corner2 (world->cam) + crop affine cloud = w @ A + c
        cid = int(sim.model.camera_name2id(cam))
        self.Rcam = np.asarray(sim.model.cam_mat0).reshape(-1, 3, 3)[cid]
        self.cam_pos = np.asarray(sim.model.cam_pos).reshape(-1, 3)[cid]
        pcT = np.asarray(pc_transform, float)
        self.A = self.Rcam @ pcT.T
        self.c = -(self.cam_pos @ self.A)
        self.cmi = np.asarray(min_bound, float)
        self.cma = np.asarray(max_bound, float)
        self.img = int(image_size)
        fovy = float(np.ravel(sim.model.cam_fovy)[cid])
        self.f = (self.img / 2.0) / np.tan(np.radians(fovy) / 2.0)
        # principal point: same convention as green_cells_metaworld._cam (and as the mask built there)
        self.cx = (self.img - 1) / 2.0
        self.HS = np.arange(_H_STEP, _H_MAX + 1e-9, _H_STEP)
        self.seed = 0
        self.ep = 0
        self.k = 0
        self.r = float(self.r_set[0])
        self.pos = None
        self.records = []
        self.cells_r = self.cells
        self._fx = np.zeros(0)
        self._fy = np.zeros(0)
        self._fh = np.zeros(0)
        if verbose:
            once("dynsphere_mw." + str(name),
                 f"[dynsphere_mw] {name}: body={self.body_id} geom={self.geom_id} "
                 f"cells={len(self.cells)} r_set={self.r_set} img={self.img} "
                 f"f={self.f:.1f} crop={self.cmi.tolist()}..{self.cma.tolist()} "
                 f"free={'on' if self.free is not None else 'off'} "
                 f"screen={'on' if self.obj_dist is not None else 'off'} "
                 f"vis_frac={'on' if self.vis is not None else 'off'}")

    def _rng(self, channel):
        s = (int(self.seed) * 2654435761 + int(self.ep) * 1000003
             + (int(channel) + 2) * 7919) % (2 ** 32)
        return np.random.RandomState(s)

    def _aabb_visible(self, xs, ys, h, r):
        """Bool per (xs,ys): the whole AABB (x±r, y±r, h±r) is visible (crop affine + frustum)."""
        xs = np.asarray(xs, float)
        ys = np.asarray(ys, float)
        n = xs.shape[0]
        S = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], float) * r
        w = np.stack([xs[:, None] + S[None, :, 0],
                      ys[:, None] + S[None, :, 1],
                      np.full((n, 8), float(h)) + S[None, :, 2]], axis=2)  # (n,8,3)
        cloud = w @ self.A + self.c                                   # crop affine
        crop_ok = (np.all(cloud >= self.cmi, axis=2) &
                   np.all(cloud <= self.cma, axis=2))
        cam = (w - self.cam_pos) @ self.Rcam                          # corner2 view frustum
        z = cam[:, :, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.where(z < -1e-6, -z, np.nan)
            u = self.cx + self.f * cam[:, :, 0] / depth
            v = self.cx + self.f * cam[:, :, 1] / depth
        frus_ok = (np.isfinite(u) & np.isfinite(v) &
                   (u >= 0.0) & (u <= self.img - 1.0) &
                   (v >= 0.0) & (v <= self.img - 1.0))
        return np.all(crop_ok & frus_ok, axis=1)

    def _hits_free(self, xs, ys, h, r):
        """True where the sphere AABB intersects the 3D occupancy (arm / object, dilated)."""
        occ, lo, gs = self.free["occ"], self.free["lo"], self.free["gs"]
        nx, ny, nz = occ.shape
        iz0 = int(np.clip(np.floor((h - r - lo[2]) / gs), 0, nz - 1))
        iz1 = int(np.clip(np.floor((h + r - lo[2]) / gs), 0, nz - 1))
        out = np.zeros(len(xs), dtype=bool)
        for sx in (-r, 0.0, r):
            ix = np.clip(np.floor((xs + sx - lo[0]) / gs).astype(int), 0, nx - 1)
            for sy in (-r, 0.0, r):
                iy = np.clip(np.floor((ys + sy - lo[1]) / gs).astype(int), 0, ny - 1)
                out |= occ[ix, iy, iz0:iz1 + 1].any(axis=1)
        return out

    def _screen_blocks(self, xs, ys, h, r):
        """True where the sphere's screen disc covers the task-object footprint (screen_mask).

        Pixel convention must match green_cells_metaworld._compute_dyn (which builds the mask):
        u = c + f * x_cam / (-z_cam), v = c + f * y_cam / (-z_cam), c = (img - 1) / 2.
        """
        n = len(xs)
        p = (np.stack([xs, ys, np.full(n, float(h))], axis=1) - self.cam_pos) @ self.Rcam
        depth = -p[:, 2]
        out = np.zeros(n, dtype=bool)
        ok = depth > 1e-6
        with np.errstate(invalid="ignore", divide="ignore"):
            u = self.cx + self.f * p[:, 0] / depth
            v = self.cx + self.f * p[:, 1] / depth
            rpx = self.f * float(r) / np.maximum(depth, 1e-6) + self.screen_margin_px
        ok &= np.isfinite(u) & np.isfinite(v)
        if not ok.any():
            return out
        H, W = self.obj_dist.shape
        cu = np.clip(np.round(u[ok]).astype(int), 0, W - 1)
        cv = np.clip(np.round(v[ok]).astype(int), 0, H - 1)
        out[ok] = self.obj_dist[cv, cu] <= rpx[ok]
        return out

    def _vis_ok(self, h, r):
        """Candidate-cell mask where 'visible fraction >= min_frac and visible at reset'."""
        if self.vis is None:
            return np.ones(len(self.cells), dtype=bool)
        j = int(np.argmin(np.abs(self.vis["heights"] - float(h))))
        k = int(np.argmin(np.abs(self.vis["radii"] - float(r))))
        ok = self.vis["frac"][:, j, k] >= self.vis["min_frac"]
        if self.vis.get("reset") is not None:
            ok = ok & self.vis["reset"][:, j, k]
        return ok

    def _build_placements(self, r):
        """Scan (candidate cells x height grid) for feasible placements (x, y, h); returns count.

        Filters in order: AABB 8 corners in crop and frustum (fixed) -> no 3D occupancy (data)
        -> screen does not cover the task object (data) -> data visibility (data). The last
        three apply only when the caller passes free/screen_mask.
        """
        hs = self.HS[self.HS >= r - 1e-9]
        fx, fy, fh = [], [], []
        n_aabb = n_free = n_screen = n_vis = 0
        for h in hs:
            m = self._aabb_visible(self.cells[:, 0], self.cells[:, 1], float(h), r)
            n_aabb += int(m.sum())
            if self.free is not None:
                drop = m & self._hits_free(self.cells[:, 0], self.cells[:, 1],
                                           float(h), r)
                n_free += int(drop.sum())
                m &= ~drop
            if self.obj_dist is not None:
                drop = m & self._screen_blocks(self.cells[:, 0], self.cells[:, 1],
                                               float(h), r)
                n_screen += int(drop.sum())
                m &= ~drop
            if self.vis is not None:
                drop = m & ~self._vis_ok(float(h), r)
                n_vis += int(drop.sum())
                m &= ~drop
            if m.any():
                fx.append(self.cells[m, 0])
                fy.append(self.cells[m, 1])
                fh.append(np.full(int(m.sum()), float(h)))
        if not fx:
            self._fx = np.zeros(0)
            self._fy = np.zeros(0)
            self._fh = np.zeros(0)
            self.diag = dict(n_aabb=n_aabb, n_drop_free=n_free,
                             n_drop_screen=n_screen, n_drop_vis=n_vis, n_final=0)
            return 0
        self._fx = np.concatenate(fx)
        self._fy = np.concatenate(fy)
        self._fh = np.concatenate(fh)
        self.cells_r = np.unique(np.stack([self._fx, self._fy], axis=1), axis=0)
        self.diag = dict(n_aabb=n_aabb, n_drop_free=n_free,
                         n_drop_screen=n_screen, n_drop_vis=n_vis,
                         n_final=len(self._fx))
        return len(self._fx)

    def _write(self, x, y, h):
        self.sim.model.body_pos[self.body_id] = [x, y, h]     # static body: pose lives in model
        self.sim.model.geom_size[self.geom_id, 0] = self.r    # episode-level radius
        self.sim.forward()

    def _place(self, channel):
        rng = self._rng(channel)
        i = int(rng.randint(len(self._fx)))
        x, y, h = float(self._fx[i]), float(self._fy[i]), float(self._fh[i])
        self._write(x, y, h)
        self.pos = (x, y, h)
        self.records.append((int(channel), x, y, h, float(self.r)))
        return (x, y, h, float(self.r))

    def start_episode(self, seed, ep):
        """Episode start (must run before env.reset()): draw the episode radius, place the sphere."""
        self.seed, self.ep = int(seed), int(ep)
        self.records = []
        self.k = 0
        self.r = float(self.r_set[self._rng(-1).randint(len(self.r_set))])
        if self._build_placements(self.r) == 0:       # fallback: table centre, lowest visible h
            self.cells_r = np.array([[0.0, 0.6]])
            self._fx = np.array([0.0])
            self._fy = np.array([0.6])
            self._fh = np.array([max(self.r, 0.05)])
        return self._place(self.k)

    def plan(self):
        """Decision boundary: teleport to the next visible placement (any feasible height)."""
        self.k += 1
        return self._place(self.k)

    def finish(self):
        return None

    def tick(self, t):
        return self.pos

    def reapply(self):
        """The static body pose lives in model.body_pos and reset does not touch it -> no-op."""
        return 0.0

    def history(self):
        return [list(r) for r in self.records]
