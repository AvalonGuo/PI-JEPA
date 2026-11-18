#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d suite: the single implementation (plan / per-episode scene + distractor / hooks).

Shared by AdroitRunner (DP3) / MetaworldRunner (DP3), replacing the per-suite
branches each of them used to carry: one semantics, written once -- fix it here
and all callers are fixed.

  catgen    : fixed P0, random type per episode (cube/can/bottle/cereal/mug)
  posgen    : 'cube' on a random green cell each episode
  dynsphere : kinematic sphere (adroit=mocap / mw=static body); radius once per
              episode, position teleported every decision step
  dynhand   : kinematic inspire left hand, smooth ping-pong per substep

family='adroit'   : scene_runtime private shim (third_party read-only) + in-place
                    swap of this task's base XML
family='metaworld': compose.ensure_scene(base=<this task's base scene>, free=False)
                    + the env is rebuilt from the composed MJCF

Callers only provide the "how to rebuild the env" callback; everything else (plan /
green-cell checks / scene composition / distractor construction / per-substep hooks)
lives in this class. Distance semantics (radius / path / visibility) stay in
dynamic*.py -- no numeric decisions are made here.
"""
import os

SUITES = ("catgen", "posgen", "dynsphere", "dynhand")
FAMILIES = ("adroit", "metaworld")
P_KEYS = ("type", "pos", "source")
_QUIET = os.environ.get("DR3D_QUIET", "0") == "1"

class Suite:
    """Full run logic for one suite (name + task + family + seed + n)."""

    def __init__(self, name, task, seed, n_episodes, family="adroit", scale=None,
                 verbose=True):
        if name not in SUITES:
            raise ValueError(f"suite: unknown name {name!r} (available: {SUITES})")
        if family not in FAMILIES:
            raise ValueError(f"suite: unknown family {family!r} "
                             f"(available: {FAMILIES})")
        self.name = str(name)
        self.task = str(task)
        self.family = str(family)
        self.mw = (self.family == "metaworld")
        self.seed_raw = seed
        self.seed = 0 if seed is None else int(seed)
        self.n = int(n_episodes)
        self.scale = scale
        self.verbose = bool(verbose) and not _QUIET
        self.plan = []
        self.ep_types = [None] * self.n
        self.ep_pos = [None] * self.n
        self.ep_size = [None] * self.n
        self.ep_hand = [None] * self.n
        self.cells = None
        self.cells_source = None
        self._scene = {}
        self._plan()

    def _plan(self):
        import sample as _S
        self._S = _S
        if self.name == "catgen":
            _, seq = _S.sample_run(self.seed, self.n)
            self.ep_types = list(seq)
            self.plan = [dict(type=t, pos=None, source="p0") for t in self.ep_types]
            return
        self._load_cells()
        if self.name == "posgen":
            pos = _S.sample_green_pos(self.seed, self.n, self.cells)
            self.ep_types = ["cube"] * self.n
            self.ep_pos = [[float(x), float(y)] for x, y in pos]
            self.plan = [dict(type="cube", pos=p, source=self.cells_source)
                         for p in self.ep_pos]
        else:
            self.ep_types = ["sphere"] * self.n if self.name == "dynsphere" \
                else ["inspire"] * self.n
            self.plan = [dict(type=self.ep_types[0], pos=None,
                              source=self.cells_source) for _ in range(self.n)]

    def _load_cells(self):
        v = self.verbose
        if self.mw:
            import green_cells_metaworld as _gcm
            if self.name in ("dynsphere", "dynhand"):
                # dynamic-suite candidates: data-driven = in-crop AND not in the task
                # footprint
                self.cells = _gcm.cells_for(self.task, self.name)
                self.cells_source = _gcm.cells_source(self.task, self.name)
            else:
                # posgen: green cells per task (assembly=legacy analytic;
                # stick-push/disassemble=data-driven from their own zarr). Details /
                # cache / missing-data hints: dr3d/bench/green_cells_metaworld.py.
                self.cells = _gcm.load_cells(self.task, verbose=v)
                self.cells_source = _gcm.cells_source(self.task)
            return
        import green_cells as _gc
        if self.name == "dynsphere":
            if _gc.dynsphere_cells_source(self.task) == "green":
                _gc.require_cells(self.task)
            self.cells = _gc.load_dynsphere_cells(task=self.task, verbose=v)
            self.cells_source = _gc.dynsphere_cells_source(self.task)
        else:
            _gc.require_cells(self.task)
            self.cells = _gc.load_green_cells(task=self.task, verbose=v)
            self.cells_source = "green"

    def _compose(self, type_name, pos=None):
        """metaworld: compose this task's base scene (free=False static body), cached
        for the process lifetime.

        The base comes from the task (compose.mw_scene): assembly -> 'assembly',
        disassemble -> 'assembly' (same MJCF), stick-push -> 'stick_push'. Never evaluate
        task A with task B's XML: sawyer_xyz_env hardcodes the manipulated object at
        qpos[9:12] (stick-push also keeps the stick at qpos[16:18]), so a wrong base
        corrupts the arm reset.
        """
        import compose as _C
        base, p0 = _C.mw_scene(self.task)
        key = (base, type_name,
               None if pos is None else (round(pos[0], 6), round(pos[1], 6)))
        if key not in self._scene:
            path, sha, fresh = _C.ensure_scene(
                type_name, p0 if pos is None else pos, free=False, base=base)
            self._scene[key] = (path, sha, fresh)
        return self._scene[key]

    def _mw_distractor(self, env):
        if self.name == "dynsphere":
            import dynamic_sphere_mw as _dynmw
            import dynamic as _dynlib
            import green_cells_metaworld as _gcm
            pcT, cmin, cmax, imgsz = _dynmw.find_pc_constants(env)
            vox, smask = _gcm.dyn_volumes(self.task)     # legacy tasks -> (None, None)
            dyn = _dynmw.DynamicSphereMW(_dynlib.unwrap_sim(env), self.cells,
                                         pcT, cmin, cmax, image_size=imgsz,
                                         free=vox, screen_mask=smask)
            if self.verbose:
                print(f"[suite:dynsphere] cells={len(self.cells)} "
                      f"({self.cells_source}) 3d_occ={'on' if vox is not None else 'off'} "
                      f"screen_guard={'on' if smask is not None else 'off'}")
            return dyn
        import dynamic_hand_mw as _dynh
        import green_cells_metaworld as _gcm
        pcT, cmin, cmax, imgsz = _dynh.find_pc_constants(env)
        vox, smask = _gcm.dyn_volumes(self.task)     # legacy tasks -> (None, None)
        work = tcenter = None
        if vox is not None and vox.get("work_dist") is not None:
            work = dict(dist=vox["work_dist"], lo=vox["work_lo"],
                        gs=float(vox["gs"]),
                        margin=float(vox.get("work_margin", 0.03)))
            tcenter = vox["task_center"]
        # occlusion criterion (optional): 3D occupancy union back-projected from the
        # zarr depth + the task object's screen footprint. Both default to None =>
        # purely analytic visibility, bit-identical to the previous behaviour.
        occ = None
        if vox is not None and vox.get("occ") is not None:
            occ = dict(occ=vox["occ"], lo=vox["lo"], gs=float(vox["gs"]))
        dyn = _dynh.HandDistractorMW(_dynh.unwrap_sim(env), self.cells,
                                     pcT, cmin, cmax, image_size=imgsz,
                                     work=work, task_center=tcenter,
                                     occ=occ, screen_mask=smask)
        if self.verbose:
            print(f"[suite:dynhand] cells={len(self.cells)} ({self.cells_source}) "
                  f"work={'zarr-footprint' if work is not None else 'assembly-const'} "
                  f"vis={dyn.vis_mode} "
                  f"occ_splat={dyn.occ_splat} "
                  f"obj_guard={'on' if dyn.obj_dist is not None else 'off'} "
                  f"task_center="
                  f"{None if tcenter is None else [round(float(v), 3) for v in tcenter]}")
        return dyn

    def _adroit_distractor(self, env):
        if self.name == "dynsphere":
            import dynamic as _dynlib
            return _dynlib.DynamicDistractor(_dynlib.unwrap_sim(env), self.cells,
                                             task_name=self.task,
                                             verbose=self.verbose)
        import dynamic_hand as _dh
        scale = _dh.HAND_SCALE if self.scale is None else float(self.scale)
        return _dh.HandDistractor(_dh.unwrap_sim(env), self.cells, scale=scale,
                                  task_name=self.task, verbose=self.verbose)

    def begin(self, ep, env=None, rebuild_env=None, reuse=None):
        """Set up episode ep: swap the scene (+rebuild env) / build and reset the
        distractors.

        Returns dict(env, dyn, scene, sha, fresh, rebuilt, desc):
          * adroit   : swap this task's base XML in place via scene_runtime, then rebuild;
          * metaworld: compose the task's base scene (see _compose), hand it to
                       rebuild_env(scene_path); `reuse` (dict(path/env/dyn)) keeps a
                       constant scene built only once.
        """
        info = self.plan[ep]
        rebuilt = False
        if self.mw:
            if self.name == "catgen":
                type_name, pos = info["type"], None
            elif self.name == "posgen":
                type_name, pos = "cube", tuple(info["pos"])
            else:
                type_name = "sphere" if self.name == "dynsphere" else "inspire"
                pos = None
            scene, sha, fresh = self._compose(type_name, pos)
            if reuse is None:
                env, rebuilt = rebuild_env(scene), True
            elif reuse.get("path") != str(scene) or reuse.get("env") is None:
                env, rebuilt = rebuild_env(scene), True
                reuse.update(path=str(scene), env=env, dyn=None)
            else:
                env = reuse["env"]
            dyn = None
            if self.name in ("dynsphere", "dynhand"):
                dyn = (reuse or {}).get("dyn") if reuse is not None else None
                if dyn is None:
                    dyn = self._mw_distractor(env)
                    if reuse is not None:
                        reuse["dyn"] = dyn
                    rebuilt = True
                dyn.start_episode(self.seed, ep)
        else:
            import scene_runtime as _SR
            if self.name == "catgen":
                _SR.activate(info["type"], task=self.task)
            elif self.name == "posgen":
                _SR.activate("cube", tuple(info["pos"]), task=self.task)
            else:
                _SR.activate("sphere" if self.name == "dynsphere" else "inspire",
                             None, mocap=True, task=self.task, **(
                                 {} if self.scale is None else {"scale": self.scale}))
            env, rebuilt, scene, sha, fresh = rebuild_env(None), True, None, None, None
            dyn = None
            if self.name in ("dynsphere", "dynhand"):
                dyn = self._adroit_distractor(env)
                dyn.start_episode(self.seed, ep)
        self.ep_types[ep] = info["type"]
        if self.name == "posgen":
            self.ep_pos[ep] = [float(v) for v in info["pos"]]
        if dyn is not None and self.name == "dynsphere":
            self.ep_size[ep] = float(dyn.r)
        if dyn is not None and self.name == "dynhand":
            self.ep_hand[ep] = dyn.spec()
        desc = (f"type={info['type']}" if self.name == "catgen" else
                f"{info['type']}@{tuple(round(float(v), 3) for v in info['pos'])}"
                if self.name == "posgen" else
                f"{info['type']} r={getattr(dyn, 'r', None)}" if self.name == "dynsphere"
                else f"{info['type']} z={getattr(dyn, 'z', None)}")
        return dict(env=env, dyn=dyn, scene=scene, sha=sha, fresh=fresh,
                    rebuilt=rebuilt, desc=desc, type=info["type"],
                    pos=info["pos"], source=info.get("source"))

    @staticmethod
    def after_reset(dyn, env=None):
        """Rewrite the distractor pose right after env.reset() (defensive);
        no-op when dyn is None."""
        if dyn is not None and hasattr(dyn, "reapply"):
            return dyn.reapply()
        return None

    @staticmethod
    def on_decision(dyn):
        return None if dyn is None else dyn.plan()

    @staticmethod
    def on_substep(dyn, t):
        return None if dyn is None else dyn.tick(t)

    @staticmethod
    def end(dyn):
        return None if dyn is None else dyn.finish()

    def summary(self):
        if self.name == "catgen":
            return f"types={self.ep_types[:8]}... (seed={self.seed})"
        return (f"{len(self.cells)} cells (source={self.cells_source}), "
                f"seed={self.seed}")

    def manifest_extra(self):
        """Common manifest extras (suite/task/family/source); existing keys unchanged."""
        return dict(suite=self.name, task=self.task, family=self.family,
                    n_cells=(None if self.cells is None else int(len(self.cells))),
                    cells_source=self.cells_source)
