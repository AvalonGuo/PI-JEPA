#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: 动态干扰物 —— kinematic 球, **每决策步瞬现**在可放置绿区(可悬空)。

语义(与用户确认的 spec 一致):
  * **尺寸 = episode 级**: 每 episode 从离散集合 R_SET 抽一次半径 r, 整段 episode 不变。
  * **位置/高度 = 决策步级**: 每个决策步(策略一次输出 n 个动作, DP3/JEPA 均
    n_action_steps=8) 球**直接出现**在绿区某处(不平滑移动, 子步期间静止); 可**悬空**。
  * **不受重力 / 不与物理交互**: mocap(kinematic, 无 joint ⇒ nq/nv 不变, 不影响
    24 维 agent_pos) 且 geom 设 `contype=0 conaffinity=0` ⇒ 零接触,
    球可穿过机械手与桌面, **只影响点云/图像**, 不影响动力学。
  * **约束**: 不许穿桌 ⇒ 球心 `h >= r`; 要求**球完整可见**(进入策略点云/图像且不被
    裁切)。判据按策略相机分两种模式(见下), 因此"给定半径 r 的可用放置集合"随 r
    变化: 远边格/高处的球在大球下不合格, 会被剔除。
  * **可见性判据模式**(由 task_name 对应的真实策略配置 ENV_POINT_CLOUD_CONFIG 决定,
    见 policy_cloud_config):
      - `mode=top`(hammer/door, 策略相机 = 顶视 `top`): 闭式
            max(|x|,|y|) + r <= tan_eff * (cam_z - h)
        (tan_eff = tan(fovy/2) 再缩 VIS_MARGIN_PX=1 像素, 防止球正好切在画幅边缘)
      - `mode=frustum`(pen, 策略相机 = 斜视 `vil_camera`): 顶视公式不成立(相机不在
        正上方), 改用 cam_geom 的 AABB 8 角点判据 —— 球外接立方体的 8 个角点必须
        同时 (1) 通过点云 crop(严格不等式, 含 transform/scale/offset) (2) 落在相机
        视锥内。可行 (x,y,h) = 候选格 × 高度网格逐点扫描(与 dynamic_sphere_mw 同思路)。
        该判据已用真实渲染逐像素验证: 见 test/verify_pen_dynsphere_vis.py
        (false-positive 必须为 0)。

可复现性: 每个 (seed, ep, 通道) 独立派生
    rng = numpy.RandomState(mix(seed, ep, channel))
  通道: -1 = 本 episode 半径; 0,1,2,... = 第 k 次放置的 (x, y, h)。
  ⇒ 序列与 rollout 长度/调用顺序无关, 同一 (seed, ep) 必然得到同一批样本。

首帧含球: `start_episode()` 必须在 `env.reset()` **之前**调用 —— 它写
    model.body_pos[body] / model.geom_size[geom] / data.mocap_pos
  (mj_envs 的 reset_model() 会先 sim.reset(), 而 mj_reset 把 model.body_pos 复制到
   data.mocap_pos) ⇒ reset 后渲染的首帧即包含球。
"""
import numpy as np

R_SET = (0.06, 0.05, 0.04, 0.03, 0.02)   # m, 离散球半径(按最大 0.06 建模型)
DEFAULT_NAME = "dr_sphere"
FALLBACK_CAM_Z, FALLBACK_FOVY = 1.1, 45.0  # 渲染相机 top 的兜底几何
IMG_SIZE = 84              # 策略观测点云/深度图边长(mjpc_wrapper + ENV_POINT_CLOUD_CONFIG)
VIS_MARGIN_PX = 1.0        # "球完整可见"额外留 1 像素余量(避免球正好切在画幅边缘)

# --- 可见性判据的两种模式 ---------------------------------------------------
#   top     : 策略相机是顶视 'top' -> 闭式判据 max(|x|,|y|)+r <= tan_eff*(cam_z-h)
#             (hammer/door 的历史行为, 本轮改动保持逐值不变)
#   frustum : 策略相机不是 'top'(如 pen 的斜视 vil_camera) -> 顶视公式不成立;
#             改用 cam_geom 的 AABB 8 角点判据: crop(严格不等式, 含 transform/
#             scale/offset) + 针孔视锥, 两者都过才算"整球进入点云/图像"。
#             可行放置点 = 候选格 × 高度网格逐点扫描的结果(与 dynamic_sphere_mw
#             同思路, 但用 cam_geom 的通用几何)。
# 模式由 task_name 对应的**真实**策略配置决定(ENV_POINT_CLOUD_CONFIG), 不做猜测。
H_GRID_DEFAULT = np.arange(0.02, 0.62 + 1e-9, 0.02)   # frustum 模式的高度扫描网格


def policy_cloud_config(task_name):
    """task_name -> 策略点云配置(cam_names/min|max_bound/transform/scale/offset)。

    判据必须与实际观测链完全一致, 所以直接读 ENV_POINT_CLOUD_CONFIG['adroit_<task>'];
    读不到(非 adroit 任务/无该键/依赖缺失)返回 None(= 维持旧行为)。
    """
    try:
        from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import (
            ENV_POINT_CLOUD_CONFIG as C)
    except Exception:
        return None
    key = f"adroit_{task_name}"
    if key not in C:
        return None
    c = C[key]
    lo = c.get("min_bound", None)
    hi = c.get("max_bound", None)
    return dict(
        cam_names=[str(n) for n in (c.get("cam_names") or ["top"])],
        min_bound=(np.full(3, -np.inf) if lo is None else np.asarray(lo, float)),
        max_bound=(np.full(3, np.inf) if hi is None else np.asarray(hi, float)),
        transform=c.get("transform", None),
        scale=c.get("scale", None),
        offset=c.get("offset", None),
        img=IMG_SIZE,
    )


def unwrap_sim(env):
    """沿 gym wrapper 链找到 MuJoCo sim。"""
    e = env
    for _ in range(12):
        if hasattr(e, "get_mujoco_sim"):
            return e.get_mujoco_sim()
        if hasattr(e, "pc_generator"):
            return e.pc_generator.sim
        if hasattr(e, "sim") and not hasattr(e, "env"):
            return e.sim
        if not hasattr(e, "env"):
            break
        e = e.env
    raise AttributeError("unwrap_sim: no mujoco sim found in env chain")


class DynamicDistractor:
    """每 episode 固定半径; 每决策步把 kinematic 球瞬现到绿区(可悬空, 零接触)。"""

    def __init__(self, sim, cells, name=DEFAULT_NAME, r_set=R_SET, verbose=True,
                 task_name=None, h_grid=None):
        self.sim = sim
        self.cells = np.asarray(cells, dtype=float)
        if len(self.cells) == 0:
            raise ValueError("DynamicDistractor: empty green-cell list")
        self.r_set = tuple(float(r) for r in r_set)
        self.name = name
        self.task_name = task_name
        self.body_id = int(sim.model.body_name2id(name))
        self.mid = int(sim.model.body_mocapid[self.body_id])
        if self.mid < 0:
            raise ValueError(f"{name} 不是 mocap body(mocap_pos 不可用)")
        # 按 body 子树反查 geom(资产的 worldbody 是 <body name="object"><geom/></body>,
        # 注入后会成为 dr_<name> 的嵌套子 body, 所以不能只看直接父 body)
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
            raise ValueError(f"{name} body 子树下没有 geom")
        self.geom_id = int(self.geom_ids[0])
        # ---- 可见性判据模式: 由策略配置决定(见模块头) --------------------------
        cfg = policy_cloud_config(task_name) if task_name is not None else None
        self.mode = "frustum" if (cfg is not None and
                                  "top" not in cfg["cam_names"]) else "top"
        self.geom_info = []          # frustum: 每台策略相机的 cam_geom.geometry
        self.crop = None             # frustum: (min_bound, max_bound, T, scale, offset)
        self.h_grid = np.asarray(h_grid if h_grid is not None else H_GRID_DEFAULT,
                                 dtype=float)
        self._fx = np.zeros(0)       # frustum: 本 episode 半径下的可行放置点
        self._fy = np.zeros(0)
        self._fh = np.zeros(0)
        if self.mode == "frustum":
            import cam_geom as _cg
            self.geom_info = [_cg.geometry(sim.model, nm, img=cfg["img"])
                              for nm in cfg["cam_names"]]
            self.crop = (cfg["min_bound"], cfg["max_bound"], cfg["transform"],
                         cfg["scale"], cfg["offset"])
            self.cam_z, self.tan_half, self.tan_eff = None, None, None
            _cams = [(gi["cam"], np.round(gi["pos"], 3).tolist(), gi["fovy"])
                     for gi in self.geom_info]
            self._vis_desc = (f"mode=frustum cams={_cams} "
                              f"crop=[{np.asarray(self.crop[0]).tolist()},"
                              f"{np.asarray(self.crop[1]).tolist()}] "
                              f"h_grid={self.h_grid[0]:.2f}..{self.h_grid[-1]:.2f}"
                              f"(-{VIS_MARGIN_PX:g}px)")
        else:
            # 渲染相机(top)的可见性几何: 方形画幅, 半宽(depth) = depth * tan(fovy/2)
            self.cam_z, self.tan_half = FALLBACK_CAM_Z, float(
                np.tan(np.radians(FALLBACK_FOVY) / 2.0))
            cid = int(sim.model.camera_name2id("top"))
            if cid >= 0:                    # camera_name2id 找不到时返回 -1, 取 [-1]
                try:                        # 会静默拿到最后一台相机 -> 必须挡住
                    self.cam_z = float(sim.model.cam_pos[cid][2]) or FALLBACK_CAM_Z
                    self.tan_half = float(np.tan(
                        np.radians(float(sim.model.cam_fovy[cid])) / 2.0))
                except Exception:
                    pass
            self.tan_eff = self.tan_half * (1.0 - VIS_MARGIN_PX / (IMG_SIZE / 2.0))
            self._vis_desc = (f"mode=top cam_z={self.cam_z:.3f} "
                              f"tan_half={self.tan_half:.4f} "
                              f"tan_eff={self.tan_eff:.4f} "
                              f"(-{VIS_MARGIN_PX:g}px)")
        # 模式相关实现在此**一次性绑定**, 调用点不再判断模式
        if self.mode == "frustum":
            self._placements_impl = self._placements_frustum
            self._place_impl = self._place_frustum
            self._prepare_episode = self._prepare_episode_frustum
        else:
            self._placements_impl = self._placements_top
            self._place_impl = self._place_top
            self._prepare_episode = self._prepare_episode_top
        self.seed = None
        self.ep = None
        self.r = None              # 本 episode 的半径(episode 级, 固定)
        self.cells_r = None        # 该半径下满足"不穿桌 + 完整可见"的绿区格
        self.k = 0                 # 已放置次数(0 = episode 起始位)
        self.records = []          # [(k, x, y, h, r), ...]
        self.pos = None            # 当前位置 (x, y, h)
        if verbose:
            print(f"[dynamic] {name}: body={self.body_id} mocap_id={self.mid} "
                  f"geom={self.geom_id} cells={len(self.cells)} r_set={self.r_set} "
                  f"{self._vis_desc}")

    # ---------------- geometry / sampling (deterministic) ----------------
    def visible_hmax(self, x, y, r):
        """[top 模式] 球**完整可见**不超出画幅的最大球心高度:
        max|.| + r <= tan_eff*(cam_z-h)。

        ★仅顶视相机成立; frustum 模式(斜视相机)请用 placements()/_aabb_visible。
        支持标量与数组(数组用于一次性筛出该半径下的可行格)。
        """
        xy = np.maximum(np.abs(np.asarray(x, dtype=float)),
                        np.abs(np.asarray(y, dtype=float)))
        return self.cam_z - (xy + float(r)) / self.tan_eff

    def _aabb_visible(self, xs, ys, h, r):
        """[frustum 模式] (xs,ys) 处 AABB 整块可见的 bool 数组。

        crop(严格不等式) + 针孔视锥, 逐角点判定; 多相机取并(球出现在任一相机的
        点云里即可)。几何全部来自 cam_geom(已用真实渲染验证)。
        """
        import cam_geom as _cg
        lo, hi, T, sc, off = self.crop
        ok = np.zeros(len(np.asarray(xs, float)), dtype=bool)
        for gi in self.geom_info:
            ok |= _cg.aabb_visible(xs, ys, float(h), float(r), gi, lo, hi, T, sc, off,
                                   margin_px=VIS_MARGIN_PX)
        return ok

    def _build_placements(self, r):
        """[frustum 模式] 扫描 (候选格 × 高度网格) 得本半径下的可行放置点; 返回数量。"""
        fx, fy, fh = [], [], []
        for h in self.h_grid:
            if float(h) < float(r) - 1e-9:
                continue                          # 球心不低于半径 => 球不穿桌
            m = self._aabb_visible(self.cells[:, 0], self.cells[:, 1], float(h), r)
            if m.any():
                fx.append(self.cells[m, 0])
                fy.append(self.cells[m, 1])
                fh.append(np.full(int(m.sum()), float(h)))
        self._fx = np.concatenate(fx) if fx else np.zeros(0)
        self._fy = np.concatenate(fy) if fy else np.zeros(0)
        self._fh = np.concatenate(fh) if fh else np.zeros(0)
        self.cells_r = (np.unique(np.stack([self._fx, self._fy], axis=1), axis=0)
                        if len(self._fx) else np.zeros((0, 2)))
        return int(len(self._fx))

    def _placements_frustum(self):
        """[frustum] 本 episode 半径下的 (M,3) 可行放置点 (x, y, h); 空则 (0,3)。"""
        if not len(self._fx):
            return np.zeros((0, 3))
        return np.stack([self._fx, self._fy, self._fh], axis=1)

    def _placements_top(self):
        """[top] 本模式是"格 + 高度区间"的闭式表达, 无离散放置点集 => (0,3)。"""
        return np.zeros((0, 3))

    def placements(self):
        """本 episode 的可行放置点(实现按模式在 __init__ 绑定)。"""
        return self._placements_impl()

    def _rng(self, channel):
        """逐 (seed, ep, channel) 的确定性 RNG(channel=-1 用于 episode 半径)。

        不使用 RandomState([seed, ep, k]): numpy 旧式播种不接受负数/超范围。
        用乘法混淆 + 取模保证 [0, 2**32) 且各通道互不碰撞。
        """
        s = (int(self.seed) * 2654435761
             + int(self.ep) * 1000003
             + (int(channel) + 2) * 7919) % (2 ** 32)
        return np.random.RandomState(s)

    def _record_place(self, x, y, h, channel, model_default):
        """写入位形 + 记账(两种模式共用)。"""
        self._write(float(x), float(y), h, model_default)
        self.pos = (float(x), float(y), float(h))
        self.records.append((int(channel), float(x), float(y), float(h),
                             float(self.r)))
        return (float(x), float(y), float(h), float(self.r))

    def _place_frustum(self, channel, model_default=False):
        """[frustum] 在已扫描出的可行放置点 (x,y,h) 中确定性抽一个。"""
        rng = self._rng(channel)
        i = int(rng.randint(len(self._fx)))
        return self._record_place(float(self._fx[i]), float(self._fy[i]),
                                  float(self._fh[i]), channel, model_default)

    def _place_top(self, channel, model_default=False):
        """[top] 在本 episode 半径下可行的绿区里抽一格, 高度 h ∈ [r, h_max]。"""
        rng = self._rng(channel)
        x, y = self.cells_r[rng.randint(len(self.cells_r))]
        hmax = self.visible_hmax(x, y, self.r)
        h = float(rng.uniform(self.r, max(self.r, hmax)))    # h>=r: 球底不穿桌
        return self._record_place(float(x), float(y), h, channel, model_default)

    def _place(self, channel, model_default=False):
        """瞬现到下一个可行位置(实现按模式在 __init__ 绑定)。"""
        return self._place_impl(channel, model_default)

    def _write(self, x, y, h, model_default=False):
        self.sim.data.mocap_pos[self.mid] = [x, y, h]        # 球心 (x, y, h), h>=r
        self.sim.model.geom_size[self.geom_id, 0] = self.r   # episode 级半径
        if model_default:                                    # 供 mj_reset 复位用
            self.sim.model.body_pos[self.body_id] = [x, y, h]
        self.sim.forward()

    # ---------------- episode / decision hooks ----------------
    def _prepare_episode_frustum(self):
        """[frustum] 扫描本半径下的可行放置点; 空则退化为最靠桌心的一格。"""
        if self._build_placements(self.r) == 0:
            i = int(np.argmin(np.max(np.abs(self.cells), axis=1)))
            c = self.cells[i]
            self.cells_r = self.cells[[i]]
            self._fx = np.array([float(c[0])])
            self._fy = np.array([float(c[1])])
            self._fh = np.array([max(self.r, 0.05)])

    def _prepare_episode_top(self):
        """[top] 闭式筛出"不穿桌 + 球完整可见"的格; 空则退化为最靠中心的一格。"""
        hmax = self.visible_hmax(self.cells[:, 0], self.cells[:, 1], self.r)
        feasible = hmax >= self.r              # 不穿桌 + 球完整可见
        self.cells_r = self.cells[feasible]
        if len(self.cells_r) == 0:             # 兜底: 用最靠中心的一格
            i = int(np.argmin(np.max(np.abs(self.cells), axis=1)))
            self.cells_r = self.cells[[i]]

    def start_episode(self, seed, ep):
        """episode 开始(须在 env.reset() 之前调用): 抽 episode 级半径 + 放初始球。"""
        self.seed, self.ep = int(seed), int(ep)
        self.records = []
        self.k = 0
        self.r = float(self.r_set[self._rng(-1).randint(len(self.r_set))])
        self._prepare_episode()                    # 按模式在 __init__ 绑定
        return self._place(self.k, model_default=True)

    def plan(self):
        """决策边界: 瞬现到下一个绿区位置(高度在该格允许范围内随机)。"""
        self.k += 1
        return self._place(self.k)

    def finish(self):
        """瞬现模式无"未到位"状态, 保留空实现以兼容 runner 调用。"""
        return None

    def tick(self, t):
        """瞬现模式无逐子步更新(球在每个**决策步**瞬现) —— 仅为与 dynhand 统一接口。"""
        return self.pos

    def reapply(self):
        """瞬现模式: 位置由 model.body_pos 承载, mj_reset 会自动复位 ⇒ 无需重写。"""
        return 0.0


    # ---------------- reporting ----------------
    def history(self):
        return [list(r) for r in self.records]
