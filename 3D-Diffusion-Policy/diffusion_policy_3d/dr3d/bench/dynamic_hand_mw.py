#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: metaworld 动态干扰物 —— inspire 左手(静态根 + 12 手指 hinge),
逐 substep 在绿区线段上**平滑乒乓**。语义与 adroit 的
`dynamic_hand.HandDistractor` 完全一致(固定 open 姿态 / episode 级 yaw+z /
逐 substep 位置 / 零接触 / 只影响点云图像), 但为 metaworld 换掉两处实现:

  1) **驱动 = 静态(非 mocap)根 body**:
     metaworld sawyer_xyz_env.set_xyz_action 读【整个】data.mocap_pos 再
     set_mocap_pos('mocap', ...), 第 2 个 mocap body 会让每步 step() 抛
     "could not broadcast (2,3) into (3,)"; 而 <freejoint> 会把物体挤到
     qpos[16](_set_obj_xyz 硬编码 qpos[9:12] -> 手臂复位被写坏)。
     因此本类:
       * 根位置/朝向写 `model.body_pos / model.body_quat`(静态体 = 唯一非侵入解,
         与 dynamic_sphere_mw.py 同一机制);
       * 12 个手指 hinge 关节保留, 但由 compose 追加在 worldbody **末尾**
         (qposadr >= 16, 见 test_load.compose_hammer_object 的
         `scene.replace('</worldbody>', body + '</worldbody>')`)
         => 机械臂/物体前缀 qpos[0:16] 逐位不变, agent_pos 仍是 9 维。
       安全性已核对: sawyer_xyz_env 唯一硬编码是 qpos[9:12]; set_state 的
       `qpos.shape==(nq,)` 断言只由 _set_obj_xyz(传全量 data.qpos) 与
       set_env_state(传全量 MjSimState) 触发 => 尾部追加关节不会触碰它。

  2) **可见性 = corner2 斜视相机 + 点云 crop**(adroit 用 top 相机解析式):
     观测来自 corner2(cam_mat0, cam_pos 被 wrapper 覆盖为 (0.6,0.295,0.8)) +
     pc_transform = Rx(61.4) @ Ry(-7) + crop [-0.5,-1.5,-0.795]..[1,-0.4,100]。
     每个手表面采样点 w 同时满足两条才算可见:
       (a) crop  : cloud = w @ A + c ∈ [min_bound, max_bound],
                   A = Rcam @ pcT.T, c = -(cam_pos @ A);
       (b) 视锥  : cam = (w - cam_pos) @ Rcam, depth = -cam_z > 0,
                   u = cx + f*cam_x/depth, v = cx + f*cam_y/depth,
                   cx = (W-1)/2, f = (W/2)/tan(fovy/2), 且 u,v ∈ [0, W-1] 且
                   znear <= depth <= zfar(渲染裁剪面)。
       (u,v) 公式与 green_cells_metaworld.screen_mask 同一约定; 已用
       `MetaWorldEnv.get_point_cloud` 的深度图逐点复核(median |z - depth| = 0)。
     **遮挡(第三条判据, 可选)**: 传入 `occ`(zarr 深度反投影的 3D 占据并集)后,
       已过 (a)(b) 的顶点还要满足
        (c) 未被挡: 把 occ 体素投影成 corner2 视角的**遮挡深度图**(每像素取最近
            占用面, 体素按 `occ_splat` px 半径 splat 防孔洞); 顶点深度须早于该
            像素的遮挡面(容差 `_OCC_TOL`)。
       即把"在画幅内"收紧成"真的会出现在(点)云里"。数据占据是**多帧并集**
       => 判据**偏保守**(实测会把整条臂可达区都算作遮挡物), 故**默认关闭**;
       需要时用 `DR3D_HAND_OCC=data` 打开, 作为保守诊断/消融。
     不传 `occ`(或 `DR3D_HAND_OCC=off`)时与旧行为**逐位一致**(analytic)。

时序: 每评测子步 = frame_skip=5 个 sim 帧(metaworld 默认), dt = timestep*5。
首帧含手: `start_episode()` 必须在 `env.reset()` **之前**调用(写 body_pos/quat +
qpos + qpos0), 之后再 `reapply()` 兜底。

位置约束(与 posgen **同源**, 严格版):
    * 工作区 = green_cells_assembly 的 `_ARM`(|x|<=0.2, y∈[0.5,0.9], margin 0.05)
      ∪ 物体/peg 圆盘(`_OBJ`, r=0.16), 直接 import 该模块的常量(单一事实来源);
      注意: posgen 的绿区已改为按 task 的**数据驱动**口径
      (dr3d/bench/green_cells_metaworld.py: zarr depth 足迹 + 复位可见性)。
      dynhand 由 `suite._mw_distractor` 传入 `work`(本任务数据足迹距离场)与
      `task_center`(本任务物体质心) => assembly / stick-push 都走本任务口径;
      只有 `work=None`(legacy 调用)才回落到 `_GA` 的 assembly 解析常量。
    * 候选格与**整条 A-B 线段**(PATH_PROBES 采样)都要求"工作区按手部半宽 hr≈0.086m
      外扩后"不相交 ⇒ **手体全程不进工作区**; 且不越出 assembly 桌面网格(PARAMS,
      按 hr 内缩)。兜底优先在"同一凸安全域"(下方带/左列/右列)内取点对 ⇒ 凸域内线段
      必在域内, **构造性保证**不穿区; 极端情况退化为静止手并在日志标注。
    * `spec()` 给出审计字段: zone_ok / table_ok / work_clear_min(手体到工作区边界的
      最近距离, m) / hand_hr_used / task_center。
    * 朝向: yaw 使手指指向 assembly 任务中心 (0.05, 0.70)(取代 adroit 的桌面原点)。
    * ⚠ 不使用 adroit 的 INSET/CENTER_CLEARANCE/MID_CLEARANCE —— 那是 hammer/door
      桌面口径, 在 assembly 上会让 A/B 采样全部失败并落进兜底, 选出一条横跨桌面的
      长直线(会穿过机械臂工作区), 正是本模块要修掉的行为。

可见性(vs adroit): 用 corner2 斜视相机 + 点云 crop 的解析判据, 并以**向量化**(3x3
展开、分批)实现; `_vis_frac_batch` 与逐点 `visible_frac` 逐位一致(详见 test/ 下的
验证脚本)。单 episode 的 `start_episode` 约 5-15 s(全顶点精确; 受机器负载影响)。

性能开关(环境变量, 免改代码):
    DR3D_HAND_HULL_PTS>0  手部顶点换成凸包/等距子样(≈380x 更少点, 大幅提速;
                          **会改变可见性判定与放置结果**, 默认 0 = 精确)
    DR3D_HAND_BATCH       向量化分块大小(默认 48; 内存吃紧时调小)

可复现: 通道 A/B = mix(seed, ep, 1), z = mix(seed, ep, 2)(与 adroit 同款)。
"""
import os
import pathlib
import sys
import time

import numpy as np

_BENCH = pathlib.Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

# 复用 adroit 的语义常量 / 几何推导 + dynsphere 的相机常量
from dynamic_hand import (            # noqa: E402
    POSE_OPEN, FINGERS, SPEED, HAND_SCALE, Z_RANGE, VIS_FRAC_MIN,
    MIN_PATH_ABS, MIN_PATH_TIME, PATH_PROBES, MAX_CELL_TRIES,
    MAX_TICK_RECORDS, DEFAULT_NAME, DT_FALLBACK, hand_geometry, qmul, quat_z)
# metaworld「任务工作区」口径的**唯一来源**: 直接复用 green_cells_assembly 的常量,
# 与 posgen 的绿区定义永久一致(_ARM/_OBJ/_OBJ_R/PARAMS)。
#   ⚠ adroit 的 INSET(0.41)/CENTER_CLEARANCE/MID_CLEARANCE/ORIENT_DOT_MIN 是
#     hammer/door 桌面口径(半方 0.41m, 场景中心 = 桌面原点), 对 assembly
#     (x∈[-0.67,0.67], 任务中心 ≈ (0.05,0.7)) 不成立 —— 旧实现因此全部落进兜底
#     分支, 选出一条横跨桌面的长直线并**穿过机械臂工作区**。本模块不再使用它们。
import green_cells_assembly as _GA     # noqa: E402
from dynamic_sphere_mw import find_pc_constants   # noqa: E402
from dynamic import unwrap_sim                    # noqa: E402  (sys.path 已注入)
from memlog import VERBOSE, once                  # noqa: E402  (日志/静音共享工具)

_CAM = "corner2"
FRAMES_PER_SUBSTEP_MW = 5      # metaworld MujocoEnv(frame_skip=5); 无 num_repeats
_Z_STEP = 0.02                 # 高度扫描步长(m)
_BATCH = int(os.environ.get("DR3D_HAND_BATCH", "48"))     # 向量化分块(cells/块)
# 0 = 用全部手部顶点(精确, 默认); >0 = 凸包/等距子样(大幅提速, 但**会改变放置结果**)
_HULL_PTS = int(os.environ.get("DR3D_HAND_HULL_PTS", "0"))
_PATH_FINE = 201               # 中轴路径细采样点数(工作区/桌面约束的严谨判据用)
# --- 遮挡判据(可选; 默认关闭) -----------------------------------------------------
#   DR3D_HAND_OCC       "off"(默认) | "data"(用 zarr 3D 占据并集作遮挡体)
#   DR3D_HAND_OCC_PTS   遮挡判定用的手部顶点子样数(均匀 stride)
#   DR3D_HAND_OCC_SPLAT 遮挡深度图的体素 splat 半径上限(px)
# 为什么默认关闭(实测, stick-push, seed=0/ep=0): 数据占据是 500 帧**并集**, 机械臂
# 扫过的整片桌面都被算成遮挡物 => 可行格 584 -> 5、所选路径 path_vis_min_occ=0.235
# (analytic 0.725), 即"手大部分落在画幅内"被判成"大部分被挡"。这条判据适合当
# **保守诊断/消融**, 不适合默认开启; 真正按 episode 的遮挡需要"当前臂位形"下渲染
# corner2 depth, 再按手轮廓像素统计可见比例 —— 该真值门控尚未实装。
_OCC_MODE = os.environ.get("DR3D_HAND_OCC", "off").strip().lower()
_OCC_PTS = int(os.environ.get("DR3D_HAND_OCC_PTS", "4096"))
_OCC_SPLAT = int(os.environ.get("DR3D_HAND_OCC_SPLAT", "6"))
_OCC_TOL = 0.01                # 遮挡深度比较容差(m, 体素量化)
# assembly 任务中心(被操纵物体/peg 质心) —— 取代 adroit 的"桌面原点"
TASK_CENTER = np.array([0.05, 0.70])


class HandDistractorMW:
    """inspire 左手: 每 episode 定 yaw/高度, 逐 substep 在绿区线段上乒乓(零接触)。"""

    def __init__(self, sim, cells, pc_transform, min_bound, max_bound,
                 image_size=128, name=DEFAULT_NAME, scale=HAND_SCALE,
                 speed=SPEED, z_range=Z_RANGE, vis_frac_min=VIS_FRAC_MIN,
                 frames_per_substep=FRAMES_PER_SUBSTEP_MW, cam=_CAM,
                 verbose=True, work=None, task_center=None,
                 occ=None, screen_mask=None, occ_pts=None, occ_splat=None):
        self.sim = sim
        self.cells = np.asarray(cells, dtype=float)
        if len(self.cells) == 0:
            raise ValueError("HandDistractorMW: empty green-cell list")
        self.name = name
        # --- 数据驱动的工作区/任务中心(可选; 不给 = 旧行为, assembly 基线逐位不变) ----
        #   work = dict(dist=(ny,nx) float32 到"数据足迹"的距离(m), lo=(x0,y0),
        #               gs=float, margin=float(额外余量,m))
        #         -> 安全 = dist_to_footprint >= hr + margin(取代 _GA._ARM/_OBJ 常量)
        #   task_center = 数据质心(被操纵物 com 均值) -> 手指朝向 & "朝中心"方向
        self.work = None
        if work is not None:
            self.work = dict(dist=np.asarray(work["dist"], np.float32),
                             lo=np.asarray(work["lo"], float).ravel()[:2],
                             gs=float(work["gs"]),
                             margin=float(work.get("margin", 0.0)))
        self.task_center = (np.asarray(TASK_CENTER, float) if task_center is None
                            else np.asarray(task_center, float).ravel()[:2])
        # --- 遮挡/shape 禁区(可选; 全 None = 纯解析判据, 与旧行为逐位一致) -------
        #   occ = dict(occ=(nx,ny,nz) bool 已膨胀, lo=(3,), gs=float)  zarr 深度占据
        #         -> 判据加第三条: 相机->顶点线段不得先穿过占用体素(见 _occ_mask)
        #   screen_mask = (img,img) bool = 任务物体(stick/thermos)的屏幕足迹
        #         -> 单独统计 obj_frac(手压在物体屏幕上的比例, 不作硬约束)
        self.occ = None
        if occ is not None and _OCC_MODE != "off":
            self.occ = dict(occ=np.asarray(occ["occ"], bool),
                            lo=np.asarray(occ["lo"], float).ravel()[:3],
                            gs=float(occ["gs"]))
        self.occ_pts = max(64, int(_OCC_PTS if occ_pts is None else occ_pts))
        self.occ_splat = max(1, int(_OCC_SPLAT if occ_splat is None else occ_splat))
        self._occ_idx = None            # 遮挡判定用的顶点子样索引(惰性)
        self._occ_z = None              # 遮挡深度图(惰性; (img,img) float32)
        self.obj_dist = None            # 物体屏幕足迹的 EDT(px)
        if screen_mask is not None:
            from scipy import ndimage
            self.obj_dist = ndimage.distance_transform_edt(
                ~np.asarray(screen_mask, bool)).astype(np.float32)
        self.obj_margin_px = 2.0
        self.vis_mode = "occlusion-aware" if self.occ is not None else "analytic"
        self.vis_relax = 1.0            # 候选筛选实际用的可见率阈值(绝对值)
        self.path_vis_min_occ = None    # 审计: 遮挡感知版 path_vis_min
        self.occluded_frac = None       # 审计: 已进画幅但被挡的顶点比例(路径最大)
        self.obj_frac = None            # 审计: 手顶点落在物体屏幕足迹内的比例(路径最大)
        self.occ_scan_s = 0.0           # 审计: 遮挡判定的累计耗时
        self.speed = float(speed)
        self.z_lo_cfg, self.z_hi_cfg = float(z_range[0]), float(z_range[1])
        self.vis_min = float(vis_frac_min)
        self.scale = float(scale)
        self.geom = hand_geometry(scale=self.scale, verbose=False)
        self.hr = float(self.geom["hr"])       # 手最远水平半宽(工作区外扩用)
        if _HULL_PTS and _HULL_PTS > 0:        # 可选: 子样(会改变可见性判定)
            _p = self.geom["points_rel"]
            if len(_p) > _HULL_PTS:
                try:
                    from scipy.spatial import ConvexHull
                    _v = np.asarray(ConvexHull(_p).vertices, int)
                except Exception:
                    _v = np.arange(len(_p))
                if len(_v) > _HULL_PTS:
                    _v = _v[np.linspace(0, len(_v) - 1, _HULL_PTS).astype(int)]
                self.geom = dict(self.geom, points_rel=_p[_v])
        # --- 静态(非 mocap)根 body ---
        self.body_id = int(sim.model.body_name2id(name))
        self.mid = int(sim.model.body_mocapid[self.body_id])
        if self.mid >= 0:
            raise ValueError(
                f"{name} 必须是静态体(非 mocap): metaworld set_xyz_action 读整个 "
                "mocap_pos, 第 2 个 mocap body 会崩(bbox (2,3))")
        # 按 body 子树反查 geom(注入后是 dr_inspire 的嵌套子 body)
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
        # --- 手部手指关节 qpos 地址(必须用 joint_name2id 按名字查!)
        #     ⚠ mujoco_py 的 `model.joint_names` 会**过滤未命名关节**
        #     (metaworld 物体是 `<freejoint/>` -> 无名; 这里 njnt=22 但
        #      len(joint_names)=21), 因此 `enumerate(joint_names)` 与
        #     `jnt_qposadr[j]` 配对会整体错位一位 —— 把 POSE_OPEN 写进
        #     qpos[9](=物体的 x)等错误槽位。按名字查 id 才安全。
        self.jadr, self.jfinger = {}, {}
        for nm in POSE_OPEN:
            try:
                jid = int(sim.model.joint_name2id(nm))
            except Exception:                       # 名字不存在
                continue
            if jid < 0:
                continue
            k = next((f for f in FINGERS if f in nm), None)
            if k is None:
                continue
            self.jadr[nm] = int(sim.model.jnt_qposadr[jid])
            self.jfinger[nm] = k
        if not self.jadr:
            raise ValueError("HandDistractorMW: 手部手指关节未找到(名字未匹配)")
        if len(self.jadr) != len(POSE_OPEN):
            raise ValueError(f"HandDistractorMW: 手部关节不完整 "
                             f"{len(self.jadr)}/{len(POSE_OPEN)}")
        self.pose = {nm: float(POSE_OPEN[nm]) for nm in self.jadr}
        self.min_path = max(MIN_PATH_ABS, self.speed * MIN_PATH_TIME)
        # --- corner2 相机(world->cam) + crop 仿射 cloud = w @ A + c ---
        cid = int(sim.model.camera_name2id(cam))
        self.Rcam = np.asarray(sim.model.cam_mat0).reshape(-1, 3, 3)[cid]
        self.cam_pos = np.asarray(sim.model.cam_pos).reshape(-1, 3)[cid]
        pcT = np.asarray(pc_transform, float)
        # crop 仿射: cloud = w @ aff + aff_c(与 dynamic_sphere_mw 同一标定)
        self.aff = self.Rcam @ pcT.T
        self.aff_c = -(self.cam_pos @ self.aff)
        self.cmi = np.asarray(min_bound, float)
        self.cma = np.asarray(max_bound, float)
        self.img = int(image_size)
        # Pinhole pixel center, same convention as green_cells_metaworld (u = cx + f*x/(-z))
        self.cx = (self.img - 1.0) / 2.0
        fovy = float(np.ravel(sim.model.cam_fovy)[cid])
        self.f = (self.img / 2.0) / np.tan(np.radians(fovy) / 2.0)
        try:                        # 渲染裁剪面(不在 [znear, zfar] 不进深度图)
            self.znear = float(np.ravel(np.asarray(sim.model.vis.map.znear))[0])
            self.zfar = float(np.ravel(np.asarray(sim.model.vis.map.zfar))[0])
        except Exception:
            self.znear, self.zfar = 0.0, 1e9
        # --- 子步时长: 每评测子步 = frames_per_substep 个 sim 帧 ---
        try:
            self.dt = float(sim.model.opt.timestep) * float(frames_per_substep)
        except Exception:
            self.dt = DT_FALLBACK
        # --- episode 级状态(由 start_episode 填) ---
        self.seed = self.ep = None
        self.yaw = 0.0
        self.A = self.B = None      # 路径端点(与 adroit 同名, 供日志/manifest)
        self.z = None
        self.L = 0.0
        self.cells_g = None
        self.cells_full = 0
        # 常量缓存: 「候选格筛选」只依赖 self.cells / hr / z_lo / TASK_CENTER,
        # 与 (seed, ep) 无关 ⇒ 只算一次(见 _ensure_cells_g)即可逐位复现每个 episode
        # 的结果, 从而消掉每 episode 重算的 560 格 × P 可见性扫描(metaworld 侧
        # dynhand 的主要固定开销)。scan_cells_once_s 记录那唯一一次的实测耗时。
        self._cells_g_cache = None
        self._cells_full_cache = 0
        self.scan_cells_once_s = 0.0
        self.z_full = None
        self.path_vis_min = None
        self.work_clear_min = None      # 审计: 手体 到工作区边界的最近距离(min 两条证据)
        self.center_clear_min = None    # 审计: 中轴 到工作区边界距离 - hr(定理级)
        self.body_clear_min = None      # 审计: 手体表面点 到工作区边界距离(证据级)
        self.zone_ok = False            # 审计: 手体是否全程不进工作区
        self.table_ok = False           # 审计: 手体是否全程在桌面范围内
        self.last_scan_s = 0.0          # 最近一次可见性批量扫描耗时
        self.scan_cells_s = 0.0         # 本 episode 候选格扫描耗时
        self.scan_z_s = 0.0             # 本 episode 高度表扫描耗时
        self.scan_s = 0.0               # 本 episode 可见性扫描总耗时
        self.k = 0
        self.records = []
        self.pos = None
        if verbose:
            once("dynhand_mw." + str(name),
                 f"[dynamic_hand_mw] {name}: body={self.body_id}(static) "
                 f"joints={len(self.jadr)} cells={len(self.cells)} "
                 f"scale={self.scale:g} speed={self.speed} "
                 f"min_path={self.min_path:.3f}m dt={self.dt:.4f}s "
                 f"z_range={z_range} vis_min={self.vis_min:.2f} img={self.img} "
                 f"f={self.f:.1f} znear={self.znear:g} zfar={self.zfar:g}")

    # ---------------- deterministic RNG ----------------
    def _rng(self, channel):
        """逐 (seed, ep, channel) 的确定性 RNG(与 adroit/dynamic.py 同款混淆)。"""
        s = (int(self.seed) * 2654435761
             + int(self.ep) * 1000003
             + (int(channel) + 2) * 7919) % (2 ** 32)
        return np.random.RandomState(s)

    # ---------------- geometry / visibility ----------------
    def _pts_world(self, x, y, z, yaw=None, pts=None):
        """表面采样点(相对几何中心)经 yaw 旋转、平移到中心 (x,y,z) 后的 (px,py,pz)。"""
        p = self.geom["points_rel"] if pts is None else np.asarray(pts, float)
        yaw = self.yaw if yaw is None else float(yaw)
        cy, sy = np.cos(yaw), np.sin(yaw)
        px = cy * p[:, 0] - sy * p[:, 1] + x
        py = sy * p[:, 0] + cy * p[:, 1] + y
        return px, py, p[:, 2] + z

    def _vis_ok(self, px, py, pz):
        """Per-point visibility: crop (affine) AND corner2 frustum (with znear/zfar).

        Pixel convention matches green_cells_metaworld (u = cx + f*x/(-z), cx=(img-1)/2),
        so the hand's screen footprint and the object screen mask live in the same frame.
        Kept bit-identical to the inline version in `_vis_frac_batch`; factored out so the
        occlusion path can reuse it.
        """
        w = np.stack([px, py, pz], axis=1)
        cloud = w @ self.aff + self.aff_c
        ok = (np.all(cloud >= self.cmi, axis=1) &
              np.all(cloud <= self.cma, axis=1))
        cam = (w - self.cam_pos) @ self.Rcam
        zc = cam[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.where(zc < -1e-6, -zc, np.nan)
            u = self.cx + self.f * cam[:, 0] / depth
            v = self.cx + self.f * cam[:, 1] / depth
        ok &= (np.isfinite(u) & np.isfinite(v) &
               (depth >= self.znear) & (depth <= self.zfar) &
               (u >= 0.0) & (u <= self.img - 1.0) &
               (v >= 0.0) & (v <= self.img - 1.0))
        return ok

    def _vis_mask(self, x, y, z, yaw=None, pts=None):
        """逐点可见性: crop(仿射) ∧ corner2 视锥(含 znear/zfar); 返回 (P,) bool。"""
        px, py, pz = self._pts_world(x, y, z, yaw, pts=pts)
        return self._vis_ok(px, py, pz)

    def visible_frac(self, x, y, z, yaw=None):
        """"大部分在范围内": 采样点同时过 crop 与 corner2 视锥的比例。"""
        return float(np.mean(self._vis_mask(x, y, z, yaw)))

    def _vis_frac_batch(self, cx, cy, z, yaw, batch=_BATCH, timed=False):
        """向量化 `visible_frac`: cx/cy/yaw 等长数组 (N,), z 为标量 -> (N,) 比例。

        与逐点 `_vis_mask` **完全同一判据**(crop 仿射 ∧ corner2 视锥 ∧ znear/zfar),
        只是把 cell 维一起放进张量, 消除逐格 Python 循环 —— 手部 7.6 万顶点 × 560 格
        时这一项就是 4 min/episode 的全部来源。分块(batch)控峰值内存 ~batch×P×3。
        """
        cx = np.atleast_1d(np.asarray(cx, float))
        cy = np.atleast_1d(np.asarray(cy, float))
        zz = np.atleast_1d(np.asarray(z, float))
        yw = np.atleast_1d(np.asarray(yaw, float))
        if yw.size == 1 and cx.size > 1:
            yw = np.full(cx.size, float(yw.ravel()[0]))
        if zz.size == 1 and cx.size > 1:
            zz = np.full(cx.size, float(zz.ravel()[0]))
        if not (zz.size == cx.size == yw.size):
            raise ValueError("_vis_frac_batch: cx/cy/z/yaw 长度必须一致")
        p = self.geom["points_rel"]
        P = p.shape[0]
        px = p[:, 0][None, :]
        py = p[:, 1][None, :]
        pz = p[:, 2][None, :]
        # 3x3 显式乘加: 小 GEMM 走 BLAS 会因线程抖动/调度开销反而慢几个数量级,
        # 展开成标量乘加后无 BLAS 调用、无 (m,P,3) 临时数组, 判据与 _vis_mask 逐位一致。
        A = self.aff
        Ac = self.aff_c
        R = self.Rcam
        cp = self.cam_pos
        hi = self.img - 1.0
        n = cx.size
        out = np.zeros(n, float)
        t0 = time.time() if timed else 0.0
        for s in range(0, n, int(batch)):
            e = min(n, s + int(batch))
            cs = np.cos(yw[s:e])[:, None]
            sn = np.sin(yw[s:e])[:, None]
            zr = zz[s:e, None]
            xw = cs * px - sn * py + cx[s:e, None]
            yh = sn * px + cs * py + cy[s:e, None]
            # crop 仿射: cloud = w @ aff + aff_c  (w = (xw, yh, pz + z))
            c0 = xw * A[0, 0] + yh * A[1, 0] + pz * A[2, 0] + Ac[0] + zr * A[2, 0]
            c1 = xw * A[0, 1] + yh * A[1, 1] + pz * A[2, 1] + Ac[1] + zr * A[2, 1]
            c2 = xw * A[0, 2] + yh * A[1, 2] + pz * A[2, 2] + Ac[2] + zr * A[2, 2]
            ok = ((c0 >= self.cmi[0]) & (c0 <= self.cma[0]) &
                  (c1 >= self.cmi[1]) & (c1 <= self.cma[1]) &
                  (c2 >= self.cmi[2]) & (c2 <= self.cma[2]))
            # corner2 视锥: cam = (w - cam_pos) @ Rcam
            rx = xw - cp[0]
            ry = yh - cp[1]
            rz = pz + zr - cp[2]
            ax = rx * R[0, 0] + ry * R[1, 0] + rz * R[2, 0]
            ay = rx * R[0, 1] + ry * R[1, 1] + rz * R[2, 1]
            az = rx * R[0, 2] + ry * R[1, 2] + rz * R[2, 2]
            zneg = az < -1e-6
            ok &= zneg
            dz = np.where(zneg, -az, 1.0)          # avoid div-by-zero; zneg=False already dropped
            u = self.cx + self.f * ax / dz
            v = self.cx + self.f * ay / dz
            ok &= ((dz >= self.znear) & (dz <= self.zfar) &
                   (u >= 0.0) & (u <= hi) & (v >= 0.0) & (v <= hi))
            out[s:e] = ok.mean(axis=1)
        if timed:
            self.last_scan_s = time.time() - t0
            if VERBOSE:
                print(f"[dynamic_hand_mw] vis scan {n} cells x {P} pts "
                      f"= {n * P / 1e6:.1f}M pts in {self.last_scan_s:.2f}s")
        return out

    # ---------------- 遮挡判据(仅在传入 occ 时启用; 否则逐位回到解析判据) --------
    def _pts_sub(self):
        """遮挡判定用的顶点子样(均匀 stride)。

        crop/视锥仍用**全部**顶点(见 `_vis_frac_batch`); 遮挡是空间低频现象,
        4096 个均匀顶点足以估计比例, 同时把射线 march 的成本从 7.6 万点降到 4 千点。
        """
        if self._occ_idx is None:
            P = int(len(self.geom["points_rel"]))
            n = int(min(P, self.occ_pts))
            self._occ_idx = np.linspace(0, P - 1, n).astype(int)
        return self._occ_idx

    def _occ_depth(self):
        """occ 体素 -> corner2 视角的**遮挡深度图** `(img,img)` float32(m)。

        每像素取"最近的占用面深度"; 体素按其在像面上的尺寸 splat(半径
        `0.5*f*gs/depth`, 上限 `occ_splat` px), 避免点投影留下孔洞导致的漏检。
        与逐点射线 march 等价但快 3 个数量级(每点 1 次查表)。
        Self-consistent frame: this map and `_occ_mask` share the SAME pixel formula
        (`u = img/2 - f*x/depth`), so the occlusion test is exact inside that frame;
        it is the 180-deg-mirrored image w.r.t. `_vis_ok` (whose frustum test is
        symmetric under that mirror, so both agree on "inside the frame").
        """
        if self._occ_z is not None:
            return self._occ_z
        img = int(self.img)
        if self.occ is None:                     # 未启用: 空遮挡图(全部 inf)
            self._occ_z = np.full((img, img), np.inf, np.float32)
            return self._occ_z
        o = self.occ
        zmap = np.full((img, img), np.inf, np.float32)
        ii = np.argwhere(o["occ"])
        if ii.size == 0:
            self._occ_z = zmap
            return zmap
        w = o["lo"][None, :] + (ii + 0.5) * float(o["gs"])
        cam = (w - self.cam_pos) @ self.Rcam
        zc = cam[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.where(zc < -1e-6, -zc, np.nan)
            u = img / 2.0 - self.f * cam[:, 0] / depth
            v = img / 2.0 - self.f * cam[:, 1] / depth
        good = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth)
        good &= (depth > 0) & (u > -8) & (u < img + 8) & (v > -8) & (v < img + 8)
        ui = np.rint(np.nan_to_num(u, nan=-1e6)).astype(np.int64)[good]
        vi = np.rint(np.nan_to_num(v, nan=-1e6)).astype(np.int64)[good]
        dz = depth[good].astype(np.float32)
        if ui.size:
            rad = np.clip(np.rint(0.5 * self.f * float(o["gs"]) / dz).astype(np.int64),
                          1, int(self.occ_splat))
            order = np.argsort(dz, kind="stable")       # 由近到远: 近的面先占像素
            ui, vi, dz, rad = ui[order], vi[order], dz[order], rad[order]
            for r in np.unique(rad):
                sel = np.flatnonzero(rad == r)
                r = int(r)
                for dv_ in range(-r, r + 1):
                    for du_ in range(-r, r + 1):
                        uu = np.clip(ui[sel] + du_, 0, img - 1)
                        vv = np.clip(vi[sel] + dv_, 0, img - 1)
                        free = zmap[vv, uu] == np.inf       # 已占用的像素保留更近的面
                        if free.any():
                            zmap[vv[free], uu[free]] = dz[sel][free]
        self._occ_z = zmap
        return zmap

    def _occ_mask(self, px, py, pz, ok):
        """已过 crop/视锥的点: 顶点是否被遮挡物挡在**后面**; 返回 (M,) bool。

        `True = 未被遮挡`。查 `_occ_depth()`: 若该像素的遮挡面比顶点更近
        (差值 > `_OCC_TOL`), 判为被挡。占用来 zarr 深度反投影的**多帧并集**
        (`green_cells_metaworld.DYN`), 因此判据偏保守。
        """
        if self.occ is None:
            return np.ones(len(px), bool)
        out = np.ones(len(px), bool)
        idx = np.flatnonzero(np.asarray(ok, bool))
        if idx.size == 0:
            return out
        zmap = self._occ_depth()
        img = int(self.img)
        cam = (np.stack([px[idx], py[idx], pz[idx]], axis=1)
               - self.cam_pos) @ self.Rcam
        zc = cam[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.where(zc < -1e-6, -zc, np.nan)
            u = img / 2.0 - self.f * cam[:, 0] / depth
            v = img / 2.0 - self.f * cam[:, 1] / depth
        good = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth)
        ui = np.where(good, np.rint(np.nan_to_num(u, nan=-1.0)), -1.0).astype(np.int64)
        vi = np.where(good, np.rint(np.nan_to_num(v, nan=-1.0)), -1.0).astype(np.int64)
        inset = good & (ui >= 0) & (ui < img) & (vi >= 0) & (vi < img)
        occ = np.zeros(idx.size, bool)
        if inset.any():
            zd = zmap[vi[inset], ui[inset]]
            occ[inset] = zd <= (depth[inset].astype(np.float32) - _OCC_TOL)
        out[idx] = ~occ
        return out

    def _occ_frac_batch(self, cx, cy, z, yaw):
        """遮挡感知的可见比例(子样顶点): crop ∧ 视锥 ∧ ¬被挡; 返回 (N,)。

        `occ is None` 时直接退化为 `_vis_frac_batch`(逐位一致)。
        """
        if self.occ is None:
            return self._vis_frac_batch(cx, cy, z, yaw)
        cx = np.atleast_1d(np.asarray(cx, float))
        cy = np.atleast_1d(np.asarray(cy, float))
        zz = np.atleast_1d(np.asarray(z, float))
        yw = np.atleast_1d(np.asarray(yaw, float))
        if yw.size == 1 and cx.size > 1:
            yw = np.full(cx.size, float(yw.ravel()[0]))
        if zz.size == 1 and cx.size > 1:
            zz = np.full(cx.size, float(zz.ravel()[0]))
        if not (zz.size == cx.size == yw.size):
            raise ValueError("_occ_frac_batch: cx/cy/z/yaw 长度必须一致")
        p = self.geom["points_rel"][self._pts_sub()]
        out = np.zeros(cx.size, float)
        t0 = time.time()
        for i in range(cx.size):
            px, py, pz = self._pts_world(cx[i], cy[i], zz[i], yw[i], pts=p)
            ok = self._vis_ok(px, py, pz)
            ok &= self._occ_mask(px, py, pz, ok)
            out[i] = float(ok.mean())
        self.occ_scan_s += time.time() - t0
        return out

    def _obj_frac(self, x, y, z, yaw=None):
        """Fraction of hand vertices landing on the task-object screen footprint.

        Only available when the caller passed `screen_mask`; None otherwise. Diagnostic
        only (does the hand cover the object pixels?) — the hard constraint is the work
        distance field. Pixel convention MUST match green_cells_metaworld, which built
        the mask as u = cx + f*x_cam/(-z_cam) with cx = (img-1)/2.
        """
        if self.obj_dist is None:
            return None
        px, py, pz = self._pts_world(x, y, z, yaw)
        cam = (np.stack([px, py, pz], axis=1) - self.cam_pos) @ self.Rcam
        zc = cam[:, 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            depth = np.where(zc < -1e-6, -zc, np.nan)
            uu = self.cx + self.f * cam[:, 0] / depth
            vv = self.cx + self.f * cam[:, 1] / depth
        good = np.isfinite(uu) & np.isfinite(vv)
        u = np.where(good, np.rint(np.nan_to_num(uu, nan=-1.0)), -1.0).astype(np.int64)
        v = np.where(good, np.rint(np.nan_to_num(vv, nan=-1.0)), -1.0).astype(np.int64)
        ok = good & (u >= 0) & (u < self.img) & (v >= 0) & (v < self.img)
        d = np.full(len(u), 1e9, np.float32)
        if ok.any():
            d[ok] = self.obj_dist[np.clip(v[ok], 0, self.img - 1),
                                  np.clip(u[ok], 0, self.img - 1)]
        return float(np.mean(d <= self.obj_margin_px))

    # ---------------- 任务工作区(与 posgen / green_cells_assembly 同源) ----------
    def _in_work_zone(self, xy, extra=0.0):
        """`xy` (N,2) 是否落在「机械臂工作区 ∪ 物体/peg 邻域」内(按 extra 外扩)。

        常量直接取自 `green_cells_assembly`(_ARM/_OBJ/_OBJ_R), 与 posgen 的绿区
        定义永久一致; `extra` 传手部半宽 `hr` 时得到"手**体**绝不进入工作区"的严格
        判据(是 posgen 格口径的超集)。
        """
        xy = np.atleast_2d(np.asarray(xy, float))
        if self.work is not None:
            # 数据口径: 中心到"数据足迹"距离(m, 已扣 margin) < extra 即在(外扩)工作区内
            return self._dist_zone(xy) < float(extra)
        mg = float(_GA._ARM["margin"]) + float(extra)
        ax1 = float(_GA._ARM["x"][1])
        ay0, ay1 = float(_GA._ARM["y"][0]), float(_GA._ARM["y"][1])
        zone = ((np.abs(xy[:, 0]) <= ax1 + mg) &
                (xy[:, 1] >= ay0 - mg) & (xy[:, 1] <= ay1 + mg))
        for (ox, oy) in _GA._OBJ:
            zone |= (np.hypot(xy[:, 0] - float(ox), xy[:, 1] - float(oy))
                     < float(_GA._OBJ_R) + float(extra))
        return zone

    def _off_table(self, xy, extra=0.0):
        """`xy` (N,2) 是否越出 assembly 桌面网格范围(按 extra 内缩)。"""
        xy = np.atleast_2d(np.asarray(xy, float))
        x0, x1 = float(_GA.PARAMS["x"][0]), float(_GA.PARAMS["x"][1])
        y0, y1 = float(_GA.PARAMS["y"][0]), float(_GA.PARAMS["y"][1])
        return ((xy[:, 0] < x0 + extra) | (xy[:, 0] > x1 - extra) |
                (xy[:, 1] < y0 + extra) | (xy[:, 1] > y1 - extra))

    def _dist_raw(self, xy):
        """到"数据足迹"(工作区)的距离(m); 在 0.03 桌面格上最近邻取值。"""
        w = self.work
        xy = np.atleast_2d(np.asarray(xy, float))
        ix = np.rint((xy[:, 0] - w["lo"][0]) / w["gs"]).astype(int)
        iy = np.rint((xy[:, 1] - w["lo"][1]) / w["gs"]).astype(int)
        ny, nx = w["dist"].shape
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        return w["dist"][iy, ix].astype(float)

    def _dist_zone(self, xy):
        """`xy` (N,2) 到「工作区」边界的最近**有符号**距离(<0 = 在工作区内)。"""
        xy = np.atleast_2d(np.asarray(xy, float))
        if self.work is not None:
            return self._dist_raw(xy) - float(self.work["margin"])
        ax1 = float(_GA._ARM["x"][1])
        ay0, ay1 = float(_GA._ARM["y"][0]), float(_GA._ARM["y"][1])
        dx = np.maximum(np.abs(xy[:, 0]) - ax1, 0.0)
        dy = np.maximum(np.maximum(ay0 - xy[:, 1], xy[:, 1] - ay1), 0.0)
        d = np.hypot(dx, dy)                       # 到臂矩形的距离(0 = 在内)
        for (ox, oy) in _GA._OBJ:
            d = np.minimum(d, np.hypot(xy[:, 0] - float(ox),
                                       xy[:, 1] - float(oy))
                           - float(_GA._OBJ_R))
        return d

    def _audit_zone(self, seg):
        """审计: 手**体**沿整条路径到工作区边界的最近距离(>=0 即全程不进工作区)。

        两条独立证据取 min:
          (a) 定理级: 中轴路径细采样, `dist_zone(center) - hr`。因手体脚印 ⊆ 半径 hr
              的圆盘, `(a) >= 0` 即**严格蕴含**"手体不进工作区"(任意子步位置);
          (b) 证据级: PATH_PROBES 个中轴点 × 手体全部表面点直接算距离(交叉验证)。
        """
        u = np.linspace(0.0, 1.0, max(int(_PATH_FINE), 2))[:, None]
        path = self.A[None, :] + (self.B - self.A)[None, :] * u
        self.center_clear_min = float(self._dist_zone(path).min() - self.hr)
        p = self.geom["points_rel"]
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        rx = cy * p[:, 0] - sy * p[:, 1]          # 手体相对中心(已按 yaw 旋转)
        ry = sy * p[:, 0] + cy * p[:, 1]
        xy = np.stack([(seg[:, 0][:, None] + rx[None, :]).ravel(),
                       (seg[:, 1][:, None] + ry[None, :]).ravel()], axis=1)
        self.body_clear_min = float(self._dist_zone(xy).min())
        self.work_clear_min = min(self.center_clear_min, self.body_clear_min)
        self.zone_ok = bool(self.work_clear_min >= 0.0)
        self.table_ok = bool(not self._off_table(path, self.hr).any())
        return self.zone_ok, self.table_ok, self.work_clear_min

    def z_cap(self, x, y, lo, target=None):
        """自 lo 向上扫描, 返回 `visible_frac >= target` 的最高中心高度。

        斜视相机下 frac(z) 未必严格单调, 故用网格扫描 + 首个不满足即止(保连续);
        `target=1.0` 求"整只手都在画幅内"的高度。返回 None 表示 lo 处就不达标。
        """
        target = self.vis_min if target is None else float(target)
        if self.visible_frac(x, y, lo) < target:
            return None
        best = float(lo)
        for z in np.arange(float(lo), self.z_hi_cfg + 1e-9, _Z_STEP):
            if z <= lo:
                continue
            if self.visible_frac(x, y, float(z)) >= target:
                best = float(z)
            else:
                break
        return best

    # ---------------- 路径 / 写入 ----------------
    def _path_xy(self, t):
        """substep t 的手几何中心 (x, y): A->B 线段上的三角波(乒乓), 纯函数。"""
        d = self.B - self.A
        s = (self.speed * self.dt * int(t)) / max(self.L, 1e-9)
        ph = 1.0 - abs((s % 2.0) - 1.0)                # 三角波 [0, 1]
        return self.A + d * ph

    def _write(self, x, y, z, model_default=False):
        """写【静态根】body_pos/body_quat + 12 手指 qpos(固定 open)。

        静态体(非 mocap): `model.body_pos/body_quat` 就是运行期位形, mj_reset 不动它;
        手指是模型关节, mj_reset 会把 qpos 复位到 qpos0 ⇒ model_default 时同步写
        qpos0(否则 reset 后首帧会退回模型默认姿态)。
        位置 = 目标几何中心减去 yaw 旋转后的中心偏移 ⇒ "几何中心落在 (x, y, z)"。
        """
        c = self.geom["center"]
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        off = np.array([cy * c[0] - sy * c[1], sy * c[0] + cy * c[1], c[2]])
        pos = np.array([x, y, z], dtype=float) - off
        quat = qmul(quat_z(self.yaw), self.geom["base_quat"])
        self.sim.model.body_pos[self.body_id] = pos
        self.sim.model.body_quat[self.body_id] = quat
        for nm, val in self.pose.items():
            self.sim.data.qpos[self.jadr[nm]] = val
        if model_default:
            for nm, val in self.pose.items():
                self.sim.model.qpos0[self.jadr[nm]] = val
        self.sim.forward()

    # ---------------- 候选格筛选(常量; 只算一次) ----------------
    def _ensure_cells_g(self):
        """计算「可行格」并缓存 —— 首次调用算一次, 之后直接复用。

        原先这段(候选格 = 绿区 ∩ ¬工作区(hr) ∩ ¬桌外(hr) ∩ z_lo 处可见)被放在
        `start_episode` 里, **每个 episode 都重算一遍** 560 格 × P 的手部可见性扫描。
        但它的输入(格中心、每格朝向 yaw_c = 指向 TASK_CENTER、hr、z_lo)全部与
        (seed, ep) 无关 ⇒ 结果逐位相同, 属纯固定开销。这里改为惰性一次性计算并缓存,
        放置结果与逐 episode 重算**完全一致**, 只是不再重复付出这份代价。

        返回 (cells_g, cells_full)。
        """
        if self._cells_g_cache is not None:
            return self._cells_g_cache, self._cells_full_cache
        hr = float(self.hr)
        z_lo = max(self.z_lo_cfg, -float(self.geom["zmin"]))    # 手底不穿桌
        safe = (~self._in_work_zone(self.cells, hr) &
                ~self._off_table(self.cells, hr))
        yaw_c = np.arctan2(self.task_center[1] - self.cells[:, 1],
                           self.task_center[0] - self.cells[:, 0])
        frac = self._vis_frac_batch(self.cells[:, 0], self.cells[:, 1],
                                    z_lo, yaw_c, timed=True)
        frac_occ = (self._occ_frac_batch(self.cells[:, 0], self.cells[:, 1],
                                         z_lo, yaw_c)
                    if self.occ is not None else frac)
        self.scan_cells_once_s = float(self.last_scan_s)
        keep = np.zeros(len(self.cells), bool)
        relax_used = float(self.vis_min)
        for relax in (self.vis_min, 0.5 * self.vis_min, 0.0):
            keep = safe & (frac >= relax) & (frac_occ >= relax)
            relax_used = float(relax)
            if keep.any():
                break
        self.vis_relax = relax_used
        if relax_used < self.vis_min:
            # 不再静默放宽: 记录到 spec.vis_relax, 非静音模式下打一次 WARN
            once("dynhand_mw.relax." + str(self.name),
                 f"[dynamic_hand_mw] WARN 候选格无格满足 vis>={self.vis_min:.2f}"
                 f"(mode={self.vis_mode}), 已放宽到 {relax_used:.2f}")
        if not keep.any():                       # 兜底: 安全域内最靠外的一格
            idx = np.flatnonzero(safe)
            if idx.size == 0:                    # 连安全域都空(不应发生)
                idx = np.arange(len(self.cells))
            i = idx[int(np.argmax(np.max(np.abs(self.cells[idx]), axis=1)))]
            keep = np.zeros(len(self.cells), bool)
            keep[i] = True
        self._cells_g_cache = self.cells[keep]
        self._cells_full_cache = int(len(self._cells_g_cache))
        return self._cells_g_cache, self._cells_full_cache

    # ---------------- episode / decision / substep hooks ----------------
    def start_episode(self, seed, ep):
        """episode 开始(须在 `env.reset()` 之前): 抽 A-B/z, 定朝向并放 t=0 位形。

        与 posgen **同源**的工作区约束(严格版: 按手部半宽 hr 外扩):
          ① 候选格 = posgen 绿区(调用方传入) ∩ ¬工作区(hr) ∩ ¬桌外(hr) ∩ z_lo 处可见;
          ② A/B 线段**全部采样点**满足 ¬工作区(hr) ∧ ¬桌外(hr) ⇒ 手**体**全程不进
             工作区; 优先在"同一凸安全域"(下方带 / 左列 / 右列)内取点对 ⇒ 凸域内线段
             必在域内, **构造性保证**不穿区;
          ③ 朝向: yaw 使手指指向 assembly 任务中心 TASK_CENTER=(0.05, 0.70);
          ④ 高度: z = U(z_lo, min(z_hi, z_full)), 由 corner2 可见性网格扫描给出。

        可复现通道: A/B = mix(seed, ep, 1); z = mix(seed, ep, 2)。
        """
        self.seed, self.ep = int(seed), int(ep)
        self.records, self.k = [], 0
        g = self.geom
        hr = float(self.hr)
        z_lo = max(self.z_lo_cfg, -float(g["zmin"]))           # 手底不穿桌
        # --- ① 候选格: posgen 绿区 ∩ ¬工作区(hr) ∩ ¬桌外(hr) ∩ z_lo 处可见 ---
        # 该筛选只依赖常量(格中心 / 每格朝向指向 TASK_CENTER / hr / z_lo), 与
        # (seed, ep) 无关 ⇒ 只算一次并缓存(见 _ensure_cells_g), 不再每 episode
        # 重算 560 格 × P 的可见性扫描(metaworld 侧 dynhand 的主要固定开销)。
        _cache_hit = self._cells_g_cache is not None
        self.cells_g, self.cells_full = self._ensure_cells_g()
        # 本 episode 的格扫描耗时: 首次为实测值, 缓存命中为 0(不再重算)
        self.scan_cells_s = 0.0 if _cache_hit else float(self.scan_cells_once_s)
        # --- ② A/B: 线段全探点 ¬工作区(hr) ∧ ¬桌外(hr), 长度 >= min_path ---
        rng_ab = self._rng(1)
        ts = np.linspace(0.0, 1.0, PATH_PROBES)[:, None]
        _PF = np.linspace(0.0, 1.0, max(int(_PATH_FINE), 2))[:, None]
        mg = float(_GA._ARM["margin"]) + hr

        def _seg_ok(aa, bb, need):
            if np.hypot(*(bb - aa)) < need:
                return None
            # 中轴细采样做工作区/桌面约束: 手体脚印 ⊆ 半径 hr 圆盘 ⇒ 只要中轴处处
            # 距工作区 >= hr, 即可**保证**任何子步位置上"手体都不进工作区"。
            fine = aa[None, :] + (bb - aa)[None, :] * _PF
            if self._in_work_zone(fine, hr).any() or self._off_table(fine, hr).any():
                return None
            return aa[None, :] + (bb - aa)[None, :] * ts

        a = b = np.asarray(self.cells_g[0], dtype=float)
        seg = None
        for _ in range(MAX_CELL_TRIES):                     # 通用拒绝采样
            aa = self.cells_g[rng_ab.randint(len(self.cells_g))]
            bb = self.cells_g[rng_ab.randint(len(self.cells_g))]
            seg = _seg_ok(aa, bb, self.min_path)
            if seg is not None:
                a, b = aa, bb
                break
        if seg is None:                                     # 构造性兜底
            if self.work is not None:
                # 数据口径: 工作区不是矩形, 没有"带"的概念 -> 用整个安全格集合
                # (线段本身已由 _seg_ok 逐点校验: ¬工作区(hr) ∧ ¬桌外(hr))
                bands = [self.cells_g]
            else:
                bands = [self.cells_g[self.cells_g[:, 1] <=
                                      float(_GA._ARM["y"][0]) - mg - 1e-9],  # 下方带
                         self.cells_g[self.cells_g[:, 0] <=
                                      -float(_GA._ARM["x"][1]) - mg - 1e-9],  # 左列
                         self.cells_g[self.cells_g[:, 0] >=
                                      float(_GA._ARM["x"][1]) + mg + 1e-9]]   # 右列
            for need in (self.min_path, 0.9 * self.min_path,
                         0.6 * self.min_path, 0.0):
                for band in bands:
                    if len(band) == 0:
                        continue
                    for _ in range(MAX_CELL_TRIES):
                        aa = band[rng_ab.randint(len(band))]
                        bb = band[rng_ab.randint(len(band))]
                        seg = _seg_ok(aa, bb, need)
                        if seg is not None:
                            a, b = aa, bb
                            break
                    if seg is not None:
                        break
                if seg is not None:
                    break
        if seg is None:                                     # 终极兜底: 静止的手
            print("[dynamic_hand_mw] WARN: 无可行 A/B 对, 退化为静止手(位移 0)")
            a = b = np.asarray(self.cells_g[0], dtype=float)
        self.A = np.asarray(a, dtype=float)
        self.B = np.asarray(b, dtype=float)
        self.L = float(np.hypot(*(self.B - self.A)))
        seg = self.A[None, :] + (self.B - self.A)[None, :] * ts
        # --- ③ 朝向: 手指指向任务中心(assembly 常量 / 数据质心) ---
        inward = self.task_center - 0.5 * (self.A + self.B)
        nrm = float(np.linalg.norm(inward))
        self.yaw = (float(np.arctan2(inward[1], inward[0]))
                    if nrm > 1e-9 else 0.0)
        # --- 审计: 手体(全部表面点)沿整条路径是否越界/进工作区 ---
        self._audit_zone(seg)
        # --- ④ z: episode 级; 上限 = 整条路径上最紧的"可见"高度 ---
        # 一次批量算出 (高度网格 × 路径探点) 的可见比例表 —— 与逐点 z_cap 等价
        # (逐探点连续区间取 min == "所有探点同时满足"的连续区间), 但省掉 ~200 次
        # 单点调用(那是 start_episode 在向量化之后剩下的主要开销)。
        zs = np.arange(z_lo, self.z_hi_cfg + 1e-9, _Z_STEP)
        nz, npr = len(zs), len(seg)
        F = self._vis_frac_batch(
            np.tile(seg[:, 0], nz), np.tile(seg[:, 1], nz),
            np.repeat(zs, npr), self.yaw).reshape(nz, npr)
        if self.occ is not None:
            # 遮挡感知的同一判据(子样顶点)取 AND: 高度上限不再只看"在画幅内"
            F = np.minimum(F, self._occ_frac_batch(
                np.tile(seg[:, 0], nz), np.tile(seg[:, 1], nz),
                np.repeat(zs, npr), self.yaw).reshape(nz, npr))
        self.scan_z_s = float(self.last_scan_s)
        self.scan_s = self.scan_cells_s + self.scan_z_s
        self.z_hi = float(z_lo)
        for i, ok in enumerate((F >= self.vis_min).all(axis=1)):
            if not ok:                       # 自 z_lo 向上, 首个不满足即止
                break
            self.z_hi = float(zs[i])
        self.z_full = None
        for i, ok in enumerate((F >= 1.0).all(axis=1)):
            if not ok:
                break
            self.z_full = float(zs[i])
        z_hi = (min(self.z_hi, self.z_full)
                if self.z_full is not None else self.z_hi)
        self.z = float(self._rng(2).uniform(z_lo, max(z_lo, z_hi)))
        self.path_vis_min = float(self._vis_frac_batch(
            seg[:, 0], seg[:, 1], self.z, self.yaw).min())
        if self.occ is not None:
            # 审计: 遮挡感知版本 + "已进画幅但被挡"的比例
            fo = self._occ_frac_batch(seg[:, 0], seg[:, 1], self.z, self.yaw)
            self.path_vis_min_occ = float(fo.min())
            self.occluded_frac = float(self.path_vis_min - self.path_vis_min_occ)
        else:
            self.path_vis_min_occ = self.path_vis_min
            self.occluded_frac = 0.0
        # 审计: 手顶点压在任务物体屏幕足迹上的比例(与遮挡/可见性解耦)
        self.obj_frac = None
        if self.obj_dist is not None:
            of = [self._obj_frac(float(sx), float(sy), self.z, self.yaw)
                  for sx, sy in seg]
            of = [v for v in of if v is not None]
            self.obj_frac = float(max(of)) if of else None
        # t=0 位形(必须在 env.reset() 之前写, 首帧才含手)
        x, y = self._path_xy(0)
        self._write(float(x), float(y), self.z, model_default=True)
        self.pos = (float(x), float(y), self.z)
        self.records.append((0, float(x), float(y), self.z))
        return self.pos

    def plan(self):
        """决策边界: 仅记账(运动是逐 substep 的纯函数, 无需重采样)。"""
        self.k += 1
        return self.pos

    def tick(self, t):
        """逐 substep 写 t 时刻位形(纯函数 ⇒ 同 (seed, ep) 逐位可复现)。"""
        if int(t) == 0:
            e = self.pose_err()
            if e > 1e-6:      # reset/set_env_state 等外部路径改写过关节 -> 暴露
                print(f"[dynamic_hand_mw] WARN tick(0) 前姿态偏差 "
                      f"pose_err={e:.2e} (即将重写)")
        x, y = self._path_xy(t)
        self._write(float(x), float(y), self.z)
        self.pos = (float(x), float(y), self.z)
        if len(self.records) < MAX_TICK_RECORDS:
            self.records.append((int(t), float(x), float(y), self.z))
        return self.pos

    def finish(self):
        """平滑运动无"未到位"状态, 保留空实现以兼容 runner 调用。"""
        return None

    def reapply(self):
        """重写 t=0 位形(建议在 `env.reset()` 之后立刻调用)并返回 pose_err。

        静态体: body_pos/body_quat 不会被 mj_reset 改动; 但 qpos 会被写回 qpos0
        (start_episode 已把手指 qpos0 同步为 POSE_OPEN)。本方法纯属防御性兜底 ——
        若某些 wrapper 路径改写了几何/关节, 返回值会立刻暴露出来。
        """
        x, y = self._path_xy(0)
        self._write(float(x), float(y), self.z, model_default=True)
        self.pos = (float(x), float(y), self.z)
        return self.pose_err()

    # ---------------- reporting ----------------
    def pose_err(self):
        """实际 qpos 与目标(固定 open)的最大偏差(证明姿态没被物理/动作改写)。"""
        d = [abs(float(self.sim.data.qpos[self.jadr[nm]]) - v)
             for nm, v in self.pose.items()]
        return float(max(d)) if d else 0.0

    def spec(self):
        """episode 级规格(供 manifest 记录); A/B 转 list 便于 json.dump。"""
        g = self.geom
        return dict(
            pose="open", yaw=float(self.yaw), z=float(self.z or 0.0),
            speed=float(self.speed), dt=float(self.dt), path_len=float(self.L),
            min_path=float(self.min_path),
            A=None if self.A is None else [float(v) for v in self.A],
            B=None if self.B is None else [float(v) for v in self.B],
            hand_hr=float(g.get("hr", 0.0)), vis_frac_min=float(self.vis_min),
            z_full=None if self.z_full is None else float(self.z_full),
            path_vis_min=(None if self.path_vis_min is None
                          else float(self.path_vis_min)),
            # 工作区审计(与 posgen 同源): 手体全程不进工作区 / 不越桌
            zone_ok=bool(self.zone_ok), table_ok=bool(self.table_ok),
            work_clear_min=(None if self.work_clear_min is None
                            else float(self.work_clear_min)),
            center_clear_min=(None if self.center_clear_min is None
                              else float(self.center_clear_min)),
            body_clear_min=(None if self.body_clear_min is None
                            else float(self.body_clear_min)),
            hand_hr_used=float(self.hr),
            task_center=[float(v) for v in self.task_center],
            work_src=("assembly-const" if self.work is None
                      else "zarr-footprint"),
            work_margin=(None if self.work is None
                         else float(self.work["margin"])),
            # 可见性/遮挡审计: vis_mode=analytic|occlusion-aware
            vis_mode=str(self.vis_mode),
            vis_relax=float(self.vis_relax),
            path_vis_min_occ=(None if self.path_vis_min_occ is None
                              else float(self.path_vis_min_occ)),
            occluded_frac=(None if self.occluded_frac is None
                           else float(self.occluded_frac)),
            obj_frac=(None if self.obj_frac is None else float(self.obj_frac)),
            occ_src=("none" if self.occ is None else "zarr-union"),
            occ_pts=int(self.occ_pts), occ_splat=int(self.occ_splat),
            occ_scan_s=float(self.occ_scan_s),
            scan_s=float(self.scan_s),
            pose_err=self.pose_err(),
            green_cells_feasible=int(0 if self.cells_g is None
                                     else len(self.cells_g)),
            green_cells_full=int(self.cells_full),
            green_cells_all=int(len(self.cells)))

    def history(self):
        return [list(r) for r in self.records]




