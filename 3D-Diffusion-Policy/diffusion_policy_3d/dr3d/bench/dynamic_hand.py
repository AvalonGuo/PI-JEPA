#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: 动态干扰物 —— inspire 左手(kinematic 平移, 固定 open 姿态),
逐 substep 在可放置绿区内**来回移动**(平滑乒乓)。

语义(与用户确认的 spec 一致):
  * **姿态 = 固定 open**: 全程用 `POSE_OPEN`(12 个手指 hinge 的显式角度)写 qpos:
    五指自然张开, **不是全 0**(每指约 0.36~0.42; 拇指 0.19/0.42/0.39/0.38)。
    asset 的 5 处关节上限已按该值放宽(见 distractors/inspire/model.xml 头注释)。
  * **位置 = substep 级(平滑)**: episode 级抽两个绿区端点 A/B, 沿线段**乒乓**
    (三角波)往返; 每评测子步位移 = SPEED * dt(dt = timestep *
    FRAMES_PER_SUBSTEP); 相位只依赖 substep 序号 ⇒ 与 rollout 长度/调用顺序无关,
    同一 (seed, ep) 逐位可复现。
  * **高度 = episode 级**: z = U(z_lo, z_cap); z_lo = max(Z_RANGE[0], 手底离桌高度);
    z_cap = 整条路径上"手在画幅内比例 >= VIS_FRAC_MIN"的最高中心高度(bisection)。
  * **朝向 = 基准 + 每 episode 随机 yaw**: 基准为"掌心朝下、手指指向 +x"
    (BASE_QUAT 由手掌 mesh PCA 法向 + 指节方向现场推导, 命名无关, 见 hand_geometry);
    yaw 通过 mocap_quat 叠加。
  * **零物理交互**: 手挂在 mocap(kinematic) 根下(平移/朝向由 mocap_pos /
    mocap_quat 驱动), 且所有 geom contype=conaffinity=0 ⇒ 不产生任何接触,
    可穿过机械手/锤子/桌面, **只影响点云/图像**(同时避开 mesh-mesh 碰撞开销)。
  * **nq/nv +12**(手指关节): agent_pos 仍为 24 维且逐位不变 —— 见
    env/adroit/rrl_local/rrl_multicam.py `if sim.model.nq > 33: qp = qp[:24]`,
    注入体追加在 worldbody 末尾 ⇒ 新增关节的 qpos 在最后, 24 维前缀不受影响。

首帧含手: `start_episode()` 必须在 `env.reset()` **之前**调用 —— 它写
    model.body_pos / model.body_quat(供 mj_reset 复制到 data.mocap_pos/quat)
    + data.mocap_pos / mocap_quat + 12 个 qpos, 再 `sim.forward()`
  ⇒ reset 后渲染的首帧即包含手且姿态/朝向正确。

资产: `dr3d/distractors/inspire/model.xml`(inspire 左手; 12 关节/拇指 4 节;
vendor 原件快照见 git 历史)。

door 兼容性(2026-09 验证): 本类无 hammer 专属假设 —— 相机几何(cam_z/fovy)
与时步(opt.timestep * FRAMES_PER_SUBSTEP)均在 __init__ 从 sim model 运行时读取
(常量仅作兜底), door 的 top 相机 z=1.5(45°) / frame_skip=5 自动适配, 0.90 m/s
语义与 hammer 一致; 观测侧 door 分支(nq>30 -> qp[:24])与 hammer 同构, 注入体
qpos 追加在 worldbody 末尾, 24 维手状态逐位不变(见 rrl_multicam.py); 绿区按
task 分派(door 372 格, 同 posgen 源)。评测冒烟/正式见 eval_policy.sh 的 dynhand 场景。

相机模式(2026-10, pen 适配): 可见性判据按**该 task 的真实策略相机**分派 —— 直接读
`ENV_POINT_CLOUD_CONFIG['adroit_<task>']` 的 cam_names(与 dynamic.py 同源, 不做猜测):
  * 含 'top'(hammer/door)    -> mode='top': 保留原顶视闭式画幅判据, 逐值不变;
  * 不含 'top'(pen 只有斜视 vil_camera) -> mode='frustum': 手的世界顶点先过 point-cloud
    crop(transform/scale/offset + min/max_bound, 严格不等式)再过该相机针孔视锥
    (画幅内留 VIS_MARGIN_PX); 两项都过才算"该顶点进入策略观测"(见 cam_geom.py ★恒等式)。
  ⚠ mujoco_py 的 `camera_name2id()` 对**不存在**的相机是**抛 ValueError**(不是返回 -1),
    所以绝不能裸调 `camera_name2id('top')` —— 必须先用 `model.camera_names` 判定存在性。
"""
import os
import pathlib

import numpy as np

from dynamic import unwrap_sim, policy_cloud_config   # noqa: F401 (bench/ 已注入 sys.path)

DR3D = pathlib.Path(__file__).resolve().parents[1]


def _cg():
    """bench 内的通用相机几何(cam_geom) —— 惰性导入, 与 dynamic.py 的用法保持一致。"""
    import cam_geom
    return cam_geom


# 评测脚本默认 DR3D_QUIET=1: 常规信息/审计告警不打印(与 clean 评测输出一致; 0 恢复)
_QUIET = os.environ.get("DR3D_QUIET", "0") == "1"


def _crop_frustum_ok(w, g, lo, hi, T, sc, off, margin_px):
    """(N,3) 世界点 -> (N,) bool: crop(严格不等式) ∧ 针孔视锥(画幅内留 margin)。

    与 cam_geom.to_cloud / cam_geom.project **逐式等价**(判据同源, 见 cam_geom.py 头
    注释的 ★恒等式与 mujoco_point_cloud.py 的观测链), 但刻意用**显式标量乘加**而不是
    `@`: 7.6 万点 × 3×3 的小 GEMM 会走 BLAS 多线程, 线程同步开销让单次判据慢一个数量级
    (同 dynamic_hand_mw._vis_frac_batch 的经验)。只读, 不改 sim。
    """
    x = w[:, 0] - g["pos"][0] + g["body_pos"][0]
    y = w[:, 1] - g["pos"][1] + g["body_pos"][1]
    z = w[:, 2] - g["pos"][2] + g["body_pos"][2]
    if T is not None:
        T = np.asarray(T, dtype=float)
        x, y, z = (x * T[0, 0] + y * T[1, 0] + z * T[2, 0],
                   x * T[0, 1] + y * T[1, 1] + z * T[2, 1],
                   x * T[0, 2] + y * T[1, 2] + z * T[2, 2])
    if sc is not None:
        sc = np.ravel(np.asarray(sc, dtype=float))
        x, y, z = x * sc[0], y * sc[1], z * sc[2]
    if off is not None:
        off = np.ravel(np.asarray(off, dtype=float))
        x, y, z = x + off[0], y + off[1], z + off[2]
    crop_ok = ((x > lo[0]) & (x < hi[0]) & (y > lo[1]) & (y < hi[1]) &
               (z > lo[2]) & (z < hi[2]))
    R = g["R"]
    rx = w[:, 0] - g["pos"][0]
    ry = w[:, 1] - g["pos"][1]
    rz = w[:, 2] - g["pos"][2]
    ax = rx * R[0, 0] + ry * R[1, 0] + rz * R[2, 0]      # == (p-pos) @ R 的第 0 列
    ay = rx * R[0, 1] + ry * R[1, 1] + rz * R[2, 1]
    az = rx * R[0, 2] + ry * R[1, 2] + rz * R[2, 2]
    front = az < 0.0                                     # z_cam<0 = 在相机前方
    dz = np.where(front, -az, 1.0)                       # 背面点不参与除法
    u = g["loc"][0] + g["f"] * ax / dz
    v = g["loc"][1] - g["f"] * ay / dz
    m = float(margin_px)
    hi_px = float(g["img"]) - 1.0 - m
    frus_ok = front & (u >= m) & (u <= hi_px) & (v >= m) & (v <= hi_px)
    return crop_ok & frus_ok

ASSET = DR3D / "distractors" / "inspire" / "model.xml"

DEFAULT_NAME = "dr_inspire"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# --------------------------------------------------------------------------- #
# 固定姿态(open): 12 维关节角(rad), 按**关节名**给出 —— 五指自然张开, 不是全 0。
# 来源 = 用户提供的参考姿态(MuJoCo MJDATA 导出的 QPOS 段, 12 值, 与模型 qpos 完全同序:
#        thumb_proximal_1 -> pinky_distal); 数值 1:1 固化于此。
# --------------------------------------------------------------------------- #
POSE_OPEN = {
    "joint_l_thumb_proximal_1": 0.19, "joint_l_thumb_proximal_2": 0.42,
    "joint_l_thumb_middle": 0.39, "joint_l_thumb_distal": 0.38,
    "joint_l_index_proximal": 0.36, "joint_l_index_distal": 0.37,
    "joint_l_middle_proximal": 0.36, "joint_l_middle_distal": 0.37,
    "joint_l_ring_proximal": 0.36, "joint_l_ring_distal": 0.37,
    "joint_l_pinky_proximal": 0.36, "joint_l_pinky_distal": 0.37,
}
SPEED = 0.90                 # m/s 匀速(用户要求 ×3: 0.30 -> 0.90; 18 mm/子步)
MIN_PATH_ABS = 0.30          # 路径最短长度(m)兜底
MIN_PATH_TIME = 0.45         # 单趟最短时长(s): min_path = max(ABS, SPEED*TIME) = 0.405m
CENTER_CLEARANCE = 0.22      # A/B 选点: 手中心距场景中心(桌面原点) >= 22cm(≈24px)
MID_CLEARANCE = 0.15         # A/B 中点也必须离中心足够远, 否则"朝中心"方向不明确
ORIENT_DOT_MIN = 0.6         # 路径采样点(端点+中点)上 "手指方向·朝中心方向" 的最小值
HAND_SCALE = 1.0             # mesh 缩放(1.0 = 真实手尺寸 ~0.18 m, 在 84x84 里 12-21 px);
                             # 设 2.0 可让手更大更清晰 —— 必须与场景 compose 的 scale
                             # 一致(eval 链路已把本常量传给 scene_runtime.activate 与
                             # HandDistractor, 缓存名会带上 scale, 不会串档)

Z_RANGE = (0.12, 0.40)       # 手"几何中心"高度的配置窗口(m; 桌面顶 z=0) —— 抬高
VIS_FRAC_MIN = 0.70          # "大部分在范围内": 手表面采样点落在画幅内的最小比例
FRAME_SKIP = 5               # hammer_v0 的 MujocoEnv(frame_skip=5)
NUM_REPEATS = 2              # AdroitEnv(num_repeats=2) 默认(每评测子步重复 2 次)
FRAMES_PER_SUBSTEP = FRAME_SKIP * NUM_REPEATS
DT_FALLBACK = 0.002 * FRAMES_PER_SUBSTEP        # 兜底 dt(hammer 用 MuJoCo 默认 timestep)
CAM_Z_FALLBACK, FOVY_FALLBACK = 1.1, 45.0       # 渲染相机 top 的兜底几何
IMG_SIZE = 84                # 策略观测点云/深度图边长(与 dynamic.py 一致)
VIS_MARGIN_PX = 6.0          # 画幅内判定余量(px): 解析判据与渲染轮廓实测差 ~3-5px
                             # (像素覆盖 + 取整), 取 6px 保证"整只手真的在画幅内"
N_SURF_PTS = 0               # 可见性采样点数: 0 = 用**全部 mesh 顶点**(=投影轮廓的极值点,
                             # 判据才等价于"整只手在画幅内"); >0 则等距抽样(更快但偏乐观)
INSET = 0.41                 # 桌面可放置内缩界(与 green_cells.PARAMS['inset'] 一致)
PATH_PROBES = 7              # 路径上用于"整条轨迹都满足可见性"的采样点数
MAX_CELL_TRIES = 256         # A/B 拒绝采样上限
MAX_TICK_RECORDS = 4096      # 逐子步位形日志上限


# --------------------------------------------------------------------------- #
# 小工具: 四元数(wxyz; MuJoCo 约定) 与确定性 RNG
# --------------------------------------------------------------------------- #
def qmul(a, b):
    """Hamilton 积 a⊗b(wxyz): 先施加 b 再施加 a(用于 mocap_quat 的 yaw⊗基准)。"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_z(yaw):
    """绕 z 轴 yaw 的四元数(wxyz)。"""
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def finger_of(joint_name):
    """关节名 -> 手指键(thumb/index/middle/ring/pinky); 未识别返回 None。"""
    for k in FINGERS:
        if k in joint_name:
            return k
    return None


_GEOM_CACHE = {}


def hand_geometry(asset=ASSET, scale=1.0, verbose=False):
    """(进程内缓存) 从 standalone 资产推导基准朝向与每档手势的局部几何。

    scale: mesh 缩放(与 compose_scene/activate 的 scale 一致; 只缩放 mesh 顶点,
           不动关节链 —— 与 compose 的 per-mesh scale 语义相同), 进缓存 key。

    返回 dict:
        base_quat : (4,) wxyz, 把"掌心朝下、手指指向 +x"叠到模型自带 root quat;
        center(3,) / zmin / zmax / dz_up / hr / n_vert
        points_rel = mesh 表面采样点 **相对几何中心**(基准朝向帧, 固定 open 姿态)。
    推导方式(确定性, 不依赖手工常量):
        手指方向 u = (四指中节质心 - 掌根) 归一化;
        掌心内法向 n = 掌 mesh 顶点 PCA 最小奇异向量(符号取朝向拇指一侧);
        目标: u -> +x, n -> -z  ⇒ R = [x̂ | -ẑ | ŷ] · [u | v | w]^T, w = u × v
        再与模型自带 root quat 复合(基准姿态里的 u/n 已含 root quat)。
    """
    scale = float(scale)
    key = str(asset) + "|s%g" % scale
    if key in _GEOM_CACHE:
        return _GEOM_CACHE[key]
    from mujoco_py import load_model_from_path, MjSim
    from scipy.spatial.transform import Rotation as Rot

    m = load_model_from_path(str(asset))
    s = MjSim(m)
    d = s.data
    # ---- 命名无关的结构发现(不硬编码 vendor 名字; 同族手掌资产通用) ----
    root = next(b for b in range(1, m.nbody) if int(m.body_parentid[b]) == 0)
    palm_bid = next((b for b in range(m.nbody) if "palm" in m.body_names[b]), root)
    palm_gid = next((g for g in range(m.ngeom) if "palm" in m.geom_names[g]), None)
    desc, changed = {root}, True
    while changed:                       # 根的子树(= 独立资产里的整只手)
        changed = False
        for b in range(m.nbody):
            if b not in desc and int(m.body_parentid[b]) in desc:
                desc.add(b)
                changed = True
    q0 = np.array(m.body_quat[root], dtype=float)

    def set_pose(q):
        d.qpos[:] = q
        d.qvel[:] = 0.0
        s.forward()

    def verts():
        out = []
        for g in range(m.ngeom):
            if int(m.geom_type[g]) != 7:          # 7 = mjGEOM_MESH
                continue
            mid = int(m.geom_dataid[g])
            if mid < 0:
                continue
            v = m.mesh_vert[m.mesh_vertadr[mid]:
                            m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].reshape(-1, 3)
            out.append((v * scale) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.vstack(out)

    def tip_body(key):
        """该指最远端的 body(零位形下离掌心最远) —— 命名无关, 需先 set_pose()。"""
        cands = [b for b in desc if key in m.body_names[b]]
        if not cands:
            raise ValueError(f"hand_geometry: 未找到 {key} 对应的 body")
        palm = d.body_xpos[palm_bid]
        return max(cands, key=lambda b: float(np.linalg.norm(d.body_xpos[b] - palm)))

    def palm_normal():
        g = palm_gid
        if g is None:                        # 兜底: 掌 body 上的第一个 geom
            g = next(x for x in range(m.ngeom)
                     if int(m.geom_bodyid[x]) == palm_bid)
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:
                        m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].reshape(-1, 3)
        v = (v * scale) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        c = v.mean(0)
        _, _, vt = np.linalg.svd(v - c, full_matrices=False)
        n = vt[2]
        thumb = d.body_xpos[tip_body("thumb")] - c      # 内法向: 指向拇指一侧
        n = n if n @ thumb > 0 else -n
        return n / np.linalg.norm(n)

    def finger_dir():
        """掌心 -> 四指(食/中/无名/小指)最远端 body 质心 的方向(零位形 FK)。"""
        tips = [tip_body(k) for k in ("index", "middle", "ring", "pinky")]
        return (np.mean([d.body_xpos[b] for b in tips], 0)
                - d.body_xpos[palm_bid])

    set_pose(np.zeros(m.nq))
    u = finger_dir()
    u = u / np.linalg.norm(u)
    n0 = palm_normal()
    v = n0 - (n0 @ u) * u
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    m_tgt = np.stack([np.array([1.0, 0.0, 0.0]),
                      np.array([0.0, 0.0, -1.0]),
                      np.array([0.0, 1.0, 0.0])], axis=1)
    r = m_tgt @ np.stack([u, v, w], axis=1).T
    qr = Rot.from_matrix(r).as_quat()
    base_quat = qmul(np.array([qr[3], qr[0], qr[1], qr[2]]), q0)
    m.body_quat[root] = base_quat                 # 后续测量都在基准朝向下

    table = POSE_OPEN
    q = np.zeros(m.nq)
    for j, nm in enumerate(m.joint_names):
        if nm in table:
            q[int(m.jnt_qposadr[j])] = float(table[nm])
    set_pose(q)
    val = verts()
    center = 0.5 * (val.max(0) + val.min(0))
    if N_SURF_PTS <= 0:
        idx = np.arange(len(val))                      # 全部顶点(=真实轮廓)
    else:
        idx = np.linspace(0, len(val) - 1,
                          min(N_SURF_PTS, len(val))).astype(int)
    pose_geom = dict(
        center=center, zmin=float(val[:, 2].min()), zmax=float(val[:, 2].max()),
        dz_up=float(val[:, 2].max() - center[2]),   # 顶点最高处相对中心(视锥最紧处)
        hr=float(np.max(np.abs(val[:, :2] - center[:2]))),
        n_vert=int(len(val)), points_rel=val[idx] - center)

    # 自检: 基准朝向下应为 手指->+x, 掌心内法向->-z
    set_pose(np.zeros(m.nq))
    f_chk = finger_dir()
    f_chk = f_chk / np.linalg.norm(f_chk)
    n_chk = palm_normal()
    geom = dict(base_quat=base_quat, **pose_geom,
                check=dict(finger_dir=[float(x) for x in f_chk],
                           palm_normal=[float(x) for x in n_chk]))
    if verbose:
        print(f"[dynamic_hand] geometry: base_quat(wxyz)="
              f"{np.round(base_quat, 6).tolist()} finger_dir="
              f"{np.round(f_chk, 3).tolist()} palm_normal={np.round(n_chk, 3).tolist()}")
        print(f"[dynamic_hand]   pose=open center="
              f"{np.round(pose_geom['center'], 3).tolist()} "
              f"hr={pose_geom['hr']:.3f} "
              f"z[{pose_geom['zmin']:+.3f},{pose_geom['zmax']:+.3f}] "
              f"verts={pose_geom['n_vert']}")
    _GEOM_CACHE[key] = geom
    return geom


class HandDistractor:
    """inspire 左手: 逐 substep 在绿区线段上乒乓移动(零接触, 只影响点云/图像)。

    与 `dynamic.py::DynamicDistractor` 同款接口:
    `start_episode / plan / tick / finish / history / pos`, 另有 `spec()` 供 manifest。
    """

    def __init__(self, sim, cells, name=DEFAULT_NAME, speed=SPEED,
                 z_range=Z_RANGE, vis_frac_min=VIS_FRAC_MIN, scale=HAND_SCALE,
                 frames_per_substep=FRAMES_PER_SUBSTEP, task_name=None,
                 verbose=True):
        self.sim = sim
        self.cells = np.asarray(cells, dtype=float)
        if len(self.cells) == 0:
            raise ValueError("HandDistractor: empty green-cell list")
        self.name = name
        self.speed = float(speed)
        self.z_lo_cfg, self.z_hi_cfg = float(z_range[0]), float(z_range[1])
        self.vis_min = float(vis_frac_min)
        self.scale = float(scale)
        self.geom = hand_geometry(scale=self.scale, verbose=verbose)
        # --- kinematic(mocap) 根 ---
        self.body_id = int(sim.model.body_name2id(name))
        self.mid = int(sim.model.body_mocapid[self.body_id])
        if self.mid < 0:
            raise ValueError(f"{name} 不是 mocap body(mocap_pos/quat 不可用)")
        # 按 body 子树反查 geom(注入后是 dr_inspire 的嵌套子 body, 不能只看直接父)
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
        # --- 11 个手指关节的 qpos 地址(按名字查, 不依赖关节顺序) ---
        self.jadr, self.jfinger = {}, {}
        for j, nm in enumerate(sim.model.joint_names):
            for k in FINGERS:
                if k in nm:
                    self.jadr[nm] = int(sim.model.jnt_qposadr[j])
                    self.jfinger[nm] = k
        if not self.jadr:
            raise ValueError("HandDistractor: 手部手指关节未找到")
        # 固定姿态 -> 本场景 qpos 地址; 关节名必须与姿态表一致(不一致就早失败)
        missing = [nm for nm in self.jfinger if nm not in POSE_OPEN]
        if missing:
            raise ValueError(f"POSE_OPEN 缺少关节: {missing}")
        self.pose = {nm: float(POSE_OPEN[nm]) for nm in self.jfinger}
        self.min_path = max(MIN_PATH_ABS, self.speed * MIN_PATH_TIME)
        # --- 可见性: 按 task 的真实策略相机分派(见模块头"相机模式") ---
        #   hammer/door: cam_names=['top']      -> 保留旧顶视闭式(逐值不变)
        #   pen        : cam_names=['vil_camera'] (斜视) -> frustum(crop ∧ 针孔视锥)
        self.task_name = None if task_name is None else str(task_name)
        self.mode = "top"
        self.cam_names = ["top"]
        self.geom_info = []          # frustum: 每台策略相机的外参/内参(cam_geom)
        self.crop = None             # frustum: (min_bound, max_bound, T, scale, offset)
        cfg = (policy_cloud_config(self.task_name)
               if self.task_name is not None else None)
        if cfg is not None and "top" not in cfg["cam_names"]:
            self.mode = "frustum"
            self.cam_names = [str(n) for n in cfg["cam_names"]]
            self.geom_info = [_cg().geometry(sim.model, nm, img=IMG_SIZE)
                              for nm in self.cam_names]
            self.crop = (cfg["min_bound"], cfg["max_bound"], cfg["transform"],
                         cfg["scale"], cfg["offset"])
            self.cam_z, self.tan_half, self.tan_eff = None, None, None
        else:
            # top 模式(hammer/door, 或未给 task_name 的旧链路): 方形画幅闭式
            self.cam_z, self.tan_half = CAM_Z_FALLBACK, float(
                np.tan(np.radians(FOVY_FALLBACK) / 2.0))
            _cams = [str(n) for n in sim.model.camera_names]   # 存在性判定(不裸调 id 查询)
            if "top" in _cams:
                cid = int(sim.model.camera_name2id("top"))
                try:
                    self.cam_z = float(sim.model.cam_pos[cid][2]) or CAM_Z_FALLBACK
                    self.tan_half = float(
                        np.tan(np.radians(float(sim.model.cam_fovy[cid])) / 2.0))
                except Exception:
                    pass
            elif verbose:
                print(f"[dynamic_hand] 模型里没有 'top' 相机(available={_cams}); "
                      f"顶视闭式退回兜底几何 cam_z={CAM_Z_FALLBACK} "
                      f"fovy={FOVY_FALLBACK}(如需严格判据请传 task_name)")
            self.tan_eff = self.tan_half * (1.0 - VIS_MARGIN_PX / (IMG_SIZE / 2.0))
        # 判据实现按模式在此**一次性绑定**, 调用点不再判断模式
        if self.mode == "frustum":
            self._visible_frac = self._visible_frac_frustum
            self._cell_filter = self._cell_filter_frustum
            self._audit = self._audit_frustum
        else:
            self._visible_frac = self._visible_frac_top
            self._cell_filter = self._cell_filter_top
            self._audit = self._audit_top
        _cam_list = (None if self.mode == "top" else
                     [(gi['cam'], np.round(gi['pos'], 3).tolist(), gi['fovy'])
                      for gi in self.geom_info])
        self._vis_desc = (f"cam_z={self.cam_z:.3f} tan_eff={self.tan_eff:.4f}"
                          if self.mode == "top" else
                          f"mode=frustum cams={_cam_list}")
        # --- 子步时长: 每评测子步 = FRAMES_PER_SUBSTEP 个 sim 帧 ---
        try:
            self.dt = float(sim.model.opt.timestep) * float(frames_per_substep)
        except Exception:
            self.dt = DT_FALLBACK
        # --- episode 级状态(由 start_episode 填) ---
        self.seed = self.ep = None
        self.yaw = 0.0
        self.A = self.B = None
        self.z = None
        self.L = 0.0
        self.cells_g = None        # 该 yaw 下满足可见性阈值的绿区格
        self.cells_full = 0        # 其中"整只手都在画幅内"(vis_frac==1.0) 的格数
        self.z_full = None         # 本 episode(路径) 下"整手在画幅内"的最高 z
        self.path_vis_min = None   # 最终 z 下整条路径的最小可见比例(审计)
        self.k = 0                 # 决策步计数(plan 调用次数)
        self.records = []          # [(t, x, y, z), ...] 逐 substep 位形
        self.pos = None            # 当前位置 (x, y, z)(手几何中心)
        if verbose:
            print(f"[dynamic_hand] {name}: body={self.body_id} mocap_id={self.mid} "
                  f"joints={len(self.jadr)} cells={len(self.cells)} scale={self.scale:g} "
                  f"speed={self.speed} min_path={self.min_path:.3f}m dt={self.dt:.4f}s "
                  f"z_range={z_range} vis_min={self.vis_min:.2f} task={self.task_name} "
                  f"{self._vis_desc}")

    # ---------------- deterministic RNG ----------------
    def _rng(self, channel):
        """逐 (seed, ep, channel) 的确定性 RNG(与 dynamic.py 同款混淆, 通道互不碰撞)。"""
        s = (int(self.seed) * 2654435761
             + int(self.ep) * 1000003
             + (int(channel) + 2) * 7919) % (2 ** 32)
        return np.random.RandomState(s)

    # ---------------- geometry / visibility ----------------
    def _pts_world(self, x, y, z):
        """表面采样点(相对中心)平移到中心 (x, y, z) 后的 (px, py, pz)。"""
        p = self.geom["points_rel"]
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        px = cy * p[:, 0] - sy * p[:, 1] + x
        py = sy * p[:, 0] + cy * p[:, 1] + y
        return px, py, p[:, 2] + z

    def _vis_mask(self, x, y, z):
        """[frustum 模式] 逐顶点可见性: crop(严格不等式) ∧ 策略相机针孔视锥(含 margin)。

        与观测链完全同源(见 dr3d/bench/cam_geom.py 头注释的 ★恒等式):
            cloud = pcT @ (p_world - cam_pos + cam_body_pos) 落在 (min_bound, max_bound) 内;
            u,v   = 针孔投影(IMG_SIZE×IMG_SIZE) 落在 [margin, IMG_SIZE-1-margin] 且 z_cam<0。
        多相机取**并**(手出现在任一策略相机的点云/图像里即算可见)。
        返回 (P,) bool, P = mesh 顶点数(见 hand_geometry: N_SURF_PTS<=0 时=全部顶点)。
        """
        px, py, pz = self._pts_world(x, y, z)
        w = np.stack([px, py, pz], axis=1)
        lo, hi, T, sc, off = self.crop
        ok = np.zeros(len(w), dtype=bool)
        for gi in self.geom_info:
            ok |= _crop_frustum_ok(w, gi, lo, hi, T, sc, off, VIS_MARGIN_PX)
        return ok

    def _aabb_visible(self, xs, ys, z, margin_px=None):
        """[frustum 模式] (xs,ys) 处、手几何中心高度 z 的**手 AABB** 是否完整可见(bool 数组)。

        保守且 **yaw 无关**: 水平取圆包围盒 r=hr(任意 yaw 的手都含在内), 竖直取 mesh 实际
        z 跨度 [zmin, zmax](相对几何中心)。判据与 `_vis_mask` 同一套(crop ∧ 视锥), 只是
        用 8 个角点代替 7.6 万顶点 —— 供候选格粗筛, 与 dynamic.py 的球 AABB 同思路。
        精确的 yaw 感知判据由 `visible_frac` / `z_cap` 在选定路径上给出。
        """
        g = self.geom
        rxy = float(g["hr"])
        z0, z1 = float(g["zmin"]), float(g["zmax"])
        xs = np.atleast_1d(np.asarray(xs, dtype=float))
        ys = np.atleast_1d(np.asarray(ys, dtype=float))
        n = int(xs.size)
        N = 8                                     # 4 个水平角 × 2 个竖直层
        w = np.empty((n, N, 3), dtype=float)
        k = 0
        for zz in (z0, z1):
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                w[:, k, 0] = xs + sx * rxy
                w[:, k, 1] = ys + sy * rxy
                w[:, k, 2] = float(z) + zz
                k += 1
        wf = w.reshape(-1, 3)
        lo, hi, T, sc, off = self.crop
        m = VIS_MARGIN_PX if margin_px is None else float(margin_px)
        ok = np.zeros(len(wf), dtype=bool)
        for gi in self.geom_info:
            ok |= _crop_frustum_ok(wf, gi, lo, hi, T, sc, off, m)
        return ok.reshape(n, N).all(axis=1)

    def _visible_frac_top(self, x, y, z):
        """[top] 顶视方形画幅闭式(max(|x|,|y|) <= tan_eff*(cam_z-z))。"""
        px, py, pz = self._pts_world(x, y, z)
        half = self.tan_eff * (self.cam_z - pz)
        return float(np.mean(np.maximum(np.abs(px), np.abs(py)) <= half))

    def _visible_frac_frustum(self, x, y, z):
        """[frustum] point-cloud crop ∧ 真实策略相机针孔视锥(见 `_vis_mask`)。"""
        return float(np.mean(self._vis_mask(x, y, z)))

    def _cell_filter_top(self, z_lo, dz_up):
        """[top] 圆包围盒闭式, 取手**最高顶点**处的高度算(逐值不变)。"""
        r = float(self.geom["hr"])
        xy = np.max(np.abs(self.cells), axis=1)
        return ((xy + r) <= self.tan_eff * (self.cam_z - z_lo - dz_up),
                (xy + r) <= self.tan_half * (self.cam_z - z_lo - dz_up))

    def _cell_filter_frustum(self, z_lo, dz_up):
        """[frustum] 手 AABB(crop ∧ 真实策略相机视锥) 在最低/最高两个中心高度上都要过。"""
        xs, ys = self.cells[:, 0], self.cells[:, 1]
        return ((self._aabb_visible(xs, ys, z_lo, margin_px=VIS_MARGIN_PX) &
                 self._aabb_visible(xs, ys, z_lo + dz_up, margin_px=VIS_MARGIN_PX)),
                (self._aabb_visible(xs, ys, z_lo, margin_px=0.0) &
                 self._aabb_visible(xs, ys, z_lo + dz_up, margin_px=0.0)))

    def _audit_top(self, caps, path_vis_min):
        return None

    def _audit_frustum(self, caps, path_vis_min):
        """[frustum] episode 级审计告警(两条独立判定, 文案与以往一致)。"""
        if _QUIET:
            return None
        if not caps:
            print("[dynamic_hand] WARN frustum: 路径上没有任何格的 z 满足可见性阈值; "
                  "z 退回配置上限(z_hi_cfg), 本 episode 的手可能部分出画幅")
        if path_vis_min + 1e-9 < self.vis_min:
            print(f"[dynamic_hand] WARN frustum: path_vis_min="
                  f"{path_vis_min:.3f} < vis_frac_min={self.vis_min:.2f} "
                  f"(斜视相机下二分非严格单调; spec/manifest 里已如实记录)")

    def visible_frac(self, x, y, z):
        """"大部分在范围内"判据: 采样点(手表面顶点)落在可见区内的比例。

        实现按模式在 __init__ 绑定: top=顶视方形画幅闭式(逐值不变);
        frustum=point-cloud crop ∧ 真实策略相机针孔视锥(pen 的 vil_camera)。
        """
        return self._visible_frac(x, y, z)

    def z_cap(self, x, y, lo, target=None):
        """`visible_frac >= target` 的最高中心高度(bisection; frac 关于 z 单调不增)。

        target 默认 self.vis_min(70%); 传 1.0 可求"整只手都在画幅内"的最高高度。
        ⚠ frustum 模式(斜视相机)下 frac 对 z 只是**近似**单调(抬高会同时改变深度与投影
          位置), 二分给出的是可靠上界附近的值; 最终以 start_episode 里的 path_vis_min
          审计为准(低于阈值会打印 WARN)。
        """
        target = self.vis_min if target is None else float(target)
        if self.visible_frac(x, y, lo) < target:
            return None
        if self.visible_frac(x, y, self.z_hi_cfg) >= target:
            return self.z_hi_cfg
        hi = self.z_hi_cfg
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if self.visible_frac(x, y, mid) >= target:
                lo = mid
            else:
                hi = mid
        return float(lo)

    # ---------------- 路径 / 写入 ----------------
    def _path_xy(self, t):
        """substep t 的手几何中心 (x, y): A->B 线段上的三角波(乒乓), 纯函数。"""
        d = self.B - self.A
        s = (self.speed * self.dt * int(t)) / max(self.L, 1e-9)   # 以线段长为单位的相位
        ph = 1.0 - abs((s % 2.0) - 1.0)                            # 三角波 [0, 1]
        return self.A + d * ph

    def _write(self, x, y, z, model_default=False):
        """写根 mocap_pos/mocap_quat + 12 个手指 qpos(固定 open); model_default 供 mj_reset。"""
        c = self.geom["center"]
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        off = np.array([cy * c[0] - sy * c[1], sy * c[0] + cy * c[1], c[2]])
        pos = np.array([x, y, z], dtype=float) - off      # 让"几何中心"落在 (x, y, z)
        quat = qmul(quat_z(self.yaw), self.geom["base_quat"])
        self.sim.data.mocap_pos[self.mid] = pos
        self.sim.data.mocap_quat[self.mid] = quat
        for nm, val in self.pose.items():
            self.sim.data.qpos[self.jadr[nm]] = val
        if model_default:
            self.sim.model.body_pos[self.body_id] = pos
            self.sim.model.body_quat[self.body_id] = quat
            # mj_reset 把 qpos0 写回 qpos ⇒ 手势必须同时写进 qpos0, 否则 reset 后的
            # 首帧会是模型默认(全张开), 与 episode 手势不符
            for nm, val in self.pose.items():
                self.sim.model.qpos0[self.jadr[nm]] = val
        self.sim.forward()

    # ---------------- episode / decision / substep hooks ----------------
    def start_episode(self, seed, ep):
        """episode 开始(须在 `env.reset()` 之前): 抽 A-B/z, 定朝向并放 t=0 位形。

        可复现通道: A/B = mix(seed, ep, 1); z = mix(seed, ep, 2)。
                    **朝向不再随机**: yaw 由路径中点决定(手指指向场景中心, 见下)。
        选点规则: ① 手中心距桌面原点 >= CENTER_CLEARANCE(不占画幅中心/不压任务物体);
                  ② "圆包围盒"判据 max(|x|,|y|) + hr <= tan_eff*(cam_z - z_lo)
                     (与 yaw 无关, 因此可以先选 A/B 再定朝向) ⇒ 整只手在画幅内;
                  ③ A/B 连线采样点都在桌面内缩区(INSET)内, 且单趟时长 >= MIN_PATH_TIME。
        朝向:     基准帧里手指沿 +x; yaw 使手指方向 = -normalize(路径中点) ⇒ **指向场景中心**
                  (手从外围"伸进来", 像人把手探进工作区)。
        """
        self.seed, self.ep = int(seed), int(ep)
        self.records, self.k = [], 0
        g = self.geom
        z_lo = max(self.z_lo_cfg, -g["zmin"])             # 手底不穿桌
        # --- ① 选点: 离中心足够远 + 整手在画幅内(包围盒判据, yaw 无关) ---
        #  top 模式   : 圆包围盒闭式, 取手**最高顶点**处的高度算(含 dz_up), 逐值不变;
        #  frustum 模式: 手 AABB(crop ∧ 真实策略相机视锥, 见 _aabb_visible), 在最低/最高
        #               两个中心高度上都要过 ⇒ 对手高的两个极端都成立(保守, yaw 无关)。
        dz_up = float(g.get("dz_up", 0.0))
        far = np.hypot(self.cells[:, 0], self.cells[:, 1]) >= CENTER_CLEARANCE
        full, loose = self._cell_filter(z_lo, dz_up)      # 按模式在 __init__ 绑定
        self.cells_full = int((far & full).sum())
        keep = far & full
        if not keep.any():
            keep = far & loose                            # 放宽到真实视锥(无边距)
        if not keep.any():
            keep = far
        if not keep.any():                                # 兜底: 最靠外的一格
            keep = np.zeros(len(self.cells), bool)
            keep[int(np.argmax(np.hypot(self.cells[:, 0], self.cells[:, 1])))] = True
        self.cells_g = self.cells[keep]
        # --- ② A/B: 连线在可放置内缩区, 单趟 >= MIN_PATH_TIME, 且朝向"朝中心"成立 ---
        rng_ab = self._rng(1)
        ts = np.linspace(0.0, 1.0, PATH_PROBES)[:, None]
        a = b = np.asarray(self.cells_g[0], dtype=float)
        inward = np.array([-1.0, 0.0])
        for _ in range(MAX_CELL_TRIES):
            a = self.cells_g[rng_ab.randint(len(self.cells_g))]
            b = self.cells_g[rng_ab.randint(len(self.cells_g))]
            if np.hypot(*(b - a)) < self.min_path:
                continue
            seg = a[None, :] + (b - a)[None, :] * ts
            if not np.all(np.max(np.abs(seg), axis=1) <= INSET):
                continue
            mid = 0.5 * (a + b)
            nrm = float(np.linalg.norm(mid))
            if nrm < MID_CLEARANCE:            # 中点太靠中心 -> "朝中心"方向不明确
                continue
            cand = -mid / nrm
            ok = True
            for p in (a, b, mid):              # 端点+中点都要"朝中心"
                pn = float(np.linalg.norm(p))
                if pn > 1e-9 and float(cand @ (-p / pn)) < ORIENT_DOT_MIN:
                    ok = False
                    break
            if not ok:
                continue
            inward = cand
            break
        else:                                  # 兜底: 取离 cells_g[0] 最远的格 + 中点定向
            d = np.linalg.norm(self.cells_g - self.cells_g[0], axis=1)
            b = self.cells_g[int(np.argmax(d))]
            mid = 0.5 * (np.asarray(self.cells_g[0], dtype=float) + b)
            nrm = float(np.linalg.norm(mid))
            if nrm > 1e-9:
                inward = -mid / nrm
        self.A, self.B = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        self.L = float(np.hypot(*(self.B - self.A)))
        # --- ③ 朝向: 手指指向场景中心(基准帧手指沿 +x ⇒ yaw = atan2(内向)) ---
        self.yaw = float(np.arctan2(inward[1], inward[0]))
        # --- ④ z: episode 级; 上限 = 整条路径上最紧的"整手在画幅内"高度 ---
        seg = self.A[None, :] + (self.B - self.A)[None, :] * ts
        caps = [c for c in (self.z_cap(px, py, z_lo) for px, py in seg)
                if c is not None]
        z_hi = min([self.z_hi_cfg] + caps) if caps else self.z_hi_cfg
        # 在**实际 z** 下也尽量"整只手在画幅内" —— 否则 z 抬高后视锥变小, 手会被
        # 画幅边缘裁掉(实测过: 只剩半只手, 姿态看不出来)
        caps_full = [c for c in (self.z_cap(px, py, z_lo, target=1.0)
                                 for px, py in seg) if c is not None]
        self.z_full = min(caps_full) if caps_full else None
        if self.z_full is not None:
            z_hi = min(z_hi, self.z_full)
        self.z = float(self._rng(2).uniform(z_lo, max(z_lo, z_hi)))
        # 审计: 最终 z 下整条路径的最小可见比例(1.0 = 整只手全程在画幅内)
        self.path_vis_min = float(min(self.visible_frac(px, py, self.z)
                                      for px, py in seg))
        self._audit(caps, self.path_vis_min)              # 按模式在 __init__ 绑定
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
            if e > 1e-6 and not _QUIET:   # reset/set_env_state 等外部路径改写过关节 -> 暴露
                print(f"[dynamic_hand] WARN tick(0) 前姿态偏差 pose_err={e:.2e} (即将重写)")
        x, y = self._path_xy(t)
        self._write(float(x), float(y), self.z)
        self.pos = (float(x), float(y), self.z)
        if len(self.records) < MAX_TICK_RECORDS:
            self.records.append((int(t), float(x), float(y), self.z))
        return self.pos

    def reapply(self):
        """把 t=0 位形重新写一遍(防御性; 建议在 `env.reset()` **之后**立刻调用)。

        首帧渲染发生在 `env.reset()` 内部, 所以 `start_episode()` 里"reset 之前"的写入
        仍然是必需的; 本方法只是兜底 —— 若某些 env/wrapper 路径(例如 GymEnv.reset 的
        set_env_state 分支)改写了关节, 重写 + 返回的 pose_err 会立刻暴露出来。
        同时重写 `model.qpos0`, 使后续任何 mj_reset 也保持该姿态。
        """
        x, y = self._path_xy(0)
        self._write(float(x), float(y), self.z, model_default=True)
        self.pos = (float(x), float(y), self.z)
        return self.pose_err()

    def finish(self):
        """平滑运动无"未到位"状态, 保留空实现以兼容 runner 调用。"""
        return None

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
            path_vis_min=None if self.path_vis_min is None else float(self.path_vis_min),
            pose_err=self.pose_err(),
            green_cells_feasible=int(0 if self.cells_g is None else len(self.cells_g)),
            green_cells_full=int(self.cells_full),
            green_cells_all=int(len(self.cells)),
            camera_mode=str(self.mode),
            cam_names=[str(n) for n in self.cam_names],
            crop_min=None if self.crop is None
            else [float(v) for v in np.ravel(self.crop[0])],
            crop_max=None if self.crop is None
            else [float(v) for v in np.ravel(self.crop[1])])


    def history(self):
        return [list(r) for r in self.records]


