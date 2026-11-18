#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: 「绿区」可放置格候选 (供 posgen suite 随机放置 cube)。

绿区定义复用同目录的 workspace_traffic_map.py::green_cells —— 即
    桌面 ±0.44 去掉「手/锤足迹」(depth 免标签判据: 高于桌面 5cm)
    → 足迹膨胀 1 格(2cm) 作安全边距 → 内缩到 ±0.41(6cm cube 不悬空)
实测 1022 个候选格。

候选只跟 ID 数据(zarr)有关、与 eval seed 无关, 因此首次使用时算一次并缓存
(资源/缓存按 task 分派, door/hammer/pen 互不覆盖):
    hammer: dr3d/tmp/green_cells.npz       + green_cells.json
            (adroit_hammer_expert.zarr + DAPG_hammer.xml)
    door:   dr3d/tmp/green_cells_door.npz  + green_cells_door.json
            (adroit_door_expert.zarr, same 84px protocol as hammer + DAPG_door.xml)
    pen:    dr3d/tmp/green_cells_pen.npz   + green_cells_pen.json
            (adroit_pen_expert.zarr + DAPG_pen.xml, footprint from the oblique
             vil_camera; see green_cells_pen.py / workspace_traffic_map.green_cells_cam)

dynsphere 专用入口见本文件 load_dynsphere_cells(): hammer/door 走上面的数据绿区;
pen 在 zarr 缺失时退化为解析整桌网格(green_cells_pen.table_grid), 保证无数据集
的机器也能跑 suite(来源由 dynsphere_cells_source() 报告)。

CLI (可选, 手动预热/查看):
    python dr3d/bench/green_cells.py --task door   # 有缓存则 hit, 否则生成(需数据)
    python dr3d/bench/green_cells.py --force       # 强制重算
"""
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

DR3D = pathlib.Path(__file__).resolve().parents[1]
if str(DR3D) not in sys.path:
    sys.path.insert(0, str(DR3D))
# 本目录非包(平铺模块): 保证 `import dataroot/test_load` 在被按路径加载时也能用
_BENCH = pathlib.Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

CACHE = DR3D / "tmp" / "green_cells.npz"
CACHE_META = DR3D / "tmp" / "green_cells.json"

# 与 workspace_traffic_map.py::green_cells 默认值保持一致
PARAMS = dict(gs=0.02, table_half=0.44, arm_height=0.05,
              margin_cells=1, inset=0.41)

# --- 任务表(每任务一行数据, 只放"代码事实", 不含大文件本身) --------------------
#   footprint      'top'=顶视闭式足迹 | 'cam'=通用相机逐点反投影(斜视, 见 green_cells_pen)
#   cam_name       footprint=='cam' 时的相机名(实际以真实策略配置为准, 见 _cam_name)
#   cache          绿区缓存文件(hammer 沿用历史名 green_cells.npz ⇒ 既有缓存不失效)
#   table_fallback 数据/缓存都不可用时是否允许整桌网格兜底(dynsphere 兜底路径; 仅 pen)
# zarr 路径由 dr3d/bench/dataroot.py 约定(DR3D_DATA_ROOT 可覆盖); xml 以 test_load.BASE_XML 为准。
import dataroot as _dr                                    # noqa: E402
import test_load as _TL                                   # noqa: E402

TASKS = {
    "hammer": dict(footprint="top", cam_name=None, cache=CACHE,
                   table_fallback=False),
    "door":   dict(footprint="top", cam_name=None,
                   cache=DR3D / "tmp" / "green_cells_door.npz",
                   table_fallback=False),
    "pen":    dict(footprint="cam", cam_name="vil_camera",
                   cache=DR3D / "tmp" / "green_cells_pen.npz",
                   table_fallback=True),
}


def _rec(task):
    """task -> 表行; task=None 兼容旧调用(=hammer)。"""
    return TASKS["hammer" if task is None else task]


def _cam_name(task):
    """足迹相机名: footprint=='top' -> 'top'; 'cam' -> 真实策略配置(pen=vil_camera)。"""
    if _rec(task)["footprint"] != "cam":
        return "top"
    import green_cells_pen as _gcp
    return str(_gcp.cam_name())


# task -> (xml, cache); zarr path is resolved per call (DR3D_DATA_ROOT aware)
TASK_RESOURCES = {t: (_TL.BASE_XML[t], TASKS[t]["cache"]) for t in TASKS}


def resources_for(task):
    """task -> (zarr_path, xml_path, cache_path); task=None kept for old callers (=hammer).

    The zarr path is recomputed every call, so switching `DR3D_DATA_ROOT` at runtime works.
    """
    if task is None:
        task = "hammer"
    if task not in TASK_RESOURCES:
        raise ValueError(f"green_cells: unknown task {task!r} "
                         f"(available: {sorted(TASK_RESOURCES)})")
    xml, cache = TASK_RESOURCES[task]
    return (_dr.adroit_expert_zarr(task), xml, cache)


def _wtm():
    """惰性 import 同目录的 workspace_traffic_map.py (非包, 用 importlib 按路径加载)。"""
    import importlib.util
    path = pathlib.Path(__file__).resolve().parent / "workspace_traffic_map.py"
    if not path.exists():
        raise FileNotFoundError(path)
    key = "_wtm_for_green_cells"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def params_key(zarr_path=None, hammer_path=None, task=None, **kw):
    """候选有效性指纹: zarr 路径 + 帧数 + 参数 + xml 路径 (+ pen 的相机名)。

    pen 的绿区依赖相机(vil_camera 斜视足迹), 故指纹里必须带相机名; hammer/door
    仍用顶视闭式, 不加该字段 => 既有缓存指纹逐字不变(不会无故失效)。
    """
    if task is not None:
        dz, dx, _ = resources_for(task)
        zarr_path = zarr_path or dz
        hammer_path = hammer_path or dx
    wtm = _wtm()
    zarr_path = str(zarr_path or wtm.DEFAULT_ZARR)
    hammer_path = str(hammer_path or wtm.DEFAULT_HAMMER)
    t = _zarr_frames(zarr_path)          # 不可读 -> None -> 用 -1(与旧指纹一致)
    if t is None:
        t = -1
    p = dict(PARAMS)
    p.update(kw)
    extra = {}
    cam = _cam_name(task)
    if cam != "top":
        extra["cam"] = cam
    blob = json.dumps({"zarr": zarr_path, "T": t, "hammer": hammer_path,
                       "params": p, **extra}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_green_cells(zarr_path=None, hammer_path=None, task=None, **kw):
    if task is not None:
        dz, dx, _ = resources_for(task)
        zarr_path = zarr_path or dz
        hammer_path = hammer_path or dx
    wtm = _wtm()
    p = dict(PARAMS)
    p.update(kw)
    if _rec(task)["footprint"] == "cam":
        # 斜视相机: 顶视闭式不成立 -> 通用相机逐点反投影足迹(见 workspace_traffic_map)
        if not pathlib.Path(str(zarr_path)).exists():
            raise FileNotFoundError(
                f"green_cells(task={task!r}) 需要 ID 数据集: {zarr_path}\n"
                f"  dynsphere 在缺数据时可走 load_dynsphere_cells() 的"
                f"整桌网格兜底(table_grid)")
        return wtm.green_cells_cam(zarr_path=zarr_path, xml_path=hammer_path,
                                   cam_name=_cam_name(task), **p)
    return wtm.green_cells(zarr_path=zarr_path, hammer_path=hammer_path, **p)


def _zarr_frames(zarr_path):
    """数据集帧数; 不可读(不在位/损坏) -> None(用于"别因缺数据而作废缓存")。"""
    try:
        import zarr as _zarr
        return int(_zarr.open(str(zarr_path), mode="r")[
            "data"]["raw_point_cloud_len"].shape[0])
    except Exception:
        return None


def _meta_compatible(meta, zarr_path, xml_path, task):
    """数据不在位时的缓存接受判定: 只比"task + zarr 路径 + xml 路径 + 参数", 不看 T。

    T(数据集帧数) 只有在数据在位时才能复算, 它的作用是"重新采集过数据 -> 绿区可能变
    -> 让缓存失效"; 数据不在位时无从比较, 因此**信任既有缓存**(调用方会打印提示),
    避免"数据搬走 -> 指纹不匹配 -> 静默重算 -> 崩"。
    """
    return (str(meta.get("task") or "hammer") == str(task or "hammer")
            and str(meta.get("zarr") or "") == str(zarr_path)
            and str(meta.get("hammer") or "") == str(xml_path)
            and dict(meta.get("params") or {}) == dict(PARAMS))


def _missing_msg(task, zarr_path):
    """缺数据且无缓存时的可执行提示(不含 traceback)。"""
    return (f"task={task} 需要 ID 数据集(绿区): {zarr_path}\n"
            f"  未找到, 且没有可用缓存。\n"
            f"  {_dr.missing_hint(task)}\n"
            f"  (或: DR3D_DATA_ROOT=<已有数据的根目录> 复用别处数据集; --zarr 亦可覆盖)")


def cells_status(task):
    """(ok, source, msg): task 的绿区候选是否可用。

    source='data' 数据集在位(可算/缓存命中); 'cache' 数据集不在位但既有缓存可用;
    ok=False 时 msg 是可直接照抄执行的可执行提示。
    """
    zarr, xml, cache = resources_for(task)
    if pathlib.Path(str(zarr)).exists():
        return True, "data", f"zarr 在位: {zarr}"
    cache = pathlib.Path(cache) if cache else CACHE
    meta_f = cache.with_suffix(".json")
    if cache.exists() and meta_f.exists():
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            if _meta_compatible(meta, zarr, xml, task):
                n = int(np.load(cache)["cells"].shape[0])
                return True, "cache", (f"数据不在位 -> 使用既有缓存 {n} cells <- {cache}")
        except Exception:
            pass
    return False, None, _missing_msg(task, zarr)


def require_cells(task):
    """suite 启动时调用一次: 绿区不可用 -> 打印可执行提示并退出(无 traceback)。"""
    ok, source, msg = cells_status(task)
    if ok:
        return source
    raise SystemExit(f"[green_cells] {msg}")


def load_green_cells(zarr_path=None, hammer_path=None, cache=None,
                     force=False, verbose=True, task=None):
    """读缓存(校验 params_key) 或就地计算并写入缓存; 返回 (M,2) world xy。

    task: 'hammer' | 'door' | 'pen'; None 兼容旧调用(=hammer 专属默认)。
          pen 需要 ID 数据集(zarr)在场 —— 无数据时请用 load_dynsphere_cells()。
    """
    if task is not None:
        dz, dx, dcache = resources_for(task)
        zarr_path = zarr_path or dz
        hammer_path = hammer_path or dx
        cache = cache or dcache
    cache = pathlib.Path(cache) if cache else CACHE
    meta_f = cache.with_suffix(".json")
    key = params_key(zarr_path, hammer_path, task=task)
    data_present = pathlib.Path(str(zarr_path)).exists()
    if not force and cache.exists() and meta_f.exists():
        try:
            meta = json.loads(meta_f.read_text(encoding="utf-8"))
            if meta.get("params_key") == key:
                cells = np.load(cache)["cells"]
                if verbose:
                    print(f"[green_cells] cache hit: {cells.shape[0]} cells <- {cache}")
                return cells
            if not data_present and _meta_compatible(meta, zarr_path, hammer_path,
                                                     task):
                # 数据集不在位 -> 指纹里的 T 无从复算; 信任既有缓存(绝不静默重算)
                cells = np.load(cache)["cells"]
                if verbose:
                    print(f"[green_cells] 数据不在位({zarr_path}) -> 使用既有缓存: "
                          f"{cells.shape[0]} cells <- {cache}")
                return cells
        except Exception as e:                       # 损坏则重算
            if verbose:
                print(f"[green_cells] cache unreadable ({e}); recomputing")
    if not data_present:
        raise SystemExit(f"[green_cells] {_missing_msg(task, zarr_path)}")
    cells = compute_green_cells(zarr_path, hammer_path, task=task)
    cache.parent.mkdir(parents=True, exist_ok=True)
    # 并发安全: 先写 pid 临时文件再 os.replace(原子), 读侧永远不会看到半写文件
    npz_tmp = cache.with_name(f".{cache.name}.tmp.{os.getpid()}")
    # 注意: np.savez* 对不以 .npz 结尾的名字会静默追加后缀 -> 必须用文件句柄写
    with open(npz_tmp, "wb") as _f:
        np.savez_compressed(_f, cells=cells)
    os.replace(str(npz_tmp), str(cache))
    meta_tmp = meta_f.with_name(f".{meta_f.name}.tmp.{os.getpid()}")
    meta_tmp.write_text(json.dumps(
        {"params_key": key, "n_cells": int(len(cells)), "task": task,
         "zarr": str(zarr_path or _wtm().DEFAULT_ZARR),
         "hammer": str(hammer_path or _wtm().DEFAULT_HAMMER),
         "zarr_frames": _zarr_frames(zarr_path),
         "params": dict(PARAMS)}, indent=2), encoding="utf-8")
    os.replace(str(meta_tmp), str(meta_f))
    if verbose:
        print(f"[green_cells] computed {cells.shape[0]} cells -> {cache}")
    return cells


def dynsphere_cells_source(task=None):
    """dynsphere 候选格来源: 'green'(ID 数据集绿区) | 'table'(解析整桌网格兜底)。

    规则(查表, 不认任务名): 数据集在位 -> 'green'; 不在位 -> 允许整桌兜底的
    任务(表里 table_fallback=True, 目前仅 pen)给 'table', 否则仍给 'green'
    (由 load_green_cells 报出"需先采集数据"的可执行提示)。
    """
    zarr_path = _dr.adroit_expert_zarr(task if task is not None else "hammer")
    if pathlib.Path(str(zarr_path)).exists():
        return "green"
    # 数据不在位: 只有允许整桌兜底的任务(表里 table_fallback=True, 目前仅 pen)给
    # 'table'; 否则仍给 'green'(由 load_green_cells 报出"需先采集数据"的可执行提示)
    return "table" if _rec(task)["table_fallback"] else "green"


def load_dynsphere_cells(task=None, verbose=True, force=False):
    """dynsphere 专用候选格入口(纯 (M,2) world xy, 可直接喂 DynamicDistractor)。

    - hammer/door: 与既往完全一致 -> load_green_cells(task=...)
    - pen        : 有数据集 -> 数据绿区(斜视 vil_camera 足迹, 与 posgen 同口径);
                   无数据集 -> green_cells_pen.table_grid() 整桌网格兜底
                   (只保证 suite 可跑; 来源用 dynsphere_cells_source(task) 报告)
    """
    src = dynsphere_cells_source(task)
    if src == "green":
        return load_green_cells(force=force, verbose=verbose, task=task)
    import green_cells_pen as _gcp
    cells = _gcp.table_grid()
    if verbose:
        print(f"[green_cells] dynsphere task={task}: table_grid fallback "
              f"{len(cells)} cells (ID dataset missing: {resources_for(task)[0]})")
    return cells


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=None, choices=["hammer", "door", "pen"],
                    help="按 task 分派默认 zarr/xml/缓存(缺省 hammer)")
    ap.add_argument("--zarr", default=None)
    ap.add_argument("--hammer", default=None)
    ap.add_argument("--force", action="store_true", help="强制重算(忽略缓存)")
    ap.add_argument("--show", type=int, default=5, help="打印前 N 个候选(默认 5)")
    a = ap.parse_args()
    cells = load_green_cells(a.zarr, a.hammer, force=a.force, task=a.task)
    print(f"[green_cells] n={len(cells)}  x[{cells[:,0].min():+.3f},{cells[:,0].max():+.3f}]"
          f" y[{cells[:,1].min():+.3f},{cells[:,1].max():+.3f}]")
    for x, y in cells[:max(0, a.show)]:
        print(f"   x={x:+.3f} y={y:+.3f}")


if __name__ == "__main__":
    main()
