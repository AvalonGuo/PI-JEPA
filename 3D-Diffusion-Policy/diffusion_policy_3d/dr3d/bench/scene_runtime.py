#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared scene runtime used by the eval chain (runner imports this, no CLI).

\u65e0\u4fb5\u5165\u7ea6\u675f(\u91cd\u8981): \u672c\u6a21\u5757\u5bf9 third_party **\u96f6\u5199\u5165**\u3002
* \u5408\u6210\u573a\u666f\u4e3a"\u81ea\u5305\u542b XML"(\u89c1 test_load.selfcontain_base_text):
  <include> \u539f\u4f4d\u5c55\u5f00 + meshdir/texturedir \u7edd\u5bf9\u5316, \u53ef\u4ece\u4efb\u610f\u8def\u5f84\u52a0\u8f7d
  => \u4e0d\u518d\u4f9d\u8d56"\u4e0e\u771f assets \u540c\u6df1\u5ea6"\u7684\u76f8\u5bf9\u8def\u5f84\u89e3\u6790\u3002
* \u79c1\u6709\u76ee\u5f55(\u6309 pid+task \u9694\u79bb) \u5168\u90e8\u843d\u5728 dr3d/tmp/scene_swap/:
  <shim>.py + assets/<target>.xml; \u5c06 <task>_v0.__file__ \u6307\u5230 shim,
  gym.make \u5c31\u8bfb\u672c\u8fdb\u7a0b\u79c1\u6709\u573a\u666f(\u771f assets \u53ea\u8bfb, \u591a\u8fdb\u7a0b\u5e76\u884c\u5b89\u5168)\u3002
"""
import atexit
import hashlib
import importlib
import os
import pathlib
import shutil
import sys

DR3D = pathlib.Path(__file__).resolve().parents[1]
if str(DR3D) not in sys.path:
    sys.path.insert(0, str(DR3D))
import compose as C

REPO = DR3D.parents[2]                                   # /root/3DP
# 评测脚本默认 DR3D_QUIET=1: 不打印常规信息(与 clean 评测输出一致; 0 恢复)
_QUIET = os.environ.get("DR3D_QUIET", "0") == "1"
POS = C.POS_DEFAULT          # P0; 三任务共用 (-0.22, -0.15)
PRIVATE_BASE = DR3D / "tmp" / "scene_swap"    # \u5168\u90e8\u843d\u5728 dr3d \u5185, \u4e0d\u78b0 third_party

# \u4efb\u52a1\u6ce8\u518c\u8868: \u4f9d\u8d56\u6a21\u5757 / shim \u6587\u4ef6\u540d / gym.make \u5b9e\u9645\u8bfb\u53d6\u7684\u573a\u666f\u6587\u4ef6\u540d
#   三者都读本任务自己的 base XML(shim 后读私有副本)
TASKS = {
    "hammer": dict(shim_mod="mj_envs.hand_manipulation_suite.hammer_v0",
                    shim_name="hammer_v0.py",
                    target_name="DAPG_hammer.xml"),
    "door":   dict(shim_mod="mj_envs.hand_manipulation_suite.door_v0",
                    shim_name="door_v0.py",
                    target_name="DAPG_door.xml"),
    "pen":    dict(shim_mod="mj_envs.hand_manipulation_suite.pen_v0",
                    shim_name="pen_v0.py",
                    target_name="DAPG_pen.xml"),
}

_state = {}    # task -> dict(dir, assets, target)


def _task_state(task):
    return _state.setdefault(task, dict(dir=None, assets=None, target=None))


def _cleanup(task):
    st = _state.get(task)
    if st and st.get("dir") and pathlib.Path(st["dir"]).exists():
        shutil.rmtree(str(st["dir"]), ignore_errors=True)
    _state.pop(task, None)



def _install_shim(task):
    """(\u5e42\u7b49) \u5efa dr3d \u79c1\u6709\u573a\u666f\u76ee\u5f55 + <task>_v0.__file__ \u91cd\u5b9a\u5411; \u8fd4\u56de assets \u76ee\u5f55\u3002"""
    st = _task_state(task)
    if st["assets"] is not None:
        return st["assets"]
    cfg = TASKS[task]
    d = PRIVATE_BASE / f"eval_{os.getpid()}_{task}"
    a = d / "assets"
    a.mkdir(parents=True, exist_ok=True)
    shim = d / cfg["shim_name"]
    if not shim.exists():
        shim.write_text("# per-process shim: dirname(__file__)/assets is the private "
                        "(self-contained) scene dir.\n", encoding="utf-8")
    mod = importlib.import_module(cfg["shim_mod"])
    mod.__file__ = str(shim)                    # gym.make \u636e\u6b64\u89e3\u6790 \"curr_dir/assets/<target>\"
    st.update(dir=d, assets=a, target=a / cfg["target_name"])
    atexit.register(_cleanup, task)
    if not _QUIET:
        print(f"[scene] private scene dir ({task}): {a}  (third_party \u5168\u7a0b\u53ea\u8bfb, \u53ef\u5b89\u5168\u5e76\u884c)",
              flush=True)
    return a


def _paths(task="hammer", create=True):
    """-> 私有 target: dr3d/tmp/scene_swap/eval_<pid>_<task>/assets/<target>.xml"""
    st = _task_state(task)
    if create:
        _install_shim(task)
        return st["target"]
    if st["assets"] is not None:
        return st["target"]
    d = PRIVATE_BASE / f"eval_{os.getpid()}_{task}"
    return d / "assets" / TASKS[task]["target_name"]


def target_path(task="hammer"):
    return _paths(task)


def ensure_scene(type_name, pos=None, mocap=False, scale=1.0, task="hammer"):
    """Shared lazy cache -> (path, sha, fresh); base \u968f task\u3002"""
    return C.ensure_scene(type_name, POS if pos is None else pos, mocap=mocap,
                          scale=scale, base=task)


def activate(type_name, pos=None, mocap=False, scale=1.0, task="hammer"):
    """\u628a\u5408\u6210\u573a\u666f(\u81ea\u5305\u542b) \u5199\u8fdb\u672c\u8fdb\u7a0b\u79c1\u6709 target; env \u91cd\u5efa\u540e\u751f\u6548\u3002"""
    path, sha, _ = ensure_scene(type_name, pos, mocap, scale, task)
    text = pathlib.Path(path).read_text(encoding="utf-8")
    target = _paths(task, create=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return path, sha


def current_sha(task="hammer"):
    return hashlib.sha256(pathlib.Path(_paths(task, create=False)).read_bytes()).hexdigest()
