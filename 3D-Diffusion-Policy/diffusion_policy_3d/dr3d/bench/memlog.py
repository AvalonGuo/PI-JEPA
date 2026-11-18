#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench: 共享的日志/内存小工具(纯标准库, 极轻量)。

动机:
  * suite 模式**每 episode 重建 env**(MuJoCo model + PointCloudGenerator + EGL 渲染
    上下文), 每个都打一遍 banner -> 日志爆炸; 用 `once()` 压成"每进程一次"。
  * 容器内存上限常远小于宿主(`/sys/fs/cgroup/memory/*`), 被 cgroup OOM killer 杀掉时
    只看到 `Killed`(无 traceback) —— `proc_mem()` 给出 rss/cgroup 两路读数, 便于定位。

环境变量:
    DR3D_QUIET   默认 "0" = 不静音; 评测入口(eval_policy.sh)显式设 "1"
                 -> 只留每 episode 一行摘要(训练链路不受影响)
    DR3D_VERBOSE 默认 "0"  = "1" 时额外打印扫描计时/审计明细
"""
import gc
import os

QUIET = os.environ.get("DR3D_QUIET", "0") == "1"
VERBOSE = os.environ.get("DR3D_VERBOSE", "0") == "1"

_SEEN = set()
_CG_FILES = ("/sys/fs/cgroup/memory/memory.usage_in_bytes",   # cgroup v1
             "/sys/fs/cgroup/memory.current")                  # cgroup v2


def once(key, msg):
    """每进程只打印一次(同 key 后续静默); 静音模式(DR3D_QUIET=1)下**完全不打印**。"""
    if QUIET:
        return False
    if key in _SEEN:
        return False
    _SEEN.add(key)
    print(msg, flush=True)
    return True


def vprint(msg):
    """仅在 DR3D_VERBOSE=1 时打印。"""
    if VERBOSE:
        print(msg, flush=True)


def qprint(msg):
    """仅在非静音(DR3D_QUIET=0)时打印。"""
    if not QUIET:
        print(msg, flush=True)


def proc_mem():
    """-> (rss_bytes|None, cgroup_used_bytes|None)。读 /proc/self/status 与 cgroup。"""
    rss = None
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    rss = int(ln.split()[1]) * 1024      # kB -> B
                    break
    except Exception:
        rss = None
    cg = None
    for p in _CG_FILES:
        try:
            with open(p) as f:
                cg = int(f.read().strip())
            break
        except Exception:
            continue
    return rss, cg


def mem_str():
    """-> `rss=4.21G cgroup=11.30G`(读不到则用 ? 占位)。"""
    rss, cg = proc_mem()
    r = "rss=%.2fG" % (rss / 1e9) if rss else "rss=?"
    c = "cgroup=%.2fG" % (cg / 1e9) if cg else "cgroup=?"
    return "%s %s" % (r, c)


def gc_cuda():
    """回收 Python 垃圾 + 清 CUDA 缓存(env 重建/释放后调用)。"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def soft_close(env):
    """尽力释放一个 env 的渲染/模型资源(失败静默; 调用方随后要断引用)。"""
    if env is None:
        return
    try:
        env.close()
    except Exception:
        pass
