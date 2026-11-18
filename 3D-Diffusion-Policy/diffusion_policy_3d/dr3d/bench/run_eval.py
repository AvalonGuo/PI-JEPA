#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d bench V0 smoke/cache runner (delegates to the standard eval chain).

No third-party source edits. Strategy:
  1. --mode all  composes the 5 hammer+distractor scenes into the SHARED cache
     dr3d/tmp/scenes on first use (git-ignored); later runs reuse (sha-verified);
  2. --mode one  runs the real catgen suite through scripts/eval_policy.sh:
     per-episode type is sampled inside the runner from RandomState(seed), the
     runner swaps the shared scene around each episode and writes
     media/<subdir>/dr3d_suite_manifest.json with episode->type + success.

Usage:
  # compose+cache all 5 scenes (no eval; cache primer)
  python dr3d/bench/run_eval.py --mode all

  # real smoke: catgen suite for eval seed 0, eval_episodes=N (default 1)
  python dr3d/bench/run_eval.py --mode one --seed 0 --eval-episodes 1
"""
import argparse
import pathlib
import subprocess
import sys

DR3D = pathlib.Path(__file__).resolve().parents[1]
if str(DR3D) not in sys.path:
    sys.path.insert(0, str(DR3D))
import scene_runtime as SR
from sample import TYPE_ORDER, sample_run

REPO = pathlib.Path("/root/3DP")


def run_catgen_eval(gpu=0, seed=0, eval_episodes=1, save_video=False):
    cmd = ["bash", "scripts/eval_policy.sh", "dp3", "adroit_hammer",
           "0112", str(seed), str(gpu),
           "true" if save_video else "false",
           str(eval_episodes), "catgen"]
    print("[eval]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO), check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "one"], default="one")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--save-video", action="store_true")
    a = ap.parse_args()

    if a.mode == "all":
        for t in TYPE_ORDER:
            path, sha, fresh = SR.ensure_scene(t)
            print(f"[cache] {t}: {'composed' if fresh else 'hit'} "
                  f"-> {path}  sha={sha[:16]}")
        return

    # show the deterministic per-episode types the suite will use
    _, ep_types = sample_run(a.seed, a.eval_episodes)
    print(f"[sample] seed={a.seed} ep={a.eval_episodes} -> {ep_types}")
    run_catgen_eval(gpu=a.gpu, seed=a.seed,
                    eval_episodes=a.eval_episodes,
                    save_video=a.save_video)
    print("[done] see media/<subdir>/dr3d_suite_manifest.json for episode types")


if __name__ == "__main__":
    main()

