#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d V0.5 sampling-only tool (random per-episode category, no SR run).

Rule (deterministic, differs across eval seeds only):
    rng = numpy.RandomState(seed)
    type(ep) = TYPE_ORDER[rng.randint(0, 5)]      # i.i.d. per episode

No block-cyclic, no extra "run" dimension: within one eval the sequence is a
random draw from the (seed)-seeded MT19937 stream; n_episodes=5/50/200 all
start from the same stream so prefixes are stable. Different eval seeds give
different sequences.

Output (test-only, removed in V1):
    <out>/seed{s:02d}_manifest.json
    <out>/sample_summary.json
"""
import argparse
import json
import pathlib

import numpy as np

TYPE_ORDER = ["cube", "can", "bottle", "cereal", "mug"]


def sample_run(seed, n_episodes):
    """Return (seed, [type0..typeN-1]) for the given eval seed.

    numpy.RandomState is a legacy MT19937 stream whose output is stable across
    numpy versions, so (seed) -> sequence is reproducible on any rerun.
    """
    seed = int(seed)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(TYPE_ORDER), size=int(n_episodes))
    return seed, [TYPE_ORDER[i] for i in idx]


GREENPOS_OFFSET = 7919      # RNG offset, independent of the catgen type stream


def sample_green_pos(seed, n_episodes, cells, offset=GREENPOS_OFFSET):
    """Deterministic per-episode (x, y) drawn uniformly from the green-cell list.

    rng = numpy.RandomState(seed + offset): independent of sample_run(seed)'s type
    stream; the sequence is stable per eval seed and differs across seeds.
    `cells` = (M, 2) world xy (see green_cells.load_green_cells).
    Returns [(x, y), ...] of length n_episodes.
    """
    cells = np.asarray(cells, dtype=float)
    if len(cells) == 0:
        raise ValueError("sample_green_pos: empty green-cell list")
    rng = np.random.RandomState(int(seed) + int(offset))
    idx = rng.randint(0, len(cells), size=int(n_episodes))
    return [(float(cells[i, 0]), float(cells[i, 1])) for i in idx]




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="/hy-tmp/3DP/dr3d_bench/samples")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"type_order": TYPE_ORDER,
               "rule": "randint(RandomState(seed))",
               "seeds": [], "manifest_dir": str(out)}
    for seed in [int(s) for s in a.seeds.split(",") if s.strip()]:
        rng_seed, ep_types = sample_run(seed, a.episodes)
        counts = {t: ep_types.count(t) for t in TYPE_ORDER}
        manifest = {
            "seed": seed, "rng_seed": rng_seed,
            "rule": summary["rule"], "type_order": TYPE_ORDER,
            "n_episodes": len(ep_types),
            "episodes": [{"episode": i, "type": t}
                         for i, t in enumerate(ep_types)],
        }
        fn = out / f"seed{seed:02d}_manifest.json"
        fn.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[seed {seed}] {ep_types}")
        print(f"[seed {seed}] total={len(ep_types)} per_type={counts}")
        summary["seeds"].append(
            {"seed": seed, "total_episodes": len(ep_types),
             "type_counts": counts, "manifest": str(fn)})
    (out / "sample_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[save] manifests -> {out}")


if __name__ == "__main__":
    main()

