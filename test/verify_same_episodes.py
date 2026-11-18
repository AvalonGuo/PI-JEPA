#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only gate: do two Metaworld collector zarrs hold the same episodes?

Two uses:

1. **reproducibility** - re-collect the same task/seed/``--seed-rule`` into a second
   directory and check both runs are bit-identical;
2. **regeneration** - compare a fresh run against an archived zarr of the same rule.

    python test/verify_same_episodes.py \\
        --a /hy-tmp/3DP/data/run1/metaworld_stick-push_expert.zarr \\
        --b /hy-tmp/3DP/data/run2/metaworld_stick-push_expert.zarr

Checks, in order:

1. ``meta/episode_ends`` identical (same episode count and boundaries);
2. every shared ``data/<field>`` exactly equal (``np.array_equal``) with the same
   sha256-16 digest; the first differing frame is reported on failure;
3. per-episode first ``full_state[:, :6]`` (agent position) identical - that is
   the first thing a mismatched reset seed breaks;
4. ``raw_point_cloud`` / ``raw_point_cloud_len`` when both files carry them
   (row count per frame, then exact rows) - skip with ``--no-raw``.

``point_cloud_core`` is a derived field and is always skipped; ``--fields``
narrows or extends the compared list.
Nothing is written: the gate only reads.

Exit codes: 0 = PASS (same episodes), 1 = FAIL, 2 = bad input.
"""
import argparse
import hashlib
import sys

import numpy as np
import zarr

DEFAULT_FIELDS = ('img', 'state', 'full_state', 'point_cloud', 'depth', 'action')
IGNORED_FIELDS = ('point_cloud_core',)
RAW_KEY = 'data/raw_point_cloud'
RAW_LEN_KEY = 'data/raw_point_cloud_len'


def sha16(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def first_diff_frame(a, b):
    if a.shape[0] == 0 or a.shape != b.shape:
        return None
    neq = np.asarray(a) != np.asarray(b)
    per_frame = np.any(neq.reshape(neq.shape[0], -1), axis=1)
    idx = np.flatnonzero(per_frame)
    return int(idx[0]) if idx.size else None


def episode_starts(ends):
    return np.concatenate([[0], np.asarray(ends)[:-1]]).astype(np.int64)


def describe(path, root):
    attrs = dict(root.attrs)
    keys = ('collector', 'seed', 'seed_rule', 'num_episodes_kept', 'attempts',
            'kept_seeds', 'failed_seeds', 'num_points', 'pc_resolution')
    print('[zarr] {}'.format(path))
    for k in keys:
        if k in attrs:
            print('       {:<20s} {}'.format(k, attrs[k]))


def compare_raw(ra, rb, failures):
    """Exact comparison of the object-array raw clouds (rows per frame, then rows)."""
    if RAW_KEY not in ra or RAW_KEY not in rb:
        print('[skip] raw_point_cloud missing in {}'.format('A' if RAW_KEY not in ra else 'B'))
        return
    la = np.asarray(ra[RAW_LEN_KEY][:], np.int64)
    lb = np.asarray(rb[RAW_LEN_KEY][:], np.int64)
    if la.shape != lb.shape:
        failures.append('raw_point_cloud_len')
        print('[FAIL] raw_point_cloud_len shape A={} B={}'.format(la.shape, lb.shape))
        return
    if not np.array_equal(la, lb):
        failures.append('raw_point_cloud_len')
        bad = np.flatnonzero(la != lb)
        print('[FAIL] raw_point_cloud_len differs at frames {}: A={} B={}'.format(
            bad[:10].tolist(), int(la[bad[0]]), int(lb[bad[0]])))
        return

    digest = hashlib.sha256()
    for i, n in enumerate(la):
        a = np.asarray(ra[RAW_KEY][i], np.float32).reshape(int(n), 6)
        b = np.asarray(rb[RAW_KEY][i], np.float32).reshape(int(n), 6)
        if not np.array_equal(a, b):
            failures.append('raw_point_cloud')
            print('[FAIL] {:<14s} first_diff_frame={} (N={})'.format(
                'raw_point_cloud', i, int(n)))
            return
        digest.update(np.ascontiguousarray(a).tobytes())
    print('[ok]   {:<14s} frames={} rows={} sha256_16={}'.format(
        'raw_point_cloud', len(la), int(la.sum()), digest.hexdigest()[:16]))


def main(args):
    try:
        ra = zarr.open(args.a, mode='r')
        rb = zarr.open(args.b, mode='r')
    except Exception as exc:
        print('[FAIL] cannot open zarr: {}'.format(exc))
        return 2

    describe(args.a, ra)
    describe(args.b, rb)

    failures = []
    ea = np.asarray(ra['meta/episode_ends'][:])
    eb = np.asarray(rb['meta/episode_ends'][:])
    if np.array_equal(ea, eb):
        print('[ok]   episode_ends identical: {} episodes, {} frames'.format(
            len(ea), int(ea[-1]) if len(ea) else 0))
    else:
        failures.append('episode_ends')
        print('[FAIL] episode_ends differ: A={} episodes / B={} episodes'.format(len(ea), len(eb)))
        if len(ea) == len(eb):
            bad = np.flatnonzero(ea != eb)
            print('       first differing episode {}: A={} B={}'.format(
                int(bad[0]), int(ea[bad[0]]), int(eb[bad[0]])))

    fields = [f for f in args.fields if f not in IGNORED_FIELDS]
    for field in fields:
        ka, kb = 'data/' + field, 'data/' + field
        if ka not in ra or kb not in rb:
            only = args.a if ka not in ra else args.b
            print('[skip] {:<14s} missing in {}'.format(field, only))
            continue
        a = np.asarray(ra[ka][:])
        b = np.asarray(rb[kb][:])
        if a.shape != b.shape:
            failures.append(field)
            print('[FAIL] {:<14s} shape A={} B={}'.format(field, a.shape, b.shape))
            continue
        da, db = sha16(a), sha16(b)
        if np.array_equal(a, b):
            print('[ok]   {:<14s} shape={} sha256_16={}'.format(field, a.shape, da))
        else:
            failures.append(field)
            frame = first_diff_frame(a, b)
            print('[FAIL] {:<14s} sha256_16 A={} B={}{}'.format(
                field, da, db, '' if frame is None else ' first_diff_frame={}'.format(frame)))

    if not args.no_raw:
        compare_raw(ra, rb, failures)

    if 'full_state' in ra and 'full_state' in rb and len(ea) == len(eb):
        starts = episode_starts(ea)
        fa = np.asarray(ra['data/full_state'][:])[starts][:, :6]
        fb = np.asarray(rb['data/full_state'][:])[starts][:, :6]
        if np.array_equal(fa, fb):
            print('[ok]   reset agent_pos identical over {} episodes'.format(len(starts)))
        else:
            failures.append('reset_agent_pos')
            bad = np.flatnonzero(np.any(fa != fb, axis=1))
            print('[FAIL] reset agent_pos differs at episodes {}'.format(bad[:10].tolist()))

    if failures:
        print('[FAIL] {} -> {}'.format(args.a, failures))
        return 1
    print('[PASS] {} == {} (fields: {})'.format(args.a, args.b, ','.join(fields)))
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, help='reference zarr (e.g. first collection run)')
    p.add_argument('--b', required=True, help='zarr to compare (e.g. re-collection run)')
    p.add_argument('--fields', nargs='+', default=list(DEFAULT_FIELDS),
                   help='data/<field> entries to compare (default: %(default)s)')
    p.add_argument('--no-raw', action='store_true',
                   help='skip raw_point_cloud(_len) comparison')
    sys.exit(main(p.parse_args()))
