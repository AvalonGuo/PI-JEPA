# -*- coding: utf-8 -*-
"""Metaworld expert demonstrations (single collector: point_cloud + raw cloud).

    cd third_party/Metaworld
    python gen_demonstration_expert.py --env_name assembly --num_episodes 10 \\
        --root_dir /hy-tmp/3DP/data --seed 42 [--save_video] [--overwrite]

Output ``<root_dir>/metaworld_<task>_expert.zarr``::

    data/ img, state, full_state, point_cloud(=FPS(--num_points)), depth, action,
          raw_point_cloud     object[VLenArray(float32)]  (N,6) flattened to (N*6,)
          raw_point_cloud_len int64 (T,)
    meta/ episode_ends

Rebuild: ``frame = raw_point_cloud[i].reshape(raw_point_cloud_len[i], 6)``

Notes
-----
* One zarr serves both consumers: the DP3 baseline reads
  ``state/action/point_cloud``, JEPA reads ``raw_point_cloud(_len)``.
* No 512 adaptation: ``point_cloud`` keeps ``--num_points`` (default 1024) rows;
  ``raw_point_cloud`` is the full cloud *after crop and before FPS*.
* Reproducibility: ``SawyerXYZEnv.seeded_rand_vec = False`` means the reset
  configuration comes from the global ``np.random`` stream, so ``seed_all`` is
  called before every ``reset`` (see ``mw_seeding``).
* Existing zarr is refused unless ``--overwrite`` is given.
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import zarr
from numcodecs import VLenArray
from termcolor import cprint

from diffusion_policy_3d.env import MetaWorldEnv
from metaworld.policies import *

from mw_seeding import add_seed_args, resolve_max_attempts, seed_all, seed_for

SUCCESS_TIMES_MIN = 5


def load_mw_policy(task_name):
    if task_name == 'peg-insert-side':
        return SawyerPegInsertionSideV2Policy()
    parts = [s.capitalize() for s in task_name.split('-')]
    return eval('Sawyer' + ''.join(parts) + 'V2Policy')()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--env_name', type=str, default='basketball')
    p.add_argument('--num_episodes', type=int, default=10)
    p.add_argument('--root_dir', type=str, default='/hy-tmp/3DP/data')
    p.add_argument('--num_points', type=int, default=1024,
                   help='point_cloud rows after FPS (raw_point_cloud is unaffected)')
    p.add_argument('--pc_resolution', type=int, default=128,
                   help='render/point-cloud resolution (MetaWorldEnv.image_size)')
    p.add_argument('--save_video', action='store_true',
                   help='export one mp4 per kept episode (matplotlib + imageio, headless)')
    p.add_argument('--overwrite', action='store_true',
                   help='replace an existing zarr (default: refuse)')
    return add_seed_args(p)


def main(args):
    save_dir = os.path.join(args.root_dir, 'metaworld_' + args.env_name + '_expert.zarr')
    if os.path.exists(save_dir):
        if not args.overwrite:
            cprint('[mw] {} exists; pass --overwrite to replace'.format(save_dir), 'red')
            return 1
        cprint('[mw] --overwrite: removing {}'.format(save_dir), 'red')
        os.system('rm -rf {}'.format(save_dir))

    e = MetaWorldEnv(args.env_name, device='cuda:0', use_point_crop=True,
                     num_points=args.num_points, image_size=args.pc_resolution)
    mw_policy = load_mw_policy(args.env_name)
    max_attempts = resolve_max_attempts(args.max_attempts, args.num_episodes)

    cprint('[mw] task={} episodes={} seed={} rule={} num_points={} pc_resolution={}'.format(
        args.env_name, args.num_episodes, args.seed, args.seed_rule,
        args.num_points, args.pc_resolution), 'yellow')
    cprint('[mw] zarr={}'.format(save_dir), 'yellow')

    total_count = 0
    img_arrays, point_cloud_arrays, depth_arrays = [], [], []
    state_arrays, full_state_arrays, action_arrays = [], [], []
    raw_arrays, raw_len_arrays = [], []
    raw_bytes_est = 0
    episode_ends_arrays = []
    kept_seeds, failed_seeds = [], []
    kept, attempt = 0, 0
    t0 = time.time()

    while kept < args.num_episodes:
        seed = seed_for(args.seed_rule, args.seed, kept, attempt)
        seed_all(e.env, seed)
        raw_state = e.reset()['full_state']
        obs_dict = e.get_visual_obs()

        done = False
        ep_reward = 0.
        ep_success = False
        ep_success_times = 0

        img_sub, pc_sub, depth_sub = [], [], []
        state_sub, full_state_sub, action_sub = [], [], []
        raw_sub, raw_len_sub = [], []
        total_count_sub = 0

        while not done:
            total_count_sub += 1
            img_sub.append(obs_dict['image'])
            pc_sub.append(obs_dict['point_cloud'])
            depth_sub.append(obs_dict['depth'])
            state_sub.append(obs_dict['agent_pos'])
            full_state_sub.append(raw_state)

            full_pc, _ = e.get_full_point_cloud()
            raw_sub.append(full_pc)
            raw_len_sub.append(int(full_pc.shape[0]))

            action = mw_policy.get_action(raw_state)
            action_sub.append(action)
            obs_dict, reward, done, info = e.step(action)
            raw_state = obs_dict['full_state']
            ep_reward += reward
            ep_success = ep_success or info['success']
            ep_success_times += info['success']

        attempt += 1
        if not ep_success or ep_success_times < SUCCESS_TIMES_MIN:
            failed_seeds.append(int(seed))
            cprint('[skip] attempt={} seed={} reward={:.2f} success_times={}'.format(
                attempt - 1, seed, ep_reward, ep_success_times), 'red')
            if attempt >= max_attempts:
                cprint('[mw] abort: {} attempts, {} / {} episodes kept'.format(
                    attempt, kept, args.num_episodes), 'red')
                return 1
            continue

        kept += 1
        kept_seeds.append(int(seed))
        total_count += total_count_sub
        episode_ends_arrays.append(copy.deepcopy(total_count))
        img_arrays.extend(copy.deepcopy(img_sub))
        point_cloud_arrays.extend(copy.deepcopy(pc_sub))
        depth_arrays.extend(copy.deepcopy(depth_sub))
        state_arrays.extend(copy.deepcopy(state_sub))
        full_state_arrays.extend(copy.deepcopy(full_state_sub))
        action_arrays.extend(copy.deepcopy(action_sub))
        raw_arrays.extend(copy.deepcopy(raw_sub))
        raw_len_arrays.extend(copy.deepcopy(raw_len_sub))
        n_mean = float(np.mean(raw_len_sub))
        raw_bytes_est += int(np.sum(raw_len_sub)) * 6 * 4
        cprint('[keep] ep={} seed={} reward={:.2f} success_times={} frames={} raw_mean={:.0f}'.format(
            kept - 1, seed, ep_reward, ep_success_times, total_count_sub, n_mean), 'green')
        if kept == 1:
            cprint('[mw] raw estimate: {} frames, {:.1f} MB so far -> ~{:.2f} GB '
                   'for {} episodes (single write at the end)'.format(
                       total_count_sub, raw_bytes_est / 1e6,
                       raw_bytes_est / 1e9 * args.num_episodes, args.num_episodes), 'cyan')

        if args.save_video and raw_sub:
            try:
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                if root not in sys.path:
                    sys.path.insert(0, root)
                from visualizer.visualizer.pointcloud import Visualizer
                video_dir = os.path.join(args.root_dir, 'video')
                os.makedirs(video_dir, exist_ok=True)
                vp = os.path.join(video_dir, 'metaworld_{}_episode_{:04d}.mp4'.format(
                    args.env_name, kept - 1))
                Visualizer().save_pointcloud_video(raw_sub, vp, fps=10)
                cprint('[save_video] -> {}'.format(vp), 'green')
            except Exception as exc:
                cprint('[save_video] failed for episode {}: {}'.format(kept - 1, exc), 'yellow')

    if len(img_arrays) == 0:
        cprint('[mw] no successful episode, nothing written', 'red')
        return 1


    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3:
        img_arrays = np.transpose(img_arrays, (0, 2, 3, 1))
    state_arrays = np.stack(state_arrays, axis=0)
    full_state_arrays = np.stack(full_state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    raw_len = np.array(raw_len_arrays, dtype=np.int64)
    raw_obj = np.empty(len(raw_arrays), dtype=object)
    for i, pc in enumerate(raw_arrays):
        raw_obj[i] = np.asarray(pc, np.float32).reshape(-1)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    zarr_data.create_dataset('img', data=img_arrays, chunks=(100,) + img_arrays.shape[1:],
                             dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=(100,) + state_arrays.shape[1:],
                             dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('full_state', data=full_state_arrays,
                             chunks=(100,) + full_state_arrays.shape[1:],
                             dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays,
                             chunks=(100,) + point_cloud_arrays.shape[1:],
                             dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=(100,) + depth_arrays.shape[1:],
                             dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=(100,) + action_arrays.shape[1:],
                             dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('raw_point_cloud', data=raw_obj, dtype=object, chunks=(100,),
                             overwrite=True, compressor=compressor,
                             object_codec=VLenArray(np.float32))
    zarr_data.create_dataset('raw_point_cloud_len', data=raw_len, chunks=(100,),
                             dtype='int64', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64',
                             overwrite=True, compressor=compressor)

    zarr_root.attrs.update({
        'env_name': str(args.env_name),
        'collector': 'fullpc',
        'raw_pc_source': 'pre-FPS, post crop (MetaWorldEnv.get_full_point_cloud)',
        'seed': int(args.seed),
        'seed_rule': str(args.seed_rule),
        'num_episodes_requested': int(args.num_episodes),
        'num_episodes_kept': int(kept),
        'attempts': int(attempt),
        'kept_seeds': [int(s) for s in kept_seeds],
        'failed_seeds': [int(s) for s in failed_seeds],
        'num_points': int(args.num_points),
        'pc_resolution': int(args.pc_resolution),
        'success_times_min': SUCCESS_TIMES_MIN,
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    })

    cprint('-' * 50, 'cyan')
    cprint('img shape: {}, range: [{}, {}]'.format(img_arrays.shape, np.min(img_arrays), np.max(img_arrays)), 'green')
    cprint('point_cloud shape: {}, range: [{:.3f}, {:.3f}]'.format(point_cloud_arrays.shape, np.min(point_cloud_arrays), np.max(point_cloud_arrays)), 'green')
    cprint('depth shape: {}, range: [{:.3f}, {:.3f}]'.format(depth_arrays.shape, np.min(depth_arrays), np.max(depth_arrays)), 'green')
    cprint('state shape: {} | full_state shape: {} | action shape: {}'.format(state_arrays.shape, full_state_arrays.shape, action_arrays.shape), 'green')
    cprint('raw_point_cloud: {} frames, N min/mean/max = {}/{:.0f}/{} (float32 xyz+rgb)'.format(
        len(raw_len), int(raw_len.min()), float(raw_len.mean()), int(raw_len.max())), 'green')
    cprint('episodes: {} | total frames: {} | attempts: {} | raw ~{:.2f} MB | elapsed {:.1f}s'.format(
        len(episode_ends_arrays), int(episode_ends_arrays[-1]), attempt, raw_bytes_est / 1e6,
        time.time() - t0), 'cyan')
    cprint('Saved zarr file to {}'.format(save_dir), 'green')

    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del raw_obj, raw_len, raw_arrays, raw_len_arrays
    del zarr_root, zarr_data, zarr_meta
    del e
    return 0


if __name__ == '__main__':
    sys.exit(main(build_parser().parse_args()))
