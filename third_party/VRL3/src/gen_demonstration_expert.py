# usage:
#       bash scripts/gen_demonstration_adroit.sh <task> [save_video=0] [num_episodes=10]
#            [seed=42] [gpu=0] [output_dir=/hy-tmp/3DP/data] [pc_resolution=84]
#            [extra_steps_after_success=0] [success_goal_threshold=10] [min_episode_steps=100]
# Adroit side of the single-collector contract (same shape as
# third_party/Metaworld/gen_demonstration_expert.py): writes
# <output_dir>/adroit_<task>_expert.zarr for DP3 training, JEPA (raw_point_cloud)
# and the dr3d green cells (depth 84x84).
import mj_envs
from adroit import AdroitEnv
import argparse
import os
import shutil
import torch
from vrl3_agent import VRL3Agent
import utils
from termcolor import cprint
import zarr
from copy import deepcopy
import numpy as np
from numcodecs import VLenArray

from diffusion_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='door', help='environment to run')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run')
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--output_dir', type=str, default='/hy-tmp/3DP/data',
                        help='output directory (overrides root_dir); default /hy-tmp/3DP/data')
    parser.add_argument('--expert_ckpt_path', type=str, default=None, help='path to expert ckpt')
    parser.add_argument('--img_size', type=int, default=84, help='image size')
    parser.add_argument('--pc_resolution', type=int, default=84, choices=[84, 128, 256],
                        help='point cloud render resolution (84/128/256), used as img_size of PointCloudGenerator')
    parser.add_argument('--not_use_multi_view', action='store_true', help='not use multi view')
    parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
    parser.add_argument('--save_video', action='store_true',
                        help='if set, export each accepted episode as an mp4 video of its full point clouds')
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for reproducible env initial states '
                             '(nail_board Z); each episode uses seed+episode_idx')
    parser.add_argument('--success_goal_threshold', type=float, default=10.0,
                        help='episode is kept once cumulative n_goal_achieved reaches this value')
    parser.add_argument('--extra_steps_after_success', type=int, default=0,
                        help='max steps to run after success (trims the long static tail); '
                             '0 = default, no truncation (run the full env horizon)')
    parser.add_argument('--min_episode_steps', type=int, default=100,
                        help='minimum steps per episode (keep collecting up to this many '
                             'steps after env done; VRL3 default 100)')
    parser.add_argument('--allow_mixed_truncation', action='store_true',
                        help='allow --append onto a zarr with a different truncation policy '
                             '(default: error; mixing truncated/full data shifts the distribution)')
    parser.add_argument('--append', action='store_true',
                        help='append new episodes to an existing zarr instead of overwriting; '
                             'episode_ends are offset by the existing frame count')
    parser.add_argument('--overwrite', action='store_true',
                        help='replace an existing zarr (default: refuse and exit)')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # load env
    action_repeat = 2
    frame_stack = 1
    def create_env(seed):
        env = AdroitEnv(env_name=args.env_name+'-v0', test_image=False, num_repeats=action_repeat,
                        num_frames=frame_stack, env_feature_type='pixels',
                                            device='cuda', reward_rescale=True)
        env = MujocoPointcloudWrapperAdroit(env=env, env_name='adroit_'+args.env_name, use_point_crop=args.use_point_crop,
                                            img_size=args.pc_resolution)
        # fix HammerEnvV0.np_random (nail_board Z) via set_seed chain
        if seed is not None:
            try:
                env.env._env.set_seed(seed)
            except Exception as e:
                cprint('WARN: set_seed failed: {}'.format(e), 'yellow')
        return env
    num_episodes = args.num_episodes
    zarr_name = 'adroit_'+args.env_name+'_expert.zarr'
    out_dir = args.output_dir if args.output_dir else args.root_dir
    save_dir = os.path.join(out_dir, zarr_name)
    # truncation policy of the data being written (recorded in zarr attrs)
    truncated = bool(args.extra_steps_after_success > 0)
    episode_len_policy = 'success+tail' if truncated else 'full'
    cprint('[adroit] task={} zarr={} | success_goal_threshold={} '
           'extra_steps_after_success={} -> episode_len_policy={} '
           '(truncated={})'.format(
               args.env_name, zarr_name, args.success_goal_threshold,
               args.extra_steps_after_success, episode_len_policy, truncated),
           'cyan')
    append_offset = 0
    if args.append and os.path.exists(save_dir):
        cprint('Appending to existing {}'.format(save_dir), 'yellow')
        old_root = zarr.open(save_dir, mode='r')
        old_trunc = old_root.attrs.get('truncated', None)
        if old_trunc is not None and bool(old_trunc) != truncated \
                and not args.allow_mixed_truncation:
            raise SystemExit(
                "[adroit] refusing to mix truncation policies: existing zarr has "
                "truncated={} but this run writes truncated={} "
                "(pass --allow_mixed_truncation to override)".format(
                    bool(old_trunc), truncated))
        append_offset = int(len(old_root['data']['state']))
        cprint('existing frames: {}, new frames will start after offset'.format(append_offset), 'yellow')
    elif os.path.exists(save_dir):
        if not args.overwrite:
            cprint('Data already exists at {}'.format(save_dir), 'red')
            cprint('Refusing to overwrite: pass --overwrite (OVERWRITE=1) or --append, '
                   'or delete the directory first.', 'red')
            raise SystemExit(1)
        cprint('Overwriting {}'.format(save_dir), 'red')
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    # load expert ckpt
    loaded_dict = torch.load(args.expert_ckpt_path, map_location='cpu')
    expert_agent = loaded_dict['agent']
    expert_agent.to('cuda')
    
    cprint('Loaded expert ckpt from {}'.format(args.expert_ckpt_path), 'green')

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    full_point_cloud_arrays = []
    full_point_cloud_len_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    success_steps_arrays = []
    

    # loop over episodes
    minimal_episode_length = int(args.min_episode_steps)
    episode_idx = 0
    while episode_idx < num_episodes:
        env = create_env(args.seed + episode_idx)
        time_step = env.reset()
        input_obs_visual = time_step.observation # (3n,84,84), unit8
        input_obs_sensor = time_step.observation_sensor # float32, door(24,)q        

        total_reward = 0.
        n_goal_achieved_total = 0.
        step_count = 0
        success_step = -1   # step count when success_goal_threshold is first reached (-1 = never)
        
        img_arrays_sub = []
        point_cloud_arrays_sub = []
        full_point_cloud_arrays_sub = []
        full_point_cloud_len_arrays_sub = []
        depth_arrays_sub = []
        state_arrays_sub = []
        action_arrays_sub = []
        total_count_sub = 0
        
        while (not time_step.last()) or step_count < minimal_episode_length:
            # stop early once extra_steps are collected after success (avoids the static tail)
            if (args.extra_steps_after_success > 0 and success_step >= 0
                    and step_count - success_step >= args.extra_steps_after_success):
                cprint(f"Episode {episode_idx}: success@{success_step} steps, "
                       f"kept {step_count - success_step} post-success tail steps -> early stop "
                       f"(total {step_count} frames)", 'green')
                break
            with torch.no_grad(), utils.eval_mode(expert_agent):
                input_obs_visual = time_step.observation
                input_obs_sensor = time_step.observation_sensor
                # cam: top, vil_camera, fixed
                    
                action = expert_agent.act(obs=input_obs_visual, step=0,
                                        eval_mode=True, 
                                        obs_sensor=input_obs_sensor) # (28,) float32
                
                if args.not_use_multi_view:
                    input_obs_visual = input_obs_visual[:3] # (3,84,84)
                

                        
                # save data
                total_count_sub += 1
                if args.pc_resolution > 84:
                    # high-res capture: data/img stores the pc_resolution color render
                    # (same render/orientation as the raw_point_cloud rgb) so the edge
                    # chain (MobileSAM) really runs at that size; 84 path is unchanged.
                    cam = env.pc_generator.cam_names[0]
                    img_cap, _ = env.pc_generator.captureImage(cam, capture_depth=True)
                    img_arrays_sub.append(np.ascontiguousarray(img_cap))
                else:
                    img_arrays_sub.append(input_obs_visual)
                state_arrays_sub.append(input_obs_sensor)
                action_arrays_sub.append(action)
                point_cloud_arrays_sub.append(time_step.observation_pointcloud)
                depth_arrays_sub.append(time_step.observation_depth)
                # full point cloud (before FPS downsampling), variable length
                full_pc, _ = env.get_full_point_cloud()
                full_point_cloud_arrays_sub.append(full_pc)
                full_point_cloud_len_arrays_sub.append(full_pc.shape[0])
                
            time_step = env.step(action)
            obs = time_step.observation # np array, (3,84,84)
            obs = obs[:3] if obs.shape[0] > 3 else obs # (3,84,84)
            n_goal_achieved_total += time_step.n_goal_achieved
            total_reward += time_step.reward
            step_count += 1
            if success_step < 0 and n_goal_achieved_total >= args.success_goal_threshold:
                success_step = step_count
            
        if n_goal_achieved_total < 10.:
            cprint(f"Episode {episode_idx} has {n_goal_achieved_total} goals achieved and {total_reward} reward. Discarding.", 'red')
        else:
            total_count += total_count_sub
            episode_ends_arrays.append(deepcopy(total_count)) # the index of the last step of the episode    
            img_arrays.extend(deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
            full_point_cloud_arrays.extend(deepcopy(full_point_cloud_arrays_sub))
            full_point_cloud_len_arrays.extend(deepcopy(full_point_cloud_len_arrays_sub))
            depth_arrays.extend(deepcopy(depth_arrays_sub))
            state_arrays.extend(deepcopy(state_arrays_sub))
            action_arrays.extend(deepcopy(action_arrays_sub))
            success_steps_arrays.append(deepcopy(int(success_step)))  # local success step; -1 = none
            print('Episode: {}, Reward: {}, Goal Achieved: {}'.format(episode_idx, total_reward, n_goal_achieved_total)) 
            # export per-episode video of full point clouds (matplotlib + imageio, no Chrome needed)
            if args.save_video and len(full_point_cloud_arrays_sub) > 0:
                try:
                    import sys as _sys
                    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
                    if ROOT_DIR not in _sys.path:
                        _sys.path.insert(0, ROOT_DIR)
                    from visualizer.visualizer.pointcloud import Visualizer
                    video_dir = os.path.join(out_dir, 'video')
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, 'adroit_{}_episode_{:04d}.mp4'.format(args.env_name, episode_idx))
                    Visualizer().save_pointcloud_video(
                        full_point_cloud_arrays_sub, video_path, fps=10)
                except Exception as e:
                    cprint(f"[save_video] failed for episode {episode_idx}: {e}", 'yellow')
            episode_idx += 1

    # tracemalloc.stop()
    ###############################
    # save data
    ###############################
    # create zarr file (append-aware)
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3: # make channel last
        img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays) + append_offset

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])

    def _write_or_append(key, arr, chunks, dtype, codec=None):
        if key in zarr_data and append_offset > 0:
            existing = np.asarray(zarr_data[key][:])
            merged = np.concatenate([existing, arr], axis=0)
            del zarr_data[key]
            zarr_data.create_dataset(key, data=merged, chunks=chunks, dtype=dtype,
                                     overwrite=True, compressor=compressor, object_codec=codec)
        else:
            zarr_data.create_dataset(key, data=arr, chunks=chunks, dtype=dtype,
                                     overwrite=True, compressor=compressor, object_codec=codec)

    _write_or_append('img', img_arrays, img_chunk_size, 'uint8')
    _write_or_append('state', state_arrays, state_chunk_size, 'float32')
    _write_or_append('point_cloud', point_cloud_arrays, point_cloud_chunk_size, 'float32')
    _write_or_append('depth', depth_arrays, depth_chunk_size, 'float32')
    _write_or_append('action', action_arrays, action_chunk_size, 'float32')

    if 'episode_ends' in zarr_meta and append_offset > 0:
        old_ends = zarr_meta['episode_ends'][:]
        new_ends = np.concatenate([old_ends, episode_ends_arrays])
        zarr_meta.create_dataset('episode_ends', data=new_ends, dtype='int64',
                                 overwrite=True, compressor=compressor)
    else:
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64',
                                 overwrite=True, compressor=compressor)

    # per-episode success step (local frame index; -1 = success_goal_threshold never reached).
    # after the success frame s only extra_steps_after_success frames are kept.
    success_steps = np.array(success_steps_arrays, dtype=np.int64)
    if append_offset > 0 and 'success_steps' not in zarr_meta:
        # legacy zarr has no such field: cannot align with episode_ends, skip writing
        cprint('[meta] legacy zarr has no success_steps; skip writing to avoid '
               'index misalignment with episode_ends (append mode)', 'yellow')
    elif 'success_steps' in zarr_meta and append_offset > 0:
        old_s = zarr_meta['success_steps'][:]
        zarr_meta.create_dataset('success_steps',
                                 data=np.concatenate([old_s, success_steps]),
                                 dtype='int64', overwrite=True, compressor=compressor)
    else:
        zarr_meta.create_dataset('success_steps', data=success_steps, dtype='int64',
                                 overwrite=True, compressor=compressor)
    zarr_root.attrs['success_goal_threshold'] = float(args.success_goal_threshold)
    zarr_root.attrs['extra_steps_after_success'] = int(args.extra_steps_after_success)
    # truncation provenance: lets training/eval refuse (or warn about) mixing a
    # success-truncated dataset with a full-length one in one run
    zarr_root.attrs['truncated'] = bool(truncated)
    zarr_root.attrs['episode_len_policy'] = str(episode_len_policy)
    zarr_root.attrs['success_steps_semantics'] = (
        'per-episode LOCAL frame index where n_goal_achieved first reached '
        'success_goal_threshold (-1 = not reached); global = episode_start + value')

    # full point clouds (before FPS downsampling), variable length per frame.
    # VLenArray only supports 1-D arrays, so each frame (N, 6) is flattened to (N*6,).
    # reconstruct with: frame.reshape(raw_point_cloud_len[i], 6)
    full_pc_obj = np.empty(len(full_point_cloud_arrays), dtype=object)
    for i, pc in enumerate(full_point_cloud_arrays):
        full_pc_obj[i] = pc.astype(np.float32).reshape(-1)
    full_pc_len = np.array(full_point_cloud_len_arrays, dtype=np.int64)

    def _write_or_append_obj(key, arr_obj, codec):
        if key in zarr_data and append_offset > 0:
            existing = zarr_data[key][:]
            merged = np.concatenate([existing, arr_obj], axis=0)
            del zarr_data[key]
            zarr_data.create_dataset(key, data=merged, dtype=object, chunks=(100,),
                                     overwrite=True, compressor=compressor, object_codec=codec)
        else:
            zarr_data.create_dataset(key, data=arr_obj, dtype=object, chunks=(100,),
                                     overwrite=True, compressor=compressor, object_codec=codec)

    _write_or_append_obj('raw_point_cloud', full_pc_obj, VLenArray(np.float32))
    _write_or_append('raw_point_cloud_len', full_pc_len, (100,), 'int64')

    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'raw_point_cloud (full, before FPS): n_frames={len(full_pc_obj)}, n_points_per_frame min={full_pc_len.min() if len(full_pc_len)>0 else 0}, max={full_pc_len.max() if len(full_pc_len)>0 else 0}', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # clean up
    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del full_pc_obj, full_pc_len
    del full_point_cloud_arrays, full_point_cloud_len_arrays
    del zarr_root, zarr_data, zarr_meta
    del env, expert_agent
    
    
if __name__ == '__main__':
    main()