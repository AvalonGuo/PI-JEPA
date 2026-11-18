import wandb
import numpy as np
import torch
import collections
import tqdm
import json
import os
from diffusion_policy_3d.env import MetaWorldEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner, save_episode_video, save_episode_success
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

_QUIET = os.environ.get("DR3D_QUIET", "0") == "1"

def _vinfo(msg, color='cyan'):
    """Routine suite info, printed only when not quiet (DR3D_QUIET=0)."""
    if not _QUIET:
        cprint(msg, color)

def _dr3d_bench():
    import pathlib as _pl
    import sys as _sys
    _bench = _pl.Path(__file__).resolve().parents[1] / "dr3d" / "bench"
    if str(_bench) not in _sys.path:
        _sys.path.insert(0, str(_bench))
    import sample as _sample
    import compose as _compose
    return _sample, _compose

def _memlog():
    """dr3d/bench/memlog.py (log dedup / quiet / memory readings); imported lazily
    so the bench dependency stays optional."""
    import pathlib as _pl
    import sys as _sys
    _bench = _pl.Path(__file__).resolve().parents[1] / "dr3d" / "bench"
    if str(_bench) not in _sys.path:
        _sys.path.insert(0, str(_bench))
    import memlog as _m
    return _m

def should_stop_after_success(executed, success_substep, extra_steps):
    """True once ``extra_steps`` env substeps have run since the success crossing.

    Mirrors the Adroit runner (--extra-steps-after-success):
    the check is done at the TOP of the rollout loop, so the substep at the limit is
    neither executed nor recorded. ``executed``/``success_substep`` are substep counts
    (``len(env.get_rewards())``). ``extra_steps <= 0`` disables the tail stop (the
    episode then runs to max_steps, matching the pre-truncation behaviour).
    """
    return bool(extra_steps and extra_steps > 0 and success_substep >= 0
                and (executed - success_substep) >= extra_steps)

class MetaworldRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=1000,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 task_name=None,
                 n_train=None,
                 n_test=None,
                 device="cuda:0",
                 use_point_crop=True,
                 num_points=512,
                 video_subdir=None,
                 seed=None,
                 distractor_suite=None,
                 success_goal_threshold=10.0,
                 extra_steps_after_success=15,
                 cluster_points=0,
                 cluster_clusters=1,
                 cluster_radius_mode="fps",
                 cluster_radius_cm=None,
                 cluster_cells=None,
                 cluster_spec=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.video_subdir = video_subdir
        self.seed = seed
        self.distractor_suite = distractor_suite
        self._suite = None
        self.success_goal_threshold = float(success_goal_threshold)
        self.extra_steps_after_success = int(extra_steps_after_success)
        if seed is not None:
            cprint(f"[MetaworldRunner] deterministic eval enabled: seed={seed}", 'green')
        if distractor_suite:
            _vinfo(f"[MetaworldRunner] distractor suite '{distractor_suite}' "
                   f"per-episode scene swap, type sequence by eval seed", 'green')

        # optional pre-FPS distractor cluster (default off: cluster_points=0 keeps
        # the wrapper's point_cloud_sampling path byte-identical)
        self.cluster_points = int(cluster_points or 0)
        self.cluster_clusters = int(cluster_clusters or 1)
        self._cluster = None
        if self.cluster_points > 0:
            from diffusion_policy_3d.cluster_distractor import ClusterInjector
            self._cluster = ClusterInjector(
                task=task_name, seed=seed or 0, points=self.cluster_points,
                clusters=self.cluster_clusters, radius_mode=cluster_radius_mode,
                radius_cm=cluster_radius_cm, cells=cluster_cells,
                spec_path=cluster_spec, num_points=num_points,
                verbose=not _QUIET)
            _vinfo(f"[MetaworldRunner] cluster injection: N={self.cluster_points} "
                   f"x{self.cluster_clusters} r={self._cluster.radius * 100:.2f} cm "
                   f"({self._cluster.radius_mode})", 'green')

        def env_fn(task_name, model_xml_path=None):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MetaWorldEnv(task_name=task_name, device=device,
                                 use_point_crop=use_point_crop,
                                 num_points=num_points,
                                 model_xml_path=model_xml_path,
                                 cluster_injector=self._cluster)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )
        self._env_fn = env_fn
        self.eval_episodes = eval_episodes
        self.env = env_fn(self.task_name)

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy, save_video=False):
        device = policy.device
        dtype = policy.dtype

        all_traj_rewards = []
        all_success_rates = []
        suite_ep_types = []
        suite_ep_pos = []
        suite_ep_size = []
        suite_ep_hand = []
        suite_ep_refresh = []
        suite_cells_source = None      # posgen: 'zarr:<task>' | 'legacy' (goes to manifest)
        env = self.env

        env.reset()

        self._reuse = {'path': None, 'env': None, 'dyn': None}

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Metaworld {self.task_name} Pointcloud Env", leave=False, mininterval=self.tqdm_interval_sec):
            
            dyn = None
            if self.distractor_suite:
                _dr3d_bench()
                if self._suite is None:
                    import suite as _SU
                    self._suite = _SU.Suite(self.distractor_suite, self.task_name,
                                            self.seed, self.eval_episodes,
                                            family='metaworld')
                _ml = _memlog()

                def _rebuild(scene_path):
                    """Rebuild only when the scene changed: soft_close the old env
                    first (memory), then rebuild from the new MJCF.

                    free=False injection (static, joint-free body) is the only
                    workable option on the metaworld side: sawyer_xyz_env hardcodes
                    the object at qpos[9:12] so a <freejoint> would shift that index,
                    and a second mocap body breaks set_xyz_action every step.
                    """
                    _ml.soft_close(self._reuse['env'])
                    _ml.gc_cuda()
                    return self._env_fn(self.task_name, model_xml_path=scene_path)

                _inf = self._suite.begin(episode_idx, rebuild_env=_rebuild,
                                         reuse=self._reuse)
                self.env = _inf['env']
                env = self.env
                dyn = _inf['dyn']
                suite_cells_source = getattr(self._suite, 'cells_source', None)
                suite_ep_types.append(_inf['type'])
                if _inf['pos'] is not None:
                    suite_ep_pos.append([float(_inf['pos'][0]),
                                         float(_inf['pos'][1])])
                if dyn is not None and self.distractor_suite == 'dynsphere':
                    suite_ep_size.append(float(dyn.r))
                elif dyn is not None:
                    suite_ep_hand.append(dyn.spec())
                if not _ml.QUIET:
                    cprint(f"[MetaworldRunner] ep={episode_idx} "
                           f"type={_inf['type']} "
                           f"xml={os.path.basename(str(_inf['scene']))} "
                           f"sha={str(_inf['sha'])[:10]} "
                           f"{'composed' if _inf['fresh'] else 'cached'}", 'cyan')
            else:
                env = self.env

            if self.seed is not None:
                import random as _random
                s = int(self.seed) + episode_idx
                torch.manual_seed(s)
                torch.cuda.manual_seed_all(s)
                np.random.seed(s)
                _random.seed(s)
                try:
                    env.seed(s)
                except Exception as e:
                    cprint(f"[WARN] env.seed failed: {e}", 'yellow')

            if self._cluster is not None:
                # per-episode cluster geometry; drawn before the first observation
                # of the episode (env.reset -> get_point_cloud -> FPS)
                self._cluster.begin(episode_idx)

            obs = env.reset()
            if dyn is not None:
                dyn.reapply()
            policy.reset()

            done = False
            traj_reward = 0
            is_success = False
            num_goal_achieved = 0.0
            actual_step_count = 0
            success_substep = -1
            act_chunk = None
            j = 0
            while not done:
                if should_stop_after_success(actual_step_count, success_substep,
                                             self.extra_steps_after_success):
                    break

                if act_chunk is None or j >= len(act_chunk):
                    if actual_step_count % self.n_action_steps != 0:
                        break
                    np_obs_dict = dict(obs)
                    obs_dict = dict_apply(np_obs_dict,
                                          lambda x: torch.from_numpy(x).to(
                                              device=device))

                    with torch.no_grad():
                        obs_dict_input = {}
                        obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                        obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                        action_dict = policy.predict_action(obs_dict_input)

                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                    act_chunk = np_action_dict['action'].squeeze(0)
                    j = 0
                    if dyn is not None:
                        dyn.plan()

                if dyn is not None:
                    dyn.tick(actual_step_count)

                obs, reward, done, info = env.step(act_chunk[j:j + 1])
                j += 1
                actual_step_count = len(env.get_rewards())
                ga = 0.0
                if 'success' in info and np.asarray(info['success']).size:
                    ga = float(np.asarray(info['success']).ravel()[-1])
                num_goal_achieved += ga
                if (success_substep < 0 and
                        num_goal_achieved >= self.success_goal_threshold):
                    success_substep = int(actual_step_count)
                traj_reward += reward
                done = np.all(done)
                is_success = is_success or max(info['success'])

            if dyn is not None:
                dyn.finish()
                suite_ep_refresh.append(dyn.history())
                _ms = _memlog().mem_str()
                if self.distractor_suite == 'dynhand':
                    _clr = dyn.work_clear_min
                    _vis = dyn.path_vis_min
                    _vinfo(f"[suite:dynhand] ep={episode_idx:03d} "
                           f"t={actual_step_count} zone_ok={int(dyn.zone_ok)} "
                           f"clear={'na' if _clr is None else format(_clr, '+.3f')} "
                           f"L={dyn.L:.3f} z={dyn.z:.3f} "
                           f"vis={'na' if _vis is None else format(_vis, '.2f')} "
                           f"scan={dyn.scan_s:.1f}s "
                           f"pose_err={dyn.pose_err():.1e} | {_ms}")
                else:
                    _vinfo(f"[suite:dynsphere] ep={episode_idx:03d} "
                           f"t={actual_step_count} "
                           f"placements={len(dyn.records)} "
                           f"radius={dyn.r:.2f} | {_ms}")

            all_success_rates.append(is_success)
            all_traj_rewards.append(traj_reward)

            if getattr(self, 'save_episode_video', True):
                save_episode_video(env.env.get_video(), self.output_dir,
                                   episode_idx, self.fps,
                                   subdir=self.video_subdir)

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        log_data['mean_traj_rewards'] = np.mean(all_traj_rewards)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)
        
        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        if getattr(self, 'save_episode_video', True):
            save_episode_success(
                all_success_rates, self.output_dir,
                subdir=self.video_subdir,
                extra_metrics={'mean_traj_rewards':
                               float(np.mean(all_traj_rewards))})

        if self.distractor_suite is not None:
            if self.video_subdir:
                suite_dir = os.path.join(self.output_dir, 'media',
                                         self.video_subdir)
            else:
                suite_dir = os.path.join(self.output_dir, 'media')
            os.makedirs(suite_dir, exist_ok=True)
            per_episode_success = [float(np.mean(s)) for s in all_success_rates]
            suite_data = {
                f'episode_{i:03d}': per_episode_success[i]
                for i in range(len(per_episode_success))}
            suite_data.update({
                'suite': self.distractor_suite,
                'seed': int(self.seed) if self.seed is not None else None,
                'mean_success_rate': float(np.mean(all_success_rates)),
                'n_episodes': len(per_episode_success),
                'episode_types': {f'episode_{i:03d}': t
                                  for i, t in enumerate(suite_ep_types)},
            })
            if suite_ep_pos:
                suite_data['episode_pos'] = {f'episode_{i:03d}': p
                                             for i, p in enumerate(suite_ep_pos)}
            if suite_cells_source:
                suite_data['cells_source'] = suite_cells_source
            if suite_ep_size:
                suite_data['episode_size'] = {f'episode_{i:03d}': r
                                              for i, r in enumerate(suite_ep_size)}
            if suite_ep_hand:
                suite_data['episode_hand_specs'] = {
                    f'episode_{i:03d}': h for i, h in enumerate(suite_ep_hand)}
            if suite_ep_refresh:
                suite_data['episode_refresh'] = {
                    f'episode_{i:03d}': h for i, h in enumerate(suite_ep_refresh)}
            suite_path = os.path.join(suite_dir, 'dr3d_suite_manifest.json')
            with open(suite_path, 'w') as f:
                json.dump(suite_data, f, indent=2)
            _vinfo(f"[suite:{self.distractor_suite}] manifest -> "
                   f"{suite_path}", 'magenta')

        if self._cluster is not None:
            # provenance of the pre-FPS cluster injection (geometry is fully
            # reproducible from seed + episode; see cluster_distractor.py)
            if self.video_subdir:
                cluster_dir = os.path.join(self.output_dir, 'media', self.video_subdir)
            else:
                cluster_dir = os.path.join(self.output_dir, 'media')
            os.makedirs(cluster_dir, exist_ok=True)
            cluster_data = self._cluster.manifest()
            cluster_data['mean_success_rate'] = float(np.mean(all_success_rates))
            cluster_data['per_episode_success'] = [
                float(np.mean(s)) for s in all_success_rates]
            cluster_path = os.path.join(cluster_dir, 'cluster_manifest.json')
            with open(cluster_path, 'w') as f:
                json.dump(cluster_data, f, indent=2)
            _vinfo(f"[cluster] manifest -> {cluster_path}", 'magenta')

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]
        
        if save_video:
            videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
            log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None

        return log_data
