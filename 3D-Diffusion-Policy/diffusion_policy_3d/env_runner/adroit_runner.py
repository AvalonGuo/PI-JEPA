import wandb
import numpy as np
import torch
import tqdm
import json
import os
from diffusion_policy_3d.env import AdroitEnv
from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import MujocoPointcloudWrapperAdroit
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner, save_episode_video, save_episode_success
import diffusion_policy_3d.common.logger_util as logger_util
from termcolor import cprint

_QUIET = os.environ.get("DR3D_QUIET", "0") == "1"

def _vinfo(msg, color='cyan'):
    """非静音(DR3D_QUIET=0)时才打印的 suite 常规信息。"""
    if not _QUIET:
        cprint(msg, color)

def _dr3d_bench():
    import pathlib as _pl
    import sys as _sys
    _bench = _pl.Path(__file__).resolve().parents[1] / "dr3d" / "bench"
    if str(_bench) not in _sys.path:
        _sys.path.insert(0, str(_bench))
    import sample as _sample
    import scene_runtime as _runtime
    return _sample, _runtime

def should_stop_after_success(executed, success_substep, extra_steps):
    """True once ``extra_steps`` env substeps have run since the success crossing.

    Same semantics as ``eval_rollout`` (`--extra-steps-after-success`):
    the check runs at the TOP of the rollout loop, so the substep at the limit is
    neither executed nor recorded. ``executed``/``success_substep`` are substep
    counts (``len(env.get_rewards())``).
    """
    return bool(extra_steps and extra_steps > 0 and success_substep >= 0
                and (executed - success_substep) >= extra_steps)

class AdroitRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 eval_episodes=20,
                 max_steps=200,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 render_size=84,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 use_point_crop=True,
                 video_subdir=None,
                 seed=None,
                 distractor_suite=None,
                 success_goal_threshold=10.0,
                 extra_steps_after_success=15,
                 core_n=None,
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
        self._suite = None
        self.core_n = None if core_n in (None, 0, "0") else int(core_n)
        if self.core_n:
            cprint(f"[AdroitRunner] 方案D fixed quota: core_n={self.core_n}", 'green')
        self.seed = seed
        if seed is not None:
            cprint(f"[AdroitRunner] deterministic eval enabled: seed={seed}", 'green')

        steps_per_render = max(10 // fps, 1)

        # optional pre-FPS distractor cluster (default off: cluster_points=0 keeps
        # the wrapper's point_cloud_sampling path byte-identical)
        self.cluster_points = int(cluster_points or 0)
        self.cluster_clusters = int(cluster_clusters or 1)
        self._cluster = None
        if self.cluster_points > 0:
            from diffusion_policy_3d.cluster_distractor import ClusterInjector
            from diffusion_policy_3d.gym_util.mjpc_diffusion_wrapper import \
                ENV_POINT_CLOUD_CONFIG
            _env_key = 'adroit_' + str(task_name)
            self._cluster = ClusterInjector(
                task=_env_key, seed=seed or 0, points=self.cluster_points,
                clusters=self.cluster_clusters, radius_mode=cluster_radius_mode,
                radius_cm=cluster_radius_cm, cells=cluster_cells,
                spec_path=cluster_spec,
                num_points=ENV_POINT_CLOUD_CONFIG[_env_key].get('num_points', 512),
                verbose=not _QUIET)
            _vinfo(f"[AdroitRunner] cluster injection: N={self.cluster_points} "
                   f"x{self.cluster_clusters} r={self._cluster.radius * 100:.2f} cm "
                   f"({self._cluster.radius_mode}, "
                   f"cells={self._cluster.cells_visible}/"
                   f"{self._cluster.cells_total})", 'green')

        def env_fn():
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(
                    MujocoPointcloudWrapperAdroit(env=AdroitEnv(env_name=task_name, use_point_cloud=True),
                                                  env_name='adroit_'+task_name, use_point_crop=use_point_crop,
                                                  core_n=self.core_n,
                                                  cluster_injector=self._cluster)),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.eval_episodes = eval_episodes
        self._env_fn = env_fn
        self.env = env_fn()
        self.distractor_suite = distractor_suite
        self.success_goal_threshold = float(success_goal_threshold)
        self.extra_steps_after_success = int(extra_steps_after_success)
        if distractor_suite:
            _vinfo(f"[AdroitRunner] distractor suite '{distractor_suite}' "
                   f"per-episode scene swap, type sequence by eval seed", 'green')

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_test = logger_util.LargestKRecorder(K=3)
        self.logger_util_test10 = logger_util.LargestKRecorder(K=5)

    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        all_goal_achieved = []
        all_success_rates = []
        suite_ep_types = []
        suite_ep_pos = []
        suite_ep_refresh = []
        suite_ep_size = []
        suite_ep_hand = []
        suite_cells_source = None
        

        for episode_idx in tqdm.tqdm(range(self.eval_episodes), desc=f"Eval in Adroit {self.task_name} Pointcloud Env",
                                     leave=False, mininterval=self.tqdm_interval_sec):

            dyn = None
            if self.distractor_suite is not None:
                _dr3d_bench()
                if self._suite is None:
                    import suite as _SU
                    self._suite = _SU.Suite(self.distractor_suite, self.task_name,
                                            self.seed, self.eval_episodes)
                _inf = self._suite.begin(
                    episode_idx, rebuild_env=lambda _p=None: self._env_fn())
                self.env = _inf["env"]
                env = self.env
                dyn = _inf["dyn"]
                suite_ep_types.append(_inf["type"])
                suite_cells_source = self._suite.cells_source
                if _inf["pos"] is not None:
                    suite_ep_pos.append([float(_inf["pos"][0]),
                                         float(_inf["pos"][1])])
                if dyn is not None and self.distractor_suite == 'dynsphere':
                    suite_ep_size.append(float(dyn.r))
                    _vinfo(f"[suite:dynsphere] ep={episode_idx} start "
                           f"pos=({dyn.pos[0]:+.3f},{dyn.pos[1]:+.3f},"
                           f"z={dyn.pos[2]:.3f}) episode_radius={dyn.r:.2f} "
                           f"cells={len(dyn.cells_r)}/{len(dyn.cells)} "
                           f"mode={dyn.mode} cells_src={suite_cells_source}")
                elif dyn is not None:
                    suite_ep_hand.append(dyn.spec())
                    _vinfo(f"[suite:dynhand] ep={episode_idx} start "
                           f"z={dyn.z:.3f} path={dyn.L:.3f}m "
                           f"A=({dyn.A[0]:+.3f},{dyn.A[1]:+.3f}) "
                           f"B=({dyn.B[0]:+.3f},{dyn.B[1]:+.3f}) "
                           f"cells={len(dyn.cells_g)}(full={dyn.cells_full})/"
                           f"{len(dyn.cells)}")
                else:
                    _vinfo(f"[suite:{self.distractor_suite}] ep={episode_idx} "
                           f"{_inf['desc']} seed={self.seed}")

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
                _pe = dyn.reapply()
                if _pe > 1e-6:
                    cprint(f"[suite:{self.distractor_suite}] post-reset reapply "
                           f"pose_err={_pe:.2e}", 'yellow')
                else:
                    _vinfo(f"[suite:{self.distractor_suite}] post-reset reapply "
                           f"pose_err={_pe:.2e}")
            policy.reset()

            done = False
            num_goal_achieved = 0.0
            actual_step_count = 0
            success_substep = -1
            act_chunk = None
            j = 0
            last_info = None
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
                if 'goal_achieved' in info and np.asarray(info['goal_achieved']).size:
                    ga = float(np.asarray(info['goal_achieved']).ravel()[-1])
                num_goal_achieved += ga
                if (success_substep < 0 and
                        num_goal_achieved >= self.success_goal_threshold):
                    success_substep = int(actual_step_count)
                done = bool(np.all(done))
                last_info = info

            if dyn is not None:
                dyn.finish()
                suite_ep_refresh.append(dyn.history())
                if self.distractor_suite == 'dynhand':
                    _vinfo(f"[suite:dynhand] ep={episode_idx} ticks="
                           f"{len(dyn.records)} z={dyn.z:.3f} "
                           f"path={dyn.L:.3f}m pose_err={dyn.pose_err():.1e}")
                else:
                    _vinfo(f"[suite:dynsphere] ep={episode_idx} placements="
                           f"{len(dyn.records)} episode_radius={dyn.r:.2f}")

            all_success_rates.append(last_info['goal_achieved'])
            all_goal_achieved.append(num_goal_achieved)

            if getattr(self, 'save_episode_video', True):
                save_episode_video(env.env.get_video(), self.output_dir,
                                   episode_idx, self.fps,
                                   subdir=self.video_subdir)

        if self.distractor_suite is not None:
            _vinfo("[suite] private scene dir removed (third_party untouched)", 'green')

        log_data = dict()
        

        log_data['mean_n_goal_achieved'] = np.mean(all_goal_achieved)
        log_data['mean_success_rates'] = np.mean(all_success_rates)

        log_data['test_mean_score'] = np.mean(all_success_rates)

        cprint(f"test_mean_score: {np.mean(all_success_rates)}", 'green')

        self.logger_util_test.record(np.mean(all_success_rates))
        self.logger_util_test10.record(np.mean(all_success_rates))
        log_data['SR_test_L3'] = self.logger_util_test.average_of_largest_K()
        log_data['SR_test_L5'] = self.logger_util_test10.average_of_largest_K()

        per_episode_success = [float(np.mean(s)) for s in all_success_rates]

        if getattr(self, 'save_episode_video', True):
            save_episode_success(
                per_episode_success, self.output_dir,
                subdir=self.video_subdir,
                extra_metrics={'mean_n_goal_achieved':
                               float(np.mean(all_goal_achieved))})

        if self.distractor_suite is not None:
            if self.video_subdir:
                suite_dir = os.path.join(self.output_dir, 'media',
                                         self.video_subdir)
            else:
                suite_dir = os.path.join(self.output_dir, 'media')
            os.makedirs(suite_dir, exist_ok=True)
            suite_data = {
                f'episode_{i:03d}': per_episode_success[i]
                for i in range(len(per_episode_success))}
            suite_data.update({
                'suite': self.distractor_suite,
                'seed': int(self.seed) if self.seed is not None else None,
                'mean_success_rate': float(np.mean(per_episode_success)),
                'n_episodes': len(per_episode_success),
                'mean_n_goal_achieved': float(np.mean(all_goal_achieved)),
                'episode_types': {f'episode_{i:03d}': t
                                  for i, t in enumerate(suite_ep_types)},
            })
            if suite_ep_pos:
                suite_data['episode_positions'] = {
                    f'episode_{i:03d}': p
                    for i, p in enumerate(suite_ep_pos)}
            if suite_ep_size:
                suite_data['episode_sizes'] = {
                    f'episode_{i:03d}': r for i, r in enumerate(suite_ep_size)}
            if suite_ep_refresh:
                suite_data['episode_refresh'] = {
                    f'episode_{i:03d}': h
                    for i, h in enumerate(suite_ep_refresh)}
            if suite_cells_source:
                suite_data['cells_source'] = suite_cells_source
            if suite_ep_hand:
                suite_data['episode_hand_specs'] = {
                    f'episode_{i:03d}': s for i, s in enumerate(suite_ep_hand)}
            suite_path = os.path.join(suite_dir, 'dr3d_suite_manifest.json')
            with open(suite_path, 'w') as f:
                json.dump(suite_data, f, indent=2)
            _vinfo(f"[suite:{self.distractor_suite}] manifest -> "
                   f"{suite_path}", 'magenta')

        if self._cluster is not None:
            # provenance of the pre-FPS cluster injection (geometry is fully
            # reproducible from seed + episode; see cluster_distractor.py)
            if self.video_subdir:
                cluster_dir = os.path.join(self.output_dir, 'media',
                                           self.video_subdir)
            else:
                cluster_dir = os.path.join(self.output_dir, 'media')
            os.makedirs(cluster_dir, exist_ok=True)
            cluster_data = self._cluster.manifest()
            cluster_data['mean_success_rate'] = float(np.mean(all_success_rates))
            cluster_data['per_episode_success'] = per_episode_success
            cluster_path = os.path.join(cluster_dir, 'cluster_manifest.json')
            with open(cluster_path, 'w') as f:
                json.dump(cluster_data, f, indent=2)
            _vinfo(f"[cluster] manifest -> {cluster_path}", 'magenta')

        videos = env.env.get_video()
        if len(videos.shape) == 5:
            videos = videos[:, 0]
        videos_wandb = wandb.Video(videos, fps=self.fps, format="mp4")
        log_data[f'sim_video_eval'] = videos_wandb

        _ = env.reset()
        videos = None
        del env

        return log_data
