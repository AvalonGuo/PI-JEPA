import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import metaworld
import random
import time

from natsort import natsorted
from termcolor import cprint
from gym import spaces
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling

TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.795, 1, -0.4, 100],
}

# --- log de-duplication (same semantics as dr3d/bench/memlog.py, without its
# sys.path injection). DR3D_QUIET=1 (default) silences it; otherwise each key
# prints once per process, since suite mode rebuilds the env every episode.
_ONCE_SEEN = set()


def _once(key, msg):
    if os.environ.get("DR3D_QUIET", "0") == "1":
        return
    if key in _ONCE_SEEN:
        return
    _ONCE_SEEN.add(key)
    print(msg, flush=True)

def _build_task_env(task_name, model_xml_path=None):
    """Instantiate a metaworld v2 goal-observable env, optionally forcing the
    mujoco XML to ``model_xml_path`` (a self-contained composed scene).

    metaworld's env classes expose ``model_name`` as a *property* that resolves
    to the packaged XML path. The goal-observable wrapper metaworld builds in
    ``create_observable_goal_envs`` hardcodes ``super(type(env), env).__init__()``,
    so a naive subclass of the goal-observable class would make that ``initialize``
    recurse into itself. Instead we subclass the underlying v2 class to override
    ``model_name`` and then rebuild the same observable-goal wrapper on top,
    mirroring metaworld's own ``create_observable_goal_envs`` exactly.
    """
    factory = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]
    if model_xml_path is None:
        return factory()

    base_cls = factory.__mro__[1]   # the raw v2 env class (model_name property here)

    class ModelOverride(base_cls):
        @property
        def model_name(self):
            return str(model_xml_path)

    def _initialize(env, seed=None):
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)
        super(type(env), env).__init__()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.reset()
        env._freeze_rand_vec = True
        if seed is not None:
            env.seed(seed)
            np.random.set_state(st0)

    cls = type(base_cls.__name__ + 'GoalObservableCustom',
               (ModelOverride,), {'__init__': _initialize})
    return cls()

class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 use_point_crop=True,
                 num_points=1024,
                 image_size=128,
                 model_xml_path=None,
                 cluster_injector=None,
                 ):
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = _build_task_env(task_name, model_xml_path=model_xml_path)
        self.env._freeze_rand_vec = False
        # optional pre-FPS distractor cluster (diffusion_policy_3d.cluster_distractor).
        # None -> the historical point_cloud_sampling call, unchanged.
        self.cluster_injector = cluster_injector

        # https://arxiv.org/abs/2212.05698
        # self.env.sim.model.cam_pos[2] = [0.75, 0.075, 0.7]
        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        

        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        
        self.image_size = int(image_size)
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size)
        self.use_point_crop = use_point_crop
        _once("mwenv.use_point_crop",
              "[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop))
        self.num_points = num_points # 512
        
        x_angle = 61.4
        y_angle = -7
        self.pc_transform = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
            [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        ]) @ np.array([
            [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        ])
        
        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]
        
    
        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        
    
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, ),
                dtype=np.float32
            ),
        })

    def get_robot_state(self):
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        # cam names: ('topview', 'corner', 'corner2', 'corner3', 'behindGripper', 'gripperPOV')
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2", device_id=self.device_id)
        return img

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
        
        
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        
        
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        if self.cluster_injector is None:
            point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        else:
            # inject distractor rows into the full cloud, then let the original
            # FPS sample them (no quota / no repair: see cluster_distractor.py)
            point_cloud = self.cluster_injector(
                point_cloud,
                lambda pc: point_cloud_sampling(pc, self.num_points, 'fps'))
        
        depth = depth[::-1]
        
        return point_cloud, depth


    def get_full_point_cloud(self, use_rgb=True):
        """Same as ``get_point_cloud`` but WITHOUT the FPS downsampling step.

        Returns the full (un-sampled) point cloud after transform / scale /
        offset / crop; used by the fullpc collector to store ``raw_point_cloud``
        (mirrors ``gym_util/mjpc_wrapper.py::MujocoPointcloudWrapperAdroit.
        get_full_point_cloud`` on the Adroit side). No behaviour change for the
        existing ``point_cloud`` path.
        """
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(
            device_id=self.device_id)

        if not use_rgb:
            point_cloud = point_cloud[..., :3]

        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        if self.pc_offset is not None:
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset

        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        depth = depth[::-1]
        return point_cloud, depth
        

    def get_visual_obs(self):
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        raw_state, reward, done, env_info = self.env.step(action)
        self.cur_step += 1


        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }

        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
        }


        return obs_dict

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        """Drop render/model references.

        PointCloudGenerator only forwards ``sim.render``; the real
        MjRenderContext is cached on the MjSim and freed by GC, so clearing the
        references is enough. Suite mode calls this when it rebuilds the env
        (otherwise each rebuild accumulates a MuJoCo model + EGL context and the
        container memory cap eventually triggers the OOM killer).
        """
        try:
            self.pc_generator = None
        except Exception:
            pass
        try:
            self.env.close()
        except Exception:
            pass

