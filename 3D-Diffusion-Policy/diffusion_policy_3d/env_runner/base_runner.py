from typing import Dict, List
import os
import json
import numpy as np
import imageio
from termcolor import cprint
from diffusion_policy_3d.policy.base_policy import BasePolicy


def _media_dir(output_dir, subdir=None):
    """<output_dir>/media[/subdir]"""
    media_dir = os.path.join(output_dir, 'media')
    if subdir:
        media_dir = os.path.join(media_dir, subdir)
    return media_dir


def save_episode_video(frames, output_dir, episode_idx, fps, subdir=None):
    """
    Save a single episode's frames (T, C, H, W), uint8 as an mp4 file:
    <output_dir>/media[/subdir]/episode_{episode_idx:03d}.mp4
    """
    if frames is None or not isinstance(frames, np.ndarray):
        cprint(f"[WARN] Episode {episode_idx} video frames unavailable, skip save", 'yellow')
        return
    if frames.ndim == 5:
        frames = frames[:, 0]  # select first frame
    # (T, C, H, W) -> (T, H, W, C)
    frames = np.transpose(frames, (0, 2, 3, 1))
    media_dir = _media_dir(output_dir, subdir)
    os.makedirs(media_dir, exist_ok=True)
    path = os.path.join(media_dir, f'episode_{episode_idx:03d}.mp4')
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    cprint(f"Episode {episode_idx} video saved to {path}", 'magenta')


def save_episode_success(success_rates, output_dir, subdir=None,
                         extra_metrics=None):
    """
    Save per-episode success rates as a JSON file:
    <output_dir>/media[/subdir]/episode_success.json

    Args:
        success_rates: list of per-episode success (float 0.0 ~ 1.0)
        output_dir: run output dir
        subdir: baseline / phase_aware / None
        extra_metrics: optional dict of aggregated scalars to include
    """
    success_rates = [float(s) for s in success_rates]
    data = {
        f'episode_{i:03d}': s for i, s in enumerate(success_rates)
    }
    data['mean_success_rate'] = float(np.mean(success_rates)) \
        if len(success_rates) > 0 else 0.0
    data['n_episodes'] = len(success_rates)
    if extra_metrics:
        for k, v in extra_metrics.items():
            data[k] = float(v) if not isinstance(v, (int, float, str, bool)) \
                else v

    media_dir = _media_dir(output_dir, subdir)
    os.makedirs(media_dir, exist_ok=True)
    path = os.path.join(media_dir, 'episode_success.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    cprint(f"Episode success rates saved to {path}", 'magenta')
    return path


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
