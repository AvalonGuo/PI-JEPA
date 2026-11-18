# -*- coding: utf-8 -*-
"""Shared seeding helpers for the Metaworld collector.

``gen_demonstration_expert.py`` imports from here so that the same
(seed, rule) yields the same episodes.
"""
import random

import numpy as np

SEED_RULES = ('per_attempt', 'per_episode')


def seed_all(env, seed):
    """Reset every RNG that can influence an episode.

    ``SawyerXYZEnv.seeded_rand_vec`` is False, so the initial configuration is
    drawn from the *global* ``np.random`` stream; the remaining seeds are cheap
    insurance. Must be called before ``env.reset()``.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        env.seed(seed)
    except Exception:
        pass


def seed_for(rule, base_seed, kept_idx, attempt_idx):
    """Episode seed for a given rule.

    per_attempt: base + attempt index (kept episodes may be non-contiguous).
    per_episode: base + kept index (legacy fullpc behaviour; a failing seed
                 repeats forever, so it can stall).
    """
    if rule == 'per_attempt':
        return int(base_seed) + int(attempt_idx)
    if rule == 'per_episode':
        return int(base_seed) + int(kept_idx)
    raise ValueError('unknown seed rule {!r}; expected one of {}'.format(rule, SEED_RULES))


def add_seed_args(parser):
    parser.add_argument('--seed', type=int, default=42, help='base collection seed')
    parser.add_argument('--seed-rule', choices=SEED_RULES, default='per_attempt',
                        help='per_attempt: seed+attempt (default, never stalls); '
                             'per_episode: seed+kept (legacy fullpc behaviour)')
    parser.add_argument('--max-attempts', type=int, default=0,
                        help='abort after N attempts (0 = 20 * num_episodes)')
    return parser


def resolve_max_attempts(max_attempts, num_episodes):
    if max_attempts and int(max_attempts) > 0:
        return int(max_attempts)
    return 20 * int(num_episodes)
