import os

import gym
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import MaxAndSkipEnv, WarpFrame
from baselines.common.cmd_util import arg_parser
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from gym import wrappers

import gym_neyboy
from neyboy.baselines.neyboy_wrappers import Cropper


def make_neyboy_environment(env_id, seed=0, rank=0, allow_early_resets=False, frame_skip=4, save_video=False):
    env = gym.make(env_id)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    env.seed(seed + rank)
    logdir = logger.get_dir() and os.path.join(logger.get_dir(), str(rank))
    if save_video:
        env = wrappers.Monitor(env, logdir, force=True)
    else:
        env = Monitor(env, logdir, allow_early_resets=allow_early_resets)
    return env


def make_neyboy_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0, allow_early_resets=False, frame_skip=4, save_video=False):
    """
    Create a wrapped, monitored SubprocVecEnv for Neyboy.
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}

    def make_env(rank):
        def _thunk():
            env = make_neyboy_environment(env_id, seed, rank, allow_early_resets, frame_skip=frame_skip, save_video=save_video)
            # env = Cropper(env)
            env = WarpFrame(env)
            return env
        return _thunk

    set_global_seeds(seed)

    envs = [make_env(i + start_index) for i in range(num_env)]
    if num_env > 1:
        env = SubprocVecEnv(envs)
    else:
        env = DummyVecEnv(envs)

    return env


def neyboy_arg_parser():
    """
    Create an argparse.ArgumentParser for run_neyboy.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', default='neyboy-v0')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))

    return parser
