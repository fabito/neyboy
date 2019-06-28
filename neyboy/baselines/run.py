#!/usr/bin/env python3
import datetime
import multiprocessing
import os
import sys

import numpy as np
import tensorflow as tf
from baselines import logger
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from neyboy.baselines.ppo2 import ppo2
from neyboy.baselines.cmd_util import neyboy_arg_parser, make_neyboy_env

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['epinfobuf'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 200
    return is_solved


def train(env_id, learning_rate, max_learning_rate, num_epoch, buffer_size, batch_size, num_timesteps, num_workers, seed, policy, load_path, frame_skip):

    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    # nenv = num_workers or ncpu if not render else 1
    # alg = policy
    # rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    env = VecFrameStack(make_neyboy_env(env_id, num_workers, seed, frame_skip=frame_skip), 4)

    nsteps = buffer_size // num_workers
    nminibatches = buffer_size // batch_size

    logger.info('buffer_size={}'.format(buffer_size))
    logger.info('batch_size={}'.format(batch_size))
    logger.info('num-workers={}'.format(num_workers))
    logger.info('nsteps={}'.format(nsteps))
    logger.info('nminibatches={}'.format(nminibatches))
    logger.info('noptepochs={}'.format(num_epoch))
    logger.info('lr={}'.format(learning_rate))
    logger.info('max_lr={}'.format(max_learning_rate))
    logger.info('load_path={}'.format(load_path))
    logger.info('frame_skip={}'.format(frame_skip))

    total_timesteps = int(num_timesteps * 1.1)

    def lr_fn(frac, iteration):
        num_iterations = nminibatches
        stepsize = num_iterations // 2
        base_lr = frac * learning_rate
        max_lr = frac * max_learning_rate
        cycle = np.floor(1 + iteration / (2 * stepsize))
        x = np.abs(iteration / stepsize - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))
        return lr

    ppo2.learn(network=policy, env=env, nsteps=nsteps, nminibatches=nminibatches,
               lam=0.95, gamma=0.99, noptepochs=num_epoch, log_interval=1,
               ent_coef=0.0,
               lr=lr_fn,
               cliprange=lambda f: f * 0.1,
               total_timesteps=total_timesteps,
               save_interval=10, load_path=load_path)


def main():
    parser = neyboy_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--output-dir', help='Output dir', default='/tmp')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--buffer-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epoch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--max-lr', type=float, default=8e-4)
    parser.add_argument('--load-path', help='load path', default=None)
    args = parser.parse_args()

    dir_sufix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    dir_name = os.path.join(args.output_dir, 'w{}-b{}-buf{}-e{}-fs{}-lr{}--{}'.format(args.num_workers, args.batch_size, args.buffer_size, args.num_epoch, args.frame_skip, args.lr, dir_sufix))
    format_strs = 'stdout,log,csv,tensorboard'.split(',')
    logger.configure(dir_name, format_strs)
    train(args.env, learning_rate=args.lr, max_learning_rate=args.max_lr, num_epoch=args.num_epoch, buffer_size=args.buffer_size,
          batch_size=args.batch_size, num_timesteps=args.num_timesteps, seed=args.seed,
          num_workers=args.num_workers, policy=args.policy, load_path=args.load_path, frame_skip=args.frame_skip)


if __name__ == '__main__':
    main()
