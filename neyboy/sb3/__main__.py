import datetime
import os
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.logger import configure, Image, TensorBoardOutputFormat
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_checker import check_env

from neyboy.sb3.cmd_util import neyboy_arg_parser, make_neyboy_env


class RawStatisticsCallback(BaseCallback):
    """
    Callback used for logging raw episode data (return and episode length).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Custom counter to reports stats
        # (and avoid reporting multiple values for the same step)
        self._timesteps_counter = 0
        self._iterations_counter = 0
        self._tensorboard_writer = None
        self._rollout_info_buffer = defaultdict(list)

    def _init_callback(self) -> None:
        # Retrieve tensorboard writer to not flood the logger output
        for out_format in self.logger.output_formats:
            if isinstance(out_format, TensorBoardOutputFormat):
                self._tensorboard_writer = out_format
        assert self._tensorboard_writer is not None, "You must activate tensorboard logging when using RawStatisticsCallback"

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if all(k in info for k in ("score", "hiscore")):
                self._rollout_info_buffer["score"].append(info["score"])
                self._rollout_info_buffer["hiscore"].append(info["hiscore"])
        return True

    def _on_rollout_start(self) -> None:
        self._rollout_info_buffer = defaultdict(list)

    def _on_rollout_end(self) -> None:
        # Display training infos
        log_interval = self.locals['log_interval']
        iteration = self.locals['iteration']
        if log_interval is not None and iteration % log_interval == 0:
            if len(self._rollout_info_buffer) > 0:
                self.logger.record("rollout/ep_score_mean", safe_mean(self._rollout_info_buffer["score"]))
                self.logger.record("rollout/ep_hiscore_mean", safe_mean(self._rollout_info_buffer["hiscore"]))


class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ImageRecorderCallback, self).__init__(verbose)

    def _on_step(self):
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
        return True


def main():
    parser = neyboy_arg_parser()
    parser.add_argument('--policy', help='Policy architecture', choices=['CnnPolicy', 'MlpPolicy', 'MultiInputPolicy'],
                        default='CnnPolicy')
    parser.add_argument('--output-dir', help='Output dir', default='/tmp')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--clip-range', type=float, default=0.2)
    parser.add_argument('--checkpoint-save-freq', type=int, default=1000)
    parser.add_argument('--load-path', help='load path', default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    dir_sufix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    dir_name = os.path.join(args.output_dir,
                            'w{}-b{}-ns{}-e{}-fs{}-lr{}-{}'.format(args.num_workers, args.batch_size,
                                                                    args.n_steps, args.num_epoch,
                                                                    args.frame_skip, args.lr, dir_sufix))
    format_strs = 'stdout,log,csv,tensorboard'.split(',')
    logger = configure(dir_name, format_strs)
    verbose = 2 if args.verbose else 1
    total_timesteps = int(args.num_timesteps * 1.1)

    _env = make_neyboy_env(args.env, args.num_workers, logger, args.seed, frame_skip=args.frame_skip)

    # _eval_env = make_neyboy_env(args.env, 1, logger, args.seed, frame_skip=args.frame_skip, allow_early_resets=True)
    # eval_env = VecTransposeImage(VecFrameStack(_eval_env, 4))
    # # Use deterministic actions for evaluation
    # eval_callback = EvalCallback(eval_env, best_model_save_path=logger.get_dir(),
    #                              log_path=logger.get_dir(), eval_freq=1000,
    #                              deterministic=True, render=False)
    # Save a checkpoint every 1000 steps

    checkpoint_callback = CheckpointCallback(save_freq=args.checkpoint_save_freq, save_path=logger.get_dir(),
                                             name_prefix='rl_model')

    # Create the callback list
    callbacks = [checkpoint_callback, ImageRecorderCallback(), RawStatisticsCallback()]

    env = VecTransposeImage(VecFrameStack(_env, 4))
    model = PPO(policy=args.policy,
                env=env,
                n_steps=args.n_steps,
                verbose=verbose,
                batch_size=args.batch_size,
                clip_range=args.clip_range,
                n_epochs=args.num_epoch,
                learning_rate=args.lr,
                tensorboard_log=logger.get_dir(),
                )
    model.learn(total_timesteps=total_timesteps, callback=callbacks, log_interval=1)


if __name__ == '__main__':
    main()
