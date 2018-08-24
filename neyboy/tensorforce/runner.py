import os
import time

from neyboy_tensorforce.execution import Runner
import tensorflow as tf


class MetricsLoggingRunner(Runner):

    def __init__(self, agent, environment, repeat_actions=1, history=None, id_=0, log_dir=None):
        super(MetricsLoggingRunner, self).__init__(agent, environment, repeat_actions, history, id_)
        self.log_dir = log_dir

    def run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None, deterministic=False,
            episode_finished=None, summary_report=None, summary_interval=None, timesteps=None, episodes=None,
            testing=False, sleep=None):
        if self.log_dir is not None and episode_finished is not None:
            tf.gfile.MakeDirs(self.log_dir)
            log_file = os.path.join(self.log_dir, 'metrics.csv')
            with tf.gfile.GFile(log_file, mode='a') as csv:
                def new_episode_finished(r, id_):
                    episode = r.global_episode
                    reward = r.episode_rewards[-1]
                    steps_per_second = r.global_timestep / (time.time() - r.start_time)
                    csv.write('{:d},{:d},{:0.2f},{:f},{:d}\n'.format(episode, r.current_timestep, steps_per_second, reward, self.environment.state.score))
                    return episode_finished(r, id_)
                super().run(num_timesteps, num_episodes, max_episode_timesteps, deterministic, new_episode_finished, summary_report,
                            summary_interval, timesteps, episodes, testing, sleep)
        else:
            super().run(num_timesteps, num_episodes, max_episode_timesteps, deterministic, episode_finished,
                        summary_report,
                        summary_interval, timesteps, episodes, testing, sleep)
