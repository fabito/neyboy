from pathlib import Path
from time import sleep

from tensorforce.environments import Environment

from .neyboy import SyncGame, GAME_OVER_SCREEN, EASTER_EGG_APPEARANCE_FREQUENCY

ACTION_NAMES = ["none", "left", "right"]
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


class NeyboyEnvironment(Environment):

    def __init__(self, headless=True, frame_skip=2.5, scoring_reward=1, death_reward=-1, stay_alive_reward=0.01, easter_egg_reward=5, user_data_dir=None):
        Environment.__init__(self)
        self.frame_skip = frame_skip
        self.stay_alive_reward = stay_alive_reward
        self.death_reward = death_reward
        self.scoring_reward = scoring_reward
        self.easter_egg_reward = easter_egg_reward

        if user_data_dir is None:
            home_dir = Path.home()
            neyboy_data_dir = Path(home_dir, 'neyboy')
            user_data_dir = str(neyboy_data_dir)

        self.game = SyncGame.create(headless=headless, user_data_dir=user_data_dir)
        self.game.load()
        self._update_state()

    def _update_state(self):
        self._state = self.game.get_state()

    @property
    def states(self):
        return dict(shape=self._state.snapshot.shape, type='float32')

    @property
    def actions(self):
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def close(self):
        self.game.stop()

    def reset(self):
        self.game.restart()
        self._update_state()
        self.game.pause()
        return self._state.snapshot

    def execute(self, actions):
        score_before_action = self._state.score
        is_over = False
        self.game.resume()
        if actions == ACTION_LEFT:
            self.game.tap_left()
        elif actions == ACTION_RIGHT:
            self.game.tap_right()
        # sleep(self.frame_skip / 10)
        self._update_state()
        # self.game.pause()
        if self._state.status == GAME_OVER_SCREEN:
            reward = self.death_reward
            is_over = True
        else:
            if self._state.score > score_before_action:
                reward = self.scoring_reward
            else:
                reward = self.stay_alive_reward + self._state.score / 1000

            if (self._state.score % EASTER_EGG_APPEARANCE_FREQUENCY) == 0:
                reward += self.easter_egg_reward * self._state.score / EASTER_EGG_APPEARANCE_FREQUENCY


        print('HiScore: {}, Score: {}, Action: {}, Reward: {}, GameOver: {}'.format(self._state.hiscore, self._state.score, ACTION_NAMES[actions], reward,
                                                                       is_over))
        return self._state.snapshot, is_over, reward

    def __str__(self):
        return 'NeyboyEnvironment()'
