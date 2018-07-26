import logging
import math
from pathlib import Path

from tensorforce.environments import Environment

from .neyboy import SyncGame, GAME_OVER_SCREEN


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

ACTION_NAMES = ["none", "left", "right"]
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


class NeyboyEnvironment(Environment):

    def __init__(self, headless=True, score_threshold=0.95, death_reward=-1, user_data_dir=None):
        Environment.__init__(self)

        if user_data_dir is None:
            home_dir = Path.home()
            neyboy_data_dir = Path(home_dir, 'neyboy')
            user_data_dir = str(neyboy_data_dir)

        self.scoring_threshold = score_threshold
        self.death_reward = death_reward
        self.game = SyncGame.create(headless=headless, user_data_dir=user_data_dir)
        self.game.load()
        self._update_state()

    def _update_state(self):
        self._state = self.game.get_state()

    @property
    def state(self):
        return self._state

    @property
    def states(self):
        return dict(shape=self._state.snapshot.shape, type='float32')

    @property
    def actions(self):
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def close(self):
        self.game.stop()

    def reset(self):
        log.debug('{}'.format(self))
        self.game.restart()
        self._update_state()
        self.game.pause()
        return self.state.snapshot

    def execute(self, action):
        self.game.resume()
        if action == ACTION_LEFT:
            self.game.tap_left()
        elif action == ACTION_RIGHT:
            self.game.tap_right()
        self._update_state()
        self.game.pause()
        is_over = self.state.status == GAME_OVER_SCREEN
        if is_over:
            reward = self.death_reward
        else:
            angle = self.state.position['angle']
            cosine = math.cos(angle)
            reward = cosine if cosine > self.scoring_threshold else 0
            log.debug('HiScore: {}, Score: {}, Action: {}, position_label: {}, Reward: {}, GameOver: {}'.format(self.state.hiscore,
                                                                                            self.state.score,
                                                                                            ACTION_NAMES[action],
                                                                                            self.state.position['name'],
                                                                                            reward,
                                                                                            is_over))
        return self.state.snapshot, is_over, reward

    def __str__(self):
        return 'NeyboyEnvironment(hiscore={}, score={}, status={})'.format(self._state.hiscore, self._state.score, self._state.status)
