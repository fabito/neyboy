import logging
from pathlib import Path

from tensorforce.environments import Environment

from .neyboy import SyncGame, GAME_OVER_SCREEN

ACTION_NAMES = ["none", "left", "right"]
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


class NeyboyEnvironment(Environment):

    def __init__(self, headless=True, user_data_dir=None):
        Environment.__init__(self)

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
        logging.info('{}'.format(self))
        self.game.restart()
        self._update_state()
        self.game.pause()
        return self._state.snapshot

    def execute(self, actions):
        self.game.resume()
        if actions == ACTION_LEFT:
            self.game.tap_left()
        elif actions == ACTION_RIGHT:
            self.game.tap_right()
        self._update_state()
        self.game.pause()
        reward = 1.0
        is_over = self._state.status == GAME_OVER_SCREEN
        logging.debug('HiScore: {}, Score: {}, Action: {}, Reward: {}, GameOver: {}'.format(self._state.hiscore,
                                                                                            self._state.score,
                                                                                            ACTION_NAMES[actions],
                                                                                            reward,
                                                                                            is_over))
        return self._state.snapshot, is_over, reward

    def __str__(self):
        return 'NeyboyEnvironment(hiscore={}, score={}, status={})'.format(self._state.hiscore, self._state.score, self._state.status)
