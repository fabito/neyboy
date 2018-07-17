from time import sleep

from tensorforce.environments import Environment

from neyboy import SyncGame

ACTION_NAMES = ["none", "left", "right"]
ACTION_NONE = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


class NeyboyEnvironment(Environment):

    def __init__(self, headless=True, frame_skip=2.5, scoring_reward=1, death_reward=-1, stay_alive_reward=0.1):
        Environment.__init__(self)
        self.frame_skip = frame_skip
        self.stay_alive_reward = stay_alive_reward
        self.death_reward = death_reward
        self.scoring_reward = scoring_reward
        self.game = SyncGame.create(headless=headless)
        self.game.load()
        self._state = self.game.screenshot()
        self._score = self.game.get_score()

    def __str__(self):
        return 'NeyboyEnvironment()'

    def reset(self):
        self.game.restart()
        self._state = self.game.screenshot()
        self._score = self.game.get_score()
        self.game.pause()
        return self._state

    def execute(self, actions):
        self._score = self.game.get_score()
        is_over = False
        self.game.resume()

        if actions == ACTION_LEFT:
            self.game.tap_left(0)

        elif actions == ACTION_RIGHT:
            self.game.tap_right(0)

        sleep(self.frame_skip / 10)
        self.game.pause()

        self._state = self.game.screenshot()

        if self.game.is_over():
            reward = self.death_reward
            is_over = True
        else:
            reward = self.scoring_reward if self.game.get_score() > self._score else self.stay_alive_reward

        print('Score: {}, Action: {}, Reward: {}, GameOver: {}'.format(self._score, ACTION_NAMES[actions], reward,
                                                                       is_over))
        return self._state, is_over, reward

    @property
    def states(self):
        return dict(shape=self._state.shape, type='float32')

    @property
    def actions(self):
        return dict(num_actions=len(ACTION_NAMES), type='int')

    def close(self):
        self.game.stop()
