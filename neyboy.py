import base64
import io
import logging
from time import sleep

import numpy as np
from PIL import Image
from pyppeteer import launch
from syncer import sync
from pathlib import Path


class Game:

    def __init__(self, headless=True, user_data_dir=None):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.is_running = False
        self.browser = None
        self.page = None

    async def initialize(self):
        if self.user_data_dir is not None:
            self.browser = await launch(headless=self.headless, userDataDir=self.user_data_dir)
        else :
            self.browser = await launch(headless=self.headless)
        self.page = await self.browser.newPage()

    @staticmethod
    async def create(headless=True, user_data_dir=None):
        o = Game(headless, user_data_dir)
        await o.initialize()
        return o

    async def dimensions(self):
        # Get dimensions of the canvas element
        dimensions = await self.page.evaluate('''() => {
                const iframe = document.getElementsByTagName("iframe")[0];
                //let {x, y, width, height} = iframe.contentWindow.document.getElementById("c2canvas").getBoundingClientRect();
                let {x, y, width, height} = iframe.getBoundingClientRect();
                return {x, y, width, height};
            }''')
        return dimensions

    async def is_loaded(self):
        await self.page.waitForFunction('''() => {
            const iframe = document.getElementsByTagName("iframe")[0];
            const iframeWindow = iframe.contentWindow;
            return iframeWindow.cr_getC2Runtime !== undefined &&
                   iframeWindow.cr_getC2Runtime().tickcount > 200;
        }''')

    async def is_over(self):
        animation_name = await self._get_is_playing_status()
        return animation_name == 2

    async def _get_is_playing_status(self):
        """
        :return: 0 for start_screen, 1 for game_screen and 2 for game_over_screen
        """
        is_playing = await self.page.evaluate('''() => {
                const iframe = document.getElementsByTagName("iframe")[0];
                const iframeWindow = iframe.contentWindow;
                return iframeWindow.cr_getC2Runtime !== undefined &&
                       iframeWindow.cr_getC2Runtime().getEventVariableByName('isPlaying').data;
                }''')
        return int(is_playing)

    async def _get_cur_animation_name(self):
        animation_name = await self.page.evaluate('''() => {
                const iframe = document.getElementsByTagName("iframe")[0];
                const iframeWindow = iframe.contentWindow;
                return iframeWindow.cr_getC2Runtime !== undefined &&
                       iframeWindow.cr_getC2Runtime().getLayerByName('Game').instances[3].cur_animation.name;
                }''')
        return animation_name

    async def load(self):
        await self.page.goto('https://neyboy.com.br', {'waitUntil': 'networkidle2'})
        await self.is_loaded()
        return self

    async def start(self):
        await self.page.click('iframe.sc-htpNat')
        return self

    async def pause(self):
        await self.page.evaluate('''() => {
            const iframe = document.getElementsByTagName("iframe")[0];
            const iframeWindow = iframe.contentWindow;
            iframeWindow.cr_setSuspended(true);
        }''')
        self.is_running = False
        return self

    async def resume(self):
        await self.page.evaluate('''() => {
            const iframe = document.getElementsByTagName("iframe")[0];
            const iframeWindow = iframe.contentWindow;
            iframeWindow.cr_setSuspended(false);
        }''')
        self.is_running = True
        return self

    async def get_score(self):
        score = await self.page.evaluate('''() => {
            const iframe = document.getElementsByTagName("iframe")[0];
            const iframeWindow = iframe.contentWindow;
            const score = iframeWindow.cr_getC2Runtime().getLayerByName('Game').instances[4].text;
            return score;
        }''')
        return int(score) if score else 1

    async def get_high_score(self):
        hiscore = await self.page.evaluate('''() => {
                const iframe = document.getElementsByTagName("iframe")[0];
                const iframeWindow = iframe.contentWindow;
                return iframeWindow.cr_getC2Runtime !== undefined &&
                       iframeWindow.cr_getC2Runtime().getEventVariableByName('hiscore').data;
                }''')
        return int(hiscore) if hiscore else 1

    async def get_scores(self):
        scores = await self.page.evaluate('''() => {
                const iframe = document.getElementsByTagName("iframe")[0];
                const iframeWindow = iframe.contentWindow;
                const runtime = iframeWindow.cr_getC2Runtime();
                const score = runtime.getLayerByName('Game').instances[4].text;
                const hiscore = runtime.getEventVariableByName('hiscore').data;
                return {score, hiscore};
                }''')
        scores['score'] = int(scores['score'])
        scores['hiscore'] = int(scores['hiscore'])
        return scores

    async def tap_right(self, delay=200):
        await self.page.mouse.click(470, 500, {delay: delay})

    async def tap_left(self, delay=200):
        await self.page.mouse.click(200, 500, {delay: delay})

    async def stop(self):
        await self.browser.close()

    async def wait_until_replay_button_is_active(self):
        await self.resume()
        await self.page.waitForFunction('''() => {
            const iframe = document.getElementsByTagName("iframe")[0];
            const iframeWindow = iframe.contentWindow;
            if (iframeWindow.cr_getC2Runtime) {
                const modal = iframeWindow.cr_getC2Runtime().getLayerByName('modal');
                replay = modal.instances[0]
                if (replay.behavior_insts) {
                    return replay.behavior_insts[0].active;
                }
            }
            return false;
        }''')

    async def restart(self):
        logging.debug('Restarting game')
        playing_status = await self._get_is_playing_status()
        if playing_status == 0:
            logging.debug('Start screen')
        # elif playing_status == 1:  # game is running
        #     logging.debug('')
        elif playing_status == 2:  # game over
            await self.wait_until_replay_button_is_active()
            logging.debug('Replay button active')
            # FIXME find out why the whataspp icon is clicked sporadically
            sleep(0.5)
            await self.page.mouse.click(400, 525)
        else:
            await self.page.reload({'waitUntil': 'networkidle2'})
            await self.is_loaded()

        await self.start()

    async def _click(self, x, y):
        await self.page.mouse.click(x, y)

    async def screenshot(self, format="jpeg", quality=30, encoding='binary'):
        dims = await self.dimensions()
        dims['y'] = dims['height'] / 2
        dims['height'] = dims['height'] - dims['y'] - 30
        snapshot = await self.page.screenshot({
            'type': format,
            'quality': quality,
            'clip': dims
        })
        if encoding == 'binary':
            return snapshot
        else:
            encoded_snapshot = base64.b64encode(snapshot)
            return encoded_snapshot.decode('ascii')

    async def save_screenshot(self, path, format="jpeg", quality=30):
        dims = await self.dimensions()
        dims['y'] = dims['height'] / 2
        dims['height'] = dims['height'] - dims['y'] - 30
        await self.page.screenshot({
            'path': path,
            'type': format,
            'quality': quality,
            'clip': dims
        })

    async def is_in_start_screen(self):
        playing_status = await self._get_is_playing_status()
        return playing_status == 0


class SyncGame:

    def __init__(self, game):
        self.game = game

    @staticmethod
    async def create(headless=True, user_data_dir=None):
        o = sync(Game.create)(headless, user_data_dir)
        return SyncGame(o)

    def dimensions(self):
        return sync(self.game.dimensions)()

    def is_loaded(self):
        return sync(self.game.is_loaded)()

    def is_over(self):
        return sync(self.game.is_over)()

    def load(self):
        return sync(self.game.load)()

    def start(self):
        return sync(self.game.start)()

    def pause(self):
        return sync(self.game.pause)()

    def resume(self):
        return sync(self.game.resume)()

    def get_score(self):
        return sync(self.game.get_score)()

    def get_scores(self):
        return sync(self.game.get_scores)()

    def get_high_score(self):
        return sync(self.game.get_high_score())()

    def tap_right(self, delay=0):
        return sync(self.game.tap_right)(delay)

    def tap_left(self, delay=0):
        return sync(self.game.tap_left)(delay)

    def stop(self):
        return sync(self.game.stop)()

    def restart(self):
        return sync(self.game.restart)()

    def _click(self, x, y):
        return sync(self.game._click)(x, y)

    def screenshot(self, format="jpeg", quality=30, encoding='binary'):
        # reconstruct image as an numpy array
        img_bytes = sync(self.game.screenshot)(format, quality, encoding)
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)


class CliGame:

    def __init__(self, game):
        self.game = game

    @staticmethod
    async def create(headless=True, user_data_dir=None):
        o = sync(Game.create)(headless, user_data_dir)
        return SyncGame(o)


