import asyncio
import uuid
from pathlib import Path

from blessed import Terminal

from neyboy import Game


async def main():

    home_dir = Path.home()
    neyboy_data_dir = Path(home_dir, 'neyboy')
    neyboy_data_dir.mkdir(exist_ok=True)
    dataset_dir = Path('/tmp/neyboy_ds/')
    dataset_dir.mkdir(exist_ok=True)

    game = await Game.create(headless=False, user_data_dir=str(neyboy_data_dir))
    await game.load()

    t = Terminal()

    with t.cbreak():
        while True:
            inp = t.inkey()
            if inp.code == t.KEY_ENTER:
                await game.restart()
                game_id = uuid.uuid4()
                frame_id = 0
                while not await game.is_over():
                    inp = t.inkey()

                    action = 0

                    if inp.code == t.KEY_LEFT:
                        await game.tap_left(0)
                        action = 1

                    elif inp.code == t.KEY_RIGHT:
                        await game.tap_right(0)
                        action = 2

                    scores = await game.get_scores()
                    state_filename = '{}_{}.jpg'.format(str(game_id), frame_id)
                    state_path = Path(dataset_dir, state_filename)
                    await game.save_screenshot(str(state_path))
                    print('{},{},{},{},{}' .format(game_id, frame_id, state_filename, scores['score'], action))
                    frame_id += 1
                    # with t.location(0, t.height - 1):
                    #     print(t.center('Hi Score: {hiscore}, Score: {score}'.format(**scores)))
            elif inp.code == t.KEY_ESCAPE:
                await game.stop()
                break


asyncio.get_event_loop().run_until_complete(main())
