import asyncio
from pathlib import Path

from blessed import Terminal

from neyboy import Game


async def main():

    home_dir = Path.home()
    neyboy_data_dir = Path(home_dir, 'neyboy')
    neyboy_data_dir.mkdir(exist_ok=True)

    game = await Game.create(headless=False, user_data_dir=str(neyboy_data_dir))
    await game.load()

    t = Terminal()
    # print(t.bold('Hi there!'))
    # print(t.bold_red_on_bright_green('It hurts my eyes!'))

    with t.cbreak():
        while True:
            inp = t.inkey()
            # print('You pressed ' + repr(inp))
            if inp.code == t.KEY_LEFT:
                await game.tap_left(0)
            elif inp.code == t.KEY_RIGHT:
                await game.tap_right(0)
            elif inp.code == t.KEY_UP:
                await game.restart()
            elif inp.code == t.KEY_ESCAPE:
                await game.stop()
                break

            scores = await game.get_scores()
            with t.location(0, t.height - 1):
                print(t.center('Hi Score: {hiscore}, Score: {score}'.format(**scores)))


asyncio.get_event_loop().run_until_complete(main())
