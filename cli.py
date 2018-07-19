import asyncio
from pathlib import Path

from blessed import Terminal

from neyboy import Game
import datetime as dt
import threading


async def main():

    home_dir = Path.home()
    neyboy_data_dir = Path(home_dir, 'neyboy')
    neyboy_data_dir.mkdir(exist_ok=True)

    game = await Game.create(headless=False, user_data_dir=str(neyboy_data_dir))
    await game.load()

    t = Terminal()
    # print(t.bold('Hi there!'))
    # print(t.bold_red_on_bright_green('It hurts my eyes!'))

    def save_snapshot(state):
        print('Saving')

    def print_state(action, state):
        with t.location(0, t.height - 1):
            # noinspection PyProtectedMember
            d = state._asdict()
            d['action'] = action
            print(t.ljust('Hi Score: {hiscore}, Score: {score}, Status: {status}, Action: {action}'.format(**d)))

    start_time = dt.datetime.today().timestamp()
    frame_count = 0
    i = 0
    while True:
        action = 'none'
        state = await game.get_state(include_snapshot=True)
        with t.raw():
            inp = t.inkey(timeout=.001)
            if inp.code == t.KEY_LEFT:
                await game.tap_left()
                action = 'left'
            elif inp.code == t.KEY_RIGHT:
                await game.tap_right()
                action = 'right'
            elif inp.code == t.KEY_UP:
                await game.restart()
                frame_count = 0
            elif inp.code == t.KEY_ESCAPE:
                await game.stop()
                break
        print_state(action, state)
        threading.Thread(target=save_snapshot, args=(state,)).start()
        time_diff = dt.datetime.today().timestamp() - start_time
        i += 1
        frame_count += 1
        print(i / time_diff)

asyncio.get_event_loop().run_until_complete(main())
