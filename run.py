import asyncio

from model import Game


async def main():

    game = await Game.create(headless=False)
    await game.start()

    screenshot = await game.screenshot()
    print(f'data:image/jpeg;base64,{screenshot}')
    dims = await game.dimensions()
    print(dims)
    delay = 300
    await game.tap_left(delay)
    await game.tap_right(delay)
    screenshot = await game.screenshot()
    print(f'data:image/jpeg;base64,{screenshot}')
    await game.tap_left(delay)
    await game.tap_right(delay)
    await game.pause()

    score = await game.get_score()
    print(score)
    screenshot = await game.screenshot()
    print(f'data:image/jpeg;base64,{screenshot}')
    await asyncio.sleep(2)
    await game.resume()
    await game.restart()
    await game.stop()


asyncio.get_event_loop().run_until_complete(main())
