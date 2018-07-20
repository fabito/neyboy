import asyncio
import time
from .neyboy import Game


async def main():

    game = await Game.create(headless=False)
    await game.load()

    while not await game.is_over():
        await game.save_screenshot('{}.jpg'.format(time.strftime("%Y%m%d-%H%M%S")))


asyncio.get_event_loop().run_until_complete(main())
