import argparse
from pathlib import Path

from asciimatics.renderers import ColourImageFile, ImageFile
from asciimatics.screen import Screen

from .neyboy import SyncGame


def main(screen, args):
    home_dir = Path.home()
    neyboy_data_dir = Path(home_dir, 'neyboy')
    neyboy_data_dir.mkdir(exist_ok=True)

    game = SyncGame.create(headless=not args.non_headless, user_data_dir=str(neyboy_data_dir))
    game.load()

    while True:
        state = game.get_state(include_snapshot='bytes', crop=False)
        ev = screen.get_key()
        if ev in (Screen.KEY_LEFT, ord('A'), ord('a')):
            game.tap_left()
        elif ev in (Screen.KEY_RIGHT, ord('D'), ord('d')):
            game.tap_right()
        elif ev in (Screen.KEY_UP, ord('W'), ord('w')):
            game.restart()
        elif ev in (Screen.KEY_ESCAPE, ord('Q'), ord('q')):
            game.stop()
            break

        if args.color:
            renderer = ColourImageFile(screen, state.snapshot, height=args.height)
        else:
            renderer = ImageFile(state.snapshot, height=args.height, colours=screen.colours)

        image, colours = renderer.rendered_text
        for (i, line) in enumerate(image):
            screen.centre(line, i, colour_map=colours[i])
            # screen.paint(line, 0, i, colour_map=colours[i], transparent=False)
        screen.refresh()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--color', action='store_true', default=False, help="Enable colors")
    parser.add_argument('--non-headless', action='store_true', default=False, help="Enable colors")
    parser.add_argument('--height', type=int, default=30, help="Screen height")

    args = parser.parse_args()
    Screen.wrapper(main, arguments=[args])
