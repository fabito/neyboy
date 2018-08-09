from .neyboy import SyncGame


def main():
    game = SyncGame.create(headless=False)
    game.load()
    game.start()

    for i in range(10):
        while not game.is_over():
            game.tap_right()
        game.restart()


if __name__ == '__main__':
    main()
