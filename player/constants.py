from player.hlt import positionals

MAX_BOARD_SIZE = 64
FEATURE_SIZE = 1
OUTPUT_SIZE = 5

DIRECTION_ORDER = [positionals.Direction.West,
                   positionals.Direction.North,
                   positionals.Direction.East,
                   positionals.Direction.South]
MOVE_TO_DIRECTION = {
    "o": positionals.Direction.Still,
    "w": positionals.Direction.West,
    "n": positionals.Direction.North,
    "e": positionals.Direction.East,
    "s": positionals.Direction.South}
OUTPUT_TO_MOVE = {
    4: "o",
    0: "w",
    1: "n",
    2: "e",
    3: "s"}
MOVE_TO_OUTPUT = {v: k for k, v in OUTPUT_TO_MOVE.items()}
