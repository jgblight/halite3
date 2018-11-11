import numpy as np

from player.hlt import positionals
from player.constants import MAX_BOARD_SIZE, MOVE_TO_OUTPUT, FEATURE_SIZE

def pad_array(arr, dims, max_size):
    shape = arr.shape
    pad_width = []
    for d, size in shape:
        pad_width.append((0, max_size - size) if d in dims else (0,0))
    return np.pad(arr, pad_width, 'constant', constant_values=0)


class GameState:

    def __init__(self, turn_number, frame, moves, ships, other_ships, dropoffs, other_dropoffs):
        self.map_size = len(frame)
        self.turn_number = turn_number
        self.frame = frame
        self.moves = moves
        self.ships = ships
        self.other_ships = other_ships
        self.dropoffs = dropoffs
        self.other_dropoffs = other_dropoffs

    def get_feature_map(self):
        feature_map = np.zeros((MAX_BOARD_SIZE, MAX_BOARD_SIZE, FEATURE_SIZE), dtype=np.float32)
        # halite
        for x in range(self.map_size):
            for y in range(self.map_size):
                feature_map[x][y][0] = self.frame[x][y]
                #feature_map[x][y][1] = self.turn_number
        #for ship_id, ship in self.ships.items():
        #    feature_map[ship.position.x][ship.position.y][1] = ship.halite_amount / 1000.

        #for ship_id, ship in self.other_ships.items():
        #    feature_map[ship.position.x][ship.position.y][2] = ship.halite_amount / 1000.

        #for dropoff in self.dropoffs:
        #    feature_map[dropoff.position.x][dropoff.position.y][3] = 1
        #for dropoff in self.other_dropoffs:
        #    feature_map[dropoff.position.x][dropoff.position.y][4] = 1

        return feature_map

    def get_expected_moves(self):
        move_map = np.zeros((MAX_BOARD_SIZE, MAX_BOARD_SIZE))
        for ship_id, ship in self.ships.items():
            move_idx = MOVE_TO_OUTPUT[self.moves.get(ship_id, 'o')]
            move_map[ship.position.x][ship.position.y] = move_idx
        return move_map

    def get_ship_mask(self):
        ship_mask = np.zeros((MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=np.int32)
        for ship_id, ship in self.ships.items():
            ship_mask[ship.position.x][ship.position.y] = 1
        return ship_mask

    def get_ship_moves(self):
        move_list = []
        for ship_id, ship in self.ships.items():
            move_idx = MOVE_TO_OUTPUT[self.moves.get(ship_id, 'o')]
            move_list.append((move_idx, (ship.position.x, ship.position.y)))
        return move_list
