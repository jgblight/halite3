import time
import logging
import numpy as np

from player.hlt import positionals, constants
from player.hlt import GameMap, MapCell, Position
from player.constants import MAX_BOARD_SIZE, MOVE_TO_OUTPUT, FEATURE_SIZE
from player.utils import Timer


class GameState:

    DIRECTION_ORDER = [positionals.Direction.West,
                   positionals.Direction.North,
                   positionals.Direction.East,
                   positionals.Direction.South]

    def __init__(self, turn_number, frame, moves, ships, other_ships, dropoffs, other_dropoffs, spawn=False):
        self.map_size = len(frame)
        self.turn_number = turn_number
        self.frame = frame
        self.moves = moves
        self.ships = ships #dict
        self.other_ships = other_ships #dict
        self.dropoffs = dropoffs #list
        self.other_dropoffs = other_dropoffs #list
        self._map = None
        self._feature_map = None
        self.spawn = spawn

    @staticmethod
    def from_game_map(game_map, turn_number, ships, other_ships, dropoffs, other_dropoffs):
        state = GameState(turn_number, [], {}, ships, other_ships, dropoffs, other_dropoffs)
        state._map = game_map
        return state

    @property
    def game_map(self):
        if not self._map:
            with Timer("Build Game Map"):
                game_map = [[None for _ in range(self.map_size)] for _ in range(self.map_size)]
                for y_position in range(self.map_size):
                    for x_position in range(self.map_size):
                        game_map[y_position][x_position] = MapCell(Position(x_position, y_position),
                                                                   self.frame[y_position][x_position])
                self._map = GameMap(game_map, self.map_size, self.map_size)
        return self._map

    @property
    def feature_map(self):
        if self._feature_map is None:
            with Timer("Generate Feature Map"):
                feature_map = np.zeros((self.map_size, self.map_size, 46), dtype=np.float32)

                ships = set([ x.position for x in self.ships.values()])
                other_ships = set([ x.position for x in self.other_ships.values()])
                dropoffs = set([ x.position for x in self.dropoffs])
                other_dropoffs = set([ x.position for x in self.other_dropoffs])

                for i, objs in enumerate([ships, other_ships, dropoffs, other_dropoffs]):
                    for y in range(self.map_size):
                        for x in range(self.map_size):
                            if Position(x=x, y=y) in objs:
                                feature_map[y][x][i] = 1
                i_base = 3
                for y in range(self.map_size):
                    for x in range(self.map_size):
                        h_amount = self.game_map[Position(x=x, y=y)].halite_amount
                        for i, threshold in enumerate(range(0, 1000, 50)):
                            if h_amount <= threshold:
                                feature_map[y][x][i+i_base] = 1
                        feature_map[y][x][23] = h_amount/1000.
                i_base = 24
                for ship_id, our_ship in self.ships.items():
                    h_amount = our_ship.halite_amount
                    for i, threshold in enumerate(range(0, 1000, 50)):
                        if h_amount >= threshold:
                            feature_map[our_ship.position.y][our_ship.position.x][i+i_base] = 1
                    feature_map[our_ship.position.y][our_ship.position.x][44] = h_amount/1000.

                for y in range(self.map_size):
                    for x in range(self.map_size):
                        feature_map[y][x][45] = self.turn_number
                if self.map_size == MAX_BOARD_SIZE:
                    self._feature_map = feature_map
                else:
                    self._feature_map = np.tile(feature_map, (2, 2, 1))
        return self._feature_map

    def feature_shift(self, ship_id):
        ship = self.ships[ship_id]
        return self._shift_map(ship.position.x, ship.position.y)

    def center_shift(self):
        center_coord = int(self.map_size/2)
        return self._shift_map(center_coord, center_coord)

    def _shift_map(self, center_x, center_y):
        shift_x = int(MAX_BOARD_SIZE/2) - center_x
        shift_y = int(MAX_BOARD_SIZE/2) - center_y

        rolled = np.roll(self.feature_map, shift=(shift_y, shift_x), axis=(0,1))
        return rolled[:MAX_BOARD_SIZE,:MAX_BOARD_SIZE,:]

    def get_ship_moves(self):
        move_list = []
        for ship_id, ship in self.ships.items():
            move = self.moves.get(ship_id, 'o')
            move_list.append((ship_id, move))
        return move_list
