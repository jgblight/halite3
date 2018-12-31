import time
import logging
import numpy as np

from player.hlt import positionals, constants
from player.hlt import GameMap, MapCell, Position
from player.constants import MAX_BOARD_SIZE, MOVE_TO_OUTPUT, FEATURE_SIZE
from player.utils import Timer

def pad_array(arr, dims, max_size):
    shape = arr.shape
    pad_width = []
    for d, size in shape:
        pad_width.append((0, max_size - size) if d in dims else (0,0))
    return np.pad(arr, pad_width, 'constant', constant_values=0)


class GameState:

    DIRECTION_ORDER = [positionals.Direction.West,
                   positionals.Direction.North,
                   positionals.Direction.East,
                   positionals.Direction.South]

    def __init__(self, turn_number, frame, moves, ships, other_ships, dropoffs, other_dropoffs):
        self.map_size = len(frame)
        self.turn_number = turn_number
        self.frame = frame
        self.moves = moves
        self.ships = ships #dict
        self.other_ships = other_ships #dict
        self.dropoffs = dropoffs #list
        self.other_dropoffs = other_dropoffs #list
        self._map = None

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

    def get_features_for_ship(self, ship_id):
        with Timer("Generate Features"):
            ship = self.ships[ship_id]
            feature_map = np.zeros((MAX_BOARD_SIZE, MAX_BOARD_SIZE, 45), dtype=np.float32)

            ships = set([ x.position for x in self.ships.values()])
            other_ships = set([ x.position for x in self.other_ships.values()])
            dropoffs = set([ x.position for x in self.dropoffs])
            other_dropoffs = set([ x.position for x in self.other_dropoffs])

            for i, objs in enumerate([ships, other_ships, dropoffs, other_dropoffs]):
                for y in range(self.map_size):
                    for x in range(self.map_size):
                        if Position(x=x, y=y) in objs:
                            self._update_feature_map(feature_map, i, ship, x, y, 1)
            i_base = 3
            for y in range(self.map_size):
                for x in range(self.map_size):
                    h_amount = self.game_map[Position(x=x, y=y)].halite_amount
                    for i, threshold in enumerate(range(0, 1000, 50)):
                        if h_amount <= threshold:
                            self._update_feature_map(feature_map, i+i_base, ship, x, y, 1)
                    self._update_feature_map(feature_map, 23, ship, x, y, h_amount/1000.)
            i_base = 24
            for ship_id, our_ship in self.ships.items():
                h_amount = our_ship.halite_amount
                for i, threshold in enumerate(range(0, 1000, 50)):
                    if h_amount >= threshold:
                        self._update_feature_map(feature_map, i+i_base, ship, our_ship.position.x, our_ship.position.y, 1)
                self._update_feature_map(feature_map, 44, ship, our_ship.position.x, our_ship.position.y, h_amount/1000.)

        return feature_map

    def _get_toroid_coordinates(self, ship, x, y):
        x_1 = int((x - ship.position.x) + (MAX_BOARD_SIZE/2)) % MAX_BOARD_SIZE
        y_1 = int((y - ship.position.y) + (MAX_BOARD_SIZE/2)) % MAX_BOARD_SIZE

        x_coords = [x_1]
        y_coords = [y_1]

        if x_1 >= self.map_size:
            x_coords.append(x_1 - self.map_size)
        if x_1 + self.map_size < MAX_BOARD_SIZE:
            x_coords.append(x_1 + self.map_size)

        if y_1 >= self.map_size:
            y_coords.append(y_1 - self.map_size)
        if y_1 + self.map_size < MAX_BOARD_SIZE:
            y_coords.append(y_1 + self.map_size)

        coords = []
        for x_ in x_coords:
            for y_ in y_coords:
                coords.append((x_, y_))
        return coords

    def _update_feature_map(self, feature_map, feature_index, ship, x, y, feature_value):
        for x_, y_ in self._get_toroid_coordinates(ship, x, y):
            feature_map[y_][x_][feature_index] = feature_value

    # Generate the feature vector
    def input_for_ship(self, ship_id, rotation=0):
        result = []
        ship = self.ships[ship_id]

        # game turn
        percent_done = self.turn_number / constants.MAX_TURNS
        result.append(percent_done)
        ships = [ x.position for x in self.ships.values()]
        other_ships = [ x.position for x in self.other_ships.values()]
        dropoffs = [ x.position for x in self.dropoffs]
        other_dropoffs = [ x.position for x in self.other_dropoffs]

        # Local area stats
        for objs in [ships, other_ships, dropoffs, other_dropoffs]:
            objs_directions = []
            for d in self.DIRECTION_ORDER:
                objs_directions.append(int(self.game_map.normalize(ship.position.directional_offset(d)) in objs))
            result += self.rotate_direction_vector(objs_directions, rotation)

        # directions to highest halite cells at certain distances
        for distance in range(1, 13):
            max_halite_cell = self.max_halite_within_distance(self.game_map, ship.position, distance)
            halite_directions = self.generate_direction_vector(self.game_map, ship.position, max_halite_cell)
            result += self.rotate_direction_vector(halite_directions, rotation)

        # directions to closest drop off
        closest_dropoff = dropoffs[0]
        for dropoff in dropoffs:
            if self.game_map.calculate_distance(ship.position, dropoff) < self.game_map.calculate_distance(ship.position,
                                                                                                 closest_dropoff):
                closest_dropoff = dropoff
        dropoff_directions = self.generate_direction_vector(self.game_map, ship.position, closest_dropoff)
        result += self.rotate_direction_vector(dropoff_directions, rotation)

        # local area halite
        local_halite = []
        for d in self.DIRECTION_ORDER:
            local_halite.append(self.game_map[self.game_map.normalize(ship.position.directional_offset(d))].halite_amount / 1000)
        result += self.rotate_direction_vector(local_halite, rotation)

        # current cell halite indicators
        for i in range(0, 200, 50):
            result.append(int(self.game_map[ship.position].halite_amount <= i))
        result.append(self.game_map[ship.position].halite_amount / 1000)

        # current ship halite indicators
        for i in range(0, 200, 50):
            result.append(int(ship.halite_amount <= i))
        result.append(ship.halite_amount / 1000)
        return result

    def get_expected_moves(self):
        move_map = np.zeros((MAX_BOARD_SIZE, MAX_BOARD_SIZE))
        for ship_id, ship in self.ships.items():
            move_idx = MOVE_TO_OUTPUT[self.moves.get(ship_id, 'o')]
            move_map[ship.position.y][ship.position.x] = move_idx
        return move_map

    def get_ship_mask(self):
        ship_mask = np.zeros((MAX_BOARD_SIZE, MAX_BOARD_SIZE), dtype=np.int32)
        for ship_id, ship in self.ships.items():
            ship_mask[ship.position.y][ship.position.x] = 1
        return ship_mask

    def get_ship_moves(self):
        move_list = []
        for ship_id, ship in self.ships.items():
            move = self.moves.get(ship_id, 'o')
            move_list.append((ship_id, move))
        return move_list

    # finds cell with max halite within certain distance of location
    def max_halite_within_distance(self, game_map, location, distance):
        max_halite_cell = location
        max_halite = 0
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                loc = game_map.normalize(location + Position(dx, dy))
                if game_map.calculate_distance(location, loc) > distance:
                    continue

                # pick cell with max halite
                cell_halite = game_map[loc].halite_amount
                if cell_halite > max_halite:
                    max_halite_cell = loc
                    max_halite = cell_halite
        return max_halite_cell

    # generate vector that tells which directions to go to get from ship_location to target
    def generate_direction_vector(self, game_map, ship_location, target):
        directions = []
        for d in self.DIRECTION_ORDER:
            directions.append(
                int(game_map.calculate_distance(game_map.normalize(ship_location.directional_offset(d)), target) <
                    game_map.calculate_distance(ship_location, target)))
        return directions

    def rotate_direction_vector(self, direction_vector, rotations):
        for i in range(rotations):
            direction_vector = [direction_vector[-1]] + direction_vector[:-1]
        return direction_vector
