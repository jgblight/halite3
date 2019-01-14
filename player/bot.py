#!/usr/bin/env python3
import logging
import os
from collections import defaultdict
import random
import time

from player.model import MovementModel, SpawnModel
from player.constants import DIRECTION_ORDER
from player.hlt import constants, positionals, Game
from player.state import GameState
from player.utils import Timer, log_message


class Bot:
    def __init__(self, name, ckpt_file, params):
        with Timer("start game"):
            # During init phase: initialize the model and compile it
            with Timer('Initialize Model'):
                my_model = MovementModel(cached_model=ckpt_file, params_file=params)

            # Get the initial game state
            game = Game()
            self.my_model = my_model
            self.game = game
            self.last_move = {}
            self.avoid = set()

            game.ready(name)

    def is_dumb_move(self, game_map, ship, ml_move):
        destination = ship.position.directional_offset(ml_move)
        if game_map[destination].is_occupied:
            return True

        if destination in self.avoid:
            return True

        if ship.id in self.last_move and self.last_move[ship.id] == destination:
            return True
        return False

    def generate_state(self, game_map, me, other_players, turn_number):
        my_ships = {s.id: s for s in me.get_ships()}
        opp_ships = {s.id: s for p in other_players for s in p.get_ships()}
        my_dropoffs = list(me.get_dropoffs()) + [me.shipyard]
        opp_dropoffs = [d for p in other_players for d in p.get_dropoffs()] + \
                       [p.shipyard for p in other_players]
        frame = [[y.halite_amount for y in x] for x in game_map._cells]
        return GameState(float(turn_number)/constants.MAX_TURNS, frame, {}, my_ships, opp_ships, my_dropoffs, opp_dropoffs)


    def run(self):
        # Some minimal state to say when to go home
        go_home = defaultdict(lambda: False)
        while True:
            logging.warning("turn {}".format(self.game.turn_number))
            with Timer("update frame", self.game.turn_number < 5):
                self.game.update_frame()
                turn_start = time.time()
                me = self.game.me  # Here we extract our player metadata from the game state
                game_map = self.game.game_map  # And here we extract the map metadata
                other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]


            with Timer("create avoid set", self.game.turn_number < 5):
                self.avoid = set()
                for player in other_players:
                    for ship in player.get_ships():
                        for dir in DIRECTION_ORDER:
                            self.avoid.add(ship.position.directional_offset(dir))

            command_queue = []


            with Timer("generate state", self.game.turn_number < 5):
                state = self.generate_state(game_map, me, other_players, self.game.turn_number)

            for ship in sorted(me.get_ships(), key=lambda x: x.halite_amount, reverse=True):  # For each of our ships
                # Did not machine learn going back to base. Manually tell ships to return home
                if ship.position == me.shipyard.position:
                    go_home[ship.id] = False
                elif go_home[ship.id] or ship.halite_amount >= 900 or (constants.MAX_TURNS - self.game.turn_number  <= 25 and ship.halite_amount > 0):
                    with Timer("go home", self.game.turn_number < 5):
                        go_home[ship.id] = True
                        movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                        if movement is not None:
                            game_map[ship.position].mark_safe()
                            game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                            command_queue.append(ship.move(movement))
                        else:
                            bulldoze = False
                            has_asshole = game_map[me.shipyard.position].is_occupied and game_map[me.shipyard.position].ship.owner != me.id
                            if (constants.MAX_TURNS - self.game.turn_number  <= 25 and ship.halite_amount > 0) or has_asshole:
                                for direction in game_map.get_unsafe_moves(ship.position, me.shipyard.position):
                                    target_pos = ship.position.directional_offset(direction)
                                    if target_pos == me.shipyard.position:
                                        bulldoze = True
                                        command_queue.append(ship.move(direction))
                                        break
                            if not bulldoze:
                                ship.stay_still()
                        continue

                # Use machine learning to get a move

                with Timer("predict move", self.game.turn_number < 5):
                    ml_move, backup = self.my_model.predict_move(state, ship.id)
                #logging.warning(ml_move)

                with Timer("make move", self.game.turn_number < 5):
                    if ml_move is not None:
                        if ml_move != positionals.Direction.Still and ship.halite_amount < (game_map[ship.position].halite_amount/10):
                            ship.stay_still()
                            continue
                        if ml_move == positionals.Direction.Still and (game_map[ship.position].halite_amount == 0 or (game_map[ship.position].has_structure and ship.halite_amount == 0)):
                            #logging.warning("Choosing random direction for {}".format(ship.id))
                            ml_move = backup

                        if ml_move != positionals.Direction.Still and self.is_dumb_move(game_map, ship, ml_move):
                            i = DIRECTION_ORDER.index(ml_move)
                            stop = i + 3

                            while self.is_dumb_move(game_map, ship, ml_move) and i < stop:
                                i += 1
                                ml_move = DIRECTION_ORDER[i%4]

                        movement = game_map.get_safe_move(game_map[ship.position],
                                                          game_map[ship.position.directional_offset(ml_move)])
                        if movement is not None:
                            cell = game_map[ship.position.directional_offset(movement)]
                            game_map[ship.position].mark_safe()
                            cell.mark_unsafe(ship)
                            self.last_move[ship.id] = ship.position
                            command_queue.append(ship.move(movement))
                            continue
                    ship.stay_still()

            # Spawn some more ships
            with Timer("spawn", self.game.turn_number < 5):
                if me.halite_amount >= constants.SHIP_COST and self.game.turn_number <= constants.MAX_TURNS/2:
                    #if self.spawn_model.predict(state):
                    command_queue.append(self.game.me.shipyard.spawn())

            #logging.warning("turn took {}".format(time.time() - turn_start))
            self.game.end_turn(command_queue)  # Send our moves back to the game environment
