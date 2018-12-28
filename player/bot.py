#!/usr/bin/env python3
import logging
import os
from collections import defaultdict
import random
import time

from player import hlt
from player import model
from player.constants import DIRECTION_ORDER
from player.hlt import constants
from player.hlt import positionals


class Bot:
    def __init__(self, name, ckpt_file):
        logging.warning('initializing model')
        # During init phase: initialize the model and compile it
        my_model = model.HaliteModel(cached_model=ckpt_file)

        # Get the initial game state
        logging.warning('intializing game')
        game = hlt.Game()
        game.ready(name)

        logging.warning("bot initialized")
        self.my_model = my_model
        self.game = game

    def run(self):
        # Some minimal state to say when to go home
        go_home = defaultdict(lambda: False)
        logging.warning("beginning game")
        while True:
            self.game.update_frame()
            logging.warning("turn {}".format(self.game.turn_number))
            turn_start = time.time()
            me = self.game.me  # Here we extract our player metadata from the game state
            game_map = self.game.game_map  # And here we extract the map metadata
            other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]

            command_queue = []

            predicted_moves = self.my_model.predict_moves(game_map, me, other_players, self.game.turn_number)
            logging.warning(predicted_moves)
            for ship in me.get_ships():  # For each of our ships
                # Did not machine learn going back to base. Manually tell ships to return home
                if ship.position == me.shipyard.position:
                    go_home[ship.id] = False
                elif go_home[ship.id] or ship.halite_amount == constants.MAX_HALITE:
                    go_home[ship.id] = True
                    movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        command_queue.append(ship.move(movement))
                    else:
                        ship.stay_still()
                    continue

                # Use machine learning to get a move
                ml_move = predicted_moves.get(ship.id)
                if ml_move is not None:
                    if ml_move != positionals.Direction.Still and ship.halite_amount < (game_map[ship.position].halite_amount/10):
                        ship.stay_still()
                        continue
                    if ml_move == positionals.Direction.Still and (game_map[ship.position].halite_amount == 0 or (game_map[ship.position].has_structure and ship.halite_amount == 0)):
                        ml_move = random.choice(DIRECTION_ORDER)
                    movement = game_map.get_safe_move(game_map[ship.position],
                                                      game_map[ship.position.directional_offset(ml_move)])

                    if ml_move != positionals.Direction.Still and movement is None:
                        i = DIRECTION_ORDER.index(ml_move)
                        stop = i + 3

                        while movement is None and i < stop:
                            i += 1
                            ml_move = DIRECTION_ORDER[i%4]
                            movement = game_map.get_safe_move(game_map[ship.position],
                                                              game_map[ship.position.directional_offset(ml_move)])
                    logging.warning("ship {} moving {}".format(ship.id, ml_move))
                    if movement is not None:
                        cell = game_map[ship.position.directional_offset(movement)]
                        cell.mark_unsafe(ship)
                        command_queue.append(ship.move(movement))
                        continue
                ship.stay_still()

            # Spawn some more ships
            if me.halite_amount >= constants.SHIP_COST and \
                    not game_map[me.shipyard].is_occupied and len(me.get_ships()) <= 4:
                command_queue.append(self.game.me.shipyard.spawn())

            logging.warning("turn took {}".format(time.time() - turn_start))
            self.game.end_turn(command_queue)  # Send our moves back to the game environment
