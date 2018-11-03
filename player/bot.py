#!/usr/bin/env python3
import logging
import os
from collections import defaultdict
import random

from player import hlt
from player import model
from player.constants import DIRECTION_ORDER
from player.hlt import constants
from player.hlt import positionals


class Bot:
    def __init__(self, name, ckpt_file):
        # Get the initial game state
        game = hlt.Game()
        game.ready(name)

        # During init phase: initialize the model and compile it
        my_model = model.HaliteModel(cached_model=ckpt_file)

        self.my_model = my_model
        self.game = game

    def run(self):
        # Some minimal state to say when to go home
        go_home = defaultdict(lambda: False)
        logging.warning("beginning game")
        while True:
            self.game.update_frame()
            logging.warning("new turn")
            me = self.game.me  # Here we extract our player metadata from the game state
            game_map = self.game.game_map  # And here we extract the map metadata
            other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]

            command_queue = []

            logging.warning("making predictions")
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
                    if ml_move == positionals.Direction.Still and game_map[ship.position].halite_amount == 0:
                        ml_move = random.choice(DIRECTION_ORDER)
                    movement = game_map.get_safe_move(game_map[ship.position],
                                                      game_map[ship.position.directional_offset(ml_move)])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        command_queue.append(ship.move(movement))
                        continue
                ship.stay_still()

            # Spawn some more ships
            if me.halite_amount >= constants.SHIP_COST and \
                    not game_map[me.shipyard].is_occupied and len(me.get_ships()) <= 4:
                command_queue.append(self.game.me.shipyard.spawn())

            self.game.end_turn(command_queue)  # Send our moves back to the game environment
