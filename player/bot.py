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
from player.state import GameState
from player.utils import Timer, log_message


class Bot:
    def __init__(self, name, ckpt_file):
        with Timer("start game"):
            # During init phase: initialize the model and compile it
            with Timer('Initialize Model'):
                my_model = model.HaliteModel(cached_model=ckpt_file)

            # Get the initial game state
            game = hlt.Game()
            self.my_model = my_model
            self.game = game
            self.last_move = {}
            self.avoid = set()

            game.ready(name)

    def is_dumb_move(self, game_map, ship, ml_move):
        destination = ship.position.directional_offset(ml_move)
        if not game_map.get_safe_move(game_map[ship.position],
                                          game_map[destination]):
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
        return GameState(turn_number, frame, {}, my_ships, opp_ships, my_dropoffs, opp_dropoffs)


    def run(self):
        # Some minimal state to say when to go home
        go_home = defaultdict(lambda: False)
        while True:
            self.game.update_frame()
            #logging.warning("turn {}".format(self.game.turn_number))
            turn_start = time.time()
            me = self.game.me  # Here we extract our player metadata from the game state
            game_map = self.game.game_map  # And here we extract the map metadata
            other_players = [p for pid, p in self.game.players.items() if pid != self.game.my_id]

            self.avoid = set()
            for player in other_players:
                for ship in player.get_ships():
                    for dir in DIRECTION_ORDER:
                        self.avoid.add(ship.position.directional_offset(dir))

            command_queue = []

            state = self.generate_state(game_map, me, other_players, self.game.turn_number)

            for ship in me.get_ships():  # For each of our ships
                # Did not machine learn going back to base. Manually tell ships to return home
                if ship.position == me.shipyard.position:
                    go_home[ship.id] = False
                elif go_home[ship.id] or ship.halite_amount >= 800:
                    go_home[ship.id] = True
                    movement = game_map.get_safe_move(game_map[ship.position], game_map[me.shipyard.position])
                    if movement is not None:
                        game_map[ship.position.directional_offset(movement)].mark_unsafe(ship)
                        command_queue.append(ship.move(movement))
                    else:
                        ship.stay_still()
                    continue

                # Use machine learning to get a move
                ml_move, backup = self.my_model.predict_move(state, ship.id)
                #logging.warning(ml_move)
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
                        cell.mark_unsafe(ship)
                        self.last_move[ship.id] = ship.position
                        command_queue.append(ship.move(movement))
                        continue
                ship.stay_still()

            # Spawn some more ships
            if me.halite_amount >= constants.SHIP_COST and \
                    not game_map[me.shipyard].is_occupied and len(me.get_ships()) < 14:
                command_queue.append(self.game.me.shipyard.spawn())

            #logging.warning("turn took {}".format(time.time() - turn_start))
            self.game.end_turn(command_queue)  # Send our moves back to the game environment
