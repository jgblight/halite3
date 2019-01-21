import glob
import numpy as np
import subprocess
import random
import os
import shutil
import logging
import math
from player.parse import parse_compressed_replay_file
from player.constants import MOVE_TO_OUTPUT
from player.model import MovementModel
from player.utils import Timer

DISCOUNT = 0.9
LOOKAHEAD = 100
DEPOSIT_MULTIPLIER = 10
MAX_SAMPLES = 1000

def parse_moves(filename):
    moves = []
    with open(filename) as f:
        for line in f:
            turn, ship_id, move_label = line.strip().split(',')
            moves.append((int(turn), int(ship_id), move_label))
    return moves

def normalize_rewards(rewards):
    m = rewards.mean()
    s = rewards.std()
    return (rewards - m)/s

def get_delta_for_ship(states, ship_id, start, limit):
    max_turn = min(start+limit, len(states)-1)
    ets = [s.ships[ship_id].halite_amount if ship_id in s.ships else None for s in states]
    return [ets[i+1] - ets[i] if ets[i+1] is not None else None for i in range(start, max_turn)]

def get_deposits(states, start, limit):
    max_turn = min(start+limit, len(states)-1)
    return [0] + [states[i+1].deposit - states[i].deposit for i in range(start, max_turn)]

def rewards_for_ship(states, ship_id, turn_number, discount, limit, deposit_multiplier):
    deltas = get_delta_for_ship(states, ship_id, turn_number, limit)
    deposits = get_deposits(states, turn_number, limit)
    rewards = []
    for delta, deposit in zip(deltas, deposits):
        turn_reward = 0
        if delta is None:
            rewards.append(-1000)
            break
        elif -1*delta == deposit:
            #rewards.append(deposit*deposit_multiplier)
            break
        else:
            rewards.append(delta)
    reward_total = 0
    for r in reversed(rewards):
        reward_total = r + (reward_total*discount)
    return float(reward_total)

def get_samples(move_list, num_players):
    sample_size = int(MAX_SAMPLES/num_players)
    early_moves = move_list[:sample_size]
    late_moves = np.random.permutation(move_list[sample_size:])[:sample_size]
    return early_moves + list(late_moves)

def get_inputs(replay, num_players):
    feature_list = []
    moves = []
    rewards = []

    bad = 0

    for player_id in [0]:
        states = parse_compressed_replay_file(replay, player_id)
        move_list = parse_moves("arena/moves_{}".format(player_id))

        for turn, ship_id, move_label in get_samples(move_list, 1):
            turn_idx = int(turn)
            ship_id = int(ship_id)
            state = states[turn_idx]
            if state.moves.get(ship_id, 'o') != move_label:
                bad += 1
            reward = rewards_for_ship(states, ship_id, turn_idx, DISCOUNT, LOOKAHEAD, DEPOSIT_MULTIPLIER)

            feature_list.append(state.feature_shift(ship_id))
            moves.append(MOVE_TO_OUTPUT[move_label])
            rewards.append(reward)

    feature_arr = np.stack(feature_list, axis=0)
    move_arr = np.array(moves)
    rewards_arr = np.array(rewards)
    print(bad)
    return feature_arr, move_arr, rewards_arr

def play_game(map_size, num_players):
    args = ['./halite','--replay-directory','arena/','-vvv','--width',str(map_size),'--height',str(map_size)]
    args += ['python3 MyBot.py --learning']
    args += ['python3 ../old_bot/MyBot.py']*(num_players - 1)
    subprocess.call(args)

def episode(model):
    map_size = random.choice([32, 40, 56, 64])
    num_players = random.choice([2,4])

    if os.path.exists('arena'):
        shutil.rmtree('arena')
    os.mkdir('arena')

    with Timer("playing game", True):
        play_game(map_size, num_players)
    replay = glob.glob('arena/*.hlt')[0]
    with Timer("generating features", True):
        f, m, r = get_inputs(replay, num_players)
    with Timer("policy update", True):
        model.policy_update(f,m,normalize_rewards(r))
    model.save_model('models/policy_model.ckpt')


if __name__ == '__main__':
    model = MovementModel(
                cached_model='models/chosen3_rmrzx_82810.ckpt',
                params_file='params/rmrzx')
    model.save_model('models/policy_model.ckpt')
    i = 0
    while True:
        with Timer('episode {}'.format(i), True):
            episode(model)
        i += 1
