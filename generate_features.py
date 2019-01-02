import argparse
import os
import pickle
import numpy as np
import random
import shutil
from multiprocessing import Pool

from player.parse import parse_replay_file
from player.model import train_test_split
from player.constants import MOVE_TO_OUTPUT, OUTPUT_TO_MOVE

DATA_FOLDER = '../training_data'
player = 'TheDuck314'
DATA_SIZE = 500
SPLIT = 0.2

def get_states(i, filename, early, late):
    print("parsing {}: {}".format(i, filename))
    states = parse_replay_file(filename, player)
    late_states = states[early:]
    random.shuffle(late_states)
    selected_states = states[:early] + late_states[:late]
    return selected_states

def process_movement(i, filename, output_folder):
    for j, state in enumerate(get_states(i, filename, 30, 30)):
        moves = state.get_ship_moves()
        random.shuffle(moves)
        moves = moves[:10]

        map_size = state.map_size
        for ship_id, move_label in moves:
            move_id = MOVE_TO_OUTPUT[move_label]
            features = state.feature_shift(ship_id)
            output_file = '{}_{}_{}.pkl'.format(i, j, ship_id)
            payload = (map_size, OUTPUT_TO_MOVE[move_id], features)
            with open(os.path.join(output_folder, output_file),'wb') as f:
                pickle.dump(payload, f)

    print("done parsing " + filename)
    return filename

def process_spawn(i, filename, output_folder):
    for j, state in enumerate(get_states(i, filename, 100, 100)):
        features = state.center_shift()
        output_file = '{}_{}.pkl'.format(i, j)
        payload = (state.spawn, features)
        with open(os.path.join(output_folder, output_file),'wb') as f:
            pickle.dump(payload, f)

    print("done parsing " + filename)
    return filename

def write_states(process, files, output_folder):
    process_count = 6
    p = Pool(process_count)
    task_id = 0
    tasks = len(files)
    results = []
    while task_id < tasks:
        results.append(p.apply_async(process, (task_id, files[task_id], output_folder)))
        task_id += 1
    for r in results:
        r.wait()

def main(process, train_folder, test, data_folder, data_size, split):
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    os.mkdir(train_folder)

    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.mkdir(test_folder)

    train, test = train_test_split(data_folder, data_size, split)
    write_states(process, train, train_folder)
    write_states(process, test, test_folder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=['movement', 'spawn'])
    args = parser.parse_args()

    if args.job == 'movement':
        train_folder = '../train'
        test_folder = '../test'
        process = process_movement
    elif args.job == 'spawn':
        train_folder = '../spawn_train'
        test_folder = '../spawn_test'
        process = process_spawn
    main(process, train_folder, test_folder, DATA_FOLDER, DATA_SIZE, SPLIT)
