import os
import pickle
import numpy as np
import random
import shutil
from multiprocessing import Pool

from player.parse import parse_replay_file
from player.model import train_test_split
from player.constants import MOVE_TO_OUTPUT, OUTPUT_TO_MOVE

data_folder = '../training_data'
train_folder = '../train'
test_folder = '../test'
player = 'TheDuck314'
data_size = 500
split = 0.2

def process_file(i, filename, output_folder):
    print("parsing {}: {}".format(i, filename))
    states = parse_replay_file(filename, player)
    late_states = states[30:]
    random.shuffle(late_states)
    selected_states = states[:30] + late_states[:30]
    for j, state in enumerate(selected_states):
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

def write_states(files, output_folder):
    process_count = 6
    p = Pool(process_count)
    task_id = 0
    tasks = len(files)
    results = []
    while task_id < tasks:
        results.append(p.apply_async(process_file, (task_id, files[task_id], output_folder)))
        task_id += 1
    for r in results:
        r.wait()

if __name__ == '__main__':
    if os.path.exists(train_folder):
        shutil.rmtree(train_folder)
    os.mkdir(train_folder)

    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.mkdir(test_folder)

    train, test = train_test_split(data_folder, data_size, split)
    write_states(train, train_folder)
    write_states(test, test_folder)
