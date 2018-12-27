import os
import pickle
import numpy as np
import shutil

from player.parse import parse_replay_file
from player.model import train_test_split

data_folder = '../training_data'
train_folder = '../train'
test_folder = '../test'
player = 'TheDuck314'
data_size = 100
split = 0.2

def write_states(files, output_folder):
    for i, filename in enumerate(files):
        print("parsing " + filename)
        states = parse_replay_file(filename, player)
        selected_states = np.random.permutation(states)[:50]
        for j, state in enumerate(selected_states):
            moves = state.get_ship_moves()

            map_size = state.map_size
            for move in moves:
                ship_id, move_label = move
                features = state.input_for_ship(ship_id)
                output_file = '{}_{}_{}.pkl'.format(i, j, ship_id)
                payload = (map_size, move_label, features)
                with open(os.path.join(output_folder, output_file),'wb') as f:
                    pickle.dump(payload, f)


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
