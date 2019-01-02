#!/usr/bin/env python3
import argparse
from player.model import MovementModel, SpawnModel

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=['movement', 'spawn'])
    args = parser.parse_args()

    if args.job == 'movement':
        m = MovementModel(train_folder='../train', test_folder='../test')
        m.train_on_files('models/model_{}.ckpt')
    elif args.job == 'spawn':
        m = SpawnModel(train_folder='../spawn_train', test_folder='../spawn_test')
        m.train_on_files('models/spawn_model_{}.ckpt')
