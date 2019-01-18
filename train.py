#!/usr/bin/env python3
import argparse
from player.model import MovementModel, SpawnModel

chosen_model = 'rmrzx'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=['movement', 'paramsearch', 'spawn'])
    args = parser.parse_args()

    if args.job == 'movement':
        m = MovementModel(
            cached_model='models/chosen_rmrzx_82810.ckpt',
            params_file='params/rmrzx',
            train_folder='../train',
            test_folder='../test')
        m.train_on_files('models/chosen2_{}_{}.ckpt', 10)
    elif args.job == 'paramsearch':
        m = MovementModel(params_file='params/orig', train_folder='../train', test_folder='../test')
        m.train_on_files('models/model_{}_{}.ckpt', 2)
        for i in range(20):
            m = MovementModel(train_folder='../train', test_folder='../test')
            m.train_on_files('models/model_{}_{}.ckpt', 2)
    elif args.job == 'spawn':
        m = SpawnModel(train_folder='../spawn_train', test_folder='../spawn_test')
        m.train_on_files('models/spawn_model_{}.ckpt', 10)
