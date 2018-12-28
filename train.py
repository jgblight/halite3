#!/usr/bin/env python3

from player.model import HaliteModel

m = HaliteModel()
m.train_on_files('../train', 'models/model_{}.ckpt')
#m.save(file_name='aggressive.svc')

#m = model.HaliteModel()
#m.train_on_files('training', 'passive')
#m.save(file_name='passive.svc')
