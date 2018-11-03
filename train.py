#!/usr/bin/env python3

from bot.model import HaliteModel

m = HaliteModel()
m.train_on_files('training_data', './model.ckpt')
#m.save(file_name='aggressive.svc')

#m = model.HaliteModel()
#m.train_on_files('training', 'passive')
#m.save(file_name='passive.svc')
