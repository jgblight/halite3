import os
from player.utils import Timer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--learning", action="store_true", default=False)
args = parser.parse_args()

with Timer("Import", True):
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from player.bot import Bot
with Timer("Initializes", True):
    bot = Bot('bot', 'models/policy_model.ckpt', 'params/rmrzx', args.learning)
bot.run()
