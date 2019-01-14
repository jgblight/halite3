import os
from player.utils import Timer

with Timer("Import", True):
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from player.bot import Bot
with Timer("Initializes", True):
    bot = Bot('bot', 'models/model_82000.ckpt', 'params/orig')
bot.run()
