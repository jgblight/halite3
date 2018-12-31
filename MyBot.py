import logging
from player.bot import Bot
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    bot = Bot('bot', 'models/model_82000.ckpt')
    bot.run()
