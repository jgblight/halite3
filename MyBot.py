import logging
from player.bot import Bot

if __name__ == '__main__':
    logging.warning('initializing bot')
    bot = Bot('bot', 'models/model_61000.ckpt')
    logging.warning('initialized')
    bot.run()
