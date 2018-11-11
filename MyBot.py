import logging
from player.bot import Bot

if __name__ == '__main__':
    logging.warning('initializing bot')
    bot = Bot('bot', './model.ckpt')
    logging.warning('initialized')
    bot.run()
