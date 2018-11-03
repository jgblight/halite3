from player.bot import Bot

if __name__ == '__main__':
    bot = Bot('bot', './model.ckpt')
    bot.run()
