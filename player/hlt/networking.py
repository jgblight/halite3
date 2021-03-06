import sys


def send_commands(commands):
    """
    Sends a list of commands to the engine.
    :param commands: The list of commands to send.
    :return: nothing.
    """
    print(" ".join(commands))
    sys.stdout.flush()

def send_command(command):
    sys.stdout.write(command + ' ')
    sys.stdout.flush()
