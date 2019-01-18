import os
import time
import logging

VERBOSE_LOGGING = os.environ.get('VERBOSE_LOGGING')

def log_message(message, always=False):
    if VERBOSE_LOGGING or always:
        logging.warning(message)

class Timer:

    def __init__(self, timer_string, always=False):
        self.timer_string = timer_string
        self.always = always

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        log_message("{} took {}".format(self.timer_string, end_time - self.start_time), self.always)
