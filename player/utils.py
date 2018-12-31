import time
import logging

class Timer:

    def __init__(self, timer_string):
        self.timer_string = timer_string

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        end_time = time.time()
        #logging.warning("{} took {}".format(self.timer_string, end_time - self.start_time))
