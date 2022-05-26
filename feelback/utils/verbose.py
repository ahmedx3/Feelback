from builtins import print as _print
import cv2
import numpy as np
import enum


class Verbose:

    class Level(int, enum.Enum):
        OFF = 0
        INFO = 1
        DEBUG = 2
        TRACE = 3
        VISUAL = 4

    verbosity_level = Level.OFF

    # Singleton Pattern
    def __new__(Class):
        if not hasattr(Class, 'instance'):
            Class.instance = super(Verbose, Class).__new__(Class)
        return getattr(Class, 'instance')

    def imshow(self, image: np.ndarray, title=None, level=Level.VISUAL, delay=0):
        if level <= self.verbosity_level:
            if title is not None:
                cv2.imshow(f"DEBUG: {title}", image)
            else:
                cv2.imshow("DEBUG", image)
            cv2.waitKey(delay)

    def print(self, *args, level=Level.INFO, sep=' ', end='\n'):
        if level <= self.verbosity_level:
            _print(*args, sep=sep, end=end, flush=True)

    def print_exception_stack_trace(self, level=Level.TRACE):
        if level <= self.verbosity_level:
            import traceback
            traceback.print_exc()

    def set_verbose_level(self, level: Level):
        self.verbosity_level = level

    def is_verbose(self):
        return self.verbosity_level != self.Level.OFF


verbose = Verbose()
