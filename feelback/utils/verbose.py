from builtins import print as _print
import warnings
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

    __verbosity_level = Level.OFF
    warnings.simplefilter('ignore')

    @property
    def verbosity_level(self):
        return self.__verbosity_level

    @verbosity_level.setter
    def verbosity_level(self, level: Level):
        if level <= self.Level.INFO:
            warnings.simplefilter('ignore')
        elif level == self.Level.DEBUG:
            warnings.simplefilter('default')
        elif level >= self.Level.TRACE:
            warnings.simplefilter('always')

        self.__verbosity_level = level

    # Singleton Pattern
    def __new__(Class):
        if not hasattr(Class, 'instance'):
            Class.instance = super(Verbose, Class).__new__(Class)
            Class.instance.verbosity_level = Class.Level.INFO

        return getattr(Class, 'instance')

    def imshow(self, image: np.ndarray, title=None, level=Level.VISUAL, delay=0):
        if level <= self.verbosity_level:
            if title is not None:
                cv2.imshow(f"DEBUG: {title}", image)
            else:
                cv2.imshow("DEBUG", image)
            cv2.waitKey(delay)

    def print(self, *args, level=Level.INFO, sep=' ', end='\n', **kwargs):
        if level <= self.verbosity_level:
            _print(*args, sep=sep, end=end, flush=True, **kwargs)

    def info(self, *args, sep=' ', end='\n', **kwargs):
        self.print("[INFO]", *args, level=self.Level.INFO, sep=sep, end=end, **kwargs)

    def debug(self, *args, sep=' ', end='\n', **kwargs):
        self.print("[DEBUG]", *args, level=self.Level.DEBUG, sep=sep, end=end, **kwargs)

    def trace(self, *args, sep=' ', end='\n', **kwargs):
        self.print("[TRACE]", *args, level=self.Level.TRACE, sep=sep, end=end, **kwargs)

    def warning(self, *args, sep=' ', end='\n', **kwargs):
        self.print("[WARNING]", *args, level=self.Level.TRACE, sep=sep, end=end, **kwargs)

    def error(self, *args, sep=' ', end='\n', **kwargs):
        self.print("[ERROR]", *args, level=self.Level.TRACE, sep=sep, end=end, **kwargs)

    def print_exception_stack_trace(self, level=Level.TRACE):
        if level <= self.verbosity_level:
            import traceback
            traceback.print_exc()

    def set_verbose_level(self, level: Level):
        self.verbosity_level = level

    def is_verbose(self):
        return self.verbosity_level != self.Level.OFF


verbose = Verbose()
