#!/usr/bin/env python3

from feelback.utils.io import get_command_line_args
from feelback import Feelback

if __name__ == '__main__':
    args = get_command_line_args()
    feelback = Feelback(args.input_video, args.fps, args.verbose)
    feelback.run()
