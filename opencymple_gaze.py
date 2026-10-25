#!/usr/bin/env python3
"""Compatibility entry point: the gaze math now lives in opencymple/gaze.py.

Usage:  python3 opencymple_gaze.py --selftest [path/to/iniFile.ini]
"""
import sys

from opencymple.calibration import find_ini
from opencymple.gaze import *  # noqa: F401,F403
from opencymple.gaze import selftest

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(0 if selftest(sys.argv[2] if len(sys.argv) > 2 else find_ini()) else 1)
    print(__doc__)
