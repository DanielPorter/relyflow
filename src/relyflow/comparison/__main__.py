"""Command line: ``python -m relyflow.comparison train --out DIR``."""

import sys

from relyflow.comparison import classifier

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "train":
        sys.exit("usage: python -m relyflow.comparison train --out DIR [options]")
    classifier.main(sys.argv[2:])
