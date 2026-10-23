"""Command line: ``python -m relyflow.inference {train,package} ...``.

- ``train``: train a network (``relyflow.inference.training.main``).
- ``package``: zip training outputs for a GitHub release and print their registry
  entries (``relyflow.inference.pretrained.main``).
"""

import sys

from relyflow.inference import pretrained, training

COMMANDS = {"train": training.main, "package": pretrained.main}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python -m relyflow.inference {{{','.join(COMMANDS)}}} ...")
    COMMANDS[sys.argv[1]](sys.argv[2:])
