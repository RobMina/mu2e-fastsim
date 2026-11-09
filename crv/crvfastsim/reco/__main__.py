"""Alias: python3 -m crvfastsim.reco <cmd> ... == python3 -m crvfastsim reco-<cmd> ..."""
import sys

from .cli import main

sys.exit(main())
