#!/usr/bin/env python3
"""Extract ResNet-34 patch features and aligned level-0 coordinates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hypermil.preprocessing.features import cli

if __name__ == '__main__':
    cli()
