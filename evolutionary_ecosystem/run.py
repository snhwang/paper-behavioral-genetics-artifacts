#!/usr/bin/env python3
"""Entry point for Evolutionary Ecosystem.

Run from the evolutionary_ecosystem directory OR from anywhere:
    python examples/evolutionary_ecosystem/run.py
    python examples/evolutionary_ecosystem/run.py --creatures 6 --port 8003
"""
import sys
from pathlib import Path

# Ensure evolutionary_ecosystem/ is on sys.path so 'server' package is importable
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from server.app import main

if __name__ == "__main__":
    main()
