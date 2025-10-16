"""Locations of immutable resources shipped in the installed package.

Runtime state and externally refreshed sanctions datasets must live outside
these resources. Wheels are installed as directories by supported installers.
"""

from pathlib import Path

PACKAGE_DATA_DIR = Path(__file__).resolve().parent
LEXICONS_DIR = PACKAGE_DATA_DIR / "lexicons"
CONFIG_DIR = PACKAGE_DATA_DIR / "config"
