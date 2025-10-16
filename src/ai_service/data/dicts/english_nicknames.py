from ai_service.data.resources import PACKAGE_DATA_DIR
"""
English nicknames dictionary
Loads data from JSON file to maintain compatibility
"""

import json
from .english_names import ENGLISH_NAMES

# Load English nicknames from JSON file
_data_dir = PACKAGE_DATA_DIR
_nicknames_file = _data_dir / "lexicons" / "en_nicknames.json"

def load_english_nicknames(path=_nicknames_file):
    """Load one consistent map, preferring the curated names for clear aliases."""
    with open(path, "r", encoding="utf-8") as f:
        nicknames = json.load(f)
    if path != _nicknames_file:
        return nicknames

    candidates = {}
    canonical = {name.casefold() for name in ENGLISH_NAMES}
    for name, entry in ENGLISH_NAMES.items():
        for nickname in entry.get("diminutives", []):
            candidates.setdefault(nickname.casefold(), set()).add(name.casefold())
    for nickname, names in candidates.items():
        if len(names) == 1 and nickname not in canonical:
            nicknames[nickname] = next(iter(names))
    # A full name must not be expanded again by subsequent pipeline stages.
    return {nickname: name for nickname, name in nicknames.items() if nickname not in canonical}


ENGLISH_NICKNAMES = load_english_nicknames()

# Export for backward compatibility
__all__ = ["ENGLISH_NICKNAMES", "load_english_nicknames"]
