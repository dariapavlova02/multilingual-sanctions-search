"""Migrate legacy local snapshots from JSON source rows without loading Python objects."""

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys

from ..layers.embeddings.indexing.enhanced_vector_index_service import (
    EnhancedVectorIndexConfig,
)
from ..layers.embeddings.indexing.local_index_snapshot import read_json
from ..layers.embeddings.indexing.watchlist_index_service import WatchlistIndexService


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit JSON EnhancedVectorIndexConfig matching the legacy retrieval policy",
    )
    args = parser.parse_args(argv)
    try:
        values, _ = read_json(args.config, 64 * 1024)
        if type(values) is not dict:
            raise ValueError("Expected index configuration object")
        config = EnhancedVectorIndexConfig(**values)
        with closing(WatchlistIndexService(config)) as service:
            result = service.migrate_legacy_snapshot(args.source, args.destination)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        print(
            "Watchlist snapshot migration failed; verify the JSON source, selected configuration and destination",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
