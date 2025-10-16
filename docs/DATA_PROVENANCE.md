# Data provenance

The repository includes lexicons and sanctions snapshots for development. The project
owner identifies the current snapshots as exports from public official Ukrainian
lists. The repository does not yet contain the exact source URLs, retrieval timestamp,
or terms reference needed to reproduce that assertion independently. Treat the files
as stale until those fields are completed and revalidated.

## Bundled snapshots

Counts and checksums below describe the files currently committed to the repository.
The INN cache is derived from the person and organization exports, not an independent
source.

| File | Records | SHA-256 | Source status |
| --- | ---: | --- | --- |
| `sanctioned_persons.json` | 13,922 | `c2d82ed64920e29134431486f276e28de983c624f265e57a98f7681591d9929c` | Owner-confirmed official public source; exact URL and retrieval time pending |
| `sanctioned_companies.json` | 8,225 | `1d45a8ea474a4fa4cb5782de9fa361311217ee6cc769c03dc9b3935f4b1701e0` | Owner-confirmed official public source; exact URL and retrieval time pending |
| `terrorism_black_list.json` | 5,258 | `64851b74382d5a9467e8922f409e20a75bd648aac95a755f697c9eb7455376e3` | Owner-confirmed official public source; exact URL and retrieval time pending |
| `sanctioned_inns_cache.json` | 19,789 | `a20593a51dd67dc7d56e6aedbce2ab3347db8d103c631658d021c24c6e5e123b` | Derived cache |

These checksums establish integrity of this repository snapshot; they do not establish
authenticity, freshness, or permission to redistribute it.

Files live in [`src/ai_service/data`](../src/ai_service/data). The separate
[demo fixture](../examples/demo/README.md) contains three fictional entities and
is covered by the code's MIT license. Bundled third-party snapshots and models
retain their own terms.

## Updating data

Before adding or refreshing a dataset, record:

| Field | Required value |
| --- | --- |
| Publisher | Legal entity responsible for the source |
| Source URL | Direct, stable download or API URL |
| Retrieved at | UTC timestamp |
| Coverage | Jurisdictions and entity types |
| License or terms | Link and redistribution assessment |
| Integrity | SHA-256 checksum of the raw artifact |
| Transform | Reproducible command or script and version |

Do not commit proprietary watchlists, personal customer data, credentials, or files whose redistribution terms are unclear. Prefer download scripts and small synthetic fixtures over large derived snapshots.
