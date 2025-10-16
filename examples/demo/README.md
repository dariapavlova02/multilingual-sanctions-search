# Local demo data

These three fictional records exercise exact names, aliases, multilingual names
and organizations. They do not describe sanctioned people or companies. The source
is reported as `custom`, with `list_name: custom_portfolio_demo` and
`synthetic: true` metadata. The fixture is covered by the repository's MIT license.

Use the [root quick start](../../README.md#quick-start). The existing
bootstrap imports only this directory when given `--data-dir /demo`; it does not
mix these examples with bundled historical snapshots.

Expected examples:

| Query | Mode | Expected source entity |
| --- | --- | --- |
| `John Smith` | `ac` | `demo-person-1` |
| `John A. Smith` | `ac` | `demo-person-1` |
| `Ivan Petrenko` | `ac` | `demo-person-2` |
| `Example Trading LLC` | `ac` | `demo-company-1` |

Vector and hybrid modes also run with the pinned embedding model. Their similarity
scores are not probabilities of identity. A result demonstrates retrieval from
this fixture only.
