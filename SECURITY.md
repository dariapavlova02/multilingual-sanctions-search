# Security

This pre-1.0 portfolio project supports local experimentation. Fixes target the
latest code on `main`; older releases have no separate support policy.

## Report a vulnerability

Use this repository's GitHub private vulnerability reporting when available.
Do not put credentials or sensitive records in a public issue. Include the affected
component, reproduction steps and impact. This personal project has no response SLA.

## Local access and credentials

Compose binds the API to loopback and keeps authenticated Elasticsearch private.
A separate restricted account serves the API; the Elasticsearch administrator
credential is used only for provisioning. Public processing routes do not implement
application authentication. Administrative routes require a bearer token of at
least 32 characters.

Generate private credentials with `python3 scripts/create_local_env.py`; do not
commit `.env` or reuse example credentials. Rotate an exposed credential wherever
it was used. Removing a file or rewriting Git history does not revoke the credential.
Current environment examples contain no key. Previously exposed credentials must
be revoked even when they are absent from the published history.

Changing `ELASTIC_PASSWORD` in `.env` after first boot does not rotate an existing
Elasticsearch administrator. Use the Elasticsearch password-management API. For
service-account changes, update the private environment, rerun provisioning and
recreate the API.

## Data and logs

Configured logging redacts dynamic values and exception payloads; HTTP errors omit
internal exception details. This does not cover arbitrary custom logs, old exports
or output from third-party tools. Keep input records, ingestion state and generated
logs private. The in-memory rate limiter and bounded queues are local safeguards.

## Historical image scans

The following recorded Trivy 0.74.0 results belong to the listed image IDs and
do not certify a later build. Counts are package/advisory
records, not independently exploitable application paths.

| Image ID prefix | OS findings | Language-package findings |
| --- | --- | --- |
| API `c521e84c6c7e` | 172 Debian records: 3 critical, 51 high, 54 medium, 57 low, 7 unknown | 0 Python |
| Elasticsearch `39a1ca013b6f` | 7 Ubuntu records: 2 medium, 5 low | 48 Java records: 18 high, 30 medium |

These findings are not claimed resolved. Scan the actual images after rebuilding;
a successful build or passing test suite does not establish advisory clearance.
Raw scan reports are not distributed with the repository. Production hosting and disaster recovery are outside the verified scope.
