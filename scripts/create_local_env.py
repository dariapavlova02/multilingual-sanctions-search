"""Create a private Compose environment file without overwriting existing secrets."""

import argparse
import os
import secrets
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".env"))
    args = parser.parse_args()
    values = {
        name: secrets.token_urlsafe(40)
        for name in (
            "ELASTIC_PASSWORD",
            "ES_SERVICE_PASSWORD",
            "ADMIN_API_KEY",
        )
    }
    values.update(
        API_PORT="8001",
        ES_INDEX_PREFIX="sanctions",
        SANCTIONS_IMAGE="hybrid-sanctions:local",
        CORS_ORIGINS="[]",
    )
    try:
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        parser.error(
            f"{args.output} already exists; existing credentials were preserved"
        )
    with os.fdopen(descriptor, "w") as file:
        file.write("".join(f"{key}={value}\n" for key, value in values.items()))
    print(f"Created private Compose configuration: {args.output}")


if __name__ == "__main__":
    main()
