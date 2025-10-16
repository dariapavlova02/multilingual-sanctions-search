"""Validate every Compose configuration without reading or replacing local secrets."""

import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="sanctions-compose-") as directory:
        environment = Path(directory) / "compose.env"
        subprocess.run(
            [
                sys.executable,
                str(root / "scripts/create_local_env.py"),
                "--output",
                str(environment),
            ],
            check=True,
            cwd=root,
        )
        for files in [
            ["docker-compose.yml"],
            ["docker-compose.prod.yml"],
            ["docker-compose.yml", "docker-compose.dev.yml"],
        ]:
            command = ["docker", "compose", "--env-file", str(environment)]
            for file in files:
                command.extend(["-f", file])
            subprocess.run([*command, "config", "--quiet"], check=True, cwd=root)


if __name__ == "__main__":
    main()
