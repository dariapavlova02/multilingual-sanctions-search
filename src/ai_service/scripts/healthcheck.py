"""Container readiness probe using the Python standard library."""

import sys
import urllib.error
import urllib.request


def main():
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:8000/health/ready", timeout=4
        ) as response:
            if response.status != 200:
                raise RuntimeError("Readiness failed")
    except (OSError, urllib.error.URLError, RuntimeError):
        sys.exit(1)


if __name__ == "__main__":
    main()
