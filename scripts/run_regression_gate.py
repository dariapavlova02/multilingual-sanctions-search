#!/usr/bin/env python3
"""Run pytest against a disposable authenticated Elasticsearch container.

Uses the current Python environment. No existing cluster, index, container or
volume is reused. Docker must have capacity for a 768 MiB test server.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid


def docker(*arguments, **kwargs):
    return subprocess.run(["docker", *arguments], check=True, text=True,
                          capture_output=True, timeout=60, **kwargs).stdout.strip()


def verify_document_roundtrip(url, authorization, cluster_name):
    """An empty cluster can report healthy while disk protection prevents writes."""
    index = cluster_name + "-preflight"

    def request(path, method, body=None):
        req = urllib.request.Request(url + "/" + path, method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": authorization, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.load(response)

    created = request(index + "?wait_for_active_shards=1&timeout=5s&master_timeout=5s", "PUT",
                      {"settings": {"number_of_shards": 1, "number_of_replicas": 0}})
    if not created.get("acknowledged") or not created.get("shards_acknowledged"):
        raise RuntimeError("Disposable Elasticsearch cannot allocate a writable index")
    document = {"runner": cluster_name}
    request(index + "/_doc/runner?refresh=wait_for&timeout=5s", "PUT", document)
    stored = request(index + "/_doc/runner", "GET")
    if not stored.get("found") or stored.get("_source") != document:
        raise RuntimeError("Disposable Elasticsearch failed document verification")
    if not request(index + "?master_timeout=5s", "DELETE").get("acknowledged"):
        raise RuntimeError("Disposable Elasticsearch failed preflight cleanup")


def run(image, result_dir, pytest_args):
    result_dir = Path(result_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    name = "sanctions-regression-" + uuid.uuid4().hex
    password = secrets.token_urlsafe(36)
    summary = {"cluster": name, "image_id": docker("image", "inspect", image, "--format", "{{.Id}}"),
               "pytest_exit_code": None, "container_removed": False,
               "document_roundtrip_verified": False}
    created = False
    try:
        with tempfile.TemporaryDirectory(prefix="sanctions-regression-") as temporary:
            env_path = Path(temporary) / "elasticsearch.env"
            env_path.write_text("\n".join([
                "discovery.type=single-node", "cluster.name=" + name,
                "xpack.security.enabled=true", "xpack.security.http.ssl.enabled=false",
                "xpack.security.transport.ssl.enabled=false",
                "ES_JAVA_OPTS=-Xms256m -Xmx256m", "ELASTIC_PASSWORD=" + password,
            ]) + "\n")
            env_path.chmod(0o600)
            # Mark ownership before creation so a failed start is also cleaned up.
            docker("create", "--name", name, "--label", "sanctions.regression=" + name,
                   "--memory", "768m", "--publish", "127.0.0.1::9200",
                   "--env-file", str(env_path), image)
            created = True
            docker("start", name)
            ports = json.loads(docker("inspect", name, "--format", "{{json .NetworkSettings.Ports}}"))
            binding = ports["9200/tcp"][0]
            if binding["HostIp"] != "127.0.0.1":
                raise RuntimeError("Test cluster must bind only to loopback")
            url = "http://127.0.0.1:" + binding["HostPort"]
            authorization = "Basic " + base64.b64encode(("elastic:" + password).encode()).decode()
            deadline = time.monotonic() + 240
            print("Waiting for isolated Elasticsearch", flush=True)
            while True:
                try:
                    request = urllib.request.Request(url + "/_cluster/health?wait_for_status=yellow&timeout=2s",
                        headers={"Authorization": authorization})
                    with urllib.request.urlopen(request, timeout=5) as response:
                        health = json.load(response)
                    if health.get("cluster_name") != name:
                        raise RuntimeError("Unexpected test cluster identity")
                    if not health.get("timed_out") and health.get("status") in {"yellow", "green"}:
                        break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
                if time.monotonic() >= deadline:
                    raise RuntimeError("Disposable Elasticsearch did not become ready")
                time.sleep(2)
            verify_document_roundtrip(url, authorization, name)
            summary["document_roundtrip_verified"] = True
            env = os.environ.copy()
            env.update(SANCTIONS_TEST_ES_URL=url, SANCTIONS_TEST_ES_USERNAME="elastic",
                       SANCTIONS_TEST_ES_PASSWORD=password, SANCTIONS_TEST_CLUSTER_NAME=name)
            command = [sys.executable, "-m", "pytest", *pytest_args,
                       "--junitxml=" + str(result_dir / "tests.xml")]
            print("Running regression tests against isolated Elasticsearch", flush=True)
            with (result_dir / "pytest.log").open("w") as output:
                summary["pytest_exit_code"] = subprocess.run(command, env=env, stdout=output,
                    stderr=subprocess.STDOUT, cwd=Path(__file__).resolve().parents[1]).returncode
            # Do not publish accidental credentials from a failing test/traceback.
            for artifact in (result_dir / "tests.xml", result_dir / "pytest.log"):
                if artifact.exists():
                    raw = artifact.read_text()
                    artifact.write_text(raw.replace(password, "[REDACTED]"))
            return summary["pytest_exit_code"]
    except BaseException as exc:
        summary["failure_type"] = type(exc).__name__
        if created:
            logs = subprocess.run(["docker", "logs", "--tail", "100", name], text=True,
                                  capture_output=True, timeout=15)
            (result_dir / "backend.log").write_text((logs.stdout + logs.stderr).replace(password, "[REDACTED]"))
        raise
    finally:
        if created:
            docker("rm", "--force", "--volumes", name)
            summary["container_removed"] = True
        (result_dir / "runner.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("Regression result: " + str(summary["pytest_exit_code"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Locally built Dockerfile.elasticsearch image")
    parser.add_argument("--result-dir", default=".artifacts/regression")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.pytest_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    raise SystemExit(run(args.image, args.result_dir, arguments or ["-m", "not model"]))
