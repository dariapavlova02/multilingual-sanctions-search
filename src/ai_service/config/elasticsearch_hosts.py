"""Validate Elasticsearch addresses consistently before constructing clients."""

from urllib.parse import urlsplit


def validate_elasticsearch_hosts(hosts):
    if not hosts:
        raise ValueError("At least one Elasticsearch host must be specified")
    validated = []
    for value in hosts:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Invalid Elasticsearch host")
        host = value.strip()
        has_scheme = "://" in host
        parsed = urlsplit(host if has_scheme else "//" + host)
        if has_scheme and parsed.scheme not in {"http", "https"}:
            raise ValueError("Elasticsearch scheme must be http or https")
        if not parsed.hostname or any(char.isspace() for char in parsed.hostname):
            raise ValueError("Invalid Elasticsearch hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Configure Elasticsearch credentials separately from hosts")
        if parsed.query or parsed.fragment:
            raise ValueError("Elasticsearch hosts cannot contain a query or fragment")
        try:
            port = parsed.port
        except ValueError as error:
            if "out of range" in str(error):
                raise ValueError("Port must be between 1 and 65535") from error
            raise ValueError("Invalid port number") from error
        if port is None and not has_scheme:
            raise ValueError("Host must include port or scheme")
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        validated.append(host)
    return validated
