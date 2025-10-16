"""Keep request values and exception payloads out of application log sinks.

Existing call sites include eagerly formatted strings. Redacting names with regex
cannot safely recover their boundaries. Retain the static source template and
code location instead; dynamic messages without a known template fail closed.
"""

import ast
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_LEVEL_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "fatal", "log"}


def _literal_template(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            value.value if isinstance(value, ast.Constant) and isinstance(value.value, str)
            else "[redacted]" for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_template(node.left) + _literal_template(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        return _literal_template(node.left)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
        return _literal_template(node.func.value)
    return "[redacted]"


@lru_cache(maxsize=128)
def _source_templates(pathname):
    """Read only installed application source; never inspect arbitrary log paths."""
    try:
        path = Path(pathname).resolve()
        if not path.is_relative_to(_PACKAGE_ROOT) or path.suffix != ".py":
            return {}
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, SyntaxError, UnicodeError):
        return {}
    templates = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _LEVEL_METHODS:
            continue
        position = 1 if node.func.attr == "log" else 0
        if len(node.args) <= position:
            continue
        template = _literal_template(node.args[position])
        for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            templates[line] = template
    return templates


class LogPrivacyFilter(logging.Filter):
    """Sanitize the shared LogRecord before any built-in handler formats it."""

    def filter(self, record):
        template = _source_templates(record.pathname).get(record.lineno, "Log event")
        exception = record.exc_info or sys.exc_info()
        exception_type = exception[0] if exception and exception[0] else None
        if exception_type is None:
            exception_type = getattr(record, "_safe_exception_class", None)
        if isinstance(exception_type, type) and issubclass(exception_type, BaseException):
            record._safe_exception_class = exception_type
            template += f" [error_type={exception_type.__name__}]"
        record.msg = template
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        # Some formatters cache these before the record reaches another handler.
        record.__dict__.pop("message", None)
        return True


class JsonFormatter(logging.Formatter):
    """Serialize protected records as valid JSON, including quoted templates."""

    def format(self, record):
        LogPrivacyFilter().filter(record)
        return json.dumps({
            "timestamp": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }, ensure_ascii=False)


def protect_logger(logger):
    if not any(isinstance(value, LogPrivacyFilter) for value in logger.filters):
        logger.addFilter(LogPrivacyFilter())
    return logger


def protect_configured_handlers():
    """Include third-party output routed through the configured application sinks."""
    loggers = [logging.getLogger()] + [
        value for value in logging.Logger.manager.loggerDict.values()
        if isinstance(value, logging.Logger)
    ]
    handlers = set()
    for logger in loggers:
        protect_logger(logger)
        handlers.update(logger.handlers)
    for handler in handlers:
        if not any(isinstance(value, LogPrivacyFilter) for value in handler.filters):
            handler.addFilter(LogPrivacyFilter())
