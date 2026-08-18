from __future__ import annotations

import logging
import re

from pythonjsonlogger.json import JsonFormatter

SENSITIVE_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|secret|token|password)(\s*[:=]\s*)([^\s,;]+)"
)


def redact(value: str) -> str:
    return SENSITIVE_PATTERN.sub(r"\1\2[REDACTED]", value)


class RedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(arg) if isinstance(arg, str) else arg for arg in record.args)
        elif isinstance(record.args, dict):
            record.args = {
                key: redact(value) if isinstance(value, str) else value
                for key, value in record.args.items()
            }
        for key, value in list(record.__dict__.items()):
            if isinstance(value, str) and key.lower() in {
                "authorization",
                "api_key",
                "password",
                "prompt",
                "messages",
            }:
                setattr(record, key, "[REDACTED]")
        return True


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level", "name": "logger"},
        )
    )
    handler.addFilter(RedactionFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
