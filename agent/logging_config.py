import logging
import json
import sys
from typing import Any, Dict, Optional
from datetime import datetime


class StructuredLogger(logging.Logger):
    def _log_with_extra(self, level: int, msg: str, *args, **kwargs) -> None:
        extra = kwargs.pop('extra', {})
        for key, value in kwargs.items():
            extra[key] = value
        super()._log(level, msg, args, extra=extra if extra else None)

    def debug(self, msg, *args, **kwargs):
        self._log_with_extra(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log_with_extra(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log_with_extra(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log_with_extra(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log_with_extra(logging.CRITICAL, msg, *args, **kwargs)


logging.setLoggerClass(StructuredLogger)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "module") and record.module:
            log_data["module"] = record.module

        if hasattr(record, "funcName") and record.funcName:
            log_data["function"] = record.funcName

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in [
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info", "taskName"
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)


def get_logger(
    name: str,
    level: Optional[str] = None,
    use_json: bool = True
) -> StructuredLogger:
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    if level is None:
        level = "INFO"

    logger.setLevel(getattr(logging, level.upper()))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.propagate = False

    return logger
