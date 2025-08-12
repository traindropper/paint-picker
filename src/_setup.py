"""Setup functions."""

import logging
import logging.config
from typing import Any


def setup_logging() -> None:
    """Setup the logger."""
    default_config: dict[str, Any] = {
        "version": 1,
        "formatters": {
            "simple": {
                "format": "[%(asctime)s] {%(levelname)s} %(name)s: line #%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["stdout"]
        }
    }

    logging.config.dictConfig(default_config)