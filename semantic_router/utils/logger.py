import logging
import os

import colorlog


class CustomFormatter(colorlog.ColoredFormatter):
    """Custom formatter for the logger."""

    def __init__(self):
        super().__init__(
            "%(log_color)s%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
            reset=True,
            style="%",
        )


def add_coloured_handler(logger):
    """Add a coloured handler to the logger."""
    formatter = CustomFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def setup_custom_logger(name):
    """Setup a custom logger."""
    logger = logging.getLogger(name)

    # get log level from environment vars
    # first check for semantic_router_log_level, then log_level, then default to INFO
    log_level = (
        os.getenv("SEMANTIC_ROUTER_LOG_LEVEL") or os.getenv("LOG_LEVEL") or "INFO"
    )
    log_level = log_level.upper()

    add_coloured_handler(logger)
    logger.setLevel(log_level)
    logger.propagate = False

    return logger


logger: logging.Logger = setup_custom_logger("semantic_router")
