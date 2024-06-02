import logging

import colorlog


class CustomFormatter(colorlog.ColoredFormatter):
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
    formatter = CustomFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def setup_custom_logger(name):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        add_coloured_handler(logger)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


logger: logging.Logger = setup_custom_logger(__name__)
