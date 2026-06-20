import logging
import os


def setup_logging(logging_path):

    os.makedirs(
        os.path.dirname(logging_path),
        exist_ok=True
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        logging_path,
        mode="w"
    )

    file_handler.setFormatter(
        formatter
    )

    logger = logging.getLogger()

    logger.setLevel(
        logging.INFO
    )

    logger.handlers.clear()

    logger.addHandler(
        file_handler
    )

    return logger