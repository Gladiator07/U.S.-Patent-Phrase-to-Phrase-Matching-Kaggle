import logging
from logging import FileHandler, Formatter, Logger, StreamHandler


def setup_logger(file_path: str) -> Logger:
    """
    Creates a custom logger which writes to stdout as well as to a log file at specified `file_path`
    Args:
        file_path (str): file path for saving log file
    Returns:
        Logger: logger
    """
    try:
        logger = logging.getLogger()
        # remove loggers if exists
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
    except:
        pass

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = StreamHandler()
    file_handler = FileHandler(filename=file_path, mode="w")

    log_format = "%(message)s"
    console_handler.setFormatter(Formatter(log_format))
    file_handler.setFormatter(Formatter(log_format))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
