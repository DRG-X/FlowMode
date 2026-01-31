import logging

def setup_logger(filename: str, log_level=logging.INFO):
    logger = logging.getLogger(filename)
    logger.setLevel(log_level)

    if not logger.handlers:
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(log_level)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger
