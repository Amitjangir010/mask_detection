import logging
import os

def setup_logger():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    # File Handler
    file_handler = logging.FileHandler('logs/logs.txt')
    file_handler.setLevel(logging.INFO)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add Handlers to Logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger setup complete and working")
    return logger
