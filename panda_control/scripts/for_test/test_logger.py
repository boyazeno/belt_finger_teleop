import logging

def get_logger():
    logger = logging.getLogger('env_logger')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("test_logger.txt")
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = get_logger()
for _ in range(10):
    logger.info(f"test test")