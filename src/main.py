from _setup import setup_logging
setup_logging()

from paint_parser import parse_test_directory

if __name__ == "__main__":
    import logging
    logging.getLogger(__name__).info("Test log from main.py")
    parse_test_directory()