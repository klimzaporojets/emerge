import logging
import os

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
)
