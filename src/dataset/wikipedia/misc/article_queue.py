import logging
import multiprocessing

logger = logging.getLogger(__name__)


class ArticleReadingQueue():
    def __init__(self,
                 process_file_queue_size=20000):
        manager = multiprocessing.Manager()
        self.manager = manager
        # with file names to process
        # process_files_queue has to be big to fit all the files (see wikipedia_create_dataset.py).
        self.process_files_queue = manager.Queue(maxsize=process_file_queue_size)
