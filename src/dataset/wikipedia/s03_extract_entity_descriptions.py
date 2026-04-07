# Extract Wikipedia entity descriptions (abstracts) at specific snapshot timestamps.
# Streams through the full Wikipedia meta-history dump (7z) and, for each page,
# finds the revision active at each configured snapshot date. Cleans the wikitext
# and extracts the first ~max_token_length tokens as the entity description.
#
# Requires s03_API_wiki_mapping.py to be running (FastAPI server that resolves
# page IDs to QIDs and provides creation timestamps).
#
# Based on s08_extract_relik_dictionary.py from wikidata-temp/wikipedia-temp.

import argparse
import json
import logging
import os
import time
import traceback
from multiprocessing import Value, Process, Lock
from threading import Thread

import git
import py7zr
import xml.sax

from .misc.article_queue import ArticleReadingQueue
from .misc.seven_zip_reader import SevenZipStreamDecompressor
from .misc.wikipedia_history_dictionary_extractor_v1 import WikipediaHistoryDictionaryExtractorV1

_LOG_LEVEL = logging._nameToLevel.get(
    os.environ.get('LOGGING_LEVEL', '').strip(), logging.INFO
)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=_LOG_LEVEL)
logger = logging.getLogger(__name__)

def display(v_nr_parsed_articles,
            initial_time,
            v_nr_hit_dictionary_convert,
            set_of_processing_files):
    while True:
        try:
            logger.info('\tDISPLAY Size of queue process_files_queue: %s' % arq.process_files_queue.qsize())
            logger.info('\tDISPLAY Size of v_nr_hit_dictionary_convert: %s' % v_nr_hit_dictionary_convert.value)
            curr_time = time.time()
            lst_running_files_with_tmp = list()
            for curr_file_name, curr_file_starttime in set_of_processing_files.items():
                lst_running_files_with_tmp.append((curr_file_name, curr_file_starttime))

            lst_running_files_with_tmp.sort(key=lambda x: x[1])
            for curr_file_name, curr_file_starttime in lst_running_files_with_tmp:
                if curr_file_name in set_of_processing_files:
                    time_run = (curr_time - set_of_processing_files[curr_file_name]) / 60 / 60
                    time_run_s = f'{time_run:.2f} hours'
                    logger.info(
                        f'\t\t{curr_file_name} running for: {time_run_s} ({curr_file_starttime}:{int(initial_time)})')

            logger.info('\tDISPLAY TOT processed entities: %s  Avg. articles per minute: %s' %
                        (v_nr_parsed_articles.value,
                         v_nr_parsed_articles.value / ((curr_time - initial_time) / 60)))

            time.sleep(60 * 1)
        except Exception as e:
            logger.error('type error display: %s' % str(e))
            logger.error(traceback.format_exc())
            continue


def process_xml_parser_parallel(shutdown_xml_parser, v_nr_parsed_articles,
                                v_nr_parsed_files,
                                config,
                                convert_to_text_dictionary,
                                set_of_processing_files,
                                v_lock, dry_run, start_time):
    """

    :return:

        wikipedia_page_id,
        wikipedia_title,
        wikidata_qid,
        wikipedia_creation_time,
        type (redirect, disambiguation, real page with content, any other?),
        content_length

    """

    filter_namespace = lambda ns: ns != '' and int(ns) == 0

    reader = WikipediaHistoryDictionaryExtractorV1(
        filter_namespace=filter_namespace,
        convert_to_text_dictionary=convert_to_text_dictionary,
        v_lock=v_lock,
        start_time=start_time,
        v_nr_parsed_articles=v_nr_parsed_articles,
        config=config,
        dry_run=dry_run
    )

    output_dir_dict_data = os.path.join(config['output_dir'], 'dictionary')
    os.makedirs(output_dir_dict_data, exist_ok=True)

    while not (shutdown_xml_parser.value == 1 and arq.process_files_queue.empty()):
        try:
            if config['input_format'] == '7zip':
                curr_file, curr_filepath = arq.process_files_queue.get(block=True,
                                                                       timeout=config['queues_timeout'])

                if not curr_file.endswith('.7z'):
                    logger.warning(f'IGNORING {curr_file} input file')
                    continue
                curr_file_start_time = time.time()
                set_of_processing_files[curr_file] = int(curr_file_start_time)
                path_dir = config['wiki_history_directory']

                path_working_file = os.path.join(path_dir, curr_file)
                archive = py7zr.SevenZipFile(path_working_file, mode='r')
                file_to_decompress = archive.files[0]

                decompressor = SevenZipStreamDecompressor(file_to_decompress.folder.coders,
                                                          file_to_decompress.compressed,
                                                          file_to_decompress.folder.unpacksizes,
                                                          archive.fp,
                                                          file_to_decompress.folder.crc,
                                                          file_to_decompress.folder.password,
                                                          )
                reader.processed_size = 0
                reader.processed_file = curr_file
                reader.tot_size = file_to_decompress.uncompressed

                output_files_dict_per_snapshot = dict()
                for curr_snapshot in config['snapshots']:
                    snapshot_dir = os.path.join(output_dir_dict_data,
                                                curr_snapshot)
                    os.makedirs(snapshot_dir, exist_ok=True)
                    output_file_dictionary_path = os.path.join(
                        snapshot_dir,
                        f'dictionary_{curr_file}.jsonl'
                    )
                    output_file_dictionary = open(output_file_dictionary_path, 'wt', encoding='utf-8')
                    output_files_dict_per_snapshot[curr_snapshot] = output_file_dictionary

                reader.output_files_dict_per_snapshot = output_files_dict_per_snapshot

                logger.info('parsing: %s' % curr_file)
                logger.info(f'files currently being parsed: {set_of_processing_files.keys()}')

                logger.info('compressed file size (GB): %s, %s' %
                            (curr_file, (os.path.getsize(curr_filepath) / 1024 / 1024 / 1024)))
                logger.info('uncompressed file size (GB): %s, %s ' %
                            (curr_file, (file_to_decompress.uncompressed / 1024 / 1024 / 1024)))
                try:
                    xml.sax.parse(decompressor, reader)
                finally:
                    archive.close()
                    for _, curr_dict_file in reader.output_files_dict_per_snapshot.items():
                        curr_dict_file.flush()
                        curr_dict_file.close()

                with v_nr_parsed_files.get_lock():
                    v_nr_parsed_files.value += 1

                lapse_from_program_start = (time.time() - start_time) / 60 / 60
                lapse_from_file_start = (time.time() - set_of_processing_files[curr_file]) / 60 / 60
                logger.info(f'FINISHED TO PARSING THE FILE NR {v_nr_parsed_files.value} ({curr_file}) '
                            f'out of {tot_files_parse} with time of '
                            f' after {lapse_from_file_start:.2f} hours of starting '
                            f' and after {lapse_from_program_start:.2f} hours of program start')
                logger.info(f'Total of files left in the queue: {arq.process_files_queue.qsize()} '
                            f'is the queue empty? {arq.process_files_queue.empty()}')
                logger.info(f'Files that are currently still being processed: '
                            f'{set_of_processing_files}')

                del set_of_processing_files[curr_file]

            elif config['input_format'] == 'text':
                curr_file, curr_filepath = arq.process_files_queue.get(block=True, timeout=config['queues_timeout'])
                reader.processed_size = 0
                reader.tot_size = 0
                logger.info('parsing: %s' % curr_file)
                xml.sax.parse(curr_filepath, reader)
                with v_nr_parsed_files.get_lock():
                    v_nr_parsed_files.value += 1

                logger.info('FINISHED TO PARSING THE FILE NR %s out of %s with time of %s hours' %
                            (v_nr_parsed_files.value, tot_files_parse, (time.time() - start_time) / 60 / 60))
            else:
                raise RuntimeError('Unknown format: ' + config['input_format'])

        except Exception as e:
            logger.error('type error process_xml_parser: %s' % str(e))
            logger.error(traceback.format_exc())
            traceback.print_exc()
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='config/dataset/wikipedia/s03_extract_entity_descriptions/'
                                '20251101_slurm_english/config.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--nr_threads_processor', required=False, type=int,
                        default=4,
                        help='Nr of threads that will process the data')

    parser.add_argument('--dry_run', help='Dry run, reading all the documents, but do not '
                                          'parse anything.', action='store_true')

    parser.add_argument('--api_ports_wiki_mapping', nargs='+',
                        type=int, required=True,
                        help='List of ports on which the s03_API_wiki_mapping '
                             'is deployed.')

    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))
    config['nr_threads_processor'] = args.nr_threads_processor
    config['dry_run'] = args.dry_run
    dry_run = config['dry_run']
    config['api_ports_wiki_mapping'] = args.api_ports_wiki_mapping

    output_dir_data = config['output_dir']

    os.makedirs(output_dir_data, exist_ok=True)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir_data, 'commit_hash_content.txt'), 'wt') as outfile:
        outfile.write(sha)

    v_nr_parsed_articles = Value('i', 0)
    v_nr_parsed_files = Value('i', 0)
    shutdown_xml_parser = Value('i', 0)

    nr_threads_processor = config['nr_threads_processor']

    start_time = time.time()
    arq = ArticleReadingQueue(process_file_queue_size=config['process_file_queue_size'])
    wiki_history_directory = config['wiki_history_directory']

    tot_files_parse = len(os.listdir(wiki_history_directory))
    convert_to_text_dictionary = arq.manager.dict()
    set_of_processing_files = arq.manager.dict()
    v_lock = Lock()
    process_file_readers = list()

    for i in range(nr_threads_processor):
        t = Process(target=process_xml_parser_parallel,
                    args=(shutdown_xml_parser, v_nr_parsed_articles,
                          v_nr_parsed_files,
                          config, convert_to_text_dictionary,
                          set_of_processing_files,
                          v_lock, dry_run, start_time))
        t.start()
        process_file_readers.append(t)
        logger.info(f'wiki_history_directory is: {wiki_history_directory}')


    for curr_file in os.listdir(wiki_history_directory):
        curr_filepath = os.path.join(wiki_history_directory, curr_file)
        arq.process_files_queue.put((curr_file, curr_filepath))

    shutdown_xml_parser.value = 1


    logger.info('nr_threads: %s' % nr_threads_processor)
    v_nr_hit_dictionary_convert = Value('i', 0)
    v_nr_api_calls_convert = Value('i', 0)
    #
    logger.info('wikipedia_create_dataset: multi processing activated')
    #

    logger.info('LAUNCHING THREAD DISPLAY!')
    thread = Thread(target=display, args=(v_nr_parsed_articles,
                                          start_time,
                                          v_nr_hit_dictionary_convert,
                                          set_of_processing_files))
    thread.daemon = True
    thread.start()

    # makes sure all the processes are finished
    for t in process_file_readers:
        t.join()
