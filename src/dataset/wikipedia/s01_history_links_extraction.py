# Module used to extract wikipedia hyperlink history graph

import argparse
import csv
import json
import logging
import os
import shutil
import time
import traceback
import xml
from multiprocessing import Process, Value, Lock
from threading import Thread

import py7zr

from .misc.article_queue import ArticleReadingQueue
from .misc.seven_zip_reader import SevenZipStreamDecompressor
from .misc.wikipedia_history_reader_no_queues_post_sort import WikipediaHistoryReaderNoQueuesPostSort

logger = logging.getLogger(__name__)


def process_xml_parser_sequential(v_nr_parsed_articles,
                                  v_nr_parsed_files,
                                  config,
                                  convert_to_text_dictionary,
                                  v_lock, dry_run, start_time,
                                  curr_file, curr_filepath):
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
    if len(config['only_process_these_page_ids']) > 0:
        logger.info('setting filter_pages to: %s' % config['only_process_these_page_ids'])
        only_process_these_page_ids = set(config['only_process_these_page_ids'])
        filter_pages = lambda page_id: page_id in only_process_these_page_ids
    else:
        filter_pages = None

    reader = WikipediaHistoryReaderNoQueuesPostSort(filter_namespace=filter_namespace,
                                                    convert_to_text_dictionary=convert_to_text_dictionary,
                                                    v_lock=v_lock,
                                                    start_time=start_time,
                                                    v_nr_parsed_articles=v_nr_parsed_articles,
                                                    config=config,
                                                    filter_pages=filter_pages,
                                                    dry_run=dry_run,
                                                    # pages_ids whose parsed text is logged
                                                    log_parsing_page_ids=config['log_parsing_page_ids'],
                                                    )

    logs_parse_dir_path = os.path.join(config['output_dir_data'], 'logs_parse')
    history_dir_path = os.path.join(config['output_dir_data'], 'wikipedia_history')
    entity_dir_path = os.path.join(config['output_dir_data'], 'entity_stats')
    # title_change_dir_path = os.path.join(config['output_dir_data'], 'entity_title_changes')
    os.makedirs(history_dir_path, exist_ok=True)
    os.makedirs(entity_dir_path, exist_ok=True)
    # os.makedirs(title_change_dir_path, exist_ok=True)
    os.makedirs(logs_parse_dir_path, exist_ok=True)
    # working directory

    output_dir_data_working = config['output_dir_data']
    logs_parse_working_dir_path = os.path.join(output_dir_data_working, 'logs_parse')
    history_working_dir_path = os.path.join(output_dir_data_working, 'wikipedia_history')
    entity_working_dir_path = os.path.join(output_dir_data_working, 'entity_stats')
    # title_change_working_dir_path = os.path.join(output_dir_data_working, 'entity_title_changes')
    revisions_working_dir_path = os.path.join(output_dir_data_working, 'revisions')
    os.makedirs(history_working_dir_path, exist_ok=True)
    os.makedirs(entity_working_dir_path, exist_ok=True)
    # os.makedirs(title_change_working_dir_path, exist_ok=True)
    os.makedirs(logs_parse_working_dir_path, exist_ok=True)
    os.makedirs(revisions_working_dir_path, exist_ok=True)

    try:
        if config['input_format'] == '7zip':
            if not curr_file.endswith('.7z'):
                logger.warning(f'IGNORING {curr_file} input file')
                return
            curr_file_start_time = time.time()
            set_of_processing_files[curr_file] = int(curr_file_start_time)
            #
            path_dir = config['wiki_history_directory']
            path_working_dir = config['wiki_history_directory_working']

            if path_working_dir is not None and path_dir != path_working_dir:
                path_working_file = os.path.join(path_working_dir, curr_file)
                os.makedirs(path_working_dir, exist_ok=True)
                logger.info('input paths to history files differ between '
                            f'working and file storage: {path_dir} '
                            f'vs {path_working_dir}\n copying current file: '
                            f'{curr_filepath} to working dir: '
                            f'{path_working_file}')
                logger.info(f'copying from {curr_filepath} '
                            f'to {path_working_file}')
                shutil.copyfile(curr_filepath, path_working_file, follow_symlinks=True)
                logger.info(f'done copying from {curr_filepath} '
                            f'to {path_working_file}')
            else:
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
            reader.tot_size = file_to_decompress.uncompressed
            #
            output_file_hist_working_path = os.path.join(history_working_dir_path,
                                                         f'history_hyperlinks_{curr_file}.csv')
            output_file_ent_working_path = os.path.join(entity_working_dir_path, f'entity_stats_{curr_file}.csv')
            # output_file_title_change_working_path = os.path.join(title_change_working_dir_path,
            #                                                      f'title_changes_{curr_file}.csv')
            output_file_revisions_working_path = os.path.join(revisions_working_dir_path,
                                                              f'revisions_{curr_file}.csv')
            output_file_parse_pages_working_path = os.path.join(logs_parse_working_dir_path,
                                                                f'log_page_parsing_{curr_file}.log')

            output_file_parse_pages = open(output_file_parse_pages_working_path, 'wt')

            output_file_history_hyperlinks = open(output_file_hist_working_path, 'wt')
            output_file_entity_stats = open(output_file_ent_working_path, 'wt')
            # output_file_pages_title_change = open(output_file_title_change_working_path, 'wt')
            output_file_revisions = open(output_file_revisions_working_path, 'wt')

            csv_writer_history_hyperlinks = csv.writer(output_file_history_hyperlinks, delimiter='\t')
            csv_writer_entity_stats = csv.writer(output_file_entity_stats, delimiter='\t')
            # csv_writer_pages_title_change = csv.writer(output_file_pages_title_change, delimiter='\t')
            csv_writer_revisions = csv.writer(output_file_revisions, delimiter='\t')

            reader.csv_writer_history_hyperlinks = csv_writer_history_hyperlinks
            reader.csv_writer_entity_stats = csv_writer_entity_stats
            # reader.csv_writer_pages_title_change = csv_writer_pages_title_change
            reader.csv_writer_revisions = csv_writer_revisions

            reader.output_file_history_hyperlinks = output_file_history_hyperlinks
            reader.output_file_entity_stats = output_file_entity_stats
            # reader.output_file_pages_title_change = output_file_pages_title_change
            reader.output_file_parse_pages = output_file_parse_pages
            reader.output_file_revisions = output_file_revisions

            logger.info('parsing: %s' % curr_file)
            logger.info(f'files currently being parsed: {set_of_processing_files.keys()}')
            #

            logger.info('compressed file size (GB): %s, %s' %
                        (curr_file, (os.path.getsize(curr_filepath) / 1024 / 1024 / 1024)))
            logger.info('uncompressed file size (GB): %s, %s ' %
                        (curr_file, (file_to_decompress.uncompressed / 1024 / 1024 / 1024)))
            try:
                xml.sax.parse(decompressor, reader)
            finally:
                archive.close()
                output_file_history_hyperlinks.flush()
                output_file_history_hyperlinks.close()
                output_file_entity_stats.flush()
                output_file_entity_stats.close()
                output_file_revisions.flush()
                output_file_revisions.close()
                output_file_parse_pages.flush()
                output_file_parse_pages.close()

            with v_nr_parsed_files.get_lock():
                v_nr_parsed_files.value += 1
            #
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
            if path_working_dir is not None and path_dir != path_working_dir:
                logger.info(f'Since {path_dir} != {path_working_dir}, '
                            f'\n removing {path_working_file}')
                os.remove(path_working_file)

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


def process_xml_parser_parallel(shutdown_xml_parser, v_nr_parsed_articles,
                                # v_nr_pages_change_title_error,
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
    if len(config['only_process_these_page_ids']) > 0:
        logger.info('setting filter_pages to: %s' % config['only_process_these_page_ids'])
        only_process_these_page_ids = set(config['only_process_these_page_ids'])
        filter_pages = lambda page_id: page_id in only_process_these_page_ids
    else:
        filter_pages = None

    reader = WikipediaHistoryReaderNoQueuesPostSort(filter_namespace=filter_namespace,
                                                    convert_to_text_dictionary=convert_to_text_dictionary,
                                                    v_lock=v_lock,
                                                    start_time=start_time,
                                                    v_nr_parsed_articles=v_nr_parsed_articles,
                                                    config=config,
                                                    filter_pages=filter_pages,
                                                    dry_run=dry_run,
                                                    log_parsing_page_ids=config['log_parsing_page_ids']
                                                    )

    logs_parse_dir_path = os.path.join(config['output_dir_data'], 'logs_parse')
    history_dir_path = os.path.join(config['output_dir_data'], 'wikipedia_history')
    entity_dir_path = os.path.join(config['output_dir_data'], 'entity_stats')
    # title_change_dir_path = os.path.join(config['output_dir_data'], 'entity_title_changes')
    os.makedirs(history_dir_path, exist_ok=True)
    os.makedirs(entity_dir_path, exist_ok=True)
    # os.makedirs(title_change_dir_path, exist_ok=True)
    os.makedirs(logs_parse_dir_path, exist_ok=True)
    # working directory

    #
    output_dir_data_working = config['output_dir_data']
    #
    logs_parse_working_dir_path = os.path.join(output_dir_data_working, 'logs_parse')
    history_working_dir_path = os.path.join(output_dir_data_working, 'wikipedia_history')
    entity_working_dir_path = os.path.join(output_dir_data_working, 'entity_stats')
    # title_change_working_dir_path = os.path.join(output_dir_data_working, 'entity_title_changes')
    revisions_working_dir_path = os.path.join(output_dir_data_working, 'revisions')
    os.makedirs(history_working_dir_path, exist_ok=True)
    os.makedirs(entity_working_dir_path, exist_ok=True)
    # os.makedirs(title_change_working_dir_path, exist_ok=True)
    os.makedirs(logs_parse_working_dir_path, exist_ok=True)
    os.makedirs(revisions_working_dir_path, exist_ok=True)

    while not (shutdown_xml_parser.value == 1 and arq.process_files_queue.empty()):
        try:
            if config['input_format'] == '7zip':
                curr_file, curr_filepath = arq.process_files_queue.get(
                    block=True,
                    timeout=config['queues_timeout'])

                if not curr_file.endswith('.7z'):
                    logger.warning(f'IGNORING {curr_file} input file')
                    continue
                curr_file_start_time = time.time()
                set_of_processing_files[curr_file] = int(curr_file_start_time)
                #
                path_dir = config['wiki_history_directory']
                path_working_dir = config['wiki_history_directory_working']
                if path_working_dir is not None and path_dir != path_working_dir:
                    path_working_file = os.path.join(path_working_dir, curr_file)
                    os.makedirs(path_working_dir, exist_ok=True)
                    logger.info('input paths to history files differ between '
                                f'working and file storage: {path_dir} '
                                f'vs {path_working_dir}\n copying current file: '
                                f'{curr_filepath} to working dir: '
                                f'{path_working_file}')
                    logger.info(f'copying from {curr_filepath} '
                                f'to {path_working_file}')
                    shutil.copyfile(curr_filepath, path_working_file, follow_symlinks=True)
                    logger.info(f'done copying from {curr_filepath} '
                                f'to {path_working_file}')
                else:
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
                reader.tot_size = file_to_decompress.uncompressed

                output_file_hist_working_path = os.path.join(history_working_dir_path,
                                                             f'history_hyperlinks_{curr_file}.csv')
                output_file_ent_working_path = os.path.join(entity_working_dir_path, f'entity_stats_{curr_file}.csv')
                # output_file_title_change_working_path = os.path.join(title_change_working_dir_path,
                #                                                      f'title_changes_{curr_file}.csv')
                output_file_revisions_working_path = os.path.join(revisions_working_dir_path,
                                                                  f'revisions_{curr_file}.csv')
                output_file_parse_pages_working_path = os.path.join(logs_parse_working_dir_path,
                                                                    f'log_page_parsing_{curr_file}.log')

                output_file_parse_pages = open(output_file_parse_pages_working_path, 'wt')

                output_file_history_hyperlinks = open(output_file_hist_working_path, 'wt')
                output_file_entity_stats = open(output_file_ent_working_path, 'wt')
                # output_file_pages_title_change = open(output_file_title_change_working_path, 'wt')
                output_file_revisions = open(output_file_revisions_working_path, 'wt')

                csv_writer_history_hyperlinks = csv.writer(output_file_history_hyperlinks, delimiter='\t')
                csv_writer_entity_stats = csv.writer(output_file_entity_stats, delimiter='\t')
                # csv_writer_pages_title_change = csv.writer(output_file_pages_title_change, delimiter='\t')
                csv_writer_revisions = csv.writer(output_file_revisions, delimiter='\t')

                reader.csv_writer_history_hyperlinks = csv_writer_history_hyperlinks
                reader.csv_writer_entity_stats = csv_writer_entity_stats
                # reader.csv_writer_pages_title_change = csv_writer_pages_title_change

                reader.csv_writer_revisions = csv_writer_revisions

                reader.output_file_history_hyperlinks = output_file_history_hyperlinks
                reader.output_file_entity_stats = output_file_entity_stats
                # reader.output_file_pages_title_change = output_file_pages_title_change
                reader.output_file_parse_pages = output_file_parse_pages
                reader.output_file_revisions = output_file_revisions

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
                    output_file_history_hyperlinks.flush()
                    output_file_history_hyperlinks.close()
                    output_file_entity_stats.flush()
                    output_file_entity_stats.close()
                    output_file_revisions.flush()
                    output_file_revisions.close()
                    output_file_parse_pages.flush()
                    output_file_parse_pages.close()

                with v_nr_parsed_files.get_lock():
                    v_nr_parsed_files.value += 1
                #
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
                if path_working_dir is not None and path_dir != path_working_dir:
                    logger.info(f'Since {path_dir} != {path_working_dir}, '
                                f'\n removing {path_working_file}')
                    os.remove(path_working_file)

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
            continue


#
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s01_history_links_extraction/20240910/s01_config_content.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--nr_threads_processor', required=False, type=int,
                        default=4,
                        help='Nr of threads that will process the data')

    parser.add_argument('--dry_run', help='Dry run, reading all the documents, but do not '
                                          'parse anything.', action='store_true')

    parser.add_argument('--flush_individually',
                        help='Generally used during testing locally to obtain fast the output without waiting for the files to automatically be flushed.',
                        action='store_true')

    parser.add_argument('--sequential',
                        help='No multiprocessing, just synchronous sequential execution.',
                        action='store_true')

    args = parser.parse_args()
    logger.info('Running wikipedia_history_reader with the following arguments: %s' % args)

    logger.info('Current working directory: %s' % os.getcwd())

    config = json.load(open(args.config_file, 'rt'))
    config['nr_threads_processor'] = args.nr_threads_processor
    config['dry_run'] = args.dry_run
    config['flush_individually'] = args.flush_individually
    config['sequential'] = args.sequential
    dry_run = args.dry_run
    if dry_run:
        logger.info('dry_run in True')
    else:
        logger.info('dry_run in False')
    logger.info('THE CONFIG IS: %s' % config)
    output_dir_data = config['output_dir_data']

    os.makedirs(output_dir_data, exist_ok=True)
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir_data, 'commit_hash_content.txt'), 'wt') as outfile:
        outfile.write(sha)

    wiki_history_directory = config['wiki_history_directory']

    nr_threads_processor = config['nr_threads_processor']
    ############### BEGIN: for a particular wikipedia snapshot --> data structures that we will need to get statistics
    wikipedia_page_id_to_wikidata_qid = dict()
    wikipedia_page_title_to_wikipedia_page_id = dict()
    wikipedia_page_id_to_wikipedia_page_title = dict()
    wikipedia_page_id_to_redirected_page_id = dict()

    ############### END: for a particular wikipedia snapshot --> data structures that we will need to get statistics

    ################ BEGIN: others independent of time
    # RQ: how far apart in time are wikidata and wikipedia entity creation?
    wikidata_qid_to_wikidata_creation_time = dict()
    wikidata_qid_to_wikipedia_creation_time = dict()

    ################ END: others independent of time

    arq = ArticleReadingQueue(process_file_queue_size=config['process_file_queue_size'])

    used_qids = set()
    nr_pages_processed = 0
    idx_lines_to_reprocess = []

    ##################

    process_file_readers = []
    v_nr_parsed_articles = Value('i', 0)
    v_nr_parsed_files = Value('i', 0)
    shutdown_xml_parser = Value('i', 0)
    # v_nr_pages_with_change_of_title = Value('i', 0)
    # v_nr_pages_change_title_error = Value('i', 0)

    tot_files_parse = len(os.listdir(wiki_history_directory))

    v_lock = Lock()

    start_time = time.time()

    convert_to_text_dictionary = arq.manager.dict()
    set_of_processing_files = arq.manager.dict()
    if not config['sequential']:
        logger.info('non-sequential execution nr_threads: %s' % nr_threads_processor)
        for i in range(nr_threads_processor):
            t = Process(target=process_xml_parser_parallel,
                        args=(shutdown_xml_parser, v_nr_parsed_articles,
                              # v_nr_pages_change_title_error,
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
    else:
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
        for curr_file in os.listdir(wiki_history_directory):
            curr_filepath = os.path.join(wiki_history_directory, curr_file)
            process_xml_parser_sequential(v_nr_parsed_articles,
                                          # v_nr_pages_change_title_error,
                                          v_nr_parsed_files,
                                          config,
                                          convert_to_text_dictionary,
                                          v_lock,
                                          dry_run,
                                          start_time,
                                          curr_file,
                                          curr_filepath)

    #######
    logger.info('Done')
