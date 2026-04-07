"""
The _v8 version extends the _v7 by also accepting the field 'matched_triples_entities_to_kg'
which contains the relations between any of the entities mentioned in text and the
deltas. Indicating what other relations have to be added to the KG.
From the API side, this has on-off functionality with the following parameters in config
json:
  "match_all_emerging_relations_from_head": true,
  "match_all_emerging_relations_from_tail": true
"""

import argparse
import csv
import json
import logging
import os
from threading import Thread
from typing import Dict
import time
import xml
import traceback
import shutil
import py7zr

from dataset.wikipedia.misc.seven_zip_reader import SevenZipStreamDecompressor

from dataset.wikipedia.misc.article_queue import ArticleReadingQueue
from dataset.wikipedia.misc.wikipedia_history_snippet_extractor_v8 import WikipediaHistorySnippetExtractorV8
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

from multiprocessing import Process, Value, Lock


def display(v_nr_parsed_articles,
            initial_time,
            v_nr_hit_dictionary_convert,
            set_of_processing_files):
    while True:
        try:
            logger.info('\tDISPLAY Size of queue process_files_queue: %s' % arq.process_files_queue.qsize())
            logger.info('\tDISPLAY Size of v_nr_hit_dictionary_convert: %s' % v_nr_hit_dictionary_convert.value)
            # TODO: the files being processed sorted by the time they have been in the queue
            # first, updates the time of the files being processed:
            # logger.info(f'\tInitial time is: {initial_time}')
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

            # time.sleep(60 * 1)
            time.sleep(60 * 1)
        except Exception as e:
            logger.error('type error display: %s' % str(e))
            logger.error(traceback.format_exc())
            continue


def process_xml_parser_parallel(shutdown_xml_parser, v_nr_parsed_articles,
                                v_max_recorded_text_length,
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

    reader = WikipediaHistorySnippetExtractorV8(filter_namespace=filter_namespace,
                                                # article_callback=arq.enqueue_article,
                                                # v_nr_pages_change_title_error=v_nr_pages_change_title_error,
                                                convert_to_text_dictionary=convert_to_text_dictionary,
                                                # timespan_2_head_id_2_tail_ids=timespan_2_head_id_2_tail_ids,
                                                # head_id_2_timespans_2_tail_ids=head_id_2_timespans_2_tail_ids,
                                                # head_page_ids=head_page_ids,
                                                v_lock=v_lock,
                                                start_time=start_time,
                                                v_nr_parsed_articles=v_nr_parsed_articles,
                                                v_max_recorded_text_length=v_max_recorded_text_length,
                                                config=config,
                                                filter_pages=filter_pages,
                                                dry_run=dry_run,
                                                # pages_ids whose parsed text is logged
                                                log_parsing_page_ids=config['log_parsing_page_ids']
                                                )

    mentions_context_dir_path = os.path.join(config['output_dir_data'], 'mentions_context')
    os.makedirs(mentions_context_dir_path, exist_ok=True)
    # working directory

    output_dir_data_working = config['output_dir_data_working']

    if output_dir_data_working is None:
        output_dir_data_working = config['output_dir_data']

    mentions_context_working_dir_path = os.path.join(output_dir_data_working, 'mentions_context')
    os.makedirs(mentions_context_working_dir_path, exist_ok=True)

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
                # archive = py7zr.SevenZipFile(curr_filepath, mode='r')
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
                # output_dir_data
                output_file_mentions_context_path = os.path.join(mentions_context_dir_path,
                                                                 f'mentions_context_{curr_file}.jsonl')

                output_file_mentions_context_working_path = os.path.join(mentions_context_working_dir_path,
                                                                         f'mentions_context_{curr_file}.jsonl')

                output_file_mentions_context = open(output_file_mentions_context_working_path, 'wt')

                reader.output_file_mentions_context = output_file_mentions_context

                logger.info('parsing: %s' % curr_file)
                logger.info(f'files currently being parsed: {set_of_processing_files.keys()}')

                logger.info('compressed file size (GB): %s, %s' %
                            (curr_file, (os.path.getsize(curr_filepath) / 1024 / 1024 / 1024)))
                logger.info('uncompressed file size (GB): %s, %s ' %
                            (curr_file, (file_to_decompress.uncompressed / 1024 / 1024 / 1024)))
                xml.sax.parse(decompressor, reader)

                output_file_mentions_context.flush()
                output_file_mentions_context.close()

                with v_nr_parsed_files.get_lock():
                    v_nr_parsed_files.value += 1

                lapse_from_program_start = (time.time() - start_time) / 60 / 60
                lapse_from_file_start = (time.time() - set_of_processing_files[curr_file]) / 60 / 60
                logger.info(f'FINISHED TO PARSING THE FILE NR {v_nr_parsed_files.value} ({curr_file}) '
                            f'out of {tot_files_parse} with time of '
                            # f'{lapse_from_program_start:.2f} hours '
                            f' after {lapse_from_file_start:.2f} hours of starting '
                            f' and after {lapse_from_program_start:.2f} hours of program start')
                # logger.info('FINISHED TO PARSING THE FILE NR %s out of %s with time of %s hours from start and ' %
                #             (v_nr_parsed_files.value, tot_files_parse, (time.time() - start_time) / 60 / 60))
                logger.info(f'Total of files left in the queue: {arq.process_files_queue.qsize()} '
                            f'is the queue empty? {arq.process_files_queue.empty()}')
                logger.info(f'Files that are currently still being processed: '
                            f'{set_of_processing_files}')
                if path_working_dir is not None and path_dir != path_working_dir:
                    logger.info(f'Since {path_dir} != {path_working_dir}, '
                                f'\n removing {path_working_file}')
                    os.remove(path_working_file)
                    # TODO
                if (config['output_dir_data_working'] is not None and
                        config['output_dir_data_working'] != config['output_dir_data']):
                    logger.info('copying output from data working dir to data dir')
                    logger.info(f'copying {output_file_mentions_context_working_path} to '
                                f'{output_file_mentions_context_path}')
                    shutil.copyfile(output_file_mentions_context_working_path, output_file_mentions_context_path,
                                    follow_symlinks=True)
                    os.remove(output_file_mentions_context_working_path)

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
                        default='experiments/s03_obtain_textual_delta_snippets_v3/'
                                '20241122/s03_config_obtain_textual_delta.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--nr_threads_processor', required=False, type=int,
                        # default=40,
                        default=4,
                        # default=1,
                        help='Nr of threads that will process the data')

    # parser.add_argument('--api_ports_wiki_mapping', '--list', nargs='+',
    parser.add_argument('--api_ports_wiki_mapping', nargs='+',
                        type=int, required=True,
                        help='List of ports on which the s03_API_v6_wiki_mapping '
                             'is deployed.')

    # parser.add_argument('--api_ports_only_deltas', '--list', nargs='+',
    parser.add_argument('--api_ports_only_deltas', nargs='+',
                        type=int, required=True,
                        help='List of ports on which the s03_API_v6_only_deltas '
                             'is deployed.')

    parser.add_argument('--dry_run', help='Dry run, reading all the documents, but do not '
                                          'parse anything.', action='store_true')

    args = parser.parse_args()
    logger.info('Running wikipedia_history_reader with the following arguments: %s' % args)

    logger.info('Current working directory: %s' % os.getcwd())

    config = json.load(open(args.config_file, 'rt'))

    anchor_entities_qid_to_page_ids = dict()
    anchor_page_ids_to_qids = dict()
    config['nr_threads_processor'] = args.nr_threads_processor
    config['dry_run'] = args.dry_run

    logger.info('THE CONFIG IS: %s' % config)
    output_dir_data = config['output_dir_data']

    nr_threads_processor = config['nr_threads_processor']
    caches_dir = config['caches_dir']
    config['api_ports_wiki_mapping'] = args.api_ports_wiki_mapping
    config['api_ports_only_deltas'] = args.api_ports_only_deltas
    logger.info(f'assigned api_ports_wiki_mapping: {config["api_ports_wiki_mapping"]}')
    logger.info(f'assigned api_ports_only_deltas: {config["api_ports_only_deltas"]}')
    os.makedirs(caches_dir, exist_ok=True)
    os.makedirs(output_dir_data, exist_ok=True)
    # os.makedirs(output_dir_stats, exist_ok=True)
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir_data, 'commit_hash_content.txt'), 'wt') as outfile:
        outfile.write(sha)

    process_file_readers = []
    v_nr_parsed_articles = Value('i', 0)
    v_max_recorded_text_length = Value('i', 0)
    v_nr_parsed_files = Value('i', 0)
    shutdown_xml_parser = Value('i', 0)
    # shutdown_file_writers = Value('i', 0)
    v_nr_pages_with_change_of_title = Value('i', 0)
    v_nr_pages_change_title_error = Value('i', 0)
    # v_nr_scanned_entities = Value('i', 0)
    wiki_history_directory = config['wiki_history_directory']
    dry_run = config['dry_run']

    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    path_wikipedia_page_info = config['path_wikipedia_page_info']
    path_wikipedia_page_redirects = config['path_wikipedia_page_redirects']

    tot_files_parse = len(os.listdir(wiki_history_directory))

    v_lock = Lock()

    start_time = time.time()
    arq = ArticleReadingQueue(process_file_queue_size=config['process_file_queue_size'])

    convert_to_text_dictionary = arq.manager.dict()
    set_of_processing_files = arq.manager.dict()

    logger.info('non-sequential execution nr_threads: %s' % nr_threads_processor)
    for i in range(nr_threads_processor):
        t = Process(target=process_xml_parser_parallel,
                    args=(shutdown_xml_parser, v_nr_parsed_articles,
                          v_max_recorded_text_length,
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
    # shutdown = Value('i', 0)
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
