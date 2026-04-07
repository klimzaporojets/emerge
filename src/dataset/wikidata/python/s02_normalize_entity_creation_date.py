import argparse
import csv
import json
import logging
import os
import time

from dataset.wikidata.python.misc.load_wiki_sql_tables import load_wikidata_page_id_to_qid, load_wikidata_redirect

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s02_normalize_entity_creation_date/20240827/'
                                's02_config_normalize_entity_creation_date.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--debug_max_size_tables', required=False, type=int,
                        # default=40,
                        default=10000000,
                        help='Maximum number of rows that are being loaded from sql tables to not '
                             'run out of memory on local environment, if -1 loads everything.')

    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))

    debug_max_size_tables = int(args.debug_max_size_tables)

    caches_dir = config['caches_dir']
    os.makedirs(caches_dir, exist_ok=True)

    output_dir_data = config['output_dir_data']
    os.makedirs(output_dir_data, exist_ok=True)
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir_data, 'commit_hash_content.txt'), 'wt') as outfile:
        outfile.write(sha)

    path_cache_redirect = os.path.join(caches_dir, 'redirect.pickle')
    path_cache_wikidata_page_id_to_qid = os.path.join(caches_dir, 'page_id_to_qid.pickle')

    path_wikidata_page_info = config['path_wikidata_page']
    path_redirects = config['path_redirects']
    logger.info('loading wikidata_page_id_to_qid')
    wikidata_page_id_to_qid = load_wikidata_page_id_to_qid(
        path_cache_wikidata_page_id_to_qid,
        path_wikidata_page_info,
        max_nr_rows=debug_max_size_tables
    )

    logger.info('loading redirect_qid')
    wikidata_qid_to_redirected_qid = \
        load_wikidata_redirect(path_cache_redirect,
                               wikidata_page_id_to_qid,
                               path_redirects)
    path_extracted_entities = config['path_extracted_entities']
    path_output_entities = os.path.join(output_dir_data, 'generated_entities')
    os.makedirs(path_output_entities, exist_ok=True)

    # not the creation date of qid, but the creation date of the original entity
    # that might be redirected to this entity
    entity_qid_to_creation_date = dict()
    #
    output_filtered_history_file = os.path.join(path_output_entities, 'output_all_entities.csv')
    logger.info(f'writing output to: {output_filtered_history_file}')
    output_file = open(output_filtered_history_file, 'wt')
    #
    writer = csv.writer(output_file, delimiter='\t')
    entity_files = os.listdir(path_extracted_entities)
    nr_entity_files = len(entity_files)
    logger.info(f'processing {nr_entity_files} entity files from {path_extracted_entities}')
    start_time = time.time()
    for idx_hist_file, curr_history_file in enumerate(entity_files):
        anchor_qid_to_target_history = dict()
        file_path_to_process = os.path.join(path_extracted_entities, curr_history_file)
        if idx_hist_file % 50 == 0 or idx_hist_file == nr_entity_files - 1:
            elapsed = time.time() - start_time
            logger.info(f'[{idx_hist_file + 1}/{nr_entity_files}] processing {curr_history_file} '
                        f'(elapsed: {elapsed:.0f}s, entities so far: {len(entity_qid_to_creation_date)})')

        tsv_reader = csv.reader(open(file_path_to_process, 'rt'), delimiter='\t')

        for row in tsv_reader:
            entity_qid = row[0]
            entity_creation_timestamp = int(row[1])
            entity_qid_int = int(entity_qid[1:])
            #
            if entity_qid in entity_qid_to_creation_date:
                # always assign the lowest creation timestamp
                # if entity_qid_to_creation_date[entity_qid] > entity_creation_timestamp:
                entity_qid_to_creation_date[entity_qid] = min(
                    entity_creation_timestamp, entity_qid_to_creation_date[entity_qid])
            else:
                entity_qid_to_creation_date[entity_qid] = entity_creation_timestamp

            if entity_qid_int in wikidata_qid_to_redirected_qid:
                redir_qid = wikidata_qid_to_redirected_qid[entity_qid_int]
                redir_qid = f'Q{redir_qid}'
                # for now leave this logger in produ
                # logger.info(f'found redirected from {entity_qid} to {redir_qid}')

                if redir_qid in entity_qid_to_creation_date:
                    entity_qid_to_creation_date[redir_qid] = min(
                        entity_qid_to_creation_date[entity_qid],
                        entity_qid_to_creation_date[redir_qid]
                    )
                    entity_qid_to_creation_date[entity_qid] = min(
                        entity_qid_to_creation_date[redir_qid],
                        entity_qid_to_creation_date[entity_qid]
                    )
                else:
                    entity_qid_to_creation_date[redir_qid] = entity_qid_to_creation_date[entity_qid]
    logger.info(f'writing {len(entity_qid_to_creation_date)} entities to {output_filtered_history_file}')
    for entity_qid, creation_timestamp in entity_qid_to_creation_date.items():
        writer.writerow([entity_qid, creation_timestamp])
    output_file.flush()
    output_file.close()
    logger.info(f'done. Total time: {time.time() - start_time:.0f}s')
