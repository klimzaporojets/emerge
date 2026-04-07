import argparse
import csv
import json
import logging
import os
import time

from .misc.load_wiki_sql_tables import load_wiki_page_title_to_wiki_page_id, load_wiki_page_id_to_redirected_page_id, \
    load_wiki_page_id_to_wikidata_qid

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='config/dataset/wikidata/s02_normalize_entity_creation_date/'
                                '20250320_slurm/s02_config_normalize_entity_creation_date.json',
                        help='The config file that contains all the parameters')

    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))

    caches_dir = config['caches_dir']
    os.makedirs(caches_dir, exist_ok=True)

    output_dir_data = config['output_dir_data']
    os.makedirs(output_dir_data, exist_ok=True)
    import git

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    with open(os.path.join(output_dir_data, 'commit_hash_content.txt'), 'wt') as outfile:
        outfile.write(sha)

    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir, 'wikipedia_page_id_to_wikidata_qid.pickle')
    #
    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    #
    wikidata_page_id_to_qid = load_wiki_page_id_to_wikidata_qid(
        path_cache_wikipedia_page_id_to_wikidata_qid,
        path_wikipedia_wikidata_map
    )
    #
    wikipedia_page_id_to_redirected_page_id = dict()
    path_wikipedia_page_info = config['path_wikipedia_page_info']
    path_wikipedia_page_redirects = config['path_wikipedia_page_redirects']
    path_cache_wikipedia_page_id_to_wikidata_qid = os.path.join(caches_dir, 'wikipedia_page_id_to_wikidata_qid.pickle')
    path_cache_wikipedia_page_title_to_wikipedia_page_id = \
        os.path.join(caches_dir, 'wikipedia_page_title_to_wikipedia_page_id.pickle')
    path_cache_wikipedia_page_id_to_wikipedia_page_title = os.path.join(caches_dir,
                                                                        'wikipedia_page_id_to_wikipedia_page_title.pickle')
    path_cache_wikipedia_page_id_to_redirected_page_id = os.path.join(caches_dir,
                                                                      'wikipedia_page_id_to_redirected_page_id.pickle')

    wikipedia_page_title_to_wikipedia_page_id = dict()

    if not os.path.isfile(path_cache_wikipedia_page_id_to_redirected_page_id):
        wikipedia_page_title_to_wikipedia_page_id, _ = load_wiki_page_title_to_wiki_page_id(
            path_cache_wikipedia_page_title_to_wikipedia_page_id,
            path_cache_wikipedia_page_id_to_wikipedia_page_title,
            path_wikipedia_page_info)

    wikipedia_page_id_to_redirected_page_id = \
        load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id,
                                                path_wikipedia_page_redirects)

    path_extracted_entities = config['path_extracted_entities']
    path_output_entities = os.path.join(output_dir_data, 'generated_entities')
    os.makedirs(path_output_entities, exist_ok=True)

    # not the creation date of qid, but the creation date of the original entity
    # that might be redirected to this entity
    entity_page_id_to_creation_date = dict()
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
                        f'(elapsed: {elapsed:.0f}s, entities so far: {len(entity_page_id_to_creation_date)})')

        tsv_reader = csv.reader(open(file_path_to_process, 'rt'), delimiter='\t')

        for row in tsv_reader:
            entity_page_id = int(row[0])
            entity_creation_timestamp = int(row[2])
            #
            if entity_page_id in entity_page_id_to_creation_date:
                # always assign the lowest creation timestamp
                entity_page_id_to_creation_date[entity_page_id] = min(
                    entity_creation_timestamp, entity_page_id_to_creation_date[entity_page_id])
            else:
                entity_page_id_to_creation_date[entity_page_id] = entity_creation_timestamp

            if entity_page_id in wikipedia_page_id_to_redirected_page_id:
                redir_page_id = wikipedia_page_id_to_redirected_page_id[entity_page_id]
                logger.debug(f'found redirected from {entity_page_id} to {redir_page_id}')

                if redir_page_id in entity_page_id_to_creation_date:
                    entity_page_id_to_creation_date[redir_page_id] = min(
                        entity_page_id_to_creation_date[entity_page_id],
                        entity_page_id_to_creation_date[redir_page_id]
                    )
                    entity_page_id_to_creation_date[entity_page_id] = min(
                        entity_page_id_to_creation_date[redir_page_id],
                        entity_page_id_to_creation_date[entity_page_id]
                    )
                else:
                    entity_page_id_to_creation_date[redir_page_id] = entity_page_id_to_creation_date[entity_page_id]
    for entity_page_id, creation_timestamp in entity_page_id_to_creation_date.items():
        int_entity_page_id = int(entity_page_id)
        if int_entity_page_id in wikipedia_page_id_to_redirected_page_id:
            int_entity_page_id = wikipedia_page_id_to_redirected_page_id[int_entity_page_id]
        if int_entity_page_id in wikidata_page_id_to_qid:
            writer.writerow([wikidata_page_id_to_qid[int_entity_page_id],
                             creation_timestamp, entity_page_id])
        else:
            logger.error(f'the following entity_page_id can not be found in wikidata_page_id_to_qid: {entity_page_id}')
    output_file.flush()
    output_file.close()
