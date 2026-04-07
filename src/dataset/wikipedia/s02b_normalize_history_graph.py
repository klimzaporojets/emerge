import argparse
import csv
import json
import logging
import os
import sys
import traceback
from time import time
from typing import Dict

from .misc.load_wiki_sql_tables import (
    load_wiki_page_title_to_wiki_page_id,
    load_wiki_page_id_to_redirected_page_id,
    load_wiki_page_id_to_wikidata_qid)

logger = logging.getLogger(__name__)

from datetime import datetime


def get_page_id_of_most_recent_title(page_name, timestamp_title, page_title_changes,
                                     page_id_to_page_title: dict,
                                     page_id=None,
                                     do_not_return_disambiguations=True
                                     ):
    if page_name in page_title_changes:
        title_changes = page_title_changes.get(page_name)
        most_recent_page_id = None
        most_recent_timestamp = -1
        for curr_page_id, curr_start_timestampl in title_changes.items():
            for curr_start_timestamp in curr_start_timestampl:
                if most_recent_timestamp < curr_start_timestamp < timestamp_title:
                    if do_not_return_disambiguations:
                        if curr_page_id in page_id_to_page_title:
                            curr_page_title = page_id_to_page_title[curr_page_id].lower()
                            if not curr_page_title.endswith('(disambiguation)'):
                                most_recent_page_id = curr_page_id
                                most_recent_timestamp = curr_start_timestamp
                    else:
                        most_recent_page_id = curr_page_id
                        most_recent_timestamp = curr_start_timestamp
        page_id = most_recent_page_id
    return page_id


def merge_two_target_histories(history1_target, history2_target):
    merged_history = list()
    pointer_history1_target = 0
    pointer_history2_target = 0
    last_action_history1 = ''
    last_action_history2 = ''
    last_added_action = ''
    while pointer_history1_target < len(history1_target) or \
            pointer_history2_target < len(history2_target):

        if pointer_history1_target < len(history1_target) and \
                pointer_history2_target < len(history2_target):
            history1_target_timestamp = history1_target[pointer_history1_target][0]
            history1_target_timestamp = int(history1_target_timestamp)
            history1_target_action = history1_target[pointer_history1_target][1]

            history2_target_timestamp = history2_target[pointer_history2_target][0]
            history2_target_timestamp = int(history2_target_timestamp)
            history2_target_action = history2_target[pointer_history2_target][1]

            if history1_target_timestamp > history2_target_timestamp:
                # do not add deletion of target if the target exist for this timestamp in
                # the other target list
                if ((history2_target_action == 'A' and last_action_history1 != 'A')
                        or (history2_target_action == 'D' and last_action_history1 != 'A')):
                    if history2_target_action != last_added_action:
                        merged_history.append(history2_target[pointer_history2_target])
                        last_added_action = history2_target_action
                last_action_history2 = history2_target_action
                pointer_history2_target += 1
            else:
                # do not add deletion of target if the target exist for this timestamp in
                # the other target list
                if ((history1_target_action == 'A' and last_action_history2 != 'A')
                        or (history1_target_action == 'D' and last_action_history2 != 'A')):
                    if history1_target_action != last_added_action:
                        merged_history.append(history1_target[pointer_history1_target])
                        last_added_action = history1_target_action
                last_action_history1 = history1_target_action
                pointer_history1_target += 1
        elif pointer_history1_target < len(history1_target):
            history1_target_action = history1_target[pointer_history1_target][1]
            # do not add deletion of target if the target exist for this timestamp in
            # the other target list
            if ((history1_target_action == 'A' and last_action_history2 != 'A')
                    or (history1_target_action == 'D' and last_action_history2 != 'A')):
                if history1_target_action != last_added_action:
                    merged_history.append(history1_target[pointer_history1_target])
                    last_added_action = history1_target_action
            last_action_history1 = history1_target_action
            pointer_history1_target += 1
        elif pointer_history2_target < len(history2_target):
            history2_target_action = history2_target[pointer_history2_target][1]
            # do not add deletion of target if the target exist for this timestamp in
            # the other target list
            if ((history2_target_action == 'A' and last_action_history1 != 'A')
                    or (history2_target_action == 'D' and last_action_history1 != 'A')):
                if history2_target_action != last_added_action:
                    merged_history.append(history2_target[pointer_history2_target])
                    last_added_action = history2_target_action
            last_action_history2 = history2_target_action
            pointer_history2_target += 1
    assert len(history1_target) + len(history2_target) >= len(merged_history)
    return merged_history


def revise_history(curr_hist, stability_span):
    """

    :param curr_hist:
    :param stability_span: in hours
    :return:
    """
    curr_hist_without_duplicate = list()
    last_action = ''
    # removes duplicate actions
    for idx_elem, curr_elem in enumerate(curr_hist):
        curr_action = curr_elem[1]
        if idx_elem > 0:
            if curr_action != last_action:
                curr_hist_without_duplicate.append(curr_elem)
        else:
            curr_hist_without_duplicate.append(curr_elem)
        last_action = curr_action

    if len(curr_hist_without_duplicate) == 1:
        curr_element = curr_hist_without_duplicate[0]

        assert curr_element[1] == 'A'
        return curr_hist_without_duplicate

    revised_history = list()
    prev_action: str = ''
    prev_timestamp: int = 0
    last_added_action: str = ''
    prev_hist_element = None
    prev_timestamp_add = 0
    last_added_timestamp = 0
    for idx_hist, curr_hist_element in enumerate(curr_hist_without_duplicate):
        curr_timestamp = int(curr_hist_element[0])
        curr_action = curr_hist_element[1]

        if idx_hist == 0:
            if curr_action != 'A':
                logger.error(f'first action can not be deletion: {curr_action} - {curr_hist}')

        if idx_hist > 0:
            diff_timestamps = curr_timestamp - prev_timestamp
            diff_timestamps_h = int(diff_timestamps / 60 / 60)
            if diff_timestamps_h >= stability_span:
                # adds if the last action is empty (first element in history) or last action
                # is different from current action. For example, in history we can not have two deletions of
                # link happening one after another.
                if last_added_action == '' and prev_action == 'A':
                    revised_history.append(prev_hist_element)
                    last_added_action = prev_action
                    last_added_timestamp = prev_timestamp
                elif last_added_action != '' and last_added_action != prev_action:
                    # added wikidata:
                    # under same difference of dates, always prioritize additions over deletions
                    # can happen that additiona and deletion happen at the same time because
                    # of redirect
                    if (not (prev_action == 'D' and last_added_action == 'A'
                             and (prev_timestamp == prev_timestamp_add))):
                        revised_history.append(prev_hist_element)
                        last_added_action = prev_action
                        last_added_timestamp = prev_timestamp
        prev_action = curr_action
        prev_timestamp = curr_timestamp
        prev_hist_element = curr_hist_element

        # the last element is always added unless it represents the same action
        if idx_hist + 1 == len(curr_hist_without_duplicate):
            if last_added_action != curr_action:
                if not (curr_action == 'D' and len(revised_history) == 0):
                    # added wikidata
                    if not (curr_action == 'D' and last_added_timestamp == curr_timestamp):
                        if not (curr_action == 'D' and prev_action == 'A' and
                                prev_timestamp == curr_timestamp):
                            revised_history.append(curr_hist_element)
                            last_added_timestamp = curr_timestamp
        if curr_action == 'A':
            prev_timestamp_add = curr_timestamp

    assert len(curr_hist_without_duplicate) >= len(revised_history)
    return revised_history


def get_page_title_changes(path_extracted_title_changes):
    page_title_changes = dict()
    for idx_file, curr_title_change_file in enumerate(os.listdir(path_extracted_title_changes)):
        file_path_to_process = os.path.join(path_extracted_title_changes, curr_title_change_file)
        logger.debug(f'reading {idx_file} - {curr_title_change_file}')
        with (open(file_path_to_process, 'rt') as infile):
            reader = csv.reader(infile, delimiter='\t')
            prev_page_id = -1
            for row in reader:
                page_id = int(row[0])
                old_page_title = row[1]
                new_page_title = row[2]
                timestamp_change = int(row[3])
                if old_page_title not in page_title_changes:
                    page_title_changes[old_page_title] = dict()

                if new_page_title not in page_title_changes:
                    page_title_changes[new_page_title] = dict()

                if page_id not in page_title_changes[new_page_title]:
                    page_title_changes[new_page_title][page_id] = list()

                if page_id != prev_page_id:
                    if page_id not in page_title_changes[old_page_title]:
                        page_title_changes[old_page_title][page_id] = list()
                    page_title_changes[old_page_title][page_id].append(0)

                page_title_changes[new_page_title][page_id].append(timestamp_change)

                prev_page_id = page_id
    return page_title_changes


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s02_history_links_normalization/20240822/'
                                's02_config_links_normalization.json',
                        help='The config file that contains all the parameters')

    parser.add_argument("--second_run_cat_sorted",
                        help="To indicate that a second run is performed "
                             "on already filtered entries, but concatenated and sorted.",
                        action="store_true")
    args = parser.parse_args()
    second_run_cat_sorted = args.second_run_cat_sorted

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
    path_cache_wikipedia_page_title_to_wikipedia_page_id = os.path.join(caches_dir,
                                                                        'wikipedia_page_title_to_wikipedia_page_id.pickle')
    path_cache_wikipedia_page_id_to_wikipedia_page_title = os.path.join(caches_dir,
                                                                        'wikipedia_page_id_to_wikipedia_page_title.pickle')
    path_cache_wikipedia_page_id_to_redirected_page_id = os.path.join(caches_dir,
                                                                      'wikipedia_page_id_to_redirected_page_id.pickle')

    path_wikipedia_wikidata_map = config['path_wikipedia_wikidata_map']
    path_wikipedia_page_info = config['path_wikipedia_page_info']
    path_wikipedia_page_redirects = config['path_wikipedia_page_redirects']

    wikipedia_page_title_to_wikipedia_page_id, wikipedia_page_id_to_wikipedia_page_title = \
        load_wiki_page_title_to_wiki_page_id(
            path_cache_wikipedia_page_title_to_wikipedia_page_id,
            path_cache_wikipedia_page_id_to_wikipedia_page_title,
            path_wikipedia_page_info)

    wikipedia_page_id_to_redirected_page_id = \
        load_wiki_page_id_to_redirected_page_id(path_cache_wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id,
                                                path_wikipedia_page_redirects)

    wikipedia_page_id_to_wikidata_qid = load_wiki_page_id_to_wikidata_qid(path_cache_wikipedia_page_id_to_wikidata_qid,
                                                                          path_wikipedia_wikidata_map)

    path_history_links = config['path_history_links']

    path_output_filtered_history_links = os.path.join(output_dir_data, 'wikipedia_history_filtered')

    os.makedirs(path_output_filtered_history_links, exist_ok=True)
    path_extracted_title_changes = config['path_extracted_title_changes']

    stability_span = config['stability_span']

    page_title_changes: Dict = get_page_title_changes(path_extracted_title_changes)

    nr_non_existent_qids = 0
    nr_non_existent_targets_because_of_qids = 0
    not_found_page_ids = set()
    found_anchor_page_ids = set()
    nr_existing_targets = 0
    nr_non_existing_targets = 0

    processed_anchors = set()
    nr_processed_entities = 0
    nr_processed_targets = 0
    start_time = time()
    already_shown = False
    # field_size_limit has to be at least 231519, which is the case of one of the lines in history_hyperlinks_enwiki-20240601-pages-meta-history2.xml-p60262p61974.7z.csv
    csv.field_size_limit(sys.maxsize)

    history_files = os.listdir(path_history_links)
    nr_history_files = len(history_files)
    logger.info(f'processing {nr_history_files} history files from {path_history_links}')
    for idx_hist_file, curr_history_file in enumerate(history_files):
        target_history = dict()
        target_history['targets'] = dict()
        target_history['anchor_key'] = None

        file_path_to_process = os.path.join(path_history_links, curr_history_file)
        elapsed = time() - start_time
        logger.info(f'[{idx_hist_file + 1}/{nr_history_files}] reading {curr_history_file} (elapsed: {elapsed:.0f}s)')

        prev_anchor_page_id = -1
        prev_anchor_wikidata_qid = -1
        output_filtered_history_file = os.path.join(path_output_filtered_history_links, curr_history_file)
        output_file = open(output_filtered_history_file, 'wt')
        writer = csv.writer(output_file, delimiter='\t')

        with (open(file_path_to_process, 'rt') as infile):
            reader = csv.reader(infile, delimiter='\t')
            for idx_row, row in enumerate(reader):
                if nr_processed_entities % 100000 == 0 and not already_shown:
                    curr_time = time()
                    nr_entities_per_minute = (nr_processed_entities) / ((curr_time - start_time) / 60)
                    logger.info(f'+-+-+-+-+-+-+nr of processed entities: {nr_processed_entities} and avg per '
                                f'minute: {nr_entities_per_minute}')
                    if nr_existing_targets > 0:
                        perc_found_targers = (nr_existing_targets / (
                                nr_existing_targets + nr_non_existing_targets)) * 100
                        perc_errors_due_to_qids = (nr_non_existent_targets_because_of_qids / (
                            nr_existing_targets)) * 100
                        logger.info(f'=========== % found targets: {perc_found_targers} ; '
                                    f'% wrt nr_existing_targets due to not found qids in targets: {perc_errors_due_to_qids}')
                    already_shown = True
                elif nr_processed_entities % 100000 != 0:
                    already_shown = False
                anchor_page_id = int(row[0])
                anchor_page_name = None
                if second_run_cat_sorted:
                    pass
                else:
                    anchor_page_name = row[1]
                    anchor_page_name = anchor_page_name.replace('_', ' ')
                qid_exists = True

                nr_redirects_found = 0
                while anchor_page_id in wikipedia_page_id_to_redirected_page_id:
                    anchor_page_id = wikipedia_page_id_to_redirected_page_id[anchor_page_id]
                    nr_redirects_found += 1
                    if nr_redirects_found > 100:
                        break
                if anchor_page_id in wikipedia_page_id_to_redirected_page_id:
                    logger.error('following anchor_page_id in wikipedia_page_id_to_redirected_page_id: '
                                 f'{anchor_page_id} with value of '
                                 f'{wikipedia_page_id_to_redirected_page_id[anchor_page_id]}')
                #

                if anchor_page_id in wikipedia_page_id_to_wikidata_qid:
                    anchor_wikidata_qid = wikipedia_page_id_to_wikidata_qid[anchor_page_id]
                    curr_anchor_key = (anchor_page_id, anchor_wikidata_qid)
                    if target_history['anchor_key'] is None:
                        target_history['anchor_key'] = curr_anchor_key
                    if curr_anchor_key != target_history['anchor_key']:
                        nr_processed_entities += 1
                        targets = target_history['targets']
                        rows_to_write = []
                        for target_key, target_value in targets.items():
                            curr_history = target_value['history']
                            revised_history = revise_history(curr_history, stability_span)
                            if len(revised_history) > 0 and revised_history[0][1] == 'D':
                                logger.debug(f'========================== \n'
                                             f'STARTS WITH DELETION! \n'
                                             f'anchor key: {target_history["anchor_key"]} \n'
                                             f'target key: {target_key} \n'
                                             f'stability span: {stability_span} \n'
                                             f'original history: {curr_history} \n'
                                             f'revised history: {revised_history} \n'
                                             f'========================== \n')
                            if len(revised_history) != len(curr_history):
                                logger.debug(f'++++++++++++++++++++++++++ \n'
                                             f'LENGTH OF THE REVISED HISTORY DIFFERENT TO ORIGINAL HISTORY! \n'
                                             f'anchor key: {target_history["anchor_key"]} \n'
                                             # f'anchor value: {anchor_value} \n'
                                             f'target key: {target_key} \n'
                                             f'stability span: {stability_span} \n'
                                             f'original history: {curr_history} \n'
                                             f'revised history: {revised_history} \n'
                                             f'++++++++++++++++++++++++++ \n')
                            anchor_page_ids = target_value['anchor_page_ids']
                            target_names = target_value['target_page_names']
                            anchor_qid = target_history['anchor_key'][1]
                            curr_anchor_page_id = target_history['anchor_key'][0]
                            target_qid = target_key[1]
                            curr_target_page_id = target_key[0]

                            if len(revised_history) > 0:
                                revised_history = [f'{h[0]}:{h[1]}' for h in revised_history]
                                rows_to_write.append([curr_anchor_page_id, anchor_qid, curr_target_page_id, target_qid,
                                                      ','.join(revised_history),
                                                      anchor_page_ids, target_names])
                            else:
                                pass
                        writer.writerows(rows_to_write)
                        target_history = dict()
                        target_history['targets'] = dict()
                        target_history['anchor_key'] = curr_anchor_key

                    prev_anchor_page_id = anchor_page_id
                    prev_anchor_wikidata_qid = anchor_wikidata_qid

                    ###### BEGIN wip obtaining target page id
                    target_page_name = None
                    if second_run_cat_sorted:
                        target_page_id = int(row[2])
                    else:
                        target_page_name = row[2]
                        target_page_name = target_page_name.replace('_', ' ')

                        low_target_page_name = target_page_name.lower()
                        if low_target_page_name.endswith('(disambiguation)'):
                            nr_non_existing_targets += 1
                            continue
                    #
                    if second_run_cat_sorted:
                        history_target = row[4]
                        history_target = history_target.split(',')
                    else:
                        history_target = row[3]
                        history_target = history_target[1:-1].split(', ')
                        history_target = [ht[1:-1] for ht in history_target]

                    try:
                        history_target_parsed = [(int(elem[:elem.index(':')]),
                                                  elem[elem.index(':') + 1:])
                                                 for elem in history_target]
                    except Exception as e:
                        logger.error(f'something_wrong with the following history_target: '
                                     f'{history_target} and the following line: '
                                     f'{row}')
                        traceback.print_exc()
                        raise

                    nr_processed_targets += 1

                    prev_target_page_id = None

                    target_source = -1
                    if not second_run_cat_sorted:
                        target_page_normalized = target_page_name.replace(' ', '_')
                        if target_page_normalized not in wikipedia_page_title_to_wikipedia_page_id:
                            target_source = 2
                            target_page_id = get_page_id_of_most_recent_title(
                                target_page_name,
                                int(datetime(2099, 1, 1).timestamp()),
                                page_title_changes,
                                page_id_to_page_title=wikipedia_page_id_to_wikipedia_page_title,
                                page_id=None,
                                do_not_return_disambiguations=True
                            )
                        else:
                            target_source = 1
                            target_page_id = wikipedia_page_title_to_wikipedia_page_id[target_page_normalized]

                        target_wikidata_qid = None
                        if target_page_id is not None:
                            nr_redirects_found = 0
                            while target_page_id in wikipedia_page_id_to_redirected_page_id:
                                target_page_id = wikipedia_page_id_to_redirected_page_id[target_page_id]
                                nr_redirects_found += 1
                                if nr_redirects_found > 100:
                                    break
                            #
                            if target_page_id in wikipedia_page_id_to_redirected_page_id:
                                logger.error('following target_page_id in wikipedia_page_id_to_redirected_page_id: '
                                             f'{target_page_id} with value of {wikipedia_page_id_to_redirected_page_id[target_page_id]}')

                            if target_page_id in wikipedia_page_id_to_wikidata_qid:
                                target_wikidata_qid = wikipedia_page_id_to_wikidata_qid[target_page_id]
                                qid_exists = True
                            else:
                                qid_exists = False
                    else:
                        target_wikidata_qid = row[3]
                        target_source = 10
                    if target_wikidata_qid is None and target_source == 1:
                        # tries from the titles
                        target_source = 2
                        target_page_id = get_page_id_of_most_recent_title(
                            target_page_name,
                            int(datetime(2099, 1, 1).timestamp()),
                            page_title_changes,
                            page_id_to_page_title=wikipedia_page_id_to_wikipedia_page_title,
                            page_id=None,
                            do_not_return_disambiguations=True
                        )
                        if target_page_id is not None:
                            nr_redirects_found = 0
                            while target_page_id in wikipedia_page_id_to_redirected_page_id:
                                target_page_id = wikipedia_page_id_to_redirected_page_id[target_page_id]
                                nr_redirects_found += 1
                                if nr_redirects_found > 100:
                                    break
                            if target_page_id in wikipedia_page_id_to_redirected_page_id:
                                logger.error('following target_page_id in wikipedia_page_id_to_redirected_page_id: '
                                             f'{target_page_id} with value of {wikipedia_page_id_to_redirected_page_id[target_page_id]}')

                            if target_page_id in wikipedia_page_id_to_wikidata_qid:
                                target_wikidata_qid = wikipedia_page_id_to_wikidata_qid[target_page_id]
                                qid_exists = True
                            else:
                                qid_exists = False

                    if target_wikidata_qid is not None:
                        #### Needs to be a valid target, not a disambiguation page
                        if target_page_id not in wikipedia_page_id_to_wikipedia_page_title:
                            logger.warning(f'ignoring since the title does not exist for page id {target_page_id}')
                            nr_non_existing_targets += 1
                            continue
                        target_title = wikipedia_page_id_to_wikipedia_page_title[target_page_id].lower()

                        if target_title.endswith('(disambiguation)') and target_source == 1:
                            target_page_id = get_page_id_of_most_recent_title(
                                target_page_name,
                                int(datetime(2099, 1, 1).timestamp()),
                                page_title_changes,
                                page_id_to_page_title=wikipedia_page_id_to_wikipedia_page_title,
                                page_id=None,
                                do_not_return_disambiguations=True
                            )
                            if target_page_id is not None:
                                nr_redirects_found = 0
                                while target_page_id in wikipedia_page_id_to_redirected_page_id:
                                    target_page_id = wikipedia_page_id_to_redirected_page_id[target_page_id]
                                    nr_redirects_found += 1
                                    if nr_redirects_found > 100:
                                        break

                                if target_page_id in wikipedia_page_id_to_redirected_page_id:
                                    logger.error('following target_page_id in wikipedia_page_id_to_redirected_page_id: '
                                                 f'{target_page_id} with value of {wikipedia_page_id_to_redirected_page_id[target_page_id]}')

                                if target_page_id in wikipedia_page_id_to_wikidata_qid:
                                    target_wikidata_qid = wikipedia_page_id_to_wikidata_qid[target_page_id]
                                    qid_exists = True
                                else:
                                    qid_exists = False
                                if target_page_id in wikipedia_page_id_to_wikipedia_page_title:
                                    target_title = wikipedia_page_id_to_wikipedia_page_title[target_page_id].lower()

                            if target_page_id is None or target_wikidata_qid is None or target_title.endswith(
                                    '(disambiguation)'):
                                nr_non_existing_targets += 1
                                continue

                        ####
                        nr_existing_targets += 1

                        assert target_page_id is not None
                        assert target_wikidata_qid is not None

                        target_key = (target_page_id, target_wikidata_qid)

                        if target_wikidata_qid == anchor_wikidata_qid:
                            continue

                        if target_key not in target_history['targets']:
                            target_history['targets'][target_key] = dict()
                            if not second_run_cat_sorted:
                                target_history['targets'][target_key][
                                    'target_page_names'] = {target_page_name}
                            else:
                                tmp_to_set = row[6].strip()[2:-2].split('\', \'')
                                tmp_to_set = set(tmp_to_set)
                                target_history['targets'][target_key][
                                    'target_page_names'] = tmp_to_set

                            if not second_run_cat_sorted:
                                target_history['targets'][target_key][
                                    'anchor_page_ids'] = {int(row[0])}
                            else:
                                tmp_to_set = set([int(itt) for itt in row[5].strip()[1:-1].split(', ')])
                                target_history['targets'][target_key][
                                    'anchor_page_ids'] = tmp_to_set

                            target_history['targets'][target_key][
                                'history'] = history_target_parsed
                        else:
                            if not second_run_cat_sorted:
                                target_history['targets'][target_key][
                                    'target_page_names'].add(target_page_name)
                            else:
                                tmp_to_set = row[6].strip()[2:-2].split('\', \'')
                                tmp_to_set = set(tmp_to_set)

                                target_history['targets'][target_key][
                                    'target_page_names'] = target_history['targets'][target_key][
                                    'target_page_names'].union(tmp_to_set)

                            if not second_run_cat_sorted:
                                target_history['targets'][target_key][
                                    'anchor_page_ids'].add(int(row[0]))
                            else:
                                tmp_to_set = set([int(itt) for itt in row[5].strip()[1:-1].split(', ')])
                                target_history['targets'][target_key][
                                    'anchor_page_ids'] = \
                                    target_history['targets'][target_key][
                                        'anchor_page_ids'].union(tmp_to_set)

                            history1_target = target_history['targets'][
                                target_key]['history']
                            #
                            history2_target = history_target_parsed

                            merged_history = merge_two_target_histories(history1_target, history2_target)

                            target_history['targets'][target_key]['history'] = merged_history
                    else:
                        nr_non_existing_targets += 1

                    if not qid_exists:
                        nr_non_existent_targets_because_of_qids += 1
                    found_anchor_page_ids.add(anchor_page_id)
                else:
                    # something is wrong here, the page is not associated with qid
                    if anchor_page_id not in not_found_page_ids:
                        not_found_page_ids.add(anchor_page_id)
                        nr_non_existent_qids += 1
                        perc_not_found = (len(not_found_page_ids) /
                                          (len(found_anchor_page_ids) + len(not_found_page_ids))) * 100

            targets = target_history['targets']
            rows_to_write = []
            for target_key, target_value in targets.items():
                curr_history = target_value['history']
                revised_history = revise_history(curr_history, stability_span)
                if len(revised_history) > 0 and revised_history[0][1] == 'D':
                    logger.debug(f'========================== \n'
                                 f'STARTS WITH DELETION! \n'
                                 f'anchor key: {target_history["anchor_key"]} \n'
                                 f'target key: {target_key} \n'
                                 f'stability span: {stability_span} \n'
                                 f'original history: {curr_history} \n'
                                 f'revised history: {revised_history} \n'
                                 f'========================== \n')

                if len(revised_history) != len(curr_history):
                    logger.debug(f'++++++++++++++++++++++++++ \n'
                                 f'LENGTH OF THE REVISED HISTORY DIFFERENT TO ORIGINAL HISTORY! \n'
                                 f'anchor key: {target_history["anchor_key"]} \n'
                                 f'target key: {target_key} \n'
                                 f'stability span: {stability_span} \n'
                                 f'original history: {curr_history} \n'
                                 f'revised history: {revised_history} \n'
                                 f'++++++++++++++++++++++++++ \n')

                anchor_page_ids = target_value['anchor_page_ids']
                target_names = target_value['target_page_names']
                anchor_qid = target_history['anchor_key'][1]
                target_qid = target_key[1]
                curr_target_page_id = target_key[0]
                curr_anchor_page_id = target_history['anchor_key'][0]

                if len(revised_history) > 0:
                    revised_history = [f'{h[0]}:{h[1]}' for h in revised_history]
                    rows_to_write.append([curr_anchor_page_id, anchor_qid, curr_target_page_id, target_qid,
                                          ','.join(revised_history),
                                          anchor_page_ids, target_names])
                else:
                    pass
            writer.writerows(rows_to_write)
        output_file.flush()
        output_file.close()
