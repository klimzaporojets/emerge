import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback

from dataset.wikidata.python.misc.load_wiki_sql_tables import load_wikidata_redirect, load_wikidata_page_id_to_qid

logger = logging.getLogger(__name__)


def revise_history(curr_hist, stability_span, timestamp_precision):
    """

    :param curr_hist:
    :param stability_span: in hours
    :param timestamp_precision: either 'seconds' or 'milliseconds'
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
        # curr_action = curr_element[curr_element.index(':') + 1:]

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
            if timestamp_precision == 'milliseconds':
                diff_timestamps_h = diff_timestamps / 1000
            elif timestamp_precision == 'seconds':
                diff_timestamps_h = diff_timestamps
            else:
                raise RuntimeError(f'not recognized timestamp_precision of {timestamp_precision}')

            diff_timestamps_h = int(diff_timestamps_h / 60 / 60)
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


def is_timestamp_sorted(history_target_parsed):
    prev_edit_timestamp = 0
    is_sorted = True
    for idx_edit, curr_edit in enumerate(history_target_parsed):
        if idx_edit > 0:
            # changes have to be sorted by increasing timestamp
            if curr_edit[0] < prev_edit_timestamp:
                return False
        prev_edit_timestamp = curr_edit[0]
    return is_sorted


def merge_two_target_histories(history1_target_no_qualifiers, history2_target_no_qualifiers,
                               history1_target_qualifiers, history2_target_qualifiers,
                               timestamp_precision):
    latest_history = 0
    merged_history = list()
    pointer_history1_target = 0
    pointer_history2_target = 0
    last_action_history1 = ''
    last_action_history2 = ''
    last_added_action = ''
    while pointer_history1_target < len(history1_target_no_qualifiers) or \
            pointer_history2_target < len(history2_target_no_qualifiers):

        if pointer_history1_target < len(history1_target_no_qualifiers) and \
                pointer_history2_target < len(history2_target_no_qualifiers):
            history1_target_timestamp = history1_target_no_qualifiers[pointer_history1_target][0]
            history1_target_timestamp = int(history1_target_timestamp)
            history1_target_action = history1_target_no_qualifiers[pointer_history1_target][1]

            history2_target_timestamp = history2_target_no_qualifiers[pointer_history2_target][0]
            history2_target_timestamp = int(history2_target_timestamp)
            history2_target_action = history2_target_no_qualifiers[pointer_history2_target][1]

            if history1_target_timestamp > history2_target_timestamp:
                # do not add deletion of target if the target exist for this timestamp in
                # the other target list
                if ((history2_target_action == 'A' and last_action_history1 != 'A')
                        or (history2_target_action == 'D' and last_action_history1 != 'A')):
                    if history2_target_action != last_added_action:
                        merged_history.append(history2_target_no_qualifiers[pointer_history2_target])
                        last_added_action = history2_target_action
                        latest_history = 2
                last_action_history2 = history2_target_action
                pointer_history2_target += 1
            else:
                # do not add deletion of target if the target exist for this timestamp in
                # the other target list
                if ((history1_target_action == 'A' and last_action_history2 != 'A')
                        or (history1_target_action == 'D' and last_action_history2 != 'A')):
                    if history1_target_action != last_added_action:
                        merged_history.append(history1_target_no_qualifiers[pointer_history1_target])
                        last_added_action = history1_target_action
                        latest_history = 1
                last_action_history1 = history1_target_action
                pointer_history1_target += 1
        elif pointer_history1_target < len(history1_target_no_qualifiers):
            history1_target_action = history1_target_no_qualifiers[pointer_history1_target][1]
            # do not add deletion of target if the target exist for this timestamp in
            # the other target list
            if ((history1_target_action == 'A' and last_action_history2 != 'A')
                    or (history1_target_action == 'D' and last_action_history2 != 'A')):
                if history1_target_action != last_added_action:
                    merged_history.append(history1_target_no_qualifiers[pointer_history1_target])
                    last_added_action = history1_target_action
                    latest_history = 1
            last_action_history1 = history1_target_action
            pointer_history1_target += 1
        elif pointer_history2_target < len(history2_target_no_qualifiers):
            history2_target_action = history2_target_no_qualifiers[pointer_history2_target][1]
            # do not add deletion of target if the target exist for this timestamp in
            # the other target list
            if ((history2_target_action == 'A' and last_action_history1 != 'A')
                    or (history2_target_action == 'D' and last_action_history1 != 'A')):
                if history2_target_action != last_added_action:
                    merged_history.append(history2_target_no_qualifiers[pointer_history2_target])
                    last_added_action = history2_target_action
                    latest_history = 2
            last_action_history2 = history2_target_action
            pointer_history2_target += 1
    assert len(history1_target_no_qualifiers) + len(history2_target_no_qualifiers) >= len(merged_history)

    # logger.info(f'revise_history_with {merged_history}')
    revised_merged_history = revise_history(merged_history, stability_span, timestamp_precision)

    # passes all to str before concatenating with qualifiers
    revised_merged_history = [f'{field_timestamp}:{field_action}' for field_timestamp, field_action in
                              revised_merged_history]
    if latest_history == 0 and len(history2_target_qualifiers) > 0:
        revised_merged_history = revised_merged_history + history2_target_qualifiers
    elif latest_history == 1 and len(history1_target_qualifiers) > 0:
        revised_merged_history = revised_merged_history + history1_target_qualifiers
    elif latest_history == 2 and len(history2_target_qualifiers) > 0:
        revised_merged_history = revised_merged_history + history2_target_qualifiers
    return revised_merged_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        # default='experiments/s02_normalize_history_graph/20240705/'
                        #         's02_config_normalize_history_graph.json',
                        default='experiments/s02_normalize_history_graph/'
                                '20240808_djlama/s02_config_normalize_history_graph.json',
                        help='The config file that contains all the parameters')

    parser.add_argument('--debug_max_size_tables', required=False, type=int,
                        default=-1,
                        # default=10000000,
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
    if debug_max_size_tables > -1:
        path_cache_wikidata_page_id_to_qid = os.path.join(caches_dir,
                                                          f'page_id_to_qid_top{debug_max_size_tables}.pickle')
    else:
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
    path_extracted_history_triples = config['path_extracted_history_triples']

    stability_span = config['stability_span']
    timestamp_precision = config['timestamp_precision']
    path_output_filtered_history_links = os.path.join(output_dir_data, 's02a_generated_history_triples_filtered')
    os.makedirs(path_output_filtered_history_links, exist_ok=True)

    not_found_page_ids = set()
    found_anchor_page_ids = set()
    nr_existing_targets = 0
    nr_non_existing_targets = 0

    processed_anchors = set()
    old_anchor_qid = None
    anchor_qid = None
    nr_processed_anchor_entities = 0

    start_time = time.time()

    csv.field_size_limit(sys.maxsize)

    max_nr_redirects = 50

    history_files = os.listdir(path_extracted_history_triples)
    nr_history_files = len(history_files)
    logger.info(f'processing {nr_history_files} history files from {path_extracted_history_triples}')
    for idx_hist_file, curr_history_file in enumerate(history_files):
        target_history = dict()
        target_history['targets'] = dict()
        target_history['anchor_key'] = None

        file_path_to_process = os.path.join(path_extracted_history_triples, curr_history_file)
        if idx_hist_file % 50 == 0 or idx_hist_file == nr_history_files - 1:
            elapsed = time.time() - start_time
            logger.info(f'[{idx_hist_file + 1}/{nr_history_files}] reading {curr_history_file} (elapsed: {elapsed:.0f}s)')

        # already_obtained_recent_title = False
        prev_anchor_page_id = -1
        prev_anchor_wikidata_qid = -1
        output_filtered_history_file = os.path.join(path_output_filtered_history_links,
                                                    curr_history_file)
        output_file = open(output_filtered_history_file, 'wt')
        writer = csv.writer(output_file, delimiter='\t')
        with (open(file_path_to_process, 'rt') as infile):
            reader = csv.reader(infile, delimiter='\t')
            # logger.info(f'processing the file: {file_path_to_process}')
            for row in reader:
                try:
                    if len(row) < 5:
                        if len(' '.join(row).split(' ')) < 1000:
                            logger.error(f'Invalid ROW with only {len(row)} elements, skipping: {row}')
                        continue

                    # logger.info(f'row read: {row}')
                    if anchor_qid is not None:
                        old_anchor_qid = anchor_qid

                    if row[0].startswith('Q'):
                        anchor_qid = int(row[0][1:])
                    else:
                        anchor_qid = int(row[0])

                    if old_anchor_qid is not None and anchor_qid != old_anchor_qid:
                        nr_processed_anchor_entities += 1
                        if nr_processed_anchor_entities % 100000 == 0:
                            avg_anchor_ent_per_min = nr_processed_anchor_entities / ((time.time() - start_time) / 60)
                            logger.info(f'+++++++++nr of processed anchor entities: {nr_processed_anchor_entities} '
                                        f'avg of anchor entities / min: {avg_anchor_ent_per_min}')

                    property_id = row[1]
                    tail_type = row[3]

                    if anchor_qid in wikidata_qid_to_redirected_qid:
                        ##################
                        nr_redirects_made = 0
                        while nr_redirects_made < max_nr_redirects and anchor_qid in wikidata_qid_to_redirected_qid:
                            redirected_anchor_qid = wikidata_qid_to_redirected_qid[anchor_qid]
                            # logger.info(f'target redirect detected from {anchor_qid} to '
                            #             f'{redirected_anchor_qid}')
                            anchor_qid = redirected_anchor_qid
                            nr_redirects_made += 1
                        if anchor_qid in wikidata_qid_to_redirected_qid:
                            logger.error(
                                f'Something wrong, can not find the redirected qid after {nr_redirects_made} retries '
                                f'for the following qid: {row[2]} of the following '
                                f'line: {row}')
                        ##################

                    anchor_key = (anchor_qid, property_id)
                    if target_history['anchor_key'] is None:
                        target_history['anchor_key'] = anchor_key

                    if anchor_key != target_history['anchor_key']:
                        targets = target_history['targets']
                        for target_key, target_value in targets.items():
                            curr_history = target_value['history']
                            curr_target_type = target_value['type']
                            curr_anchor_qids = target_value['anchor_qids']
                            curr_anchor_qid_hist = target_history['anchor_key'][0]
                            curr_property_id_hist = target_history['anchor_key'][1]
                            curr_target_value = target_key
                            writer.writerow([curr_anchor_qid_hist,
                                             curr_property_id_hist,
                                             curr_target_value,
                                             curr_target_type,
                                             ','.join(curr_history),
                                             curr_anchor_qids])
                            #
                        target_history['anchor_key'] = anchor_key
                        target_history['targets'] = dict()

                    if tail_type == 'wikibase-entityid':
                        if row[2].startswith('Q'):
                            curr_target_value = int(row[2][1:])
                        else:
                            curr_target_value = int(row[2])
                    else:
                        curr_target_value = row[2]

                    history_target = row[4]

                    history_target = history_target.split(',')
                    # removes the last one

                    if history_target[-1] == '':
                        history_target = history_target[:-1]

                    prev_target_qid = None

                    if tail_type == 'wikibase-entityid' and curr_target_value in wikidata_qid_to_redirected_qid:
                        nr_redirects_made = 0
                        while nr_redirects_made < max_nr_redirects and curr_target_value in wikidata_qid_to_redirected_qid:
                            redirected_target_value = wikidata_qid_to_redirected_qid[curr_target_value]
                            # logger.info(f'target redirect detected from {curr_target_value} to '
                            #             f'{redirected_target_value}')
                            curr_target_value = redirected_target_value
                            nr_redirects_made += 1
                        if curr_target_value in wikidata_qid_to_redirected_qid:
                            logger.error(
                                f'Something wrong can not find the redirected qid after {nr_redirects_made} retries '
                                f'for the following qid: {row[2]} of the following '
                                f'line: {row}')

                    target_key = curr_target_value

                    # if curr_target_value == anchor_qid:
                    #     pass

                    if target_key not in target_history['targets']:
                        history_target_no_qualifiers = [hist for hist in history_target
                                                        if not hist.startswith('P')]
                        history_target_no_qualifiers_parsed = [(int(elem[:elem.index(':')]),
                                                                elem[elem.index(':') + 1:])
                                                               for elem in history_target_no_qualifiers]
                        history_target_no_qualifiers_parsed = revise_history(history_target_no_qualifiers_parsed,
                                                                             stability_span,
                                                                             timestamp_precision)
                        if len(history_target_no_qualifiers_parsed) > 0:
                            history_target = [f'{field_timestamp}:{field_action}' for
                                              field_timestamp, field_action in history_target_no_qualifiers_parsed] + \
                                             [hist for hist in history_target
                                              if hist.startswith('P')]

                            target_history['targets'][target_key] = dict()
                            target_history['targets'][target_key]['anchor_qids'] = {anchor_qid}
                            target_history['targets'][target_key]['history'] = history_target
                            target_history['targets'][target_key]['type'] = tail_type

                    else:
                        target_history['targets'][target_key]['anchor_qids'].add(anchor_qid)
                        history1_target = target_history['targets'][target_key]['history']

                        history2_target = history_target
                        history1_target_no_qualifiers = [hist for hist in history1_target
                                                         if not hist.startswith('P')]
                        history2_target_no_qualifiers = [hist for hist in history2_target
                                                         if not hist.startswith('P')]

                        history1_target_no_qualifiers_parsed = [(int(elem[:elem.index(':')]),
                                                                 elem[elem.index(':') + 1:])
                                                                for elem in history1_target_no_qualifiers]

                        history2_target_no_qualifiers_parsed = [(int(elem[:elem.index(':')]),
                                                                 elem[elem.index(':') + 1:])
                                                                for elem in history2_target_no_qualifiers]
                        # extracts temporal qualifiers of history1
                        history1_target_qualifiers = [hist for hist in history1_target
                                                      if hist.startswith('P')]
                        #
                        # extracts temporal qualifiers of history2
                        history2_target_qualifiers = [hist for hist in history2_target
                                                      if hist.startswith('P')]

                        revised_merged_history_l = merge_two_target_histories(history1_target_no_qualifiers_parsed,
                                                                            history2_target_no_qualifiers_parsed,
                                                                            history1_target_qualifiers,
                                                                            history2_target_qualifiers,
                                                                            timestamp_precision)
                        target_history['targets'][target_key]['history'] = revised_merged_history_l
                except:
                    logger.error(f'error when processing the following row: {row}')
                    print(traceback.format_exc())

        for target_key, target_value in target_history['targets'].items():
            curr_history = target_value['history']
            curr_anchor_qids = target_value['anchor_qids']
            curr_anchor_qid_hist = target_history['anchor_key'][0]
            curr_property_id_hist = target_history['anchor_key'][1]
            curr_target_value = target_key
            curr_target_type = target_value['type']

            writer.writerow([curr_anchor_qid_hist,
                             curr_property_id_hist,
                             curr_target_value,
                             curr_target_type,
                             ','.join(curr_history),
                             curr_anchor_qids])

        output_file.flush()
        output_file.close()
