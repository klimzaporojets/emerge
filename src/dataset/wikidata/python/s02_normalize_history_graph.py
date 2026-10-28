import argparse
import csv
import dataclasses
import json
import logging
import os
import sys
import time

from dataset.wikidata.python.misc.history_tsv import (ITEM_ID, OTHER_ENTITY_ID, MalformedHistoryRecord, ReadStats,
                                                      iter_raw_history_records, sanitize_value, split_history)
from dataset.wikidata.python.misc.load_wiki_sql_tables import load_wikidata_redirect, load_wikidata_page_id_to_qid

INPUT_FORMATS = ('raw_extraction', 'normalized')   # pass 1 reads the extractor output, pass 2 its own sorted output
MAX_NR_REDIRECTS = 50

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
                               stability_span, timestamp_precision):
    # stability_span used to be read as a global that exists only when this module runs as a script
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


def iter_history_rows(file_path: str, input_format: str, stats: ReadStats):
    """Yield (line number, [head, property, value, value_type, history]) of one input file.

    raw_extraction: the step-01 extractor output (no quoting at all) -> misc.history_tsv, which re-assembles statements
    whose value contains a line break and never interprets quotes.
    normalized: this script's own output (csv.writer, values sanitised) -> csv.reader; a statement must be exactly one
    physical line with 6 fields, otherwise a quote has swallowed lines and the run stops.
    """
    if input_format == 'raw_extraction':
        yield from iter_raw_history_records(file_path, stats)
        return
    with open(file_path, 'rt', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile, delimiter='\t')
        prev_line_num = 0
        for row in reader:
            if reader.line_num != prev_line_num + 1 or len(row) != 6:
                raise MalformedHistoryRecord(f'{file_path}:{prev_line_num + 1}: expected one line with 6 fields, got '
                                             f'{len(row)} fields over {reader.line_num - prev_line_num} lines: '
                                             f'{str(row)[:300]}')
            prev_line_num = reader.line_num
            stats.nr_records += 1
            yield prev_line_num, row[:5]


def write_targets(writer, target_history):
    anchor_qid, property_id = target_history['anchor_key']
    for target_key, target_value in target_history['targets'].items():
        curr_target_value = sanitize_value(target_key) if isinstance(target_key, str) else target_key
        writer.writerow([anchor_qid,
                         property_id,
                         curr_target_value,
                         target_value['type'],
                         ','.join(target_value['history']),
                         target_value['anchor_qids']])


def process_history_files(path_in, path_output, stats_dir, input_format, wikidata_qid_to_redirected_qid,
                          stability_span, timestamp_precision):
    """Normalizes every history file of `path_in` into `path_output` and returns (read statistics of ALL files, number
    of files, number skipped as already done). Each file's statistics are saved in `stats_dir/<file>.json` (outside
    `path_output`, which the sort step concatenates), so a resumed run still reports every file. A file is done only
    when both its output and its statistics exist; otherwise it is (re)processed (both are written atomically)."""
    os.makedirs(path_output, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    history_files = sorted(f for f in os.listdir(path_in) if not f.endswith('.partial'))
    logger.info(f'processing {len(history_files)} history files ({input_format}) from {path_in}')
    total, nr_skipped, start_time = ReadStats(), 0, time.time()
    for idx_hist_file, curr_history_file in enumerate(history_files):
        output_path = os.path.join(path_output, curr_history_file)
        stats_path = os.path.join(stats_dir, f'{curr_history_file}.json')
        if os.path.exists(output_path) and os.path.exists(stats_path):
            with open(stats_path, 'rt') as infile:
                total.merge(ReadStats(**json.load(infile)))
            nr_skipped += 1
            continue
        if idx_hist_file % 50 == 0 or idx_hist_file == len(history_files) - 1:
            logger.info(f'[{idx_hist_file + 1}/{len(history_files)}] reading {curr_history_file} '
                        f'(elapsed: {time.time() - start_time:.0f}s, {total.nr_records:,} statements so far)')
        file_stats = ReadStats()
        normalize_history_file(os.path.join(path_in, curr_history_file), output_path, input_format,
                               wikidata_qid_to_redirected_qid, stability_span, timestamp_precision, file_stats)
        with open(f'{stats_path}.partial', 'wt') as outfile:
            json.dump(dataclasses.asdict(file_stats), outfile)
        os.replace(f'{stats_path}.partial', stats_path)
        total.merge(file_stats)
    return total, len(history_files), nr_skipped


def resolve_redirect(qid, wikidata_qid_to_redirected_qid, row):
    nr_redirects_made = 0
    while nr_redirects_made < MAX_NR_REDIRECTS and qid in wikidata_qid_to_redirected_qid:
        qid = wikidata_qid_to_redirected_qid[qid]
        nr_redirects_made += 1
    if qid in wikidata_qid_to_redirected_qid:
        logger.error(f'Something wrong, can not find the redirected qid after {nr_redirects_made} retries '
                     f'for the following line: {row}')
    return qid


def normalize_history_file(file_path, output_path, input_format, wikidata_qid_to_redirected_qid, stability_span,
                           timestamp_precision, stats: ReadStats):
    """Resolves redirects, removes edits reverted within `stability_span` hours and merges the histories of the
    same (subject, property, target) within consecutive rows of one file. Writes to `<output_path>.partial` and renames
    it when the file is complete (a restart skips finished files)."""
    target_history = {'targets': dict(), 'anchor_key': None}
    partial_path = f'{output_path}.partial'
    with open(partial_path, 'wt', encoding='utf-8', newline='') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        for line_no, row in iter_history_rows(file_path, input_format, stats):
            anchor_qid = int(row[0][1:]) if row[0].startswith('Q') else int(row[0])
            property_id = row[1]
            tail_type = row[3]
            if anchor_qid in wikidata_qid_to_redirected_qid:
                anchor_qid = resolve_redirect(anchor_qid, wikidata_qid_to_redirected_qid, row)

            anchor_key = (anchor_qid, property_id)
            if target_history['anchor_key'] is None:
                target_history['anchor_key'] = anchor_key
            if anchor_key != target_history['anchor_key']:
                write_targets(writer, target_history)
                target_history['anchor_key'] = anchor_key
                target_history['targets'] = dict()

            if tail_type == 'wikibase-entityid':
                if not ITEM_ID.fullmatch(row[2]):
                    if not OTHER_ENTITY_ID.fullmatch(row[2]):
                        raise MalformedHistoryRecord(f'{file_path}:{line_no}: entity value that is not an entity id: '
                                                     f'{row[:4]}')
                    stats.add_non_item_entity_value(row[2], f'{file_path}:{line_no}')
                    continue
                curr_target_value = int(row[2][1:]) if row[2].startswith('Q') else int(row[2])
                if curr_target_value in wikidata_qid_to_redirected_qid:
                    curr_target_value = resolve_redirect(curr_target_value, wikidata_qid_to_redirected_qid, row)
            else:
                curr_target_value = row[2]

            history_actions, history_qualifiers = split_history(row[4], stats, f'{file_path}:{line_no}')
            history_parsed = [(int(elem[:elem.index(':')]), elem[elem.index(':') + 1:]) for elem in history_actions]
            target_key = curr_target_value
            if target_key not in target_history['targets']:
                history_parsed = revise_history(history_parsed, stability_span, timestamp_precision)
                if len(history_parsed) > 0:
                    target_history['targets'][target_key] = {
                        'anchor_qids': {anchor_qid},
                        'history': [f'{ts}:{action}' for ts, action in history_parsed] + history_qualifiers,
                        'type': tail_type}
            else:
                target_history['targets'][target_key]['anchor_qids'].add(anchor_qid)
                history1 = target_history['targets'][target_key]['history']
                history1_actions = [(int(elem[:elem.index(':')]), elem[elem.index(':') + 1:])
                                    for elem in history1 if not elem.startswith('P')]
                history1_qualifiers = [elem for elem in history1 if elem.startswith('P')]
                target_history['targets'][target_key]['history'] = merge_two_target_histories(
                    history1_actions, history_parsed, history1_qualifiers, history_qualifiers,
                    stability_span, timestamp_precision)
        if target_history['anchor_key'] is not None:
            write_targets(writer, target_history)
    os.replace(partial_path, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True, type=str,
                        help='The config file that contains all the parameters')
    parser.add_argument('--debug_max_size_tables', required=False, type=int, default=-1,
                        help='Maximum number of rows that are being loaded from sql tables to not '
                             'run out of memory on local environment, if -1 loads everything.')
    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))
    input_format = config['input_format']
    if input_format not in INPUT_FORMATS:
        raise ValueError(f'input_format must be one of {INPUT_FORMATS}, got {input_format!r}')
    debug_max_size_tables = int(args.debug_max_size_tables)

    caches_dir = config['caches_dir']
    os.makedirs(caches_dir, exist_ok=True)
    output_dir_data = config['output_dir_data']
    os.makedirs(output_dir_data, exist_ok=True)
    import git

    repo = git.Repo(search_parent_directories=True)
    with open(os.path.join(output_dir_data, 'commit_hash_content.txt'), 'wt') as outfile:
        outfile.write(repo.head.object.hexsha)

    path_cache_redirect = os.path.join(caches_dir, 'redirect.pickle')
    if debug_max_size_tables > -1:
        path_cache_wikidata_page_id_to_qid = os.path.join(caches_dir,
                                                          f'page_id_to_qid_top{debug_max_size_tables}.pickle')
    else:
        path_cache_wikidata_page_id_to_qid = os.path.join(caches_dir, 'page_id_to_qid.pickle')
    logger.info('loading wikidata_page_id_to_qid')
    wikidata_page_id_to_qid = load_wikidata_page_id_to_qid(path_cache_wikidata_page_id_to_qid,
                                                           config['path_wikidata_page'],
                                                           max_nr_rows=debug_max_size_tables)
    logger.info('loading redirect_qid')
    wikidata_qid_to_redirected_qid = load_wikidata_redirect(path_cache_redirect, wikidata_page_id_to_qid,
                                                            config['path_redirects'])

    path_extracted_history_triples = config['path_extracted_history_triples']
    stability_span = config['stability_span']
    timestamp_precision = config['timestamp_precision']
    path_output = os.path.join(output_dir_data, 's02a_generated_history_triples_filtered')

    # big fields are legitimate (long histories); a swallowed block of lines is caught by the one-line check
    csv.field_size_limit(sys.maxsize)
    stats, nr_files, nr_skipped = process_history_files(
        path_extracted_history_triples, path_output, os.path.join(output_dir_data, f's02_read_stats_{input_format}'),
        input_format, wikidata_qid_to_redirected_qid, stability_span, timestamp_precision)

    summary = dataclasses.asdict(stats)
    summary.update({'input_format': input_format, 'nr_files': nr_files, 'nr_files_skipped_already_done': nr_skipped})
    with open(os.path.join(output_dir_data, f's02_read_summary_{input_format}.json'), 'wt') as outfile:
        json.dump(summary, outfile, indent=2)
    logger.info(f'done: {json.dumps({k: v for k, v in summary.items() if k != "examples"})}')
    for example in stats.examples:
        logger.info(f'example: {example}')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    main()


