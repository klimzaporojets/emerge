import os
import json
import logging
import traceback
from typing import Dict, List, Any

from dataset.emerge.loader.loader_config import EMERGELoaderConfig

logger = logging.getLogger(__name__)


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


def get_page_id_func(timestamp: int,
                     page_title: str,
                     wikipedia_page_title_to_wikipedia_page_id: Dict,
                     page_title_changes,
                     wikipedia_page_id_to_wikipedia_page_title: Dict,
                     wikipedia_page_id_to_redirected_page_id: Dict,
                     wikipedia_page_id_to_wikidata_qid: Dict
                     ):
    to_ret_page_id = None
    if wikipedia_page_title_to_wikipedia_page_id is None:
        return None
    if page_title not in wikipedia_page_title_to_wikipedia_page_id:
        to_ret_page_id = get_page_id_of_most_recent_title(
            page_title,
            int(timestamp),
            page_title_changes,
            page_id_to_page_title=wikipedia_page_id_to_wikipedia_page_title,
            page_id=None,
            do_not_return_disambiguations=True
        )
    else:
        to_ret_page_id = wikipedia_page_title_to_wikipedia_page_id[page_title]
    to_ret_page_qid = None
    to_ret_page_title = None
    if to_ret_page_id is not None:
        nr_redirects_found = 0

        try:
            while to_ret_page_id in wikipedia_page_id_to_redirected_page_id:
                to_ret_page_id = wikipedia_page_id_to_redirected_page_id[to_ret_page_id]
                nr_redirects_found += 1
                if nr_redirects_found > 100:
                    break
            # assert target_page_id not in wikipedia_page_id_to_redirected_page_id
            if to_ret_page_id in wikipedia_page_id_to_redirected_page_id:
                logger.error('following target_page_id in wikipedia_page_id_to_redirected_page_id: '
                             f'{to_ret_page_id} with value of '
                             f'{wikipedia_page_id_to_redirected_page_id[to_ret_page_id]}')
        except Exception as e:
            logger.error(f'An error occurred: {e}')
            traceback.print_exc()

        if to_ret_page_id in wikipedia_page_id_to_wikidata_qid:
            to_ret_page_qid = wikipedia_page_id_to_wikidata_qid[to_ret_page_id]
        else:
            return {
                'page_id': None,
                'page_qid': None,
                'page_title': None
            }
        if to_ret_page_id in wikipedia_page_id_to_wikipedia_page_title:
            to_ret_page_title = wikipedia_page_id_to_wikipedia_page_title[to_ret_page_id]
        else:
            logger.warning(f'to_ret_page_id ({to_ret_page_id}) not in wikipedia_page_id_to_wikipedia_page_title '
                           f'when calling get_page_id_func with timestamp {timestamp} and '
                           f'page_title {page_title} , extracted qid: {to_ret_page_qid}')
            return {
                'page_id': None,
                'page_qid': None,
                'page_title': None
            }

    to_ret = {
        'page_id': to_ret_page_id,
        'page_qid': to_ret_page_qid,
        'page_title': page_title,
        'page_title_normalized': to_ret_page_title
    }
    return to_ret


def obtain_potential_qid(mention_text,
                         wikipedia_page_id_to_wikidata_qid,
                         page_title_changes,
                         wikipedia_page_id_to_wikipedia_page_title,
                         wikipedia_page_id_to_redirected_page_id,
                         wikipedia_page_title_to_wikipedia_page_id
                         ):
    all_lowercase = str(mention_text).lower()
    all_uppercase = str(mention_text).title()
    first_uppercase = str(mention_text).capitalize()

    cand_qid = (
        get_page_id_func(
            1746597701,
            all_uppercase,
            wikipedia_page_id_to_wikidata_qid=wikipedia_page_id_to_wikidata_qid,
            page_title_changes=page_title_changes,
            wikipedia_page_id_to_wikipedia_page_title=wikipedia_page_id_to_wikipedia_page_title,
            wikipedia_page_id_to_redirected_page_id=wikipedia_page_id_to_redirected_page_id,
            wikipedia_page_title_to_wikipedia_page_id=wikipedia_page_title_to_wikipedia_page_id
        )
    )
    if cand_qid is not None and cand_qid['page_qid'] is not None:
        return cand_qid['page_qid']

    cand_qid = get_page_id_func(
        1746597701,
        first_uppercase,
        wikipedia_page_id_to_wikidata_qid=wikipedia_page_id_to_wikidata_qid,
        page_title_changes=page_title_changes,
        wikipedia_page_id_to_wikipedia_page_title=wikipedia_page_id_to_wikipedia_page_title,
        wikipedia_page_id_to_redirected_page_id=wikipedia_page_id_to_redirected_page_id,
        wikipedia_page_title_to_wikipedia_page_id=wikipedia_page_title_to_wikipedia_page_id
    )
    if cand_qid is not None and cand_qid['page_qid'] is not None:
        return cand_qid['page_qid']

    cand_qid = get_page_id_func(
        1746597701,
        all_lowercase,
        wikipedia_page_id_to_wikidata_qid=wikipedia_page_id_to_wikidata_qid,
        page_title_changes=page_title_changes,
        wikipedia_page_id_to_wikipedia_page_title=wikipedia_page_id_to_wikipedia_page_title,
        wikipedia_page_id_to_redirected_page_id=wikipedia_page_id_to_redirected_page_id,
        wikipedia_page_title_to_wikipedia_page_id=wikipedia_page_title_to_wikipedia_page_id
    )
    if cand_qid is not None and cand_qid['page_qid'] is not None:
        return cand_qid['page_qid']
    return '---NME---'


class EMERGEDatasetLoader:
    """
    Loads the S14 dataset from disk and returns raw parsed records.

    This class is intentionally a thin wrapper around the original
    loading logic, preserving behavior.
    """

    def __init__(
            self,
            config: EMERGELoaderConfig,
            wikipedia_page_id_to_wikidata_qid,
            page_title_changes,
            wikipedia_page_id_to_wikipedia_page_title,
            wikipedia_page_id_to_redirected_page_id,
            wikipedia_page_title_to_wikipedia_page_id,
    ):
        self.config: EMERGELoaderConfig = config
        self.wikipedia_page_id_to_wikidata_qid = wikipedia_page_id_to_wikidata_qid
        self.page_title_changes = page_title_changes
        self.wikipedia_page_id_to_wikipedia_page_title = wikipedia_page_id_to_wikipedia_page_title
        self.wikipedia_page_id_to_redirected_page_id = wikipedia_page_id_to_redirected_page_id
        self.wikipedia_page_title_to_wikipedia_page_id = wikipedia_page_title_to_wikipedia_page_id

    def load(self) -> List[Dict[str, Any]]:
        dataset_loaded = []

        nr_canonicalized_triples = 0
        tot_nr_non_canonicalized_triples = 0
        aliases_predictions = set()

        for root, dirs, files in os.walk(self.config.input_dataset_path):
            print(f'Current Directory: {root}')
            for curr_file in files:

                if tot_nr_non_canonicalized_triples > 0:
                    logger.info(
                        f'File to be processed: {curr_file} '
                        f'nr_canonicalized_triples: {nr_canonicalized_triples} '
                        f'tot_nr_non_canonicalized_triples: {tot_nr_non_canonicalized_triples} ; '
                        f'fraction: {nr_canonicalized_triples / tot_nr_non_canonicalized_triples} ; '
                        f'size of dataset_loaded: {len(dataset_loaded)} '
                        f'aliases: {aliases_predictions}'
                    )

                if not curr_file.endswith('.jsonl'):
                    logger.info(f'ignoring {curr_file}')
                    continue

                logger.info(f'reading dataset file {curr_file}')

                curr_path = os.path.join(root, curr_file)
                with open(curr_path, 'rt', encoding='utf-8') as infile:
                    for curr_line in infile:
                        parsed_line = json.loads(curr_line)

                        if 'passage_timestamp' not in parsed_line:
                            parsed_line['passage_timestamp'] = parsed_line.pop('revision_timestamp')

                        if 'passage_revision_id' not in parsed_line:
                            parsed_line['passage_revision_id'] = parsed_line.pop('revision_id')

                        if self.config.should_add_predictions and 'predictions' in parsed_line:
                            predictions = parsed_line['predictions']

                            for curr_model_name, curr_predictions in predictions.items():
                                curr_predictions['canonicalized_kg_triples'] = list()
                                aliases_predictions.add(curr_predictions['model_type'])

                                curr_model_type = curr_predictions['model_type'].lower()

                                # curr_predicted_triples_actions = ['predicted_triples']

                                if curr_model_type == 'llm-tool':
                                    continue

                                # for curr_predicted_triple_action in curr_predicted_triples_actions:
                                pred_triples = curr_predictions['predicted_triples']
                                # pred_triples = curr_predictions[curr_predicted_triple_action]


                                if curr_predictions['model_type'].lower() == 'edc':
                                    pred_triples = (
                                            pred_triples +
                                            curr_predictions['predicted_triples_entities_to_kg']
                                    )

                                for curr_predicted_triple in pred_triples:
                                    if curr_predictions['model_type'].lower() == 'edc':
                                        if (
                                                curr_predicted_triple['extracted_relation'][2] == 'ADD' or
                                                curr_predicted_triple['extracted_relation'][2] == 'DEPRECATE'
                                        ):
                                            continue

                                    canonicalized_predicted_triple = (
                                        curr_predicted_triple['triple_qids'].copy()
                                    )

                                    mention_text_head = (
                                        str(curr_predicted_triple['extracted_relation'][0])
                                        .replace('_', ' ')
                                    )
                                    mention_text_tail = (
                                        str(curr_predicted_triple['extracted_relation'][2])
                                        .replace('_', ' ')
                                    )

                                    if curr_predictions['model_type'] in {'relik', 'edc'}:
                                        assert canonicalized_predicted_triple[1].startswith('P')

                                        if (
                                                (not canonicalized_predicted_triple[0].startswith('Q') or
                                                 not canonicalized_predicted_triple[2].startswith('Q')) and
                                                curr_predictions['model_type'] == 'edc'
                                        ):
                                            tot_nr_non_canonicalized_triples += 1

                                        if not canonicalized_predicted_triple[0].startswith('Q'):
                                            qid_head = obtain_potential_qid(
                                                mention_text=mention_text_head,
                                                wikipedia_page_id_to_wikidata_qid=self.wikipedia_page_id_to_wikidata_qid,
                                                page_title_changes=self.page_title_changes,
                                                wikipedia_page_id_to_wikipedia_page_title=self.wikipedia_page_id_to_wikipedia_page_title,
                                                wikipedia_page_id_to_redirected_page_id=self.wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id=self.wikipedia_page_title_to_wikipedia_page_id,
                                            )
                                            if qid_head is not None:
                                                canonicalized_predicted_triple[0] = qid_head

                                        if not canonicalized_predicted_triple[2].startswith('Q'):
                                            qid_tail = obtain_potential_qid(
                                                mention_text=mention_text_tail,
                                                wikipedia_page_id_to_wikidata_qid=self.wikipedia_page_id_to_wikidata_qid,
                                                page_title_changes=self.page_title_changes,
                                                wikipedia_page_id_to_wikipedia_page_title=self.wikipedia_page_id_to_wikipedia_page_title,
                                                wikipedia_page_id_to_redirected_page_id=self.wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id=self.wikipedia_page_title_to_wikipedia_page_id,
                                            )
                                            if qid_tail is not None:
                                                canonicalized_predicted_triple[2] = qid_tail

                                        if (
                                                canonicalized_predicted_triple[0].startswith('Q') and
                                                canonicalized_predicted_triple[2].startswith('Q') and
                                                canonicalized_predicted_triple[1].startswith('P')
                                        ):
                                            curr_predictions['canonicalized_kg_triples'].append(
                                                canonicalized_predicted_triple
                                            )

                                            if (
                                                    not curr_predicted_triple['triple_qids'][0].startswith('Q') and
                                                    curr_predictions['model_type'] == 'edc'
                                            ):
                                                nr_canonicalized_triples += 1

                        dataset_loaded.append(parsed_line)

        print('---DONE---')
        return dataset_loaded

    def load_old(self) -> List[Dict[str, Any]]:
        dataset_loaded = []

        nr_canonicalized_triples = 0
        tot_nr_non_canonicalized_triples = 0
        aliases_predictions = set()

        for root, dirs, files in os.walk(self.config.input_dataset_path):
            print(f'Current Directory: {root}')
            for curr_file in files:

                if tot_nr_non_canonicalized_triples > 0:
                    logger.info(
                        f'File to be processed: {curr_file} '
                        f'nr_canonicalized_triples: {nr_canonicalized_triples} '
                        f'tot_nr_non_canonicalized_triples: {tot_nr_non_canonicalized_triples} ; '
                        f'fraction: {nr_canonicalized_triples / tot_nr_non_canonicalized_triples} ; '
                        f'size of dataset_loaded: {len(dataset_loaded)} '
                        f'aliases: {aliases_predictions}'
                    )

                if not curr_file.endswith('.jsonl'):
                    logger.info(f'ignoring {curr_file}')
                    continue

                logger.info(f'reading dataset file {curr_file}')

                curr_path = os.path.join(root, curr_file)
                with open(curr_path, 'rt', encoding='utf-8') as infile:
                    for curr_line in infile:
                        parsed_line = json.loads(curr_line)

                        if 'passage_timestamp' not in parsed_line:
                            parsed_line['passage_timestamp'] = parsed_line.pop('revision_timestamp')

                        if 'passage_revision_id' not in parsed_line:
                            parsed_line['passage_revision_id'] = parsed_line.pop('revision_id')

                        if self.config.should_add_predictions and 'predictions' in parsed_line:
                            predictions = parsed_line['predictions']

                            for curr_model_name, curr_predictions in predictions.items():
                                curr_predictions['canonicalized_kg_triples'] = list()
                                aliases_predictions.add(curr_predictions['model_type'])

                                pred_triples = curr_predictions['predicted_triples']

                                if curr_predictions['model_type'].lower() == 'edc':
                                    pred_triples = (
                                            pred_triples +
                                            curr_predictions['predicted_triples_entities_to_kg']
                                    )

                                for curr_predicted_triple in pred_triples:
                                    if curr_predictions['model_type'].lower() == 'edc':
                                        if (
                                                curr_predicted_triple['extracted_relation'][2] == 'ADD' or
                                                curr_predicted_triple['extracted_relation'][2] == 'DEPRECATE'
                                        ):
                                            continue

                                    canonicalized_predicted_triple = (
                                        curr_predicted_triple['triple_qids'].copy()
                                    )

                                    mention_text_head = (
                                        str(curr_predicted_triple['extracted_relation'][0])
                                        .replace('_', ' ')
                                    )
                                    mention_text_tail = (
                                        str(curr_predicted_triple['extracted_relation'][2])
                                        .replace('_', ' ')
                                    )

                                    if curr_predictions['model_type'] in {'relik', 'edc'}:
                                        assert canonicalized_predicted_triple[1].startswith('P')

                                        if (
                                                (not canonicalized_predicted_triple[0].startswith('Q') or
                                                 not canonicalized_predicted_triple[2].startswith('Q')) and
                                                curr_predictions['model_type'] == 'edc'
                                        ):
                                            tot_nr_non_canonicalized_triples += 1

                                        if not canonicalized_predicted_triple[0].startswith('Q'):
                                            qid_head = obtain_potential_qid(
                                                mention_text=mention_text_head,
                                                wikipedia_page_id_to_wikidata_qid=self.wikipedia_page_id_to_wikidata_qid,
                                                page_title_changes=self.page_title_changes,
                                                wikipedia_page_id_to_wikipedia_page_title=self.wikipedia_page_id_to_wikipedia_page_title,
                                                wikipedia_page_id_to_redirected_page_id=self.wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id=self.wikipedia_page_title_to_wikipedia_page_id,
                                            )
                                            if qid_head is not None:
                                                canonicalized_predicted_triple[0] = qid_head

                                        if not canonicalized_predicted_triple[2].startswith('Q'):
                                            qid_tail = obtain_potential_qid(
                                                mention_text=mention_text_tail,
                                                wikipedia_page_id_to_wikidata_qid=self.wikipedia_page_id_to_wikidata_qid,
                                                page_title_changes=self.page_title_changes,
                                                wikipedia_page_id_to_wikipedia_page_title=self.wikipedia_page_id_to_wikipedia_page_title,
                                                wikipedia_page_id_to_redirected_page_id=self.wikipedia_page_id_to_redirected_page_id,
                                                wikipedia_page_title_to_wikipedia_page_id=self.wikipedia_page_title_to_wikipedia_page_id,
                                            )
                                            if qid_tail is not None:
                                                canonicalized_predicted_triple[2] = qid_tail

                                        if (
                                                canonicalized_predicted_triple[0].startswith('Q') and
                                                canonicalized_predicted_triple[2].startswith('Q') and
                                                canonicalized_predicted_triple[1].startswith('P')
                                        ):
                                            curr_predictions['canonicalized_kg_triples'].append(
                                                canonicalized_predicted_triple
                                            )

                                            if (
                                                    not curr_predicted_triple['triple_qids'][0].startswith('Q') and
                                                    curr_predictions['model_type'] == 'edc'
                                            ):
                                                nr_canonicalized_triples += 1

                        dataset_loaded.append(parsed_line)

        print('---DONE---')
        return dataset_loaded
