"""
It extends _v7 by also accepting the field 'matched_triples_entities_to_kg'
which contains the relations between any of the entities mentioned in text and the
deltas. Indicating what other relations have to be added to the KG.
From the API side, this has on-off functionality with the following parameters in config
json:
  "match_all_emerging_relations_from_head": true,
  "match_all_emerging_relations_from_tail": true
"""
import json
import logging
import multiprocessing.managers
import random
import re
import time
import traceback
import xml.sax
from datetime import datetime
from typing import List, Dict, Tuple, Set

import requests
from requests import Session
from urllib3 import Retry

from misc.cleaning import gross_clean, fine_clean, clean_random_stuff
from misc.wiki_parse import extract_mentions_with_positions_v2
# from s04_find_interesting_snippets import timestamp_to_date
from utils import wiki_logger

from misc.compiled_regexes import compiled_regexes

from requests.adapters import HTTPAdapter

from utils.wiki_utils import generate_short_hash

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=wiki_logger.logger_level)
logger = logging.getLogger(__name__)

def timestamp_to_date(timestamp: int, return_time=False):
    # Convert timestamp to datetime object
    dt_object = datetime.fromtimestamp(timestamp)

    # Format the datetime object to a string
    if return_time:
        formatted_date = dt_object.strftime('%Y-%m-%d - %H:%M')
    else:
        formatted_date = dt_object.strftime('%Y-%m-%d')

    # print(formatted_date)  # Output: 2021-10-01
    return formatted_date

def api_call_get_entities_ids(config, timestamp, page_titles, session: Session = None) -> \
        (List[str], List[str]):
    if len(config['api_ports_wiki_mapping']) == 1:
        api_port = config['api_ports_wiki_mapping'][0]
    else:
        idx_port = random.randint(0, len(config['api_ports_wiki_mapping']) - 1)
        api_port = config['api_ports_wiki_mapping'][idx_port]

    url = f'http://{config["api_host"]}:{api_port}/get_entities_ids'

    params = [
        ('timestamp', int(timestamp)),
    ]
    if len(page_titles) == 0:
        return list()

    for curr_page_title in page_titles:
        params.append(('page_titles', curr_page_title))

    if not session:
        response = requests.get(url, params=params)
    else:
        try:
            response = session.get(url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            # return response.json()  # Return the JSON response
        except Exception as e:
            logger.error(f"Request failed: {e}")
            traceback.print_exc()
            raise e
    # Check the response status and print the result
    if response.status_code == 200:
        result = response.json()
        return result['page_titles'], result['page_qids']
    else:
        raise RuntimeError(f'Failed to retrieve data using '
                           f'api_call_should_process_page invoking '
                           f'with parameters: {params}')


def api_call_get_temporal_intervals(config, session: Session = None) -> (
        (Dict[int, Tuple[int, int]], Dict[Tuple[int, int], int])):
    if len(config['api_ports_only_deltas']) == 1:
        api_port = config['api_ports_only_deltas'][0]
    else:
        idx_port = random.randint(0, len(config['api_ports_only_deltas']) - 1)
        api_port = config['api_ports_only_deltas'][idx_port]

    #
    url = f'http://{config["api_host"]}:{api_port}/get_temporal_intervals'
    logger.debug(f'invoking api_call_get_temporal_intervals with no params')
    if not session:
        response = requests.get(url, params=None)
    else:
        try:
            response = session.get(url, params=None)
            response.raise_for_status()  # Raise an error for bad responses
            # return response.json()  # Return the JSON response
        except Exception as e:
            logger.error(f'Request failed: {e}')
            traceback.print_exc()
            raise e
    # logger.info(f'invoking api_call_get_temporal_intervals '
    #             f'response: {response}')
    # Check the response status and print the result
    if response.status_code == 200:
        json_response = response.json()
        logger.debug(f'json_response to api_call_get_temporal_intervals {json_response}')
        #
        interval_ids_to_interval: Dict[int, Tuple[int, int]]
        # intervals_to_interval_id: Dict[Tuple[int, int], int]
        #
        interval_ids_to_interval = json_response['interval_ids_to_interval']

        # intervals_to_interval_id = json_response['intervals_to_interval_id']
        interval_ids_to_interval = {
            int(curr_interval_id): curr_interval for
            curr_interval_id, curr_interval in
            interval_ids_to_interval.items()
        }
        return interval_ids_to_interval
    else:
        raise RuntimeError(f'Failed to retrieve data using '
                           f'api_call_get_temporal_intervals')


def api_call_should_process_page(page_id, config, session: Session = None):
    if len(config['api_ports_wiki_mapping']) == 1:
        api_port = config['api_ports_wiki_mapping'][0]
    else:
        idx_port = random.randint(0, len(config['api_ports_wiki_mapping']) - 1)
        api_port = config['api_ports_wiki_mapping'][idx_port]

    url = f'http://{config["api_host"]}:{api_port}/should_process_page'
    params = {
        'page_id': page_id
    }

    if not session:
        response = requests.get(url, params=params)
    else:
        try:
            response = session.get(url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            # return response.json()  # Return the JSON response
        except Exception as e:
            logger.error(f"Request failed: {e}")
            traceback.print_exc()
            raise e
            # return None
    # Check the response status and print the result
    if response.status_code == 200:
        result = response.json()
        return result['should_process_page'], result['page_qid']
    else:
        raise RuntimeError(f'Failed to retrieve data using '
                           f'api_call_should_process_page invoking '
                           f'with parameters: {params}')


def api_call_get_page_id(timestamp, page_title, config, session: Session = None):
    # raise NotImplementedError('api_call_get_page_id not implemented')
    if len(config['api_ports_wiki_mapping']) == 1:
        api_port = config['api_ports_wiki_mapping'][0]
    else:
        idx_port = random.randint(0, len(config['api_ports_wiki_mapping']) - 1)
        api_port = config['api_ports_wiki_mapping'][idx_port]

    # url = f'http://{config['api_host']}:{config['api_port']}/get_page_id'
    url = f'http://{config["api_host"]}:{api_port}/get_page_id'
    params = {
        'timestamp': int(timestamp),
        'page_title': page_title
    }
    logger.debug(f'invoking api_call_get_page_id with params: {params}')
    if not session:
        response = requests.get(url, params=params)
    else:
        try:
            response = session.get(url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            # return response.json()  # Return the JSON response
        except Exception as e:
            logger.error(f'Request failed: {e}')
            traceback.print_exc()
            raise e
    logger.debug(f'invoking api_call_get_page_id with params: {params} '
                 f'response: {response}')
    # Check the response status and print the result
    if response.status_code == 200:
        json_response = response.json()
        to_ret_page_id = json_response['page_id']
        to_ret_page_qid = json_response['page_qid']
        to_ret_page_title = json_response['page_title']
        return to_ret_page_qid, to_ret_page_id, to_ret_page_title
    else:
        raise RuntimeError(f'Failed to retrieve data using '
                           f'api_call_get_revision_ids_to_tail_ids invoking '
                           f'with parameters: {params}')


def api_call_get_entities_in_triples_from_deltas(timestamp, interval_ids,
                                                 page_titles,
                                                 page_qids,
                                                 config,
                                                 session: Session = None):
    # raise NotImplementedError('api_call_get_page_id not implemented')
    # url = f'http://{config['api_host']}:{config['api_port']}/get_page_id'

    if len(config['api_ports_only_deltas']) == 1:
        api_port = config['api_ports_only_deltas'][0]
    else:
        idx_port = random.randint(0, len(config['api_ports_only_deltas']) - 1)
        api_port = config['api_ports_only_deltas'][idx_port]
    # url = f'http://{config['api_host']}:{config['api_port']}/get_entities_in_triples_from_deltas'
    url = f'http://{config["api_host"]}:{api_port}/get_entities_in_triples_from_deltas'
    # params = {
    #     'timestamp': int(timestamp),
    #     'interval_id': interval_id,
    #     'page_titles': page_titles
    # }
    params = [
        ('timestamp', int(timestamp)),
    ]
    if len(page_titles) == 0:
        return list()

    assert len(interval_ids) > 0

    for curr_interval_id in interval_ids:
        params.append(('interval_ids', curr_interval_id))

    for curr_page_title in page_titles:
        params.append(('page_titles', curr_page_title))

    for curr_page_qid in page_qids:
        params.append(('page_qids', curr_page_qid))

    logger.debug(f'invoking api_call_get_entities_in_triples_from_deltas with params: {params}')
    if not session:
        response = requests.get(url, params=params)
    else:
        try:
            response = session.get(url, params=params)
            response.raise_for_status()  # Raise an error for bad responses
            # return response.json()  # Return the JSON response
        except Exception as e:
            logger.error(f'Request failed: {e}')
            # traceback.print_exc()
            traceback.print_exc()
            # raise e
            return list()
    logger.debug(f'invoking api_call_get_entities_in_triples_from_deltas with params: {params} '
                 f'response: {response}')
    # Check the response status and print the result
    if response.status_code == 200:
        json_responses = response.json()
        return json_responses
    else:
        logger.error(f'Failed to retrieve data using '
                     f'api_call_get_revision_ids_to_tail_ids invoking '
                     f'with parameters: {params}')
        return list()


class WikipediaHistorySnippetExtractorV8(xml.sax.ContentHandler):
    def __init__(self, filter_namespace,
                 convert_to_text_dictionary,
                 config, v_lock, start_time,
                 v_nr_parsed_articles,
                 v_max_recorded_text_length,
                 do_asserts=False,
                 filter_pages=None,
                 dry_run=False,
                 log_parsing_page_ids=[],
                 test_run=False):
        super().__init__()
        self.v_nr_parsed_articles = v_nr_parsed_articles
        self.v_max_recorded_text_length = v_max_recorded_text_length
        logger.debug('Init WikipediaHistoryReader')
        self.do_asserts = do_asserts
        self.last_revision_content: Dict = dict()
        self.nr_ignored_revisions = 0
        self.stack_elements: List = list()
        self.convert_through_api = config['convert_through_api']
        self.config = config
        #
        self.max_ratio_special_characters = config['max_ratio_special_characters']
        self.min_stability_span_in_secs = config['min_stability_span_in_secs']
        self.max_ignored_due_to_stability = config['max_ignored_due_to_stability']
        #
        # self.nr_tokens_around_mentions = config['nr_tokens_around_mentions']

        self.request_session1 = requests.Session()

        self.output_file_mentions_context = None

        self.filter_pages = filter_pages
        self.log_parsing_page_ids = set(log_parsing_page_ids)
        self.should_be_processed = True
        self.nr_revisions = 0
        self.convert_to_text_dictionary: multiprocessing.managers.DictProxy = convert_to_text_dictionary
        self.v_lock = v_lock
        self.dry_run = dry_run
        self.start_time = start_time

        self.text = ''
        self.length_text = ''
        self.field_title = ''
        self.field_page_id = ''
        self.field_page_qid = ''
        self.field_revision_id = ''
        self.field_comment = ''
        self.ns = ''
        self.processed_file = ''

        self.timestamp = ''
        self.revision_date: datetime = None
        self.revision_date_str = None
        self.tmstmp = -1
        self.interval_ids = list()

        self.creation_date = None
        self.field_creation_date = None

        self.filter_namespace = filter_namespace

        # in true when we are inside the elements, the idea is to avoid doing "in self.stack_elements" operation
        # which is O(n)
        self.active_page = False
        self.active_revision = False

        # related to movement of titles
        # self.page_change_of_titles = list()
        self.mention_contexts = list()
        self.nr_processed_revisions = 0
        self.old_page_title = None
        self.new_page_title = None

        self.move_patterns = ['(^|\s)moved (\[\[.*?\]\]) to (\[\[.*?\]\])',
                              '(^|\s)moved page (\[\[.*?\]\]) to (\[\[.*?\]\])']

        self.last_finished_page_time = time.time()
        # self.revision_id_to_page_ids = dict()
        # self.revision_id_to_timestamps = dict()

        session = requests.Session()

        # Define a retry strategy
        retry_strategy = Retry(
            total=10,  # Total number of retries
            backoff_factor=6
        )

        # Mount the adapter with the retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        self.paragraphs_in_page_history = set()
        self.json_responses_cache = dict()
        self.session = session

        # self.max_recorded_text_length = 0
        #
        if not test_run:
            self.interval_ids_to_interval = api_call_get_temporal_intervals(
                config=config,
                session=session
            )
            self.interval_ids_to_interval_str = dict()
            for curr_interval_id, curr_interval in self.interval_ids_to_interval.items():
                assert len(curr_interval) == 2
                interval_from = timestamp_to_date(curr_interval[0])
                interval_to = timestamp_to_date(curr_interval[1])
                self.interval_ids_to_interval_str[curr_interval_id] = [interval_from, interval_to]

    def getIntervalIds(self, timestamp):
        # logger.info(f'value of self.interval_ids_to_interval: '
        #             f'{self.interval_ids_to_interval}')
        int_tmstmp = int(timestamp)
        # logger.info(f'int_tmstmp: {int_tmstmp} and '
        #             f'self.interval_ids_to_interval: {self.interval_ids_to_interval}')
        to_ret_interval_ids = list()
        for curr_interval_id, curr_interval in self.interval_ids_to_interval.items():
            # logger.info(f'int_tmstmp: {int_tmstmp} and '
            #             f'curr_interval: {curr_interval}')
            if curr_interval[0] <= int_tmstmp <= curr_interval[1]:
                to_ret_interval_ids.append(curr_interval_id)
                # return curr_interval_id
        return to_ret_interval_ids

    def startDocument(self):
        pass

    def endDocument(self):
        pass

    def startElementNS(self, name, qname, attrs):
        pass

    def startElement(self, name, attributes: xml.sax.xmlreader.AttributesImpl):
        try:
            if not self.should_be_processed and name != 'page':
                return

            if name == 'ns':
                if self.do_asserts:
                    assert self.stack_elements == ['page']
            elif name == 'page':
                self.should_be_processed = True
                if self.do_asserts:
                    assert self.stack_elements == []
                self.nr_revisions = 0
                self.field_title = ''
                # self.redirTarget = None
                self.active_page = True
                self.revision_date = None
                self.creation_date = None
                self.field_creation_date = None
                self.revision_date_str = None
                self.nr_processed_revisions = 0
                self.tmstmp = -1
                self.interval_ids = list()

                self.ns = ''
                # self.revision_id_to_page_ids = dict()
                # self.revision_id_to_timestamps = dict()
                self.mention_contexts = list()
                self.paragraphs_in_page_history = set()
                self.json_responses_cache = dict()
                self.last_revision_content: Dict = dict()
                self.nr_ignored_revisions = 0
            elif name == 'revision':
                if self.do_asserts:
                    assert self.stack_elements == ['page']
                logger.debug(f'self.filter_pages is: {self.filter_pages}')
                if self.nr_revisions == 0:
                    with self.v_nr_parsed_articles.get_lock():
                        self.v_nr_parsed_articles.value += 1

                    logger.debug('invoking api_call_should_process_page with '
                                 f'self.field_page_id in {self.field_page_id}')
                    if self.field_page_id.strip() == '':
                        raise RuntimeError('For some reason self.field_page_id is empty: '
                                           f'{self.field_page_id}')
                    should_process_this_page, page_qid = \
                        api_call_should_process_page(self.field_page_id, self.config, self.session)

                    logger.debug('api_call_should_process_page results: '
                                 f'{should_process_this_page} - {page_qid}')

                    if should_process_this_page:
                        logger.debug(f'Detected page to be processed: {self.field_page_id}')
                        # rev_ids_data = (
                        #     api_call_get_revision_ids_data(self.field_page_id,
                        #                                    self.config,
                        #                                    self.session))
                        # logger.debug(f'rev_ids_data for {self.field_page_id}: '
                        #              f'{rev_ids_data}')
                        self.field_page_qid = page_qid
                        # self.revision_id_to_page_ids = rev_ids_data['revision_ids_to_page_ids']
                        # self.revision_id_to_timestamps = rev_ids_data['revision_ids_to_timestamps']
                    else:
                        # belongs to a page that should not be processed
                        logger.debug(f'Ignoring page: {self.field_page_id}')
                        self.should_be_processed = False
                        self.stack_elements = list()
                        self.active_page = False
                        self.active_revision = False

                if self.should_be_processed:
                    self.active_revision = True

                self.nr_revisions += 1
                self.text = ''
                self.length_text = 0
                # self.redirTarget = None
                self.revision_date = None
                self.tmstmp = -1
                self.interval_ids = list()

                # in case the revision is a change of title of the page
                self.old_page_title = None
                self.new_page_title = None

            elif name == 'title':
                if self.do_asserts:
                    assert self.stack_elements == ['page']
                    assert self.field_title == ''
                self.field_title = ''
            elif name == 'text':
                if self.do_asserts:
                    assert self.stack_elements == ['page', 'revision']
                self.text = ''
                self.length_text = 0
            elif name == 'id':
                if self.active_page and not self.active_revision:
                    self.field_page_id = ''
                    self.field_page_qid = ''
                elif self.active_page and self.active_revision and self.stack_elements[-1] == 'revision':
                    self.field_revision_id = ''
            elif name == 'timestamp':
                if self.active_page and self.active_revision:
                    self.timestamp = ''
            elif name == 'comment':
                self.field_comment = ''

            self.stack_elements.append(name)
        except Exception as e:
            # Print the full stack trace
            logging.error(f'An error occurred inside endElement: {e}')
            traceback.print_exc()

    def obtain_entities_from_paragraph(self, mention_links, mention_link_to_page_id,
                                       tmstmp) -> (
            Tuple)[Dict[str, List[str]], Dict]:

        set_entities_titles = set()
        set_entities_qids = set()
        set_entities_page_ids = set()
        for curr_mention_link in mention_links:
            # target_page_name_orig = curr_mention_link['target_wikipedia_title_orig']
            if curr_mention_link['target_wikipedia_title_orig'] not in mention_link_to_page_id:
                # revision_date = datetime.strptime(self.timestamp, '%Y-%m-%dT%H:%M:%SZ')
                # tmstmp = revision_date.timestamp()
                target_page_qid, target_page_id, target_page_name = \
                    api_call_get_page_id(tmstmp,
                                         curr_mention_link['target_wikipedia_title_orig'],
                                         self.config,
                                         self.session)
                mention_link_to_page_id[curr_mention_link['target_wikipedia_title_orig']] = \
                    (target_page_qid, target_page_id, target_page_name)
            else:
                target_page_qid, target_page_id, target_page_name = mention_link_to_page_id[
                    curr_mention_link['target_wikipedia_title_orig']]
            set_entities_titles.add(target_page_name)
            set_entities_page_ids.add(target_page_id)
            set_entities_qids.add(target_page_qid)
            logger.debug(f'From curr_mention_link {curr_mention_link} obtained '
                         f'target_page_qid: {target_page_qid} '
                         f'target_page_id: {target_page_id} '
                         f'target_page_name: {target_page_name} '
                         )
        entities_from_paragraph = {
            'entities_titles': list(set_entities_titles),
            'entities_qids': list(set_entities_qids),
            'entities_page_ids': list(set_entities_page_ids)
        }
        return entities_from_paragraph, mention_link_to_page_id

    def complete_with_coref_regex(self, curr_paragraph: str,
                                  mentions_in_page: Dict[str, str]):
        if len(mentions_in_page) == 0:
            return curr_paragraph
        # pattern = r'(?<!\[)(Barack Obama|Donald Trump|Aarhus University|Aarhus)(?!\])'
        # pattern = r'(?<!\[\[.*?)(Barack Obama)(?!.*?\]\])'
        # pattern = r'(?<!(\[\[))('
        # pattern = r'(?<!\[\[.*?)('
        # r'(?<!\[\[)(Barack Obama|Donald Trump)(?![^\[]*\]\])'
        pattern = r'(?<!\[\[)('
        #
        # # Define the replacement function
        # def replace_match(match):
        #     if match.group(0) == "Barack Obama":
        #         return "[Barack Obama|Barack_Obama]"
        #     elif match.group(0) == "Donald Trump":
        #         return "[Donald Trump|Trump]"
        #     elif match.group(0) == "Aarhus University":
        #         return "[Aarhus University|AU]"
        #     elif match.group(0) == "Aarhus":
        #         return "[Aarhus|Aarhus]"
        # mentions_in_page_escaped = dict()
        for idx, (curr_mention_in_page, curr_target_title) in enumerate(mentions_in_page.items()):
            escaped_mention = re.escape(curr_mention_in_page)
            # mentions_in_page_escaped[escaped_mention] = curr_target_title
            if idx > 0:
                pattern += '|'
            pattern += escaped_mention

        # pattern += ')(?!(\]\]))'
        pattern += ')(?![^\[]*\]\])'

        # pattern += ')(?!(\]\]))' this one works

        # pattern += r')(?!.*?\]\])'
        # pattern += r')(?!.*?\]\])'

        # pattern = r'(?<!\[\[.*?)(Barack Obama)(?!.*?\]\])'
        # (?<!\[\[.*?)(Fortunato|Kingpin|Marvel\ Comics)(?!.*?\]\])
        # (?<!\[\[)(Fortunato|Kingpin|Marvel\ Comics)(?!.*?\]\])
        # (?<!\[\[.*?)(Fortunato|Kingpin|Marvel\ Comics)(?!\]\])
        #
        def replace_match(match):
            matched = match.group(0)
            if matched in mentions_in_page:
                # return f'[{matched}|{mentions_in_page_escaped[matched]}]'
                return f'[[{mentions_in_page[matched].replace("_", " ")}|{matched}]]'

        # r'(?<!\[\[.*?)(Fortunato|Kingpin|Marvel Comics)(?!.*?\]\])'
        # print(f'pattern is: {pattern}')
        replaced_paragraph = re.sub(pattern, replace_match, curr_paragraph)

        # if replaced_paragraph != curr_paragraph:
        #     logger.info(f'======================PARAGRAPH CHANGED======================\n'
        #                 f'ORIGINAL: \n'
        #                 f'{curr_paragraph} \n'
        #                 f'REPLACED WITH: \n'
        #                 f'{replaced_paragraph}\n'
        #                 f'=============================================================\n'
        #                 )
        return replaced_paragraph

    def count_braces(self, s: str):
        count_open = s.count('{')
        count_close = s.count('}')
        return count_open + count_close

    def count_pipes(self, s: str):
        count_pipes = s.count('|')
        return count_pipes

    # def parseText(self, text: str, revision_id: int, field_title: str,
    #               field_page_id: int, field_page_qid: str,
    #               tmstmp: int, timestamp: str, interval_ids: List[int]):
    def parseText(self):
        text: str = self.last_revision_content['text']
        revision_id: int = self.last_revision_content['revision_id']
        field_title: str = self.last_revision_content['field_title']
        field_page_id: int = self.last_revision_content['field_page_id']
        field_page_qid: str = self.last_revision_content['field_page_qid']
        tmstmp: int = self.last_revision_content['tmstmp']
        timestamp: str = self.last_revision_content['timestamp']
        interval_ids: List[int] = self.last_revision_content['interval_ids']
        prev_last_revision_content: Dict = self.last_revision_content['prev_last_revision_content']

        # start_parse_text = time.time()
        #                         'revision_timestamp': tmstmp,
        #                         'revision_date': timestamp,
        #                         'anchor_title': field_title,
        #                         'anchor_page_id': field_page_id,
        #                         'anchor_page_qid': field_page_qid,
        # start_time = time.time()
        simple_cleaned_text = gross_clean(text=text.strip(),
                                          regexes=compiled_regexes,
                                          convert_through_api=self.convert_through_api,
                                          convert_to_text_dictionary=self.convert_to_text_dictionary,
                                          request_session1=self.request_session1,
                                          v_lock=self.v_lock)
        # logger.info(f'{(time.time() - start_time):.5f} sec for gross_clean')
        # start_time = time.time()
        simple_cleaned_stripped_code = fine_clean(simple_cleaned_text)
        # logger.info(f'{(time.time() - start_time):.5f} sec for fine_clean')

        content_length = len(simple_cleaned_stripped_code.split(' '))

        # break into paragraphs:
        paragraphs_text = simple_cleaned_stripped_code.split('\n\n')

        mentions_in_page = dict()

        # if the previous version of the page comes from outside the delta boundary
        # then add to paragraph_hashes, so we do not add the text coming from previous versions
        if prev_last_revision_content is not None and \
                len(prev_last_revision_content['interval_ids']) == 0:
            prev_text = prev_last_revision_content['text']
            prev_simple_cleaned_text = gross_clean(text=prev_text.strip(),
                                                   regexes=compiled_regexes,
                                                   convert_through_api=self.convert_through_api,
                                                   convert_to_text_dictionary=self.convert_to_text_dictionary,
                                                   request_session1=self.request_session1,
                                                   v_lock=self.v_lock)
            prev_simple_cleaned_stripped_code = fine_clean(prev_simple_cleaned_text)
            prev_paragraphs_text = prev_simple_cleaned_stripped_code.split('\n\n')
            for idx_prev_paragraph, curr_prev_paragraph in enumerate(prev_paragraphs_text):
                prev_paragraph_hash = generate_short_hash(curr_prev_paragraph, hash_length=256)
                # logger.info(f'adding curr_prev_paragraph: {curr_prev_paragraph}')
                self.paragraphs_in_page_history.add(prev_paragraph_hash)

        for idx_paragraph, curr_paragraph in enumerate(paragraphs_text):
            paragraph_hash = generate_short_hash(curr_paragraph, hash_length=256)

            # if curr_paragraph in self.paragraphs_in_page_history:
            if paragraph_hash in self.paragraphs_in_page_history:
                continue

            nr_braces = self.count_braces(curr_paragraph)
            nr_pipes = self.count_pipes(curr_paragraph)
            nr_special = nr_braces + nr_pipes
            nr_tokens = len(curr_paragraph.split(' '))
            ratio_tokens = 0.0
            if nr_tokens > 0:
                ratio_tokens = nr_special / nr_tokens

            # if ratio_tokens > 0.5:
            if ratio_tokens > self.max_ratio_special_characters:
                # logger.info(f'ignoring: {curr_paragraph}')
                continue

            # TODO - this can be done using hash of the paragraph
            # self.paragraphs_in_page_history.add(curr_paragraph)
            self.paragraphs_in_page_history.add(paragraph_hash)
            if curr_paragraph.strip() != '':
                logger.debug(f'----------- paragraph nr {idx_paragraph}')
                logger.debug(f'{idx_paragraph} - {curr_paragraph}')

            if self.config['coreference_regex']:
                curr_paragraph = self.complete_with_coref_regex(curr_paragraph,
                                                                mentions_in_page)

            start_time = time.time()
            # mention_links, tot_detected_mentions, tot_links_errors = \
            #     get_mentions_and_links(curr_paragraph,
            #                            content_length,
            #                            field_title,
            #                            compiled_regexes['compiled_mention_finder'],
            #                            compiled_regexes['compiled_country_in_link'],
            #                            get_span_pos=True)
            if curr_paragraph.strip().startswith('{{'):
                logger.debug(f'IGNORING: {curr_paragraph}')
                continue
            # curr_paragraph = ('lets talk something about the big [[Washington state court system#Superior Court|Washington State Superior Court]] '
            #                   'or maybe some [[United_States|United States]] other interesting topic')
            if 'File:' in curr_paragraph:
                continue
            if 'Image:' in curr_paragraph:
                continue
            if 'Category:' in curr_paragraph:
                continue
            if 'Wiktionary:' in curr_paragraph:
                continue

            curr_paragraph = clean_random_stuff(curr_paragraph)
            # Excerpt from the Wikipedia page describing [[entity title]]:
            curr_paragraph = \
                (f'Excerpt from the Wikipedia page describing [[{field_title}]]: ' +
                 curr_paragraph)
            # plain_text, mentions = extract_mentions_with_positions(curr_paragraph)
            plain_text, mentions = extract_mentions_with_positions_v2(
                curr_paragraph,
                compiled_mention_finder=compiled_regexes['compiled_mention_finder'],
                compiled_country_in_link=compiled_regexes['compiled_country_in_link'],
                source_title=field_title
            )
            # mention_links, tot_detected_mentions, tot_links_errors = \
            #     get_mentions_and_links(curr_paragraph,
            #                            content_length,
            #                            field_title,
            #                            compiled_regexes['compiled_mention_finder'],
            #                            compiled_regexes['compiled_country_in_link'],
            #                            get_span_pos=True)

            # logger.info(f'{(time.time() - start_time):.5f} sec for get_mentions_and_links')
            #             mentions.append(
            #                 {
            #                     'mention_text': mention_text,
            #                     'pos_start': current_position,
            #                     'pos_end': current_position + len(mention_text),
            #                     'target_entity': link_target
            #                 }
            #             )

            len_text = len(plain_text.strip().split(' '))
            if (len_text < self.config['min_passage_length'] or
                    len_text > self.config['max_passage_length']):
                logger.debug(f'IGNORING: {plain_text}')
                continue
            # logger.debug(
            #     f'======================BEGIN_PARAGRAPH====================== \n'
            #     f'curr_paragraph: {curr_paragraph} \n'
            #     f'---------------- \n'
            #     f'plain_text: {plain_text} \n'
            #     f'---------------- \n'
            #     f'mentions: {mentions} \n'
            #     # f'---------------- \n'
            #     # f'mention_links: {mention_links} \n'
            #     f'======================END_PARAGRAPH======================'
            # )
            # logger.info('==================================================')

            # mention_to_title = set()

            mentions_in_paragraph = set()

            for curr_mention_link in mentions:
                # mention_to_title.add((curr_mention_link['mention_text'],
                #                       curr_mention_link['target_entity']))
                target_page_name_orig = curr_mention_link['target_entity']
                mentions_in_paragraph.add(target_page_name_orig)
                if self.config['coreference_regex']:
                    mentions_in_page[curr_mention_link['mention_text']] = (
                        curr_mention_link)['target_entity']
            #                 {
            #                     'mention_text': mention_text,
            #                     'pos_start': current_position,
            #                     'pos_end': current_position + len(mention_text),
            #                     'target_entity': link_target
            #                 }
            # for curr_mention_link in mention_links:
            #     mention_to_title.add((curr_mention_link['anchor_mention_text'],
            #                           curr_mention_link['target_wikipedia_title_orig']))
            #     target_page_name_orig = curr_mention_link['target_wikipedia_title_orig']
            #     mentions_in_paragraph.add(target_page_name_orig)
            #     if self.config['coreference_regex']:
            #         mentions_in_page[curr_mention_link['anchor_mention_text']] = (
            #             curr_mention_link)['target_wikipedia_title_orig']

            # also adds the anchor page
            # anchor_title_to_add = field_title.replace(' ', '_')
            # mentions_in_paragraph.add(anchor_title_to_add)
            # logger.debug(f'mentions_in_paragraph for {anchor_title_to_add}: '
            #              f'{mentions_in_paragraph}')
            # if len(mentions_in_paragraph) <= 2:
            if len(mentions_in_paragraph) < self.config['min_nr_mentions_in_paragraph']:
                continue

            sorted_tuple_titles = tuple(sorted(mentions_in_paragraph))
            sorted_tuple_intervals = tuple(sorted(interval_ids))

            entry_cache_tuple = sorted_tuple_titles + sorted_tuple_intervals

            if entry_cache_tuple in self.json_responses_cache:
                # logger.info(f'found in cache: {entry_cache_tuple}')
                json_responses: List[Dict] = self.json_responses_cache[entry_cache_tuple]['json_responses']
                # page_titles: List[str] = self.json_responses_cache[entry_cache_tuple]['page_titles']
                # page_qids: List[str] = self.json_responses_cache[entry_cache_tuple]['page_qids']
                titles_to_qid = self.json_responses_cache[entry_cache_tuple]['titles_to_qid']
            else:
                start_time = time.time()

                # def api_call_get_entities_ids(config, timestamp, page_titles, session: Session = None) -> \
                #
                page_titles, page_qids = api_call_get_entities_ids(
                    config=self.config,
                    # timestamp=timestamp,
                    timestamp=tmstmp,
                    page_titles=mentions_in_paragraph,
                    session=self.session
                )
                titles_to_qid = dict()
                for curr_title, curr_qid in zip(page_titles, page_qids):
                    titles_to_qid[curr_title] = curr_qid
                json_responses: List[Dict] = (
                    api_call_get_entities_in_triples_from_deltas(
                        tmstmp,
                        interval_ids,
                        page_titles,
                        page_qids,
                        self.config,
                        self.session)
                )
                # logger.info(f'{(time.time() - start_time):.5f} sec for get_mentions_and_links')
                self.json_responses_cache[entry_cache_tuple] = {
                    'json_responses': json_responses,
                    # 'page_titles': page_titles,
                    # 'page_qids': page_qids,
                    'titles_to_qid': titles_to_qid
                }
            # api_call_get_entities_in_triples_from_deltas(self.tmstmp,
            #                                              self.interval_ids,
            #                                              mentions_in_paragraph,
            #                                              self.config,
            #                                              self.session))
            matched_entity_qids = set()
            for curr_mention_link in mentions:
                if curr_mention_link['target_entity'] in titles_to_qid:
                    curr_mention_link['qid'] = titles_to_qid[curr_mention_link['target_entity']]
                    matched_entity_qids.add(curr_mention_link['qid'])
                else:
                    curr_mention_link['qid'] = None
            for curr_json_response in json_responses:
                # TODO: here get found_entities

                # curr_json_response['matched_triples'] =
                # filtered_matched_triples = []
                # for curr_matched_triple in curr_json_response['matched_triples']:
                #     # head and tail have to be different
                #     if curr_matched_triple['triple_qid'][0]==curr_matched_triple['triple_qid'][2]:
                #         continue
                #     filtered_matched_triples.append(curr_matched_triple)
                # curr_json_response['matched_triples'] = filtered_matched_triples
                emerging_entities_in_triples = set()
                # emerging_entities_in_triples = set([curr_triple [] curr_json_response['matched_triples']])
                for curr_triple_to_check in curr_json_response['matched_triples']:
                    if curr_triple_to_check['emerging_head'] and \
                            curr_triple_to_check['triple_qid'][0] in matched_entity_qids:
                        emerging_entities_in_triples.add(curr_triple_to_check['triple_qid'][0])
                    if curr_triple_to_check['emerging_tail'] and \
                            curr_triple_to_check['triple_qid'][2] in matched_entity_qids:
                        emerging_entities_in_triples.add(curr_triple_to_check['triple_qid'][2])

                for curr_triple_to_check in curr_json_response['matched_triples_entities_to_kg']:
                    if curr_triple_to_check['emerging_head'] and \
                            curr_triple_to_check['triple_qid'][0] in matched_entity_qids:
                        emerging_entities_in_triples.add(curr_triple_to_check['triple_qid'][0])
                    if curr_triple_to_check['emerging_tail'] and \
                            curr_triple_to_check['triple_qid'][2] in matched_entity_qids:
                        emerging_entities_in_triples.add(curr_triple_to_check['triple_qid'][2])
                if len(emerging_entities_in_triples) > 0:
                    logger.debug(f'emerging_entities_in_triples_is: {emerging_entities_in_triples}')

                if (json_responses is not None and
                        ((len(curr_json_response['matched_triples']) >=
                          self.config['min_nr_triples_in_chunk']) or
                         (len(curr_json_response['matched_triples_entities_to_kg']) >=
                          self.config['min_nr_triples_in_chunk']) or
                         len(emerging_entities_in_triples) > 0  # at least one emerging entity associated to a triple
                        )):
                    to_append = {
                        'interval_id': curr_json_response['interval_id'],  # self.interval_ids,
                        'interval': self.interval_ids_to_interval_str[
                            curr_json_response['interval_id']],
                        # 'mention_to_title': list(mention_to_title),
                        'mentions': mentions,
                        # 'entities_titles': curr_json_response['titles'],
                        # 'entities_titles': page_titles,
                        # 'tot_nr_entities': curr_json_response['tot_nr_entities'],
                        # 'nr_matched_triples': curr_json_response['nr_matched_triples'],
                        # 'nr_matched_triples_entities_to_kg': len(curr_json_response['matched_triples_entities_to_kg']),
                        # 'nr_triples_with_emerging_heads': curr_json_response[
                        #     'nr_triples_with_emerging_heads'],
                        # 'nr_triples_with_emerging_tails': curr_json_response[
                        #     'nr_triples_with_emerging_tails'],
                        # 'nr_matched_entities': curr_json_response['nr_matched_entities'],
                        # 'found_entities': curr_json_response['entities'],
                        # 'found_entities': page_qids,
                        'matched_entities': curr_json_response['matched_entities'],
                        'matched_triples': curr_json_response['matched_triples'],
                        'matched_triples_entities_to_kg': curr_json_response['matched_triples_entities_to_kg'],
                        'revision_id': revision_id,
                        'revision_timestamp': tmstmp,
                        'revision_date': timestamp,
                        'anchor_title': field_title,
                        'anchor_page_id': field_page_id,
                        'anchor_page_qid': field_page_qid,
                        # 'revision_timestamp': self.tmstmp,
                        # 'revision_date': self.timestamp,
                        # 'anchor_title': self.field_title,
                        # 'anchor_page_id': self.field_page_id,
                        # 'anchor_page_qid': self.field_page_qid,
                        # 'chunk': curr_paragraph,
                        'chunk': plain_text,
                        'paragraph_idx': idx_paragraph
                    }

                    self.mention_contexts.append(to_append)

    def endElement(self, name):
        try:
            if not self.should_be_processed:
                return

            self.stack_elements.pop()

            if name == 'ns' and not self.filter_namespace(self.ns):
                self.should_be_processed = False
            if name == 'timestamp' and self.active_page and self.active_revision:
                self.revision_date = datetime.strptime(self.timestamp, '%Y-%m-%dT%H:%M:%SZ')
                self.tmstmp = self.revision_date.timestamp()
                self.interval_ids = self.getIntervalIds(self.tmstmp)
                if len(self.interval_ids) > 0:
                    logger.debug(f'invoking self.getIntervalId with {self.tmstmp} '
                                 f'and getting {self.interval_ids}')
            if (name == 'revision' and
                    self.filter_namespace(self.ns) and
                    (not self.dry_run)):
                logger.debug(f'{self.processed_file} '
                             f'obtaining revision: {self.field_revision_id} for '
                             f'page id {self.field_page_id}')
                revision_id = int(self.field_revision_id)
                logger.debug(f'{self.processed_file} '
                             f'revision obtained good: {revision_id} for page id '
                             f'{self.field_page_id}')

                # logger.debug(f'text_to_parse: {self.text.strip()}')
                #
                ######## BEGIN - FUNCTION OF PARSING BEGIN
                if (len(self.last_revision_content) > 0) and \
                        len(self.last_revision_content['interval_ids']) > 0 and \
                        not self.last_revision_content['text'].lower().startswith('#redirect ') and \
                        ((int(self.tmstmp) - self.last_revision_content['tmstmp'] >
                          self.min_stability_span_in_secs) or
                         self.nr_ignored_revisions > self.max_ignored_due_to_stability + 1):
                    # logger.info(f'{int(self.tmstmp) - self.last_revision_content["tmstmp"]} '
                    #             f'parsing_text: {self.last_revision_content}')
                    self.parseText(
                        # text=self.last_revision_content['text'],
                        # revision_id=self.last_revision_content['revision_id'],
                        # field_title=self.last_revision_content['field_title'],
                        # field_page_id=self.last_revision_content['field_page_id'],
                        # field_page_qid=self.last_revision_content['field_page_qid'],
                        # tmstmp=self.last_revision_content['tmstmp'],
                        # timestamp=self.last_revision_content['timestamp'],
                        # interval_ids=self.last_revision_content['interval_ids']
                    )
                    self.nr_ignored_revisions = 0
                self.nr_ignored_revisions += 1

                # after something is done above, then we go here and update the
                prev_last_revision_content = None
                if len(self.last_revision_content) > 0:
                    prev_last_revision_content = self.last_revision_content
                    prev_last_revision_content['prev_last_revision_content'] = None

                self.last_revision_content = {
                    'text': self.text.strip()[:self.config['max_length_article_in_chars']],
                    'revision_id': revision_id,
                    'field_title': self.field_title,
                    'field_page_id': int(self.field_page_id),
                    'field_page_qid': self.field_page_qid,
                    'tmstmp': int(self.tmstmp),
                    'timestamp': self.timestamp,
                    'interval_ids': self.interval_ids,
                    'prev_last_revision_content': prev_last_revision_content
                }
                ### uncomment to log the maximum size of snippets
                ltext = len(self.last_revision_content['text'])
                if ltext >= self.v_max_recorded_text_length.value:
                    with self.v_max_recorded_text_length.get_lock():
                        logger.info(f'=========================')
                        logger.info(f'field_title: {self.last_revision_content["field_title"]}')
                        logger.info(f'field_page_id: {self.last_revision_content["field_page_id"]}')
                        logger.info(f'field_page_qid: {self.last_revision_content["field_page_qid"]}')
                        logger.info(f'maximum recorded text length: {ltext:,}')
                        logger.info(f'maximum recorded text length in tokens: '
                                    f'{len(self.last_revision_content["text"].split(" ")):,}')
                        self.v_max_recorded_text_length.value = ltext
                ### END uncomment to log the maximum size of snippets

                # if ltext > self.max_recorded_text_length:
                #     logger.info(f'maximum recorded text length: {ltext:,}')
                #     self.max_recorded_text_length = ltext
                ######## END - FUNCTION OF PARSING ENDS
                #
                self.active_revision = False
            if name == 'page':
                self.active_page = False

                if self.filter_namespace(self.ns):
                    if self.should_be_processed:
                        ##########################################################################
                        # BEGIN - this parseText is needed since the one above on revision works
                        #   only for previous revision, this is a way to incorporate the last revision
                        #   of a page, if this is needed
                        if (not self.dry_run and
                                not self.last_revision_content['text'].lower().startswith('#redirect ')
                                and len(self.last_revision_content['interval_ids']) > 0):
                            self.parseText(
                                # text=self.last_revision_content['text'],
                                # revision_id=self.last_revision_content['revision_id'],
                                # field_title=self.last_revision_content['field_title'],
                                # field_page_id=self.last_revision_content['field_page_id'],
                                # field_page_qid=self.last_revision_content['field_page_qid'],
                                # tmstmp=self.last_revision_content['tmstmp'],
                                # timestamp=self.last_revision_content['timestamp'],
                                # interval_ids=self.last_revision_content['interval_ids']
                            )

                            # revision_id = int(self.field_revision_id)
                            # self.parseText(
                            #     text=self.text.strip(),
                            #     revision_id=revision_id,
                            #     field_title=self.field_title,
                            #     field_page_id=int(self.field_page_id),
                            #     field_page_qid=self.field_page_qid,
                            #     tmstmp=int(self.tmstmp),
                            #     timestamp=self.timestamp,
                            #     interval_ids=self.interval_ids
                            # )
                        # END - this parseText is needed since...
                        ##########################################################################

                        for curr_mention_context in self.mention_contexts:
                            try:
                                str_json = json.dumps(curr_mention_context)
                                logger.debug(f'writing str_json: {str_json}')
                                self.output_file_mentions_context.write(str_json + '\n')
                                # TODO: remove this flush if too slow
                                self.output_file_mentions_context.flush()
                            except Exception as e:
                                logger.error(f'an json.dumps error occurred with the following '
                                             f'curr_mention_context: {curr_mention_context}')
                                traceback.print_exc()

                        curr_time = time.time()
                        logger.debug(f'finished_page {self.field_page_id} in '
                                     f'{((curr_time - self.last_finished_page_time) / 60):.4f} mins')
                        self.last_finished_page_time = curr_time
                    else:
                        if len(self.mention_contexts) > 0:
                            logger.warning(f'if 1 for_some_reason did not process: {self.mention_contexts}')
                else:
                    if len(self.mention_contexts) > 0:
                        logger.warning(f'if 2 for_some_reason did not process: {self.mention_contexts} '
                                       f'namespace: {self.ns}')

        except Exception as e:
            # Print the full stack trace
            logging.error(f'An error occurred inside endElement: {e}')
            traceback.print_exc()

    def endElementNS(self, name, qname):
        pass

    def characters_field_title(self, content):
        self.field_title += content

    def characters_field_page_id(self, content):
        self.field_page_id += content
        logger.debug(f'starting_page {self.field_page_id}')

    def characters_field_revision_id(self, content):
        self.field_revision_id += content

    def characters_timestamp(self, content):
        self.timestamp += content

    def characters_field_comment(self, content):
        self.field_comment += content

    def characters_ns(self, content):
        self.ns += content

    def characters_text(self, content):
        if self.length_text > self.config['max_length_article_in_chars']:
            return
        self.text += content
        self.length_text += len(content)

    def characters(self, content):
        if self.nr_revisions >= 1 and not self.should_be_processed:
            return

        len_stack_elements = len(self.stack_elements)
        if len_stack_elements == 0:
            return

        assert content is not None
        stack_min_1 = self.stack_elements[-1]

        stack_min_2 = None
        if len_stack_elements > 1:
            stack_min_2 = self.stack_elements[-2]

        if stack_min_1 == 'ns':
            # self.ns += content
            self.characters_ns(content)
            logger.debug(f'Set ns to {self.ns}')

        if self.ns != '' and not self.filter_namespace(self.ns):
            return

        if stack_min_1 == 'title':
            # self.field_title += content
            self.characters_field_title(content)
            logger.debug(f'Set title to {self.field_title}')

        # if stack_min_1 == 'text' and \
        #         int(self.field_revision_id) in self.revision_id_to_page_ids:

        if stack_min_1 == 'text':
            self.characters_text(content)

        if stack_min_1 == 'id':
            if self.active_page and not self.active_revision:
                # self.field_page_id += content
                self.characters_field_page_id(content)
            elif self.active_page and self.active_revision and stack_min_2 == 'revision':
                logger.debug(f'{self.processed_file} adding to revision: {content}')
                self.characters_field_revision_id(content)

        if stack_min_1 == 'timestamp':
            if self.active_page and self.active_revision:
                self.characters_timestamp(content)

        if stack_min_1 == 'comment':
            if self.active_page and self.active_revision:
                self.characters_field_comment(content)

    def startPrefixMapping(self, prefix, uri):
        pass

    def endPrefixMapping(self, prefix):
        pass
