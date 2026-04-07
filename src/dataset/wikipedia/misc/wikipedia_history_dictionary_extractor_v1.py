# Dictionary extractor, first to be used with the relik baseline

import json
import logging
import multiprocessing.managers
import os
import random
import re
import time
import traceback
import xml.sax
import calendar
from datetime import datetime, timezone
from typing import List, Dict

import mwparserfromhell
import requests
from mwparserfromhell.wikicode import Wikicode
from requests import Session
from urllib3 import Retry

from .cleaning import gross_clean, fine_clean

from .compiled_regexes import compiled_regexes

from requests.adapters import HTTPAdapter

_LOG_LEVEL = logging._nameToLevel.get(
    os.environ.get('LOGGING_LEVEL', '').strip(), logging.INFO
)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=_LOG_LEVEL)
logger = logging.getLogger(__name__)


def clean_text_from_link_markers(input_text):
    parsed: Wikicode = mwparserfromhell.parse(input_text)
    # parsed.get_sections().
    text = parsed.strip_code()

    text = re.sub(r'\([\s\,\.]*\)', ' ', text)

    text = re.sub(r' +', ' ', text)

    return text.strip()


def api_call_should_process_page(page_id, config, session: Session = None):
    # return True, 'Q1'
    if len(config['api_ports_wiki_mapping']) == 1:
        api_port = config['api_ports_wiki_mapping'][0]
    else:
        idx_port = random.randint(0, len(config['api_ports_wiki_mapping']) - 1)
        api_port = config['api_ports_wiki_mapping'][idx_port]

    url = f"http://{config['api_host']}:{api_port}/should_process_page"
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
        wikidata_creation_timestamp = int(result['wikidata_creation_timestamp'])
        if config['wikidata_timestamp_format'] == 'milliseconds':
            wikidata_creation_timestamp = int(wikidata_creation_timestamp / 1000)
        return result['should_process_page'], result['page_qid'], wikidata_creation_timestamp
    else:
        raise RuntimeError(f'Failed to retrieve data using '
                           f'api_call_should_process_page invoking '
                           f'with parameters: {params}')


class WikipediaHistoryDictionaryExtractorV1(xml.sax.ContentHandler):
    def __init__(self, filter_namespace,
                 convert_to_text_dictionary,
                 config, v_lock, start_time,
                 v_nr_parsed_articles,
                 do_asserts=False,
                 dry_run=False,
                 ):
        super().__init__()
        self.v_nr_parsed_articles = v_nr_parsed_articles
        logger.debug('Init WikipediaHistoryReader')
        self.do_asserts = do_asserts
        self.last_revision_content: Dict = dict()
        self.stack_elements: List = list()
        self.convert_through_api = config['convert_through_api']
        self.config = config
        self.redirect_max_secs_from_snapshot = config['redirect_max_days_from_snapshot'] * 86400
        self.snapshots = config['snapshots']
        self.snapshots_timestamps = list()
        self.timestamps_to_dates = dict()
        self.dates_to_timestamps = dict()
        self.snapshot_to_content: Dict[str, Dict] = dict()
        self.output_dir = config['output_dir']

        self.max_token_length = config['max_token_length']

        ####
        self.output_files_dict_per_snapshot = dict()
        for curr_snapshot in self.snapshots:
            # Convert the string to a datetime object
            date_obj = datetime.strptime(curr_snapshot, '%Y-%m-%d')

            # Convert to a UTC timestamp (calendar.timegm always interprets as UTC,
            # unlike datetime.timestamp() which uses local timezone for naive datetimes)
            curr_timestamp = calendar.timegm(date_obj.timetuple())
            self.snapshots_timestamps.append(curr_timestamp)
            self.timestamps_to_dates[curr_timestamp] = curr_snapshot
            self.dates_to_timestamps[curr_snapshot] = curr_timestamp
            self.snapshot_to_content[curr_snapshot] = None
        ####
        #
        self.max_ratio_special_characters = config['max_ratio_special_characters']

        self.request_session1 = requests.Session()

        self.should_be_processed = True
        self.nr_revisions = 0
        self.convert_to_text_dictionary: multiprocessing.managers.DictProxy = convert_to_text_dictionary
        self.v_lock = v_lock
        self.dry_run = dry_run
        self.start_time = start_time

        self.text = ''
        self.field_title = ''
        self.field_page_id = ''
        self.field_page_qid = ''
        self.field_wikidata_creation_timestamp = 0
        self.field_revision_id = ''
        self.field_comment = ''
        self.ns = ''
        self.processed_file = ''

        self.timestamp = ''
        self.revision_date: datetime = None
        self.revision_timestamp = -1

        self.filter_namespace = filter_namespace

        # in true when we are inside the elements, the idea is to avoid doing "in self.stack_elements" operation
        # which is O(n)
        self.active_page = False
        self.active_revision = False

        # related to movement of titles
        # self.nr_processed_revisions = 0

        self.last_finished_page_time = time.time()

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
        self.session = session
        #

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
                self.snapshot_to_content = dict()
                for curr_snapshot in self.snapshots:
                    self.snapshot_to_content[curr_snapshot] = None

                self.should_be_processed = True
                if self.do_asserts:
                    assert self.stack_elements == []
                self.nr_revisions = 0
                self.field_title = ''
                self.active_page = True
                self.revision_date = None
                # self.nr_processed_revisions = 0
                self.revision_timestamp = -1

                self.ns = ''
                self.last_revision_content: Dict = dict()
            elif name == 'revision':
                if self.do_asserts:
                    assert self.stack_elements == ['page']
                # logger.debug(f'self.filter_pages is: {self.filter_pages}')
                if self.nr_revisions == 0:
                    with self.v_nr_parsed_articles.get_lock():
                        self.v_nr_parsed_articles.value += 1

                    logger.debug('invoking api_call_should_process_page with '
                                 f'self.field_page_id in {self.field_page_id}')
                    if self.field_page_id.strip() == '':
                        raise RuntimeError('For some reason self.field_page_id is empty: '
                                           f'{self.field_page_id}')
                    should_process_this_page, page_qid, wikidata_creation_timestamp = \
                        api_call_should_process_page(self.field_page_id, self.config, self.session)

                    # logger.debug('api_call_should_process_page results: '
                    #              f'{should_process_this_page} - {page_qid}')

                    if should_process_this_page:
                        logger.debug(f'Detected page to be processed: {self.field_page_id}')
                        self.field_page_qid = page_qid
                        self.field_wikidata_creation_timestamp = wikidata_creation_timestamp
                    else:
                        # belongs to a page that should not be processed
                        self.should_be_processed = False
                        self.stack_elements = list()
                        self.active_page = False
                        self.active_revision = False
                        raise RuntimeError('!this_should_not_happen!, all the '
                                           'pages have to be processed')
                        # logger.debug(f'Ignoring page: {self.field_page_id}')

                if self.should_be_processed:
                    self.active_revision = True

                self.nr_revisions += 1
                self.text = ''
                self.revision_date = None
                self.revision_timestamp = -1

            elif name == 'title':
                if self.do_asserts:
                    assert self.stack_elements == ['page']
                    assert self.field_title == ''
                self.field_title = ''
            elif name == 'text':
                if self.do_asserts:
                    assert self.stack_elements == ['page', 'revision']
                self.text = ''
            elif name == 'id':
                if self.active_page and not self.active_revision:
                    self.field_page_id = ''
                    self.field_page_qid = ''
                    self.field_wikidata_creation_timestamp = 0
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

    def count_braces(self, s: str):
        count_open = s.count('{')
        count_close = s.count('}')
        return count_open + count_close

    def count_pipes(self, s: str):
        count_pipes = s.count('|')
        return count_pipes

    def parseText(self, text: str) -> str:
        # start_parse_text = time.time()
        # start_time = time.time()
        # if text.strip()
        stripped_text = text.strip()
        if stripped_text.lower().startswith('#redirect ') or \
                stripped_text.startswith('REDIRECT '):
            # logger.info('====================BEGIN REDIRECT PARSING')
            # stripped_text = stripped_text.split('\n')[0]
            if ']]' not in stripped_text:
                logger.error(f'something_wrong_with_redirect "{text}", no "]]" in "{stripped_text}" #### '
                             f'for "{self.field_title}" -- {self.field_page_id} -- {self.field_page_qid}')
            else:
                stripped_text = stripped_text[:stripped_text.index(']]') + 2]
            # logger.info(f'redirect_line: "{stripped_text}" for "{self.field_title}" -- {self.field_page_id} -- {self.field_page_qid}')
            # logger.info('====================END REDIRECT PARSING')
        #
        # dddddddddddddddddd
        simple_cleaned_text = gross_clean(text=stripped_text,
                                          regexes=compiled_regexes,
                                          convert_through_api=self.convert_through_api,
                                          convert_to_text_dictionary=self.convert_to_text_dictionary,
                                          request_session1=self.request_session1,
                                          v_lock=self.v_lock)
        # logger.info(f'{(time.time() - start_time):.5f} sec for gross_clean')
        # start_time = time.time()
        simple_cleaned_stripped_code = fine_clean(simple_cleaned_text)
        # logger.info(f'{(time.time() - start_time):.5f} sec for fine_clean')

        simple_cleaned_stripped_code = clean_text_from_link_markers(simple_cleaned_stripped_code)
        # break into paragraphs:
        paragraphs_text = simple_cleaned_stripped_code.split('\n\n')

        end_content = ''
        for idx_paragraph, curr_paragraph in enumerate(paragraphs_text):
            nr_braces = self.count_braces(curr_paragraph)
            nr_pipes = self.count_pipes(curr_paragraph)
            nr_special = nr_braces + nr_pipes
            nr_tokens = len(curr_paragraph.split(' '))
            ratio_tokens = 0.0
            if nr_tokens > 0:
                ratio_tokens = nr_special / nr_tokens

            if ratio_tokens > self.max_ratio_special_characters:
                continue

            if curr_paragraph.strip() != '':
                logger.debug(f'----------- paragraph nr {idx_paragraph}')
                logger.debug(f'{idx_paragraph} - {curr_paragraph}')
            end_content += curr_paragraph.strip()
            # end_content += '\n\n'
            end_content += ' '
            end_content = re.sub(r'\r?\n', ' ', end_content)
            if len(end_content.strip().split(' ')) > self.max_token_length:
                break
        return ' '.join(end_content.strip().split(' ')[:self.max_token_length])

    def endElement(self, name):
        try:
            if not self.should_be_processed:
                return

            self.stack_elements.pop()

            if name == 'ns' and not self.filter_namespace(self.ns):
                self.should_be_processed = False
            if name == 'timestamp' and self.active_page and self.active_revision:
                self.revision_date = datetime.strptime(self.timestamp, '%Y-%m-%dT%H:%M:%SZ')
                # Use calendar.timegm to get UTC epoch (strptime produces naive datetime,
                # and the 'Z' suffix indicates UTC)
                self.revision_timestamp = calendar.timegm(self.revision_date.timetuple())
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

                logger.debug(f'text_to_parse: {self.text.strip()}')
                #
                ######## BEGIN - FUNCTION OF PARSING BEGIN

                # if (len(self.last_revision_content) > 0) and \
                #         not self.last_revision_content['text'].strip().lower().startswith('#redirect ') and \
                #         not self.last_revision_content['text'].strip().startswith('REDIRECT '):
                #
                # 2025.03.30 -- also accepting redirects, which later have to be resolved since
                # many links are to these redirect pages.
                if len(self.last_revision_content) > 0:
                    parsed_text = None
                    assert self.nr_revisions >= 2
                    for curr_snapshot, curr_content in self.snapshot_to_content.items():
                        curr_snapshot_timestamp = self.dates_to_timestamps[curr_snapshot]
                        if (self.last_revision_content['tmstmp'] <= curr_snapshot_timestamp <
                                int(self.revision_timestamp) and curr_content is None and
                                self.field_wikidata_creation_timestamp <= curr_snapshot_timestamp):
                            if parsed_text is None:
                                parsed_text = self.parseText(
                                    text=self.last_revision_content['text']
                                )
                            self.snapshot_to_content[curr_snapshot] = {
                                'parsed_text': parsed_text,
                                'revision_id': self.last_revision_content['revision_id'],
                                'revision_timestamp': self.last_revision_content['tmstmp'],
                                'revision_date': self.last_revision_content['timestamp']
                            }
                        elif (self.nr_revisions == 2 and
                              self.last_revision_content['tmstmp'] > curr_snapshot_timestamp
                              >= self.field_wikidata_creation_timestamp and curr_content is None):
                            # TODO 01.04.2025 --> this if has to be merged into above,
                            #  currently here to be able to print some extra logs and see that
                            #  it actually works
                            #  the idea is to capture the content if it is the first version, but already
                            #  created later than the snapshot date AND the wikidata entity is created
                            #  BEFORE (or on) the snapshot date

                            if parsed_text is None:
                                parsed_text = self.parseText(
                                    text=self.last_revision_content['text']
                                )

                            logger.debug(f'adding_content_even_if_after_snapshot_1 '
                                         f'{self.field_title} -- '
                                         f'{self.field_page_qid} -- '
                                         f'{self.field_page_id} (page id) -- '
                                         f'{self.timestamp} (self.timestamp) -- '
                                         f'{curr_snapshot} (curr_snapshot) -- '
                                         f'{parsed_text} ')
                            self.snapshot_to_content[curr_snapshot] = {
                                'parsed_text': parsed_text,
                                'revision_id': self.last_revision_content['revision_id'],
                                'revision_timestamp': self.last_revision_content['tmstmp'],
                                'revision_date': self.last_revision_content['timestamp']
                            }
                        elif curr_content is not None \
                                and self.last_revision_content[
                            'tmstmp'] - self.redirect_max_secs_from_snapshot < curr_snapshot_timestamp \
                                and curr_snapshot_timestamp >= self.field_wikidata_creation_timestamp \
                                and (curr_content['parsed_text'].lower().strip().startswith('redirect') or
                                     curr_content['parsed_text'].lower().strip() == ''):
                            parsed_text = self.parseText(
                                text=self.last_revision_content['text']
                            )
                            if not (parsed_text.lower().strip().startswith('redirect') or
                                    parsed_text.lower().strip() == ''):
                                logger.info(f'adding_content_even_if_after_snapshot_no_redirect_1 '
                                            f'{self.field_title} -- '
                                            f'{self.field_page_qid} -- '
                                            f'{self.field_page_id} (page id) -- '
                                            f'{self.timestamp} (self.timestamp) -- '
                                            f'{curr_snapshot} (curr_snapshot) -- '
                                            f'{parsed_text} ')
                            self.snapshot_to_content[curr_snapshot] = {
                                'parsed_text': parsed_text,
                                'revision_id': self.last_revision_content['revision_id'],
                                'revision_timestamp': self.last_revision_content['tmstmp'],
                                'revision_date': self.last_revision_content['timestamp']
                            }

                self.last_revision_content = {
                    'text': self.text.strip(),
                    'revision_id': revision_id,
                    'field_title': self.field_title,
                    'field_page_id': int(self.field_page_id),
                    'field_page_qid': self.field_page_qid,
                    'tmstmp': int(self.revision_timestamp),
                    'timestamp': self.timestamp
                }
                ######## END - FUNCTION OF PARSING ENDS
                #
            if name == 'revision':
                self.active_revision = False
            if name == 'page':
                self.active_page = False

                if self.filter_namespace(self.ns):
                    if self.should_be_processed:
                        ##########################################################################
                        # BEGIN - this parseText is needed since the one above on revision works
                        #   only for previous revision, this is a way to incorporate the last revision
                        #   of a page, if this is needed
                        # if ((not self.dry_run and
                        #      not self.last_revision_content['text'].strip().lower().startswith('#redirect ') and
                        #      not self.last_revision_content['text'].strip().startswith('REDIRECT '))):
                        #
                        # 2025.03.30 -- also accepting redirects, which later have to be resolved since
                        # many links are to these redirect pages.
                        if not self.dry_run:
                            ##########################################################################
                            parsed_text = None
                            for curr_snapshot, curr_content in self.snapshot_to_content.items():
                                curr_snapshot_timestamp = self.dates_to_timestamps[curr_snapshot]
                                if (self.last_revision_content['tmstmp'] <= curr_snapshot_timestamp and
                                        curr_content is None and
                                        self.field_wikidata_creation_timestamp <= curr_snapshot_timestamp):

                                    if parsed_text is None:
                                        parsed_text = self.parseText(
                                            text=self.last_revision_content['text']
                                        )
                                    self.snapshot_to_content[curr_snapshot] = \
                                        {
                                            'parsed_text': parsed_text,
                                            'revision_id': int(self.field_revision_id),
                                            'revision_timestamp': int(self.revision_timestamp),
                                            'revision_date': self.timestamp
                                        }
                                elif (self.nr_revisions == 1 and
                                      self.last_revision_content['tmstmp'] > curr_snapshot_timestamp
                                      >= self.field_wikidata_creation_timestamp and curr_content is None):
                                    # TODO 01.04.2025 --> this if has to be merged into above,
                                    #  currently here to be able to print some extra logs and see that
                                    #  it actually works
                                    #  the idea is to capture the content if it is the first version, but already
                                    #  created later than the snapshot date AND the wikidata entity is created
                                    #  BEFORE (or on) the snapshot date
                                    if parsed_text is None:
                                        parsed_text = self.parseText(
                                            text=self.last_revision_content['text']
                                        )
                                    logger.debug(f'adding_content_even_if_after_snapshot_2 '
                                                 f'{self.field_title} -- '
                                                 f'{self.field_page_qid} -- '
                                                 f'{self.field_page_id} (page id) -- '
                                                 f'{self.timestamp} (self.timestamp) -- '
                                                 f'{curr_snapshot} (curr_snapshot) -- '
                                                 f'{parsed_text} ')
                                    self.snapshot_to_content[curr_snapshot] = \
                                        {
                                            'parsed_text': parsed_text,
                                            'revision_id': int(self.field_revision_id),
                                            'revision_timestamp': int(self.revision_timestamp),
                                            'revision_date': self.timestamp
                                        }
                                elif curr_content is not None and \
                                        self.last_revision_content['tmstmp'] - self.redirect_max_secs_from_snapshot < \
                                        curr_snapshot_timestamp \
                                        and curr_snapshot_timestamp >= self.field_wikidata_creation_timestamp \
                                        and (curr_content['parsed_text'].lower().strip().startswith('redirect') or
                                             curr_content['parsed_text'].lower().strip() == ''):
                                    parsed_text = self.parseText(
                                        text=self.last_revision_content['text']
                                    )
                                    if not (parsed_text.lower().strip().startswith('redirect') or
                                            parsed_text.lower().strip() == ''):
                                        logger.info(f'adding_content_even_if_after_snapshot_no_redirect_2 '
                                                    f'{self.field_title} -- '
                                                    f'{self.field_page_qid} -- '
                                                    f'{self.field_page_id} (page id) -- '
                                                    f'{self.timestamp} (self.timestamp) -- '
                                                    f'{curr_snapshot} (curr_snapshot) -- '
                                                    f'{parsed_text} ')
                                    self.snapshot_to_content[curr_snapshot] = {
                                        'parsed_text': parsed_text,
                                        'revision_id': self.last_revision_content['revision_id'],
                                        'revision_timestamp': self.last_revision_content['tmstmp'],
                                        'revision_date': self.last_revision_content['timestamp']
                                    }
                            for curr_snapshot, curr_paragraph in self.snapshot_to_content.items():
                                try:
                                    if curr_paragraph is not None:
                                        curr_snapshot_timestamp = self.dates_to_timestamps[curr_snapshot]
                                        entry_to_add = {
                                            'text': self.field_title,
                                            'qid': self.field_page_qid,
                                            'page_id': self.field_page_id,
                                            'revision_id': curr_paragraph['revision_id'],
                                            'revision_timestamp': curr_paragraph['revision_timestamp'],
                                            'revision_date': curr_paragraph['revision_date'],
                                            'snapshot_date': curr_snapshot,
                                            'snapshot_timestamp': curr_snapshot_timestamp,
                                            'metadata': {
                                                'definition': curr_paragraph['parsed_text']
                                            }
                                        }
                                        str_json = json.dumps(entry_to_add, ensure_ascii=False)
                                        logger.debug(f'writing str_json: {str_json}')
                                        self.output_files_dict_per_snapshot[curr_snapshot].write(str_json + '\n')
                                        self.output_files_dict_per_snapshot[curr_snapshot].flush()
                                    # TODO: remove this flush if too slow
                                    # self.output_files_dict_per_snapshot[curr_snapshot].flush()
                                except Exception as e:
                                    logger.error(f'an json.dumps error occurred with the following '
                                                 f'curr_snapshot, curr_paragraph: {curr_snapshot}, '
                                                 f'{curr_paragraph}')
                                    traceback.print_exc()

                            curr_time = time.time()
                            logger.debug(f'finished_page {self.field_page_id} in '
                                         f'{((curr_time - self.last_finished_page_time) / 60):.4f} mins')
                            self.last_finished_page_time = curr_time
                        else:
                            if len(self.snapshot_to_content) > 0:
                                logger.warning(f'if 1 for_some_reason did not process: {self.snapshot_to_content}')
                    else:
                        if len(self.snapshot_to_content) > 0:
                            logger.warning(f'if 2 for_some_reason did not process: {self.snapshot_to_content} '
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
        self.text += content

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

        # if stack_min_1 == 'text' and len(self.text) == 0:
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
