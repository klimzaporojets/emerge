# Dictionary extractor, first to be used with the relik baseline

import json
import logging
import multiprocessing.managers
import random
import re
import time
import traceback
import xml.sax
from datetime import datetime, timezone
from typing import List, Dict

# import mwparserfromhell
# import requests
# from mwparserfromhell.wikicode import Wikicode
# from requests import Session
# from urllib3 import Retry


# from requests.adapters import HTTPAdapter

from utils import wiki_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=wiki_logger.logger_level)
logger = logging.getLogger(__name__)


class WikidataHistoryDictionaryExtractorV1(xml.sax.ContentHandler):
    def __init__(self, filter_namespace,
                 convert_to_text_dictionary,
                 config, v_lock, start_time,
                 v_nr_parsed_articles,
                 do_asserts=False,
                 dry_run=False,
                 ):
        super().__init__()
        self.v_nr_parsed_articles = v_nr_parsed_articles
        logger.debug('Init WikidataHistoryReader')
        self.do_asserts = do_asserts
        self.last_revision_content: Dict = dict()
        self.stack_elements: List = list()
        # self.convert_through_api = config['convert_through_api']
        self.config = config

        self.snapshots = config['snapshots']
        self.snapshots_timestamps = list()
        self.timestamps_to_dates = dict()
        self.dates_to_timestamps = dict()
        self.snapshot_to_content: Dict[str, Dict] = dict()
        self.output_dir = config['output_dir']

        # self.max_token_length = config['max_token_length']

        ####
        self.output_files_dict_per_snapshot = dict()
        for curr_snapshot in self.snapshots:
            # Convert the string to a datetime object
            date_obj = datetime.strptime(curr_snapshot, '%Y-%m-%d')

            # Set the time to 00:00:00 and convert to UTC
            date_obj_utc = date_obj.replace(tzinfo=timezone.utc)

            # Convert to a timestamp
            curr_timestamp = date_obj_utc.timestamp()
            self.snapshots_timestamps.append(curr_timestamp)
            self.timestamps_to_dates[curr_timestamp] = curr_snapshot
            self.dates_to_timestamps[curr_snapshot] = curr_timestamp
            self.snapshot_to_content[curr_snapshot] = None
        ####
        #
        # self.max_ratio_special_characters = config['max_ratio_special_characters']

        # self.request_session1 = requests.Session()

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
        self.nr_processed_revisions = 0

        self.last_finished_page_time = time.time()

        # session = requests.Session()

        # Define a retry strategy
        # retry_strategy = Retry(
        #     total=10,  # Total number of retries
        #     backoff_factor=6
        # )

        # Mount the adapter with the retry strategy
        # adapter = HTTPAdapter(max_retries=retry_strategy)
        # session.mount('http://', adapter)
        # session.mount('https://', adapter)
        # self.session = session
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
                self.nr_processed_revisions = 0
                self.revision_timestamp = -1

                self.ns = ''
                self.last_revision_content: Dict = dict()
                with self.v_nr_parsed_articles.get_lock():
                    self.v_nr_parsed_articles.value += 1

            elif name == 'revision':
                # if self.do_asserts:
                #     assert self.stack_elements == ['page']
                # logger.debug(f'self.filter_pages is: {self.filter_pages}')
                if self.nr_revisions == 0:
                    # with self.v_nr_parsed_articles.get_lock():
                    #     self.v_nr_parsed_articles.value += 1

                    logger.debug('invoking api_call_should_process_page with '
                                 f'self.field_page_id in {self.field_page_id}')
                    if self.field_title.strip() == '':
                        raise RuntimeError('For some reason self.field_page_id is empty: '
                                           f'{self.field_title}')
                    # should_process_this_page, page_qid = \
                    #     api_call_should_process_page(self.field_page_id, self.config, self.session)

                    # logger.debug('api_call_should_process_page results: '
                    #              f'{should_process_this_page} - {page_qid}')
                    should_process_this_page = self.field_title.startswith('P')
                    # self.should_be_processed = self.field_page_id.startswith('P')
                    # if not self.should_be_processed:
                    #     if not self.stack_elements == ['page']:
                    #         logger.info(f'these_are_stack_elements: {self.stack_elements}')
                    #     self.stack_elements = []
                    if should_process_this_page:
                        logger.info(f'Detected page to be processed: {self.field_title}, '
                                    f'{self.should_be_processed} ; '
                                    f'active_page: {self.active_page}; '
                                    f'ns: {self.ns}')
                        assert self.should_be_processed
                        # self.field_page_qid = page_qid
                    # else:
                    #     # belongs to a page that should not be processed
                    #     self.should_be_processed = False
                    #     self.stack_elements = list()
                    #     self.active_page = False
                    #     self.active_revision = False
                    #     raise RuntimeError('!this_should_not_happen!, all the '
                    #                        'pages have to be processed')

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
                elif self.active_page and self.active_revision and self.stack_elements[-1] == 'revision':
                    self.field_revision_id = ''
            elif name == 'timestamp':
                if self.active_page and self.active_revision:
                    self.timestamp = ''
                    # logger.info(f'timestamp_initialized for {self.field_title}')
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

    def endElement(self, name):
        try:
            if not self.should_be_processed:
                return

            self.stack_elements.pop()

            if name == 'title' and not self.field_title.startswith('P'):
                self.should_be_processed = False
                # logger.info(f'stack_elements in {self.stack_elements} and setting to '
                #             f'empty')
                self.stack_elements = []
            if name == 'ns' and not self.filter_namespace(self.ns):
                self.should_be_processed = False
            if name == 'timestamp' and self.active_page and self.active_revision:
                self.revision_date = datetime.strptime(self.timestamp, '%Y-%m-%dT%H:%M:%SZ')
                self.revision_timestamp = self.revision_date.timestamp()
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

                if (len(self.last_revision_content) > 0) and \
                        not self.last_revision_content['text'].strip().lower().startswith('#redirect ') and \
                        not self.last_revision_content['text'].strip().startswith('REDIRECT '):
                    parsed_text = None
                    parsed_label = None
                    for curr_snapshot, curr_content in self.snapshot_to_content.items():
                        curr_snapshot_timestamp = self.dates_to_timestamps[curr_snapshot]
                        if (self.last_revision_content['tmstmp'] <= curr_snapshot_timestamp <
                                int(self.revision_timestamp) and curr_content is None):
                            if parsed_text is None or parsed_label is None:
                                # logger.info(f'have to parse the following: '
                                #             f'{self.last_revision_content['text']}')
                                parsed_line = \
                                    json.loads(self.last_revision_content['text'])
                                parsed_label = ''
                                if 'en' in parsed_line['labels']:
                                    parsed_label = parsed_line['labels']['en']['value']

                                parsed_text = ''
                                if 'en' in parsed_line['descriptions']:
                                    parsed_text = parsed_line['descriptions']['en']['value']
                                # parsed_text = self.parseText(
                                #     text=self.last_revision_content['text']
                                #     # ,
                                #     # revision_id=self.last_revision_content['revision_id'],
                                #     # field_title=self.last_revision_content['field_title'],
                                #     # field_page_id=self.last_revision_content['field_page_id'],
                                #     # field_page_qid=self.last_revision_content['field_page_qid'],
                                #     # tmstmp=self.last_revision_content['tmstmp'],
                                #     # timestamp=self.last_revision_content['timestamp']
                                # )
                            self.snapshot_to_content[curr_snapshot] = {
                                'parsed_title': parsed_label,
                                'parsed_text': parsed_text,
                                # 'parsed_text': parsed_text,
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
                        if ((not self.dry_run and
                             # not self.text.strip().lower().startswith('#redirect ') and
                             not self.last_revision_content['text'].strip().lower().startswith('#redirect ') and
                             not self.last_revision_content['text'].strip().startswith('REDIRECT '))):
                            # revision_id = int(self.field_revision_id)
                            # self.parseText(
                            #     text=self.text.strip()
                            #     # ,
                            #     # revision_id=revision_id,
                            #     # field_title=self.field_title,
                            #     # field_page_id=int(self.field_page_id),
                            #     # field_page_qid=self.field_page_qid,
                            #     # tmstmp=int(self.revision_timestamp),
                            #     # timestamp=self.timestamp
                            # )
                            ##########################################################################
                            parsed_text = None
                            parsed_label = None
                            for curr_snapshot, curr_content in self.snapshot_to_content.items():
                                curr_snapshot_timestamp = self.dates_to_timestamps[curr_snapshot]
                                if (self.last_revision_content['tmstmp'] <= curr_snapshot_timestamp and
                                        curr_content is None):
                                    if parsed_text is None or parsed_label is None:
                                        # logger.info(f'have to parse the following: '
                                        #             f'{self.last_revision_content['text']}')
                                        parsed_line = \
                                            json.loads(self.last_revision_content['text'])
                                        parsed_label = ''
                                        if 'en' in parsed_line['labels']:
                                            parsed_label = parsed_line['labels']['en']['value']

                                        parsed_text = ''
                                        if 'en' in parsed_line['descriptions']:
                                            parsed_text = parsed_line['descriptions']['en']['value']
                                        # parsed_text = self.parseText(
                                        #     text=self.last_revision_content['text']
                                        #     # ,
                                        #     # revision_id=self.last_revision_content['revision_id'],
                                        #     # field_title=self.last_revision_content['field_title'],
                                        #     # field_page_id=self.last_revision_content['field_page_id'],
                                        #     # field_page_qid=self.last_revision_content['field_page_qid'],
                                        #     # tmstmp=self.last_revision_content['tmstmp'],
                                        #     # timestamp=self.last_revision_content['timestamp']
                                        # )
                                    self.snapshot_to_content[curr_snapshot] = {
                                        'parsed_title': parsed_label,
                                        'parsed_text': parsed_text,
                                        # 'parsed_text': parsed_text,
                                        'revision_id': self.last_revision_content['revision_id'],
                                        'revision_timestamp': self.last_revision_content['tmstmp'],
                                        'revision_date': self.last_revision_content['timestamp']
                                    }
                                    # self.snapshot_to_content[curr_snapshot] = \
                                    #     {
                                    #         'parsed_text': parsed_text,
                                    #         'revision_id': int(self.field_revision_id),
                                    #         'revision_timestamp': int(self.revision_timestamp),
                                    #         'revision_date': self.timestamp
                                    #     }
                                    # self.snapshot_to_content[curr_snapshot] = parsed_text

                        for curr_snapshot, curr_paragraph in self.snapshot_to_content.items():
                            try:
                                if curr_paragraph is not None:
                                    assert self.field_title.startswith('Property')
                                    entry_to_add = {
                                        # 'text': self.field_title,
                                        'text': curr_paragraph['parsed_title'],
                                        # 'qid': self.field_page_qid,
                                        'page_id': self.field_page_id,
                                        'revision_id': curr_paragraph['revision_id'],
                                        'revision_timestamp': curr_paragraph['revision_timestamp'],
                                        'revision_date': curr_paragraph['revision_date'],
                                        'metadata': {
                                            'definition': curr_paragraph['parsed_text'],
                                            'property': self.field_title[len('Property:'):]
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
