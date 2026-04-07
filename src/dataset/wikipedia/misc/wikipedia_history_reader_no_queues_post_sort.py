"""
The difference with wikipedia_history_reader_no_queues.py is that this class
does the sorting by timestamp of the revisions before obtaining the
differences in target entities across time.
This is because the revisions in wikipedia logs are not always sorted by time.
I have found some examples inside enwiki-20240601-pages-meta-history1.xml-p11532p12107.7z
 file for example with the following page ids:
    - 11582
    - 11535
    - 11585
    - others
"""

import logging
import multiprocessing.managers
import time
import xml.sax
from datetime import datetime
from typing import List, Set, Tuple

import requests

from .cleaning import gross_clean, fine_clean
from .wiki_parse import get_mentions_and_links

from .compiled_regexes import compiled_regexes

logger = logging.getLogger(__name__)


class WikipediaHistoryReaderNoQueuesPostSort(xml.sax.ContentHandler):
    def __init__(self, filter_namespace,
                 # v_nr_pages_change_title_error,
                 convert_to_text_dictionary,
                 config, v_lock, start_time, v_nr_parsed_articles, do_asserts=False,
                 filter_pages=None, dry_run=False,
                 log_parsing_page_ids=[]):
        super().__init__()
        self.v_nr_parsed_articles = v_nr_parsed_articles
        logger.debug('Init WikipediaHistoryReader')
        self.do_asserts = do_asserts
        self.stack_elements: List = list()
        self.convert_through_api = config['convert_through_api']
        self.request_session1 = requests.Session()

        self.csv_writer_history_hyperlinks = None
        self.csv_writer_entity_stats = None
        self.csv_writer_revisions = None

        self.output_file_history_hyperlinks = None
        self.output_file_entity_stats = None
        self.output_file_parse_pages = None
        self.output_file_revisions = None

        self.filter_pages = filter_pages
        self.log_parsing_page_ids = set(log_parsing_page_ids)
        self.should_be_processed = True
        self.nr_revisions = 0
        self.convert_to_text_dictionary: multiprocessing.managers.DictProxy = convert_to_text_dictionary
        self.v_lock = v_lock
        self.dry_run = dry_run
        self.start_time = start_time

        self.text = ''
        self.field_title = ''
        self.field_page_id = ''
        self.field_revision_id = ''
        self.field_comment = ''
        self.ns = ''

        self.timestamp = ''
        self.revision_date: datetime = None
        self.revision_date_str = None

        self.creation_date = None
        self.field_creation_date = None

        self.filter_namespace = filter_namespace

        # in true when we are inside the elements, the idea is to avoid doing "in self.stack_elements" operation
        # which is O(n)
        self.active_page = False
        self.active_revision = False

        # related to movement of titles
        self.nr_processed_revisions = 0

        self.snapshots_target_pages: List[Tuple[Tuple[int, int], Set]] = list()

        self.flush_individually = config['flush_individually']
        self.last_finished_revision_time = time.time()
        self.last_finished_page_time = time.time()

    def startDocument(self):
        pass

    def endDocument(self):
        pass

    def startElementNS(self, name, qname, attrs):
        pass

    def startElement(self, name, attributes: xml.sax.xmlreader.AttributesImpl):

        if name == 'ns':
            if self.do_asserts:
                assert self.stack_elements == ['page']
        elif name == 'page':
            self.snapshots_target_pages = list()
            self.should_be_processed = True
            if self.do_asserts:
                assert self.stack_elements == []
            self.nr_revisions = 0
            self.field_title = ''
            self.active_page = True
            self.revision_date = None
            self.creation_date = None
            self.field_creation_date = None
            self.revision_date_str = None
            self.nr_processed_revisions = 0

            self.ns = ''

        elif name == 'revision':
            if self.do_asserts:
                assert self.stack_elements == ['page']

            if self.filter_pages is not None and self.nr_revisions == 0:
                if self.filter_pages(self.field_page_id):
                    logger.info('GOOD! FOUND THE FOLLOWING: {} - {}'.format(self.field_title, self.field_page_id))
                else:
                    self.should_be_processed = False
            self.nr_revisions += 1
            self.text = ''
            self.active_revision = True
            self.revision_date = None

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
            elif self.active_page and self.active_revision and self.stack_elements[-1] == 'revision':
                self.field_revision_id = ''
        elif name == 'timestamp':
            if self.active_page and self.active_revision:
                self.timestamp = ''
        elif name == 'comment':
            self.field_comment = ''

        self.stack_elements.append(name)

    def endElement(self, name):

        if len(self.stack_elements) > 0 and name == self.stack_elements[-1]:
            self.stack_elements.pop()
        if not self.should_be_processed:
            if name == 'revision':
                self.active_revision = False
            if name == 'page':
                self.active_page = False
            return
        if name == 'comment':
            if self.filter_namespace(self.ns):
                self.field_comment = self.field_comment.strip()
        if name == 'timestamp' and self.active_revision and self.active_page:
            if self.filter_namespace(self.ns):
                self.revision_date = datetime.strptime(self.timestamp, '%Y-%m-%dT%H:%M:%SZ')
                self.revision_date_str = self.timestamp
                # if it is the first revision, then assigns the creation date
                if self.creation_date is None or self.revision_date < self.creation_date:
                    self.creation_date = self.revision_date
                    self.field_creation_date = self.timestamp

        if name == 'revision':
            if self.filter_namespace(self.ns):
                assert self.revision_date is not None
                if not self.dry_run:
                    simple_cleaned_text = gross_clean(text=self.text.strip(),
                                                      regexes=compiled_regexes,
                                                      convert_through_api=self.convert_through_api,
                                                      convert_to_text_dictionary=self.convert_to_text_dictionary,
                                                      request_session1=self.request_session1,
                                                      v_lock=self.v_lock)
                    simple_cleaned_stripped_code = fine_clean(simple_cleaned_text)

                    content_length = len(simple_cleaned_stripped_code.split(' '))

                    mention_links, tot_detected_mentions, tot_links_errors = \
                        get_mentions_and_links(simple_cleaned_stripped_code, content_length,
                                               self.field_title,
                                               compiled_regexes['compiled_mention_finder'],
                                               compiled_regexes['compiled_country_in_link'])

                    curr_snapshot_target_pages = set()
                    for curr_mention_link in mention_links:
                        curr_snapshot_target_pages.add(
                            (curr_mention_link['target_wikipedia_title_orig'])
                        )
                    tmstamp = self.revision_date.timestamp()
                    tmstamp = int(tmstamp)
                    revision_id = int(self.field_revision_id)
                    self.snapshots_target_pages.append(((tmstamp, revision_id), curr_snapshot_target_pages))
                    if self.field_page_id in self.log_parsing_page_ids:
                        logger.info(f'logging parsing for {self.field_page_id}')
                        self.output_file_parse_pages.write(f'============== {self.field_page_id} - '
                                                           f'{self.revision_date}   ===========')
                        self.output_file_parse_pages.write(f'\n BEFORE CLEANING {self.field_page_id} - '
                                                           f'{self.revision_date}: {self.text.strip()} '
                                                           f'\n\n ------------------')
                        self.output_file_parse_pages.write(f'\n simple_cleaned_text {self.field_page_id} - '
                                                           f'{self.revision_date}: {simple_cleaned_text} '
                                                           f'\n\n ------------------')
                        self.output_file_parse_pages.write(f'\n simple_cleaned_stripped_code {self.field_page_id} - '
                                                           f'{self.revision_date}: {simple_cleaned_stripped_code} '
                                                           f'\n\n ------------------')
                        self.output_file_parse_pages.write(f'\n curr_snapshot_target_pages {self.field_page_id} - '
                                                           f'{self.revision_date}: {curr_snapshot_target_pages} '
                                                           f'\n\n ------------------')
                        self.output_file_parse_pages.flush()

                self.nr_processed_revisions += 1
                curr_time = time.time()
                logger.debug(f'finished_revision {self.nr_processed_revisions} - '
                             f'{self.timestamp} of page {self.field_page_id} '
                             f'in '
                             f'{((curr_time - self.last_finished_revision_time) / 60):.8f} mins '
                             f'lc: {len(self.text)} '
                             f'lcm: {len(self.field_comment)} ')
                self.last_finished_revision_time = curr_time
            self.active_revision = False
        if name == 'page':
            self.active_page = False

            if self.filter_namespace(self.ns):

                # 20240802 --> BEGIN calculates the differences in target links
                # sorted function in python is stable, so elements with equal timestamps will remain
                # in the same order
                self.snapshots_target_pages = sorted(self.snapshots_target_pages, key=lambda x: x[0][0])

                revision_ids_w_timestamps = [(self.field_page_id, stp[0][0], stp[0][1])
                                             for stp in self.snapshots_target_pages]
                prev_target_pages = set()
                outlink_history = dict()
                for idx_snapshot, curr_target_pages in enumerate(self.snapshots_target_pages):
                    new_target_pages = curr_target_pages[1]
                    deleted_target_pages = set()
                    tmstamp = curr_target_pages[0][0]
                    if idx_snapshot > 0:
                        new_target_pages = curr_target_pages[1].difference(prev_target_pages)
                        deleted_target_pages = prev_target_pages.difference(curr_target_pages[1])

                    for curr_new_target_page in new_target_pages:
                        if curr_new_target_page not in outlink_history:
                            outlink_history[curr_new_target_page] = list()
                        outlink_history[curr_new_target_page].append(f'{tmstamp}:A')
                    for curr_target_page in deleted_target_pages:
                        outlink_history[curr_target_page].append(f'{tmstamp}:D')
                    prev_target_pages = curr_target_pages[1]

                # 20240802 --> END calculates the differences in target links

                self.field_title = self.field_title.strip()
                self.field_title = self.field_title.replace(' ', '_')
                with self.v_nr_parsed_articles.get_lock():
                    self.v_nr_parsed_articles.value += 1

                for curr_target_page, curr_history in outlink_history.items():
                    self.csv_writer_history_hyperlinks.writerow([self.field_page_id,
                                                                 self.field_title,
                                                                 curr_target_page,
                                                                 curr_history])

                self.csv_writer_entity_stats.writerow([self.field_page_id,
                                                       self.field_title,
                                                       int(self.creation_date.timestamp()),
                                                       self.creation_date.date()])

                for curr_revision_id_w_timestamp in revision_ids_w_timestamps:
                    curr_page_id = curr_revision_id_w_timestamp[0]
                    curr_timestamp = curr_revision_id_w_timestamp[1]
                    curr_revision_id = curr_revision_id_w_timestamp[2]
                    self.csv_writer_revisions.writerow([
                        curr_page_id,
                        curr_timestamp,
                        curr_revision_id
                    ])

                if self.flush_individually:
                    self.output_file_history_hyperlinks.flush()
                    self.output_file_entity_stats.flush()
                    self.output_file_revisions.flush()

                curr_time = time.time()
                logger.debug(f'finished_page {self.field_page_id} in '
                             f'{((curr_time - self.last_finished_page_time) / 60):.4f} mins')
                self.last_finished_page_time = curr_time

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

        if stack_min_1 == 'text':
            # self.text += content
            self.characters_text(content)

        if stack_min_1 == 'id':
            if self.active_page and not self.active_revision:
                # self.field_page_id += content
                self.characters_field_page_id(content)
            elif self.active_page and self.active_revision and stack_min_2 == 'revision':
                self.characters_field_revision_id(content)

        if stack_min_1 == 'timestamp':
            if self.active_page and self.active_revision:
                # self.timestamp += content
                self.characters_timestamp(content)

        if stack_min_1 == 'comment':
            if self.active_page and self.active_revision:
                # self.field_comment += content
                self.characters_field_comment(content)

    def startPrefixMapping(self, prefix, uri):
        pass

    def endPrefixMapping(self, prefix):
        pass
