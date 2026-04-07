import csv
import logging
import os
import pickle
import subprocess
from typing import Set

from torchgen.packaged.autograd.context import with_native_function_with_differentiability_info_and_key

from utils import wiki_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=wiki_logger.logger_level)
logger = logging.getLogger(__name__)


def load_wiki_page_id_to_wdata_qid_only_parsed(
        path_cache_wdata_and_wpedia_parsed_mapping,
        wpedia_entity_creation_date_path,
        wdata_entity_creation_date_path
):
    wpedia_page_id_to_wikidata_qid = dict()
    if not os.path.exists(path_cache_wdata_and_wpedia_parsed_mapping):
        logger.info(f'START load_wiki_page_id_to_wdata_qid_only_parsed from files '
                    f'(not pickled)')
        wdata_qids = set()
        # page_id_to_wikidata_qid = dict()
        with open(wdata_entity_creation_date_path, 'rt') as infile:
            csv_reader = csv.reader(infile, delimiter='\t')
            for curr_line in csv_reader:
                curr_qid = curr_line[0].strip()
                wdata_qids.add(curr_qid)
        with open(wpedia_entity_creation_date_path, 'rt') as infile:
            csv_reader = csv.reader(infile, delimiter='\t')
            for curr_line in csv_reader:
                curr_qid = curr_line[0].strip()
                if curr_qid in wdata_qids:
                    curr_page_id = int(curr_line[2])
                    wpedia_page_id_to_wikidata_qid[curr_page_id] = curr_qid
        logger.info(f'END load_wiki_page_id_to_wdata_qid_only_parsed from files '
                    f'now pickling to {path_cache_wdata_and_wpedia_parsed_mapping}')
        pickle.dump(wpedia_page_id_to_wikidata_qid, open(path_cache_wdata_and_wpedia_parsed_mapping, 'wb'))
        logger.info(f'END load_wiki_page_id_to_wdata_qid_only_parsed finished pickling'
                    f' to {path_cache_wdata_and_wpedia_parsed_mapping}')
    else:
        logger.info(f'START load_wiki_page_id_to_wdata_qid_only_parsed pickled, '
                    f'loading from pickle: {path_cache_wdata_and_wpedia_parsed_mapping} ')
        wpedia_page_id_to_wikidata_qid = (
            pickle.load(open(path_cache_wdata_and_wpedia_parsed_mapping, 'rb')))
        logger.info(f'END load_wiki_page_id_to_wdata_qid_only_parsed pickled, '
                    f'loading from pickle: {path_cache_wdata_and_wpedia_parsed_mapping} ')
    return wpedia_page_id_to_wikidata_qid


def load_wikidata_qid_to_label_qids(path_wikidata_labels, cache_path,
                               qids_to_load: Set = None):
    # Open the TSV file
    wikidata_qid_to_label = dict()
    if not os.path.exists(cache_path):
        logger.info('wikidata_qid_to_label not pickled, loading from the tsv file: '
                    f'{path_wikidata_labels}')
        with open(path_wikidata_labels, 'r', encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')  # Specify tab as the delimiter
            # Iterate through the rows
            for row in tsv_reader:
                # Process each row (for example, print the first and second columns)
                # print(f"Column 1: {row[0]}, Column 2: {row[1]}")
                wikidata_qid = row[0]
                to_check = wikidata_qid[1:]
                if not to_check.isdigit():
                    logger.warning(f'load_wikidata_qid_to_label the following '
                                   f'wikidata_qid is not numeric: {wikidata_qid}')
                    continue

                if qids_to_load is not None and wikidata_qid not in qids_to_load:
                    continue
                wikidata_label = row[2]
                wikidata_qid_to_label[wikidata_qid] = wikidata_label
        pickle.dump(wikidata_qid_to_label, open(cache_path, 'wb'))
    else:
        logger.info(f'wikidata_qid_to_label pickled, loading from pickle: {cache_path} ')
        wikidata_qid_to_label = pickle.load(open(cache_path, 'rb'))
    return wikidata_qid_to_label

def load_wikidata_qid_to_label(path_wikidata_labels, cache_path):
    # Open the TSV file
    wikidata_qid_to_label = dict()
    if not os.path.exists(cache_path):
        logger.info('wikidata_qid_to_label not pickled, loading from the tsv file: '
                    f'{path_wikidata_labels}')
        with open(path_wikidata_labels, 'r', encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')  # Specify tab as the delimiter
            # Iterate through the rows
            for row in tsv_reader:
                # Process each row (for example, print the first and second columns)
                # print(f"Column 1: {row[0]}, Column 2: {row[1]}")
                wikidata_qid = row[0]
                wikidata_label = row[2]
                wikidata_qid_to_label[wikidata_qid] = wikidata_label
        pickle.dump(wikidata_qid_to_label, open(cache_path, 'wb'))
    else:
        logger.info(f'wikidata_qid_to_label pickled, loading from pickle: {cache_path} ')
        wikidata_qid_to_label = pickle.load(open(cache_path, 'rb'))
    return wikidata_qid_to_label


def load_property_qid_to_label(path_property_labels, cache_path):
    property_qid_to_label = dict()
    if not os.path.exists(cache_path):
        with open(path_property_labels, 'r', encoding='utf8') as tsv_file:
            tsv_reader = csv.reader(tsv_file, delimiter='\t')  # Specify tab as the delimiter
            # Iterate through the rows
            for row in tsv_reader:
                # Process each row (for example, print the first and second columns)
                # print(f"Column 1: {row[0]}, Column 2: {row[1]}")
                property_qid = row[1]
                property_label = row[2]
                property_qid_to_label[property_qid] = property_label
        pickle.dump(property_qid_to_label, open(cache_path, 'wb'))
    else:
        property_qid_to_label = pickle.load(open(cache_path, 'rb'))
    return property_qid_to_label


def get_git_commit_hash():
    try:
        # Get the current git commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError:
        return None


import hashlib
import base64


def generate_short_hash(input_string: str, hash_length: int):
    # Create a SHA256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    # Convert the hash to a byte array
    hash_bytes = hash_object.digest()
    # Encode the byte array to a base64 string and strip unwanted characters
    short_hash = base64.urlsafe_b64encode(hash_bytes).decode('utf-8').rstrip('=')
    # Return the first 8 characters for a shorter hash
    return short_hash[:hash_length]

# Example usage
# input_str = "Hello, World!"
# short_hash = generate_short_hash(input_str)
# print(short_hash)
