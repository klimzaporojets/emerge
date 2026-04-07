import csv
import hashlib
import base64
import logging
import os
import pickle
from typing import Set

import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
# logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def generate_short_hash(input_string: str, hash_length: int):
    # Create a SHA256 hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    # Convert the hash to a byte array
    hash_bytes = hash_object.digest()
    # Encode the byte array to a base64 string and strip unwanted characters
    short_hash = base64.urlsafe_b64encode(hash_bytes).decode('utf-8').rstrip('=')
    # Return the first 8 characters for a shorter hash
    return short_hash[:hash_length]


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


def load_wikidata_qid_to_label(path_wikidata_labels, cache_path,
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
