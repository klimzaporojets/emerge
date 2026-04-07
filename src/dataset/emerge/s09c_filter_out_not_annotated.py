# very basic code to filter out not annotated instances due a (already fixed) bug in
#  09b_annotate_dataset_v4.py
import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import List, Dict

from tqdm import tqdm

import os
from dataset.emerge.utils.constants import ACTION_CATEGORY_DEPRECATE, ACTION_CATEGORY_ASSERT

import pandas as pd

from dataset.emerge.utils.s09_annotate_dataset_utils_v4 import obtain_property_ids_to_definitions, \
    get_instance_tkgu_types_llm_assessments, load_all_instances, stats_all_instances, get_annotation_statistics, \
    human_assessment_exists, clear_stdin, get_print_annotation_statistics, \
    show_discrepancies, exceeds_max_per_tkgu_type, update_count_annotated, update_count_annotated_per_triple

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    in_file = 'output/experiments/s09_annotate_dataset/20250902c_only_yearly_snapshots/annotated/instances_to_annotate.jsonl'
    out_file = 'output/experiments/s09_annotate_dataset/20250902c_only_yearly_snapshots/annotated/instances_to_annotate_filtered.jsonl'


    with open(out_file, 'wt', encoding='utf-8') as ofile:
        for curr_line in tqdm(open(in_file, 'rt', encoding='utf-8')):
            curr_parsed_line = json.loads(curr_line)
            leave_line = False
            for curr_triple in curr_parsed_line['tkgu_triples']:
                if 'human_assessment' in curr_triple \
                        and len(curr_triple['human_assessment']) > 0:
                    leave_line = True
                    break
            if leave_line:
                # print('leaving line')
                # ofile.write(json.dumps(curr_parsed_line,ensure_ascii=False) + '\n')
                ofile.write(curr_line)
                # ofile.flush()