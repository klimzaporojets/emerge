# s09e_annotation_stats_for_paper --

import argparse
import json
import logging
import os
# from pathlib import Path
from typing import Dict

import os

import pandas as pd

from dataset.emerge.utils.s09_annotate_dataset_utils_v4 import get_annotation_statistics, \
    merge_annotations, get_print_annotation_statistics_w_humans, return_paper_stats

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s09_annotate_dataset/20250901/'
                                's09_annotate_dataset.json',
                        help='The config file that contains all the parameters')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 20)
    pd.set_option('display.width', 200)

    args = parser.parse_args()
    config = json.load(open(args.config_file, 'rt'))

    tkgu_operations_to_check = set(config['tkgu_operations_to_check'])

    nr_annotated_per_tkgu_operation = dict()

    input_annotation_paths = config['input_annotation_paths']

    # annotator_name = config['annotator_name']
    human_annotators = config['human_annotators']
    hash_id_to_instance: Dict[str, Dict] = dict()

    # input_relation_dictionaries = config['input_relation_dictionaries']
    # root = Path(input_relation_dictionaries)
    # dir_dates = [d.name for d in root.iterdir() if d.is_dir()]
    # date_object = datetime.fromtimestamp(curr_parsed_line['delta_timestamps'][0])
    # Format the date as yyyy-mm-dd
    # formatted_date = date_object.strftime('%Y-%m-%d')
    logger.info('BEGIN loading property ids to definitions')
    date_to_property_id_to_definition: Dict[str, Dict[str, str]] = dict()
    # for formatted_date in dir_dates:
    #     if formatted_date not in date_to_property_id_to_definition:
    #         dictionary_path = os.path.join(config['input_relation_dictionaries'],
    #                                        formatted_date,
    #                                        'documents.jsonl')
    #
    #         date_to_property_id_to_definition[formatted_date] = \
    #             obtain_property_ids_to_definitions(dictionary_path)
    logger.info('END loading property ids to definitions')

    logger.info('BEGIN loading already annotated content and merging')

    continue_next = True
    for in_dir_path in input_annotation_paths:
        curr_annotated_file_names = os.listdir(in_dir_path)
        for curr_annotated_filename in curr_annotated_file_names:
            in_file_path = os.path.join(in_dir_path, curr_annotated_filename)
            for curr_instance in open(in_file_path, 'rt', encoding='utf-8'):
                parsed_line = json.loads(curr_instance)
                curr_hash_id = parsed_line['hash_id']
                if curr_hash_id not in hash_id_to_instance:
                    hash_id_to_instance[curr_hash_id] = parsed_line
                else:
                    merged_annotations = merge_annotations(
                        instance1=hash_id_to_instance[curr_hash_id],
                        instance2=parsed_line
                    )
                    hash_id_to_instance[curr_hash_id] = merged_annotations

    #
    # logger.info('BEGIN replacing already fixed')
    # output_file_path = os.path.join(output_solved_disagreements_path,
    #                                 'solved_disagreements.jsonl')
    # already_fixed_hash_ids = set()
    # if os.path.exists(output_file_path):
    #     mode_disagreements = 'at'
    #     for curr_line in open(output_file_path, 'rt', encoding='utf-8'):
    #         pars_line = json.loads(curr_line)
    #         hash_id_to_instance[pars_line['hash_id']] = pars_line
    #         already_fixed_hash_ids.add(pars_line['hash_id'])
    # else:
    #     mode_disagreements = 'wt'
    # logger.info('END replacing already fixed')
    #
    logger.info('BEGIN calculating statistics')
    human_name_to_alias = dict()
    human_names = [hum['annotator'] for hum in human_annotators]
    annotator_to_alias = {hum['annotator']: hum['alias'] for hum in human_annotators}

    df_stats = get_annotation_statistics(
        annotated_instances=list(hash_id_to_instance.values()),
        annotator_names=human_names
        # annotator_names=human_annotators
    )
    #
    logger.info('END calculating statistics')
    llms_d_triples = set(config['llms_to_compare']['triple_deprecation'])
    llms_assert_triples = set(config['llms_to_compare']['triple_assertion'])
    df_stats = \
        df_stats[
            ((df_stats['tkgu_operation'] == 'd-triples') & (df_stats['llm_name'].isin(llms_d_triples))) |
            ((df_stats['tkgu_operation'] != 'd-triples') & (df_stats['llm_name'].isin(llms_assert_triples)))
            ]
    logger.info(f'df_stats.columns: {df_stats.columns}')
    df_stats.rename(columns={'annotator_name': 'human_name'}, inplace=True)

    df_stats = df_stats[['human_name', 'human_assessment', 'llm_assessment', 'tkgu_operation', 'hash_id', 'human_readable_triple']]
    paper_stats = return_paper_stats(df_statistics=df_stats, human_names=human_names)
    logger.info(f' =============ANNOTATION STATS=================\n'
                f' {paper_stats}\n'
                f' =============ANNOTATION STATS=================')


    anno_paper_stats = return_paper_stats(df_statistics=df_stats,
                                          human_names=human_names)

    # 1) Round all numeric columns to 3 decimals
    cols = ['H-H cohen', 'H1-LLM cohen', 'H2-LLM cohen', 'H+LLM fleiss', 'H+LLM kripp']
    anno_paper_stats[cols] = anno_paper_stats[cols].round(3)

    # Optional: standardize operation names for LaTeX
    def format_operation(op):
        # Split on dashes, capitalize each part, join back
        parts = op.split('-')
        # Keep known acronyms uppercase
        acronyms = {'EE', 'KG', 'D', 'E', 'X'}
        formatted_parts = []
        for p in parts:
            p_upper = p.upper()
            if p_upper in acronyms:
                formatted_parts.append(p_upper)
            else:
                formatted_parts.append(p.capitalize())
        return '-'.join(formatted_parts)


    # Desired row order
    order = ["x-triples", "e-triples", "ee-triples", "ee-kg-triples", "d-triples", "Overall"]

    # Apply formatting + custom order
    anno_paper_stats['Operation_latex'] = anno_paper_stats['Operation'].apply(format_operation)

    # Reorder according to 'order'
    anno_paper_stats = (
        anno_paper_stats
        .set_index('Operation')
        .loc[order]
        .reset_index()
    )

    # anno_paper_stats['Operation_latex'] = anno_paper_stats['Operation'].apply(format_operation)

    # anno_paper_stats['Operation_latex'] = anno_paper_stats['Operation'].str.replace('-', '-').str.upper()

    # 2) Generate LaTeX table rows
    latex_rows = []
    for _, row in anno_paper_stats.iterrows():
        latex_row = (
            f"{row['Operation_latex']} & "
            f"{row['H-H cohen']:.3f} & "
            f"{row['H1-LLM cohen']:.3f} & "
            f"{row['H2-LLM cohen']:.3f} & "
            f"{row['H+LLM fleiss']:.3f} & "
            f"{row['H+LLM kripp']:.3f} \\\\"
        )
        latex_rows.append(latex_row)

    # 3) Combine into full LaTeX table
    latex_table = r"""
    \begin{tabular}{lccccc}
    \toprule
    \shortstack{TKGU \\ Operation } & 
    \shortstack{H-H \\ Cohen's $\kappa$} & 
    \shortstack{H1-LLM \\ Cohen's $\kappa$} & 
    \shortstack{H2-LLM \\ Cohen's $\kappa$} & 
    \shortstack{H+LLM \\ Fleiss' $\kappa$} & 
    \shortstack{H+LLM \\ Kripp. $\alpha$} \\
    \midrule
    """ + "\n".join(latex_rows) + r"""
    \bottomrule
    \end{tabular}
    """

    print(latex_table)

    # logger.info('END loading already annotated content and merging')
    # check_instance = True
    # with open(output_file_path, mode_disagreements, encoding='utf-8') as out_file:
    #     for idx_passage, (curr_hash_id, curr_instance) in enumerate(hash_id_to_instance.items()):
    #         ################################
    #         if curr_hash_id in already_fixed_hash_ids:
    #             logger.info(f'curr_hash_id was already assessed and fixed '
    #                         f'{curr_hash_id}')
    #             continue
    #         if check_instance:
    #             logger.info('*****************************************************')
    #             logger.info('showing discrepancies with new instance')
    #             logger.info('*****************************************************')
    #             curr_instance, next_action = \
    #                 show_discrepancies_and_ask_correct(
    #                     instance=curr_instance,
    #                     config=config,
    #                     df_stats=df_stats,
    #                     idx_passage = idx_passage
    #                 )
    #
    #             hash_id_to_instance[curr_hash_id] = curr_instance
    #
    #             if next_action == 2:
    #                 check_instance = False
    #                 break
    #             if next_action == 3:
    #                 logger.info('BEGIN recalculating statistics')
    #                 df_stats = get_annotation_statistics(
    #                     annotated_instances=list(hash_id_to_instance.values()),
    #                     annotator_names=[annotator_name] + \
    #                                     annotators_to_compare
    #                 )
    #                 logger.info('END recalculating statistics')
    #
    #             nr_annotated_per_tkgu_operation = update_count_annotated(
    #                 p_instance=curr_instance,
    #                 p_annotator_name=config['annotator_name'],
    #                 p_nr_annotated_per_tkgu_operation=nr_annotated_per_tkgu_operation
    #             )
    #             out_file.write(json.dumps(curr_instance, ensure_ascii=False) + '\n')
    #             out_file.flush()
    #     logger.info('END loading already annotated content')
    #     logger.info(f'nr_annotated_per_tkgu_operation: '
    #                 f'{nr_annotated_per_tkgu_operation}')

    ####################################

    logger.info('END annotating')
