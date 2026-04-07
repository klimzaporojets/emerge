# the _v3 interacts with edc_framework_emerge_llm_api, while the previous version
# interacted with edc_framework_emerge_tgi_api. edc_framework_emerge_llm_api uses
# edc/utils/unified_llm_client.py, which is a generic llm client enabling access
# to both locally as well as remotely (e.g., using openai api calls) deployed llms.
# this allows to also use larger llms such as GPT 4o-mini, GPT 4o, GPT 5.1, etc..,
# while still preserving the ability to use locally deployed llms in servers such as
# TGI or VLLM such as Llama8b-instruct or the originally used Mistral8b-instruct.
#
# This interface also allows to track already predicted instances (e.g., by running
# tiny or super-tiny versions of the datasets, so no need to make llm calls for already
# predicted EMERGE instances identified by their hash-ids).
#
import pdb
from argparse import ArgumentParser
import json
import os
import logging
from typing import Dict, List

from edc.edc_framework_emerge_llm_api import EDCEmergeLLMApi
from edc.utils.unified_llm_client import UnifiedLLMClient

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logger = logging.getLogger(__name__)


def parse_tt2kg(file_path) -> List[Dict]:
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            lines.append(
                {
                    'hash_id': data['hash_id'],
                    'passage': data['passage']
                }
            )
    return lines


def main(args):
    """
    Main entry point.
    `args` is expected to be a dict with fully resolved configuration
    (CLI args + config.json merged).
    """

    base_input_dir = args['base_input_dir']
    llm_cfg = args['llm']

    api_key = None
    if llm_cfg.get('api_key_env'):
        api_key = os.environ.get(llm_cfg['api_key_env'])
    logger.info(f'args_value_is: {args}')
    print(f'args_value_is: {args}')
    print(f'llm_cfg_is: {llm_cfg}')
    llm_client = UnifiedLLMClient(
        base_url=llm_cfg['base_url'],
        model=llm_cfg['model'],
        model_max_context=llm_cfg['model_max_context'],
        api_key=api_key,
        concurrency=llm_cfg['concurrency'],
        timeout=llm_cfg['timeout'],
        # backend=llm_cfg.get('backend', 'auto'),
        backend=llm_cfg['backend'],
        # api_version=llm_cfg.get('api_version'),
        api_version=llm_cfg['api_version'],
        mode=llm_cfg['mode'],
        authorization_header_type=llm_cfg.get('authorization_header_type')
    )

    # generation_profiles = args['generation_profiles']

    for curr_snapshot in args['snapshots_to_predict']:
        rel_snapshot_dir = curr_snapshot['input_dir']
        base_snapshot_input_dir = os.path.join(base_input_dir, rel_snapshot_dir)
        #
        args['target_schema_path'] = os.path.join(args['base_target_schema_dir'],
                                                  curr_snapshot['target_schema_path'])
        args['relations_cache_path'] = os.path.join(args['base_relations_cache_dir'],
                                                    curr_snapshot['cache_path'])
        assert os.path.exists(args['target_schema_path'])
        #
        edc = EDCEmergeLLMApi(
            llm_client=llm_client,
            **args
        )
        logger.info(f'edc_emerge_llm_api_just_created_for_snapshot: {curr_snapshot}')
        # pdb.set_trace()
        for filename in os.listdir(base_snapshot_input_dir):
            print('**************************************************************************')
            print(f'*************processing {filename} inside {base_snapshot_input_dir}')
            print('**************************************************************************')

            base_output_dir = args['base_output_dir']
            batch_size = args['batch_size']
            base_input_dir = args['base_input_dir']
            base_input_dir_last_processed = args['base_input_dir_last_processed']

            if not filename.endswith('.jsonl'):
                continue

            snapshot_delta_name = os.path.splitext(os.path.basename(filename))[0]
            input_text_file_path = os.path.join(base_snapshot_input_dir, filename)

            # last_processed_file = os.path.join(
            #     base_output_dir, rel_snapshot_dir,
            #     f'{snapshot_delta_name}_last_processed.json'
            # )
            # last_processed_json_results_list_file = os.path.join(
            #     base_output_dir, rel_snapshot_dir,
            #     f'{snapshot_delta_name}_last_processed_results_list.json'
            # )
            # last_processed_json_canon_file = os.path.join(
            #     base_output_dir, rel_snapshot_dir,
            #     f'{snapshot_delta_name}_last_processed_canon.json'
            # )
            last_processed_json_results_list_file = os.path.join(
                base_input_dir_last_processed, rel_snapshot_dir,
                f'{snapshot_delta_name}_last_processed_results_list.json'
            )
            last_processed_json_canon_file = os.path.join(
                base_input_dir_last_processed, rel_snapshot_dir,
                f'{snapshot_delta_name}_last_processed_canon.json'
            )

            output_processed_json_results_list_file = os.path.join(
                base_output_dir, rel_snapshot_dir,
                f'{snapshot_delta_name}_last_processed_results_list.json'
            )
            output_processed_json_canon_file = os.path.join(
                base_output_dir, rel_snapshot_dir,
                f'{snapshot_delta_name}_last_processed_canon.json'
            )
            # last_processed_content = dict()
            json_results_list = []

            # if not os.path.exists(last_processed_file):
            #     last_processed_dir = os.path.dirname(last_processed_file)
            #     os.makedirs(last_processed_dir, exist_ok=True)
            # else:
            #     with open(last_processed_file, 'r', encoding='utf-8') as f:
            #         last_processed_content = json.load(f)

            if os.path.exists(last_processed_json_results_list_file):
                with open(last_processed_json_results_list_file, 'r', encoding='utf-8') as f:
                    json_results_list = json.load(f)
            nr_last_processed = len(json_results_list)
            json_canon_in_text_list = []
            json_canon_not_in_text_list = []
            if os.path.exists(last_processed_json_canon_file):
                with open(last_processed_json_canon_file, 'r', encoding='utf-8') as f:
                    json_canon = json.load(f)
                    json_canon_in_text_list = json_canon['canon_in_text_list']
                    json_canon_not_in_text_list = json_canon['canon_not_in_text_list']

            # batch_offset = 0
            # if input_text_file_path in last_processed_content:
            #     batch_offset = last_processed_content[input_text_file_path]

            # pdb.set_trace()
            # assert batch_offset == len(json_results_list) == len(json_canon_in_text_list) == \
            #        len(json_canon_not_in_text_list)

            already_processed_hash_ids = set([lstt['hash_id'] for lstt in json_results_list])
            assert len(json_results_list) == \
                   len(json_canon_in_text_list) == \
                   len(json_canon_not_in_text_list)

            input_parser = args['input_parser']
            input_text_list: List[Dict]
            if input_parser is None:
                input_text_list = open(input_text_file_path, 'r').readlines()
            elif input_parser == 'tt2kg':
                input_text_list = parse_tt2kg(input_text_file_path)
            else:
                raise ValueError(f'Unknown input parser {args["input_parser"]}')

            nr_tot_input_list = len(input_text_list)
            input_text_list = [lstt for lstt in input_text_list if \
                               lstt['hash_id'] not in already_processed_hash_ids]
            nr_after_filtering_input_list = len(input_text_list)
            nr_processed_hash_ids = len(already_processed_hash_ids)
            logger.info(f'***** input_file_delta: {filename} --- '
                        f'nr_tot_input_list: {nr_tot_input_list} --- '
                        f'nr_after_filtering_input_list: {nr_after_filtering_input_list} --- '
                        f'nr_processed_hash_ids: {nr_processed_hash_ids} *****')
            #
            # pdb.set_trace()
            batch_offset = 0
            curr_batch_input_text_list = \
                input_text_list[batch_offset:batch_offset + batch_size]
            #
            nr_batch = 0
            if len(curr_batch_input_text_list) == 0:
                print(f'ignoring {filename}, probably already processed before')

            while len(curr_batch_input_text_list) > 0:
                print('starting_extract_kg')
                # pdb.set_trace()
                json_canon_in_text_list, json_canon_not_in_text_list, json_results_list = \
                    edc.extract_kg(
                        input_text_list=curr_batch_input_text_list,
                        json_results_list_final_iter=json_results_list,
                        canon_triplets_list_final=json_canon_in_text_list,
                        canon_triplets_not_in_text_list_final=json_canon_not_in_text_list,
                        # last_processed_idx=batch_offset,
                        # output_dir=None,
                        # refinement_iterations=args['refinement_iterations'],
                    )
                print('ending_extract_kg')
                # pdb.set_trace()

                # BEGIN SAVE RESULTS of the processed batches
                if not os.path.exists(output_processed_json_results_list_file):
                    os.makedirs(os.path.dirname(output_processed_json_results_list_file), exist_ok=True)
                with open(output_processed_json_results_list_file, 'w', encoding='utf-8') as f:
                    json.dump(json_results_list, f, indent=4)
                    f.flush()

                if not os.path.exists(output_processed_json_canon_file):
                    os.makedirs(os.path.dirname(output_processed_json_canon_file), exist_ok=True)
                with open(output_processed_json_canon_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'canon_in_text_list': json_canon_in_text_list,
                        'canon_not_in_text_list': json_canon_not_in_text_list
                    }, f, indent=4)
                    f.flush()
                # END SAVE RESULTS of the processed batches

                batch_offset = batch_offset + len(curr_batch_input_text_list)

                print(f'batch_offset: {batch_offset}, '
                      f'len(json_results_list): {len(json_results_list)}, '
                      f'len(json_canon_in_text_list): {len(json_canon_in_text_list)}, '
                      f'len(json_canon_not_in_text_list): {len(json_canon_not_in_text_list)}')
                # pdb.set_trace()
                assert batch_offset + nr_last_processed \
                       == len(json_results_list) == len(json_canon_in_text_list) == \
                       len(json_canon_not_in_text_list)

                nr_batch += 1
                print(f'nr_processed_batches: {nr_batch} , '
                      f'size current: {len(curr_batch_input_text_list)}')

                curr_batch_input_text_list = input_text_list[
                    batch_offset:batch_offset + batch_size
                ]
            assert batch_offset + nr_last_processed \
                   == len(json_results_list) == len(json_canon_in_text_list) == \
                   len(json_canon_not_in_text_list)
            # BEGIN SAVE RESULTS of the processed batches
            if not os.path.exists(output_processed_json_results_list_file):
                os.makedirs(os.path.dirname(output_processed_json_results_list_file), exist_ok=True)
            with open(output_processed_json_results_list_file, 'w', encoding='utf-8') as f:
                json.dump(json_results_list, f, indent=4)
                f.flush()

            if not os.path.exists(output_processed_json_canon_file):
                os.makedirs(os.path.dirname(output_processed_json_canon_file), exist_ok=True)
            with open(output_processed_json_canon_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'canon_in_text_list': json_canon_in_text_list,
                    'canon_not_in_text_list': json_canon_not_in_text_list
                }, f, indent=4)
                f.flush()
            # END SAVE RESULTS of the processed batches
    print('!! BYE BYE I am done BYE BYE!!!!')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default='experiments/s02_postprocess_results_v2/20250513/config.json',
                        help='A config file with the files to postprocess.')
    parser.add_argument('--logging_verbose', action='store_const', dest='loglevel', const=logging.INFO)
    parser.add_argument('--logging_debug', action='store_const', dest='loglevel', const=logging.DEBUG)
    parser.add_argument('--max_workers',
                        type=int,
                        default=32,
                        help='Max threads to call TGI api.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=10,
                        help='Batch size.')

    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))

    args = vars(args)
    # Merge config into args — command-line args take precedence if set
    for k, v in config.items():
        if args.get(k) is None:
            args[k] = v

    main(args)
