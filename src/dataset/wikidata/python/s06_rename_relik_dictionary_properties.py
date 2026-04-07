import argparse
import json
import logging
import os

from tqdm import tqdm

import os

# s13_rename_relik_index.json

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=getattr(logging, os.environ.get('LOGGING_LEVEL', 'INFO')))
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=False, type=str,
                        default='experiments/s06_rename_relik_dictionary_properties/20250319/s06_rename_relik_dictionary_properties.json',
                        help='The config file that contains all the parameters')

    args = parser.parse_args()

    config = json.load(open(args.config_file, 'rt'))
    # config['nr_threads_processor'] = args.nr_threads_processor
    # config['dry_run'] = args.dry_run
    # dry_run = config['dry_run']
    # config['api_ports_wiki_mapping'] = args.api_ports_wiki_mapping

    output_path = config['output_path']
    input_path = config['input_path']

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(input_path):
        if filename.endswith('.jsonl'):  # Change the extension as needed
            logger.info(f'processing {filename}')
        output_file_path = os.path.join(output_path, filename)
        input_file_path = os.path.join(input_path, filename)
        output_file = open(output_file_path, 'wt', encoding='utf-8')
        for curr_line_idx, curr_line in tqdm(enumerate(open(input_file_path, 'rt', encoding='utf-8'))):
            curr_parsed_line = json.loads(curr_line)
            # from relik.retriever.indexers.base.BaseDocumentIndex.get_passage_from_document
            # doesn't seem we need to put everything in metadata, so commenting next lines
            curr_parsed_line['metadata'] = {
                'definition': curr_parsed_line['metadata']['definition'],
                'property': curr_parsed_line['metadata']['property']
            }
            # curr_parsed_line['metadata']['qid']=curr_parsed_line['qid']
            # curr_parsed_line['metadata']['page_id']=curr_parsed_line['page_id']
            # curr_parsed_line['metadata']['revision_id']=curr_parsed_line['revision_id']
            # curr_parsed_line['metadata']['revision_timestamp']=curr_parsed_line['revision_timestamp']
            # curr_parsed_line['metadata']['revision_date']=curr_parsed_line['revision_date']

            curr_parsed_line = {
                'text':curr_parsed_line['text'],
                'id': curr_line_idx,
                'metadata': curr_parsed_line['metadata']
            }
            output_file.write(json.dumps(curr_parsed_line, ensure_ascii=False) + '\n')

        output_file.flush()
        output_file.close()
