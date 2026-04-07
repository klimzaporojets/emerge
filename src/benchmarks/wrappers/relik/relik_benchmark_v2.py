# The difference between relik_benchmark_v2.py and relik_benchmark.py
# is that _v2 adds the ability to use the wiki-temp (the dataset we introduce)
# specific indices for entities
# and relations, making the comparison more fare. The plan is also to include the ability
# to use concrete models fine-tuned on wiki-temp dataset.
import json
import logging
import os.path
from typing import Tuple, List, Dict

from tqdm import tqdm

from relik import Relik
from relik.inference.data.objects import RelikOutput

from benchmark_model import BenchmarkModel
from prediction import Prediction


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class RelikBenchmarkV2(BenchmarkModel):
    def __init__(self, config: Dict, index_to_use: int):
        super().__init__(config['name'])
        self.config = config
        # self.relik_model = Relik.from_pretrained('sapienzanlp/relik-entity-linking-large')
        # https://huggingface.co/relik-ie/relik-cie-large
        index_config = config['snapshots'][index_to_use]

        # if index_config['relation_index_path'] is None:
        #     triple_index = {}
        # else:
        #     triple_index = {}
        if config['task'] == 'BOTH':
            self.entity_index_path = os.path.join(config['entity_index_base_path'],
                                                  index_config['entity_index_path'])
        self.relation_index_path = os.path.join(config['relation_index_base_path'],
                                                index_config['relation_index_path'])

        wikidata_qid_to_label: Dict[str, str] = dict()
        property_qid_to_label: Dict[str, str] = dict()

        if config['task'] == 'BOTH':
            index_param = {
                'span': {
                    '_target_': 'relik.retriever.indexers.inmemory.InMemoryDocumentIndex.from_pretrained',
                    'name_or_path': self.entity_index_path,
                    # 'name_or_path': '/path/to/storage/wikipedia-processing/output/experiments/s08_extract_relik_index/20250220/indexes/2019-01-01',
                    # 'precision': '16',
                    'precision': config['entity_index_precision']
                },
                'triplet': {
                    '_target_': 'relik.retriever.indexers.inmemory.InMemoryDocumentIndex.from_pretrained',
                    'name_or_path': self.relation_index_path,
                    'precision': config['relation_index_precision']
                    # 'name_or_path': 'relik-ie/encoder-e5-small-v2-wikipedia-relations-index',
                    # 'precision': '16',
                    # relik-ie/encoder-e5-small-v2-wikipedia-relations-index
                }
            }
            path_entity_idx = os.path.join(self.entity_index_path, 'documents.jsonl')

            with open(path_entity_idx, 'rt', encoding='utf-8') as infile:
                for curr_line in tqdm(infile, desc='loading wikidata_qid_to_label'):
                    curr_parsed_line = json.loads(curr_line)
                    label = curr_parsed_line['text']
                    wikidata_qid = curr_parsed_line['metadata']['wikidata']
                    wikidata_qid_to_label[wikidata_qid] = label
            path_relation_idx = os.path.join(self.relation_index_path, 'documents.jsonl')

            with open(path_relation_idx, 'rt', encoding='utf-8') as infile:
                for curr_line in tqdm(infile, desc='loading property_qid_to_label'):
                    curr_parsed_line = json.loads(curr_line)
                    label = curr_parsed_line['text']
                    wikidata_qid = curr_parsed_line['metadata']['property']
                    property_qid_to_label[wikidata_qid] = label
        elif config['task'] == 'TRIPLET':
            index_param = {
                'triplet': {
                    '_target_': 'relik.retriever.indexers.inmemory.InMemoryDocumentIndex.from_pretrained',
                    'name_or_path': self.relation_index_path,
                    'precision': config['relation_index_precision']
                    # 'name_or_path': 'relik-ie/encoder-e5-small-v2-wikipedia-relations-index',
                    # 'precision': '16',
                    # relik-ie/encoder-e5-small-v2-wikipedia-relations-index
                }
            }
            path_relation_idx = os.path.join(self.relation_index_path, 'documents.jsonl')

            with open(path_relation_idx, 'rt', encoding='utf-8') as infile:
                for curr_line in tqdm(infile, desc='loading property_qid_to_label'):
                    curr_parsed_line = json.loads(curr_line)
                    label = curr_parsed_line['text']
                    wikidata_qid = curr_parsed_line['metadata']['property']
                    property_qid_to_label[wikidata_qid] = label
        else:
            raise RuntimeError(f'Config task not recognized: {config["task"]}')
        logger.info(f'Instantiating ReLiK with the following index: {index_config}')
        self.model = Relik.from_pretrained(
            config['config_name'],
            device=config['device'],
            index=index_param,
            cache_dir=config['cache_dir'],
        )
        # self.model.index
        logger.info('loading wikidata_qid_to_label')

        self.wikidata_qid_to_label: Dict[str, str] = wikidata_qid_to_label
        self.property_qid_to_label: Dict[str, str] = property_qid_to_label

        # wikidata_qid_to_label = ['wikidata_qid_to_label']
        # property_qid_to_label = ['property_qid_to_label']

        # self.model = self.model.to(config['device'])

    def run(self, texts: List[str], timestamps: List[int] = None,
            **kwargs) -> List[Prediction]:
        relik_out: List[RelikOutput] = self.model(texts)

        to_ret: List[Prediction] = list()
        for idx, curr_relik_out in enumerate(relik_out):
            prediction: Prediction = Prediction()
            text_to_property_id = dict()
            text_to_wikidata_id = dict()
            candidate_triples = curr_relik_out.candidates.triplet
            for curr_cand_triple in candidate_triples:
                for curr_triple in curr_cand_triple:
                    if type(curr_triple) == list:
                        for curr_triple2 in curr_triple:
                            # logger.info(f'curr_triple2: {curr_triple2}')
                            text_to_property_id[curr_triple2.text] = curr_triple2.metadata['property']
                    else:
                        # logger.info(f'curr_triple: {curr_triple}')
                        text_to_property_id[curr_triple.text] = curr_triple.metadata['property']

            # type(relik_out.candidates.span[0])
            candidate_spans = curr_relik_out.candidates.span
            # relik_out.candidates.span[0][0][0]
            for curr_cand_span in candidate_spans:
                for curr_cand in curr_cand_span:
                    if type(curr_cand) == list:
                        for curr_cand2 in curr_cand:
                            text_to_wikidata_id[curr_cand2.text] = curr_cand2.metadata['wikidata']
                    else:
                        text_to_wikidata_id[curr_cand.text] = curr_cand.metadata['wikidata']

            #
            for curr_predicted_triple in curr_relik_out.triplets:
                label_head = curr_predicted_triple.subject.label
                label_tail = curr_predicted_triple.object.label
                text_head = curr_predicted_triple.subject.text
                text_tail = curr_predicted_triple.object.text
                label_relation = curr_predicted_triple.label
                property_id = text_to_property_id[curr_predicted_triple.label]

                curr_textual_triple: Tuple[str, str, str]
                curr_textual_triple = (text_head, label_relation, text_tail)
                # prediction.add_predicted_relation(curr_textual_triple)

                if label_head in text_to_wikidata_id:
                    head_qid = text_to_wikidata_id[label_head]
                else:
                    head_qid = label_head

                if label_tail in text_to_wikidata_id:
                    tail_qid = text_to_wikidata_id[label_tail]
                else:
                    tail_qid = label_tail

                curr_qid_triple: Tuple[str, str, str]
                curr_qid_triple = (head_qid, property_id, tail_qid)
                curr_triple_labels = [head_qid, property_id, tail_qid]
                if curr_triple_labels[0] in self.wikidata_qid_to_label:
                    curr_triple_labels[0] = self.wikidata_qid_to_label[curr_triple_labels[0]]
                if curr_triple_labels[2] in self.wikidata_qid_to_label:
                    curr_triple_labels[2] = self.wikidata_qid_to_label[curr_triple_labels[2]]
                if curr_triple_labels[1] in self.property_qid_to_label:
                    curr_triple_labels[1] = self.property_qid_to_label[curr_triple_labels[1]]
                prediction.add_predicted_triple(predicted_relation=curr_textual_triple,
                                                predicted_triple_qids=curr_qid_triple,
                                                predicted_triple_labels=tuple(curr_triple_labels))

            # logger.info(f'The output of RELIK for {texts[idx]} is: {prediction}')

            to_ret.append(prediction)
        return to_ret
