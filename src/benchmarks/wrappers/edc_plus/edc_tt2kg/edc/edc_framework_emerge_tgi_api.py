import traceback

from edc.extract_emerge_tgi_api import ExtractorEmergeTgiApi
from edc.schema_canonicalization_emerge_tgi_api import SchemaCanonicalizerEmergeTgiApi
from edc.entity_extraction import EntityExtractor
import edc.utils.llm_utils as llm_utils
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from edc.schema_definition_tgi_api import SchemaDefinerTgiApi
from edc.schema_retriever import SchemaRetriever
from tqdm import tqdm
import csv
import pathlib
import copy
import logging
from sentence_transformers import SentenceTransformer
from importlib import reload
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

reload(logging)
logger = logging.getLogger(__name__)


class EDCEmergeTgiApi:
    def __init__(self, **edc_configuration) -> None:

        # OIE module settings
        self.oie_llm_name = edc_configuration['oie_llm']
        self.oie_prompt_template_file_path = edc_configuration['oie_prompt_template_file_path']
        self.oie_few_shot_example_file_path = edc_configuration['oie_few_shot_example_file_path']

        # Schema Definition module settings
        self.sd_llm_name = edc_configuration['sd_llm']
        self.sd_template_file_path = edc_configuration['sd_prompt_template_file_path']
        self.sd_few_shot_example_file_path = edc_configuration['sd_few_shot_example_file_path']

        # Schema Canonicalization module settings
        self.sc_llm_name = edc_configuration['sc_llm']
        self.sc_embedder_name = edc_configuration['sc_embedder']
        self.sc_template_file_path = edc_configuration['sc_prompt_template_file_path']

        # Refinement settings
        self.sr_adapter_path = edc_configuration['sr_adapter_path']

        self.sr_embedder_name = edc_configuration['sr_embedder']
        self.oie_r_prompt_template_file_path = edc_configuration['oie_refine_prompt_template_file_path']
        self.oie_r_few_shot_example_file_path = edc_configuration['oie_refine_few_shot_example_file_path']

        self.ee_llm_name = edc_configuration['ee_llm']
        self.ee_template_file_path = edc_configuration['ee_prompt_template_file_path']
        self.ee_few_shot_example_file_path = edc_configuration['ee_few_shot_example_file_path']

        self.em_template_file_path = edc_configuration['em_prompt_template_file_path']

        self.initial_schema_path = edc_configuration['target_schema_path']
        self.enrich_schema = edc_configuration['enrich_schema']

        self.relations_cache_path = edc_configuration['relations_cache_path']
        self.tgi_url = edc_configuration['tgi_url']
        self.max_workers = edc_configuration['max_workers']
        #
        self.cuda_device_apptainer = edc_configuration['cuda_device_apptainer']
        self.wait_for_restart_time = edc_configuration['wait_for_restart_time']
        self.restart_apptainer_script_name = edc_configuration['restart_apptainer_script_name']
        self.use_restart_script = edc_configuration['use_restart_script']

        if self.initial_schema_path is not None:
            self.schema = {}
            if edc_configuration['schema_parser'] is None:
                reader = csv.reader(open(self.initial_schema_path, 'r'))
                for row in reader:
                    relation, relation_definition = row
                    self.schema[relation] = relation_definition
            elif edc_configuration['schema_parser'] == 'tt2kg':
                # self.serialized_schema_name = f'{os.path.splitext(edc_configuration['target_schema_path'])[0]}.pt'
                with open(edc_configuration['target_schema_path']) as f:
                    for line in f:
                        property_data = json.loads(line)
                        qid = property_data['metadata']['property']
                        definition = property_data['metadata']['definition']
                        self.schema[qid] = definition
            else:
                raise ValueError(f'Unknown schema parser {edc_configuration["schema_parser"]}')
        else:
            self.schema = {}

        # Load the needed models and tokenizers
        self.needed_model_set = set(
            [self.oie_llm_name, self.sd_llm_name, self.sc_llm_name, self.sc_embedder_name, self.ee_llm_name]
        )

        self.loaded_model_dict = {}

        logging.basicConfig(level=edc_configuration['loglevel'])

        logger.info(f'Model used: {self.needed_model_set}')

    def oie(
            self, input_text_list: List[Dict[str, str]], previous_extracted_triplets_list: List[List[str]] = None,
            free_model=False
    ):
        extractor = ExtractorEmergeTgiApi(
            self.tgi_url,
            max_workers=self.max_workers,
            cuda_device_apptainer=self.cuda_device_apptainer,
            restart_apptainer_script_name=self.restart_apptainer_script_name,
            wait_for_restart_time=self.wait_for_restart_time
        )
        logger.info('extractor_emerge_tgi_api_initialized')
        logger.info(f'previous_extracted_triplets_list is in: {previous_extracted_triplets_list}')
        if previous_extracted_triplets_list is not None:
            # Refined OIE
            logger.info('Running Refined OIE...')

            raise RuntimeError('TODO: kzaporoj: adapt to tgi api')
        else:
            # Normal OIE
            oie_triples_list = [None] * len(input_text_list)
            oie_triples_not_in_text_list = [None] * len(input_text_list)

            entity_hint_list = ['' for _ in input_text_list]
            relation_hint_list = ['' for _ in input_text_list]
            logger.info('Running OIE...')
            oie_few_shot_examples_str = open(self.oie_few_shot_example_file_path).read()
            oie_few_shot_prompt_template_str = open(self.oie_prompt_template_file_path).read()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                if self.use_restart_script:
                    for id_txt, input_text in enumerate(input_text_list):
                        curr_input_text = input_text['passage']
                        futures.append(executor.submit(
                            extractor.extract_v2,
                            curr_input_text,
                            oie_few_shot_examples_str,
                            oie_few_shot_prompt_template_str,
                            None,
                            None,
                            id_txt
                        ))
                else:
                    for id_txt, input_text in enumerate(input_text_list):
                        curr_input_text = input_text['passage']
                        futures.append(executor.submit(
                            extractor.extract,
                            curr_input_text,
                            oie_few_shot_examples_str,
                            oie_few_shot_prompt_template_str,
                            None,
                            None,
                            id_txt
                        ))

                results = [future.result() for future in tqdm(as_completed(futures),
                                                              total=len(input_text_list))]
            nr_triples_list_in_zero = 0
            nr_not_in_text_triples_list_in_zero = 0
            for curr_result, id_txt in results:
                if len(curr_result['extracted_triplets_list']) == 0:
                    nr_triples_list_in_zero += 1
                if len(curr_result['extracted_triplets_not_in_text_list']) == 0:
                    nr_not_in_text_triples_list_in_zero += 1
            print(f'nr_triples_list_in_zero: {nr_triples_list_in_zero} --- '
                  f'nr_not_in_text_triples_list_in_zero: {nr_not_in_text_triples_list_in_zero}')

            for curr_oie_triples_result, id_txt in results:
                oie_triples_list[id_txt] = curr_oie_triples_result['extracted_triplets_list']
                oie_triples_not_in_text_list[id_txt] = curr_oie_triples_result['extracted_triplets_not_in_text_list']

        logger.info('OIE finished.')

        if free_model:
            logger.info('free_model, but not freeing is necessary as using tgi api')

        return oie_triples_list, oie_triples_not_in_text_list, entity_hint_list, relation_hint_list

    def load_model(self, model_name, model_type):
        assert model_type in ['sts', 'hf']  # Either a sentence transformer or a huggingface LLM
        if model_name in self.loaded_model_dict:
            logger.info(f'Model {model_name} is already loaded, reusing it.')
        else:
            logger.info(f'Loading model {model_name}')
            if model_type == 'hf':
                model, tokenizer = (
                    AutoModelForCausalLM.from_pretrained(model_name, device_map='auto'),
                    AutoTokenizer.from_pretrained(model_name),
                )
                self.loaded_model_dict[model_name] = (model, tokenizer)
            elif model_type == 'sts':
                model = SentenceTransformer(model_name, trust_remote_code=True)
                self.loaded_model_dict[model_name] = model
        return self.loaded_model_dict[model_name]

    def schema_definition(self, input_text_list: List[Dict[str, str]], oie_triplets_list: List[List[str]],
                          free_model=False):
        assert len(input_text_list) == len(oie_triplets_list)

        schema_definer = SchemaDefinerTgiApi(tgi_url=self.tgi_url,
                                             max_workers=self.max_workers,
                                             cuda_device_apptainer=self.cuda_device_apptainer,
                                             wait_for_restart_time=self.wait_for_restart_time,
                                             restart_apptainer_script_name=self.restart_apptainer_script_name)

        schema_definition_few_shot_prompt_template_str = open(self.sd_template_file_path).read()
        schema_definition_few_shot_examples_str = open(self.sd_few_shot_example_file_path).read()
        schema_definition_dict_list = [None] * len(input_text_list)
        logger.info('edc_framework_emerge_tgi_api: Running Schema Definition...')

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx_it, input_text in enumerate(input_text_list):
                curr_input_text = input_text['passage']
                futures.append(executor.submit(
                    schema_definer.define_schema,
                    curr_input_text,
                    oie_triplets_list[idx_it],
                    schema_definition_few_shot_examples_str,
                    schema_definition_few_shot_prompt_template_str,
                    idx_it
                ))

            results = [future.result() for future in tqdm(
                as_completed(futures),
                desc='schema_definition',
                total=len(input_text_list)
            )]
        assert len(results) == len(schema_definition_dict_list) == len(input_text_list)
        for curr_schema_definition_dict, idx_it in results:
            schema_definition_dict_list[idx_it] = curr_schema_definition_dict
        assert all(x is not None for x in schema_definition_dict_list), "List contains None values"
        assert len(results) == len(schema_definition_dict_list) == len(input_text_list)

        logger.info('Schema Definition finished.')
        return schema_definition_dict_list

    def schema_canonicalization(
            self,
            input_text_list: List[Dict[str, str]],
            oie_triplets_list: List[List[str]],
            schema_definition_dict_list: List[dict],
            free_model=False,
    ):
        assert len(input_text_list) == len(oie_triplets_list) and len(input_text_list) == len(
            schema_definition_dict_list
        )
        logger.info('Running Schema Canonicalization...')
        print('Running Schema Canonicalization...')

        sc_verify_prompt_template_str = open(self.sc_template_file_path).read()
        logger.info(f'loading self.sc_embedder_name: {self.sc_embedder_name}')
        sc_embedder = self.load_model(self.sc_embedder_name, 'sts')
        logger.info(f'end loading self.sc_embedder_name: {self.sc_embedder_name}')

        logger.info(f'loading self.sc_llm_name: {self.sc_llm_name}')
        logger.info(f'end loading self.sc_llm_name: {self.sc_llm_name}')

        schema_canonicalizer = SchemaCanonicalizerEmergeTgiApi(
            self.schema, sc_embedder,
            self.relations_cache_path,
            tgi_url_verifier=self.tgi_url,
            max_workers=self.max_workers,
            cuda_device_apptainer=self.cuda_device_apptainer,
            wait_for_restart_time=self.wait_for_restart_time,
            restart_apptainer_script_name=self.restart_apptainer_script_name,
            use_restart_script=self.use_restart_script
        )

        input_text_idxs = list()
        sd_dicts = list()
        c_oie_triplets = list()

        for idx, input_text in enumerate(tqdm(input_text_list)):
            oie_triplets = oie_triplets_list[idx]
            if len(oie_triplets) == 0:
                logger.debug(f'WARNING - NO_TRIPLES_FOR {input_text}')
            sd_dict = schema_definition_dict_list[idx]
            sd_dicts.append(sd_dict)
            for oie_triplet in oie_triplets:
                c_oie_triplets.append(oie_triplet)
                input_text_idxs.append(idx)
            ######
        canonicalized_triplets, canon_candidate_dict_list = schema_canonicalizer.canonicalize(
            input_text_list=input_text_list,
            input_text_idxs=input_text_idxs,
            c_oie_triplets=c_oie_triplets,
            open_relation_definition_dicts=sd_dicts,
            verify_prompt_template=sc_verify_prompt_template_str,
            enrich=self.enrich_schema
        )

        canonicalized_triplets_list = canonicalized_triplets
        canon_candidate_dict_per_entry_list = canon_candidate_dict_list

        logger.info('Schema Canonicalization finished.')

        if free_model:
            logger.info(f'Freeing model {self.sc_embedder_name, self.sc_llm_name} as it is no longer needed')
            llm_utils.free_model(sc_embedder)
            if self.sc_embedder_name in self.loaded_model_dict:
                del self.loaded_model_dict[self.sc_embedder_name]

        return canonicalized_triplets_list, canon_candidate_dict_per_entry_list

    def construct_refinement_hint(
            self,
            input_text_list: List[str],
            extracted_triplets_list: List[List[List[str]]],
            include_relation_example='self',
            relation_top_k=10,
            free_model=False,
    ):
        entity_extraction_few_shot_examples_str = open(self.ee_few_shot_example_file_path).read()
        entity_extraction_prompt_template_str = open(self.ee_template_file_path).read()

        entity_merging_prompt_template_str = open(self.em_template_file_path).read()

        entity_hint_list = []
        relation_hint_list = []

        # Initialize entity extractor
        if not llm_utils.is_model_openai(self.ee_llm_name):
            # Load the HF model for Schema Definition
            ee_model, ee_tokenizer = self.load_model(self.ee_llm_name, 'hf')
            entity_extractor = EntityExtractor(model=ee_model, tokenizer=ee_tokenizer)
        else:
            entity_extractor = EntityExtractor(openai_model=self.sd_llm_name)

        # Initialize schema retriever
        sr_embedding_model = self.load_model(self.sr_embedder_name, 'sts')

        schema_retriever = SchemaRetriever(
            self.schema,
            sr_embedding_model,
            None,
            finetuned_e5mistral=False,
        )

        relation_example_dict = {}
        if include_relation_example == 'self':
            # Include an example of where this relation can be extracted
            for idx in range(len(input_text_list)):
                input_text_str = input_text_list[idx]
                extracted_triplets = extracted_triplets_list[idx]
                for triplet in extracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = [{'text': input_text_str, 'triplet': triplet}]
                    else:
                        relation_example_dict[relation].append({'text': input_text_str, 'triplet': triplet})
        else:
            # Todo: allow to pass gold examples of relations
            pass

        for idx in tqdm(range(len(input_text_list))):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                previous_entities.add(triplet[0])
                previous_entities.add(triplet[2])
                previous_relations.add(triplet[1])

            previous_entities = list(previous_entities)
            previous_relations = list(previous_relations)

            # Obtain candidate entities
            extracted_entities = entity_extractor.extract_entities(
                input_text_str, entity_extraction_few_shot_examples_str, entity_extraction_prompt_template_str
            )
            merged_entities = entity_extractor.merge_entities(
                input_text_str, previous_entities, extracted_entities, entity_merging_prompt_template_str
            )
            entity_hint_list.append(str(merged_entities))

            # Obtain candidate relations
            hint_relations = previous_relations

            retrieved_relations = schema_retriever.retrieve_relevant_relations(input_text_str)

            counter = 0

            for relation in retrieved_relations:
                if counter >= relation_top_k:
                    break
                else:
                    if relation not in hint_relations:
                        hint_relations.append(relation)

            candidate_relation_str = ''
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in self.schema:
                    continue

                relation_definition = self.schema[relation]

                candidate_relation_str += f'{relation_idx + 1}. {relation}: {relation_definition}\n'
                if include_relation_example == 'self':
                    if relation not in relation_example_dict:
                        pass
                    else:
                        selected_example = None
                        if len(relation_example_dict[relation]) != 0:
                            selected_example = random.choice(relation_example_dict[relation])
                        if selected_example is not None:
                            candidate_relation_str += f'For example, {selected_example["triplet"]} can be extracted from \'{selected_example["text"]}\'\n'
                        else:
                            pass
            relation_hint_list.append(candidate_relation_str)

        if free_model:
            logger.info(f'Freeing model {self.sr_embedder_name, self.ee_llm_name} as it is no longer needed')
            llm_utils.free_model(sr_embedding_model)
            llm_utils.free_model(ee_model, ee_tokenizer)
            del self.loaded_model_dict[self.sr_embedder_name]
            del self.loaded_model_dict[self.ee_llm_name]
        return entity_hint_list, relation_hint_list

    def extract_kg(self, input_text_list: List[Dict[str, str]],
                   json_results_list_final_iter: List,
                   canon_triplets_list_final: List,
                   canon_triplets_not_in_text_list_final: List,
                   last_processed_idx: int,
                   output_dir: str = None,
                   refinement_iterations=0,
                   ):
        json_results_list = []

        if output_dir is not None:
            for iteration in range(refinement_iterations + 1):
                pathlib.Path(f'{output_dir}/iter{iteration}').mkdir(parents=True, exist_ok=True)

        # EDC run
        logger.info('EDC starts running...')

        required_model_dict = {
            'oie': self.oie_llm_name,
            'sd': self.sd_llm_name,
            'sc_embed': self.sc_embedder_name,
            'sc_verify': self.sc_llm_name,
            'ee': self.ee_llm_name,
            'sr': self.sr_embedder_name,
        }

        triplets_from_last_iteration = None
        canon_triplets_not_in_text_list = None
        canon_triplets_list = None
        for iteration in range(refinement_iterations + 1):
            logger.info(f'Iteration {iteration}:')
            iteration_result_dir = None
            if output_dir is not None:
                iteration_result_dir = f'{output_dir}/iter{iteration}'

            required_model_dict_current_iteration = copy.deepcopy(required_model_dict)

            del required_model_dict_current_iteration['oie']
            oie_triplets_list, oie_triplets_not_in_text_list, entity_hint_list, \
                relation_hint_list = self.oie(
                input_text_list,
                free_model=self.oie_llm_name not in required_model_dict_current_iteration.values()
                           and iteration == refinement_iterations,
                previous_extracted_triplets_list=triplets_from_last_iteration,
            )

            oie_merged_triples = list()

            assert len(oie_triplets_list) == len(oie_triplets_not_in_text_list) == len(input_text_list)
            for curr_oie_triplets, curr_oie_triplets_not_in_text in zip(oie_triplets_list,
                                                                        oie_triplets_not_in_text_list):
                new_oie_triplets = []
                new_oie_triplets_not_in_text = []

                for curr_entry in curr_oie_triplets:
                    try:
                        new_oie_triplets.append(
                            [str(curr_entry[0]).replace('_', ' '),
                             curr_entry[1],
                             str(curr_entry[2]).replace('_', ' ')]
                        )
                    except Exception:
                        print('!!exception in the loop of corr_oie_triplets: '
                              f'for curr_entry_in_curr_oie_triplets: {curr_entry}')
                        traceback.print_exc()
                for curr_entry in curr_oie_triplets_not_in_text:
                    try:
                        new_oie_triplets_not_in_text.append(
                            [str(curr_entry[0]).replace('_', ' '),
                             curr_entry[1],
                             str(curr_entry[2]).replace('_', ' ')]
                        )
                    except Exception:
                        print('!!exception in the loop of corr_oie_triplets_not_in_text: '
                              f'for curr_entry_in_curr_oie_triplets_not_in_text: {curr_entry}')
                        traceback.print_exc()
                curr_oie_triplets = new_oie_triplets
                curr_oie_triplets_not_in_text = new_oie_triplets_not_in_text

                oie_merged_triples.append(curr_oie_triplets + curr_oie_triplets_not_in_text)

            assert len(oie_merged_triples) == len(input_text_list)
            del required_model_dict_current_iteration['sd']
            print('about to invoke self.schema_definition')
            sd_dict_list = self.schema_definition(
                input_text_list,
                oie_merged_triples,
                free_model=self.sd_llm_name not in required_model_dict_current_iteration.values()
                           and iteration == refinement_iterations,
            )
            del required_model_dict_current_iteration['sc_embed']
            del required_model_dict_current_iteration['sc_verify']

            print('****** edc_framework_emerge_tgi_api: invoking self.schema_canonicalization')
            canon_triplets_list, canon_candidate_dict_list = self.schema_canonicalization(
                input_text_list,
                oie_triplets_list,
                sd_dict_list,
                free_model=True
            )
            print('****** edc_framework_emerge_tgi_api: invoking self.schema_canonicalization 2')
            canon_triplets_not_in_text_list, canon_candidate_not_in_text_dict_list = self.schema_canonicalization(
                input_text_list,
                oie_triplets_not_in_text_list,
                sd_dict_list,
                free_model=True
            )
            print('****** edc_framework_emerge_tgi_api: finished with self.schema_canonicalization')
            #
            non_null_triplets_list = [
                [triple for triple in triplets if triple is not None] for triplets in canon_triplets_list
            ]
            #
            triplets_from_last_iteration = non_null_triplets_list

            # Write results
            assert len(oie_triplets_list) == len(sd_dict_list) and len(sd_dict_list) == len(canon_triplets_list) == len(
                canon_triplets_not_in_text_list) == len(oie_triplets_not_in_text_list)
            json_results_list = []
            for idx in range(len(oie_triplets_list)):
                result_json = {
                    'index': idx + last_processed_idx,
                    'hash_id': input_text_list[idx]['hash_id'],
                    'input_text': input_text_list[idx]['passage'],
                    'entity_hint': entity_hint_list[idx],
                    'relation_hint': relation_hint_list[idx],
                    'oie': oie_triplets_list[idx],
                    'oie_not_in_text': oie_triplets_not_in_text_list[idx],
                    'schema_definition': sd_dict_list[idx],
                    'canonicalization_candidates': str(canon_candidate_dict_list[idx]),
                    'schema_canonicalization': canon_triplets_list[idx],
                    'canonicalization_candidates_not_in_text': str(canon_candidate_not_in_text_dict_list[idx]),
                    'schema_canonicalization_not_in_text': canon_triplets_not_in_text_list[idx],
                }
                json_results_list.append(result_json)
            result_at_each_stage_file = None
            if iteration_result_dir is not None:
                result_at_each_stage_file = open(f'{iteration_result_dir}/result_at_each_stage.json', 'w')

            def remove_ellipsis(obj):
                if isinstance(obj, dict):
                    return {k: remove_ellipsis(v) for k, v in obj.items() if v is not Ellipsis}
                elif isinstance(obj, list):
                    return [remove_ellipsis(v) for v in obj if v is not Ellipsis]
                else:
                    return obj

            json_results_list = remove_ellipsis(json_results_list)
            if result_at_each_stage_file is not None:
                json.dump(json_results_list, result_at_each_stage_file, indent=4)

            if iteration_result_dir is not None:
                final_result_file = open(f'{iteration_result_dir}/canon_kg.txt', 'w')
                for idx, canon_triplets in enumerate(non_null_triplets_list):
                    final_result_file.write(str(canon_triplets))
                    if idx != len(canon_triplets_list) - 1:
                        final_result_file.write('\n')
                    final_result_file.flush()
        json_results_list_final_iter = json_results_list_final_iter + json_results_list
        canon_triplets_list_final = canon_triplets_list_final + canon_triplets_list
        canon_triplets_not_in_text_list_final = canon_triplets_not_in_text_list_final + \
                                                canon_triplets_not_in_text_list
        return (
            canon_triplets_list_final,
            canon_triplets_not_in_text_list_final,
            json_results_list_final_iter
        )
