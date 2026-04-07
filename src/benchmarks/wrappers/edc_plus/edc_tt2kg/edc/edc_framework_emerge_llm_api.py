import asyncio
import csv
import json
import logging
import pdb
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from edc.extract_emerge_llm_api import ExtractorEmergeLLMApi
from edc.schema_definition_llm_api import SchemaDefinerLLMApi
from edc.schema_canonicalization_emerge_llm_api import SchemaCanonicalizerEmergeLLMApi
from edc.utils.unified_llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)


class EDCEmergeLLMApi:
    '''
    Async-first EDC pipeline orchestrator.

    This class is behavior-compatible with EDCEmergeTgiApi, but:
      - uses UnifiedLLMClient-based modules
      - uses asyncio instead of ThreadPoolExecutor
      - keeps a synchronous public API
    '''

    def __init__(self, llm_client: UnifiedLLMClient, **edc_configuration) -> None:
        self.llm = llm_client
        self.run_canonicalizer = edc_configuration['run_canonicalizer']
        #
        # ---------------- OIE ----------------
        self.oie_llm_generation_profile = edc_configuration['generation_profiles']['GEN_OIE']
        self.oie_prompt_template_file_path = edc_configuration['oie_prompt_template_file_path']
        self.oie_few_shot_example_file_path = edc_configuration['oie_few_shot_example_file_path']

        # ---------------- Schema Definition ----------------
        self.sd_llm_generation_profile = edc_configuration['generation_profiles']['GEN_SD']
        self.sd_template_file_path = edc_configuration['sd_prompt_template_file_path']
        self.sd_few_shot_example_file_path = edc_configuration['sd_few_shot_example_file_path']

        # ---------------- Schema Canonicalization ----------------
        self.sc_verify_llm_generation_profile = edc_configuration['generation_profiles']['GEN_SC_VERIFY']
        self.sc_embedder_name = edc_configuration['sc_embedder']
        self.sc_template_file_path = edc_configuration['sc_prompt_template_file_path']

        # ---------------- Refinement ----------------
        # self.gen_ee_llm_generation_profile = edc_configuration['generation_profiles']['GEN_EE']
        self.sr_adapter_path = edc_configuration['sr_adapter_path']
        self.sr_embedder_name = edc_configuration['sr_embedder']

        self.ee_llm_name = edc_configuration['ee_llm']
        self.ee_template_file_path = edc_configuration['ee_prompt_template_file_path']
        self.ee_few_shot_example_file_path = edc_configuration['ee_few_shot_example_file_path']
        self.em_template_file_path = edc_configuration['em_prompt_template_file_path']

        # ---------------- Schema ----------------
        self.initial_schema_path = edc_configuration['target_schema_path']
        self.enrich_schema = edc_configuration['enrich_schema']
        self.relations_cache_path = edc_configuration['relations_cache_path']

        self.max_workers = edc_configuration['max_workers']
        self.loaded_model_dict = {}

        logging.basicConfig(level=edc_configuration['loglevel'])

        # ---------------- Load schema ----------------
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

        # ---------------- Initialize modules ----------------
        self.extractor = ExtractorEmergeLLMApi(
            llm_client=self.llm,
            oie_llm_generation_profile=self.oie_llm_generation_profile
        )

        if self.run_canonicalizer:
            self.schema_definer = SchemaDefinerLLMApi(
                llm_client=self.llm,
                sd_llm_generation_profile=self.sd_llm_generation_profile
            )
            self.embedder = self.load_model(self.sc_embedder_name, 'sts')

            logger.info('begin_creating_SchemaCanonicalizerEmergeLLMApi')
            self.canonicalizer = SchemaCanonicalizerEmergeLLMApi(
                target_schema_dict=self.schema,
                embedder=self.embedder,
                relations_cache_path=self.relations_cache_path,
                llm_client=self.llm,
                max_workers=self.max_workers,
                sc_verify_llm_generation_profile=self.sc_verify_llm_generation_profile
            )
            logger.info('end_creating_SchemaCanonicalizerEmergeLLMApi')

    # ==================================================================
    # Model loading (unchanged semantics)
    # ==================================================================
    def load_model(self, model_name, model_type):
        assert model_type in ['sts', 'hf']

        if model_name in self.loaded_model_dict:
            logger.info(f'Model {model_name} already loaded, reusing.')
            return self.loaded_model_dict[model_name]

        logger.info(f'Loading model {model_name}')
        if model_type == 'hf':
            model, tokenizer = (
                AutoModelForCausalLM.from_pretrained(model_name, device_map='auto'),
                AutoTokenizer.from_pretrained(model_name),
            )
            self.loaded_model_dict[model_name] = (model, tokenizer)
        else:
            model = SentenceTransformer(model_name, trust_remote_code=True)
            self.loaded_model_dict[model_name] = model

        return self.loaded_model_dict[model_name]

    async def _run_oie(self, input_text_list):
        few_shot = open(self.oie_few_shot_example_file_path).read()
        template = open(self.oie_prompt_template_file_path).read()

        tasks = [
            self.extractor.extract_async(
                input_text_str=input_text['passage'],
                few_shot_examples_str=few_shot,
                prompt_template_str=template,
                entities_hint=None,
                relations_hint=None,
                id_txt=idx
            )
            for idx, input_text in enumerate(input_text_list)
        ]

        results = await asyncio.gather(*tasks)

        oie = [None] * len(input_text_list)
        oie_not = [None] * len(input_text_list)

        for res, idx in results:
            oie[idx] = res['extracted_triplets_list']
            oie_not[idx] = res['extracted_triplets_not_in_text_list']

        return oie, oie_not

    async def _run_schema_definition(self, input_text_list, merged_triplets):
        assert len(input_text_list) == len(merged_triplets)

        few_shot = open(self.sd_few_shot_example_file_path).read()
        template = open(self.sd_template_file_path).read()

        tasks = [
            self.schema_definer.define_schema(
                input_text_str=input_text_list[idx]['passage'],
                extracted_triplets_list=merged_triplets[idx],
                few_shot_examples_str=few_shot,
                prompt_template_str=template,
                idx_it=idx
            )
            for idx in range(len(input_text_list))
        ]

        results = await asyncio.gather(*tasks)

        schema_definition_lst = [None] * len(input_text_list)
        for d, idx in results:
            schema_definition_lst[idx] = d

        assert len(results) == len(schema_definition_lst) == len(input_text_list)
        assert all(x is not None for x in schema_definition_lst), 'List contains None values'
        assert len(results) == len(schema_definition_lst) == len(input_text_list)

        return schema_definition_lst

    async def _run_schema_canonicalization(self, input_text_list, oie_triplets_list, sd_dict_list):
        assert len(input_text_list) == len(oie_triplets_list) and len(input_text_list) == len(
            sd_dict_list
        )

        verify_template = open(self.sc_template_file_path).read()
        # embedder = self.load_model(self.sc_embedder_name, 'sts')
        #
        # logger.info('begin_creating_SchemaCanonicalizerEmergeLLMApi')
        # # pdb.set_trace()
        # canonicalizer = SchemaCanonicalizerEmergeLLMApi(
        #     target_schema_dict=self.schema,
        #     embedder=embedder,
        #     relations_cache_path=self.relations_cache_path,
        #     llm_client=self.llm,
        #     max_workers=self.max_workers,
        #     sc_verify_llm_generation_profile=self.sc_verify_llm_generation_profile
        # )
        # logger.info('end_creating_SchemaCanonicalizerEmergeLLMApi')
        # pdb.set_trace()

        input_text_idxs = []
        c_oie_triplets = []

        for idx, triplets in enumerate(oie_triplets_list):
            for t in triplets:
                c_oie_triplets.append(t)
                input_text_idxs.append(idx)
        logger.info('start_canonicalize_async')
        return await self.canonicalizer.canonicalize_async(
            input_text_list=input_text_list,
            input_text_idxs=input_text_idxs,
            c_oie_triplets=c_oie_triplets,
            open_relation_definition_dicts=sd_dict_list,
            verify_prompt_template=verify_template,
            enrich=self.enrich_schema,
        )

    # ==================================================================
    # Public API (sync, behavior-compatible)
    # ==================================================================
    def extract_kg(
            self,
            input_text_list: List[Dict[str, str]],
            json_results_list_final_iter: List,
            canon_triplets_list_final: List,
            canon_triplets_not_in_text_list_final: List,
            # last_processed_idx: int,
            # output_dir: str = None,
            # refinement_iterations: int = 0,
    ):
        async def _run():
            logger.info('start_run_oie')
            oie_triplets_list, oie_triplets_not_in_text_list = await self._run_oie(input_text_list)
            logger.info('end_run_oie')
            assert len(oie_triplets_list) == len(oie_triplets_not_in_text_list) == len(input_text_list)
            # pdb.set_trace()
            oie_merged_triples = []
            logger.info('start_merging')
            for a, b in zip(oie_triplets_list, oie_triplets_not_in_text_list):
                oie_merged_triples.append(a + b)
            logger.info('end_merging')
            # pdb.set_trace()
            assert len(oie_merged_triples) == len(input_text_list)

            if self.run_canonicalizer:
                logger.info('start_run_schema_definition')
                sd_dict_list = await self._run_schema_definition(input_text_list, oie_merged_triples)
                logger.info('end_run_schema_definition')
                # pdb.set_trace()
                logger.info('start_run_schema_canonicalization')
                #
                canon_triplets_list, canon_candidate_dict_list = await self._run_schema_canonicalization(
                    input_text_list, oie_triplets_list, sd_dict_list
                )
                #
                logger.info('end_run_schema_canonicalization')
                # pdb.set_trace()

                canon_triplets_not_in_text_list, canon_candidate_not_in_text_dict_list = \
                    await self._run_schema_canonicalization(
                        input_text_list, oie_triplets_not_in_text_list, sd_dict_list
                    )
                # Write results
                assert len(oie_triplets_list) == len(sd_dict_list) and \
                       len(sd_dict_list) == len(canon_triplets_list) == len(
                    canon_triplets_not_in_text_list) == len(oie_triplets_not_in_text_list)
            else:
                sd_dict_list = dict()
                canon_candidate_dict_list = dict()
                canon_triplets_list = dict()
                canon_candidate_not_in_text_dict_list = dict()
                canon_triplets_not_in_text_list = dict()
                for idx in range(len(oie_triplets_list)):
                    sd_dict_list[idx] = dict()
                    canon_candidate_dict_list[idx] = ''
                    canon_triplets_list[idx] = list()
                    canon_candidate_not_in_text_dict_list[idx] = list()
                    canon_triplets_not_in_text_list[idx] = list()
            ## BEGIN CHANGE
            json_results_list = []
            for idx in range(len(input_text_list)):
                result_json = {
                    # 'index': idx + last_processed_idx,
                    'hash_id': input_text_list[idx]['hash_id'],
                    'input_text': input_text_list[idx]['passage'],
                    'oie': oie_triplets_list[idx],
                    'oie_not_in_text': oie_triplets_not_in_text_list[idx],
                    'schema_definition': sd_dict_list[idx],
                    'canonicalization_candidates': str(canon_candidate_dict_list[idx]),
                    'schema_canonicalization': canon_triplets_list[idx],
                    'canonicalization_candidates_not_in_text': str(canon_candidate_not_in_text_dict_list[idx]),
                    'schema_canonicalization_not_in_text': canon_triplets_not_in_text_list[idx],
                }
                json_results_list.append(result_json)
            ## END CHANGE

            return canon_triplets_list, canon_triplets_not_in_text_list, json_results_list

        ## BEGIN CHANGE
        async def _run_with_llm():
            async with self.llm:
                return await _run()

        ## END CHANGE
        ## BEGIN CHANGE
        # canon, canon_not = asyncio.run(_run_with_llm())
        canon, canon_not, json_results_list = asyncio.run(_run_with_llm())
        # pdb.set_trace()
        ## END CHANGE
        # canon, canon_not = asyncio.run(_run())

        canon_triplets_list_final += canon
        canon_triplets_not_in_text_list_final += canon_not
        json_results_list_final_iter = json_results_list_final_iter + json_results_list

        return (
            canon_triplets_list_final,
            canon_triplets_not_in_text_list_final,
            json_results_list_final_iter,
        )
