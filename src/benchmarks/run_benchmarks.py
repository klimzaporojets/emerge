"""
Benchmark orchestrator. Reads a config file and dispatches model execution
to per-model wrappers via ModelRunner.
"""
import json
import logging
import os
from pathlib import Path

from benchmarks.configs.experiment import ExperimentConfig
from benchmarks.configs.kggen import KGGenConfig
from benchmarks.configs.rakg import RAKGConfig
from benchmarks.configs.rebel import REBELConfig
from benchmarks.model_runner import ModelRunner

# Logging level configurable via LOGGING_LEVEL env var (e.g. DEBUG, INFO, WARNING)
_log_level_name = os.environ.get('LOGGING_LEVEL', '').strip()
_log_level = logging._nameToLevel.get(_log_level_name, logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=_log_level)
logger = logging.getLogger(__name__)


def main():
    arguments_main = ExperimentConfig(explicit_bool=True).parse_args()
    logger.info(f'After parse_args(): {arguments_main.config_file}')

    # Step 2: Load JSON config if provided
    if arguments_main.config_file:
        logger.info(f'Loading JSON config from {arguments_main.config_file}')
        with open(arguments_main.config_file) as f:
            config_data = json.load(f)

        # Step 3: Merge JSON config into arguments (but keep CLI priority)
        # Only update fields not set via CLI
        for key, value in config_data.items():
            if getattr(arguments_main, key, None) == ExperimentConfig().get_default(key):
                setattr(arguments_main, key, value)

    logger.info(f'Dataset: {arguments_main.input_dataset_base_path}')
    logger.info(f'Output predictions: {arguments_main.output_base_dir}')
    models_to_run = set([cm['name'] for cm in arguments_main.models])
    logger.info(f'Models: {list(models_to_run)}')

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    # repo_root = Path(__file__).resolve().parents[1]
    # wrappers_root = repo_root / 'model_wrappers'
    # inputs_dir = repo_root / 'inputs'
    # outputs_dir = repo_root / 'outputs'

    os.makedirs(arguments_main.output_base_dir, exist_ok=True)

    runner = ModelRunner(
        wrappers_root=Path(arguments_main.wrapper_root)
    )

    # --------------------------------------------------
    # KGGen
    # --------------------------------------------------

    for arguments_to_pass in arguments_main.models:
        curr_model = arguments_to_pass['name']
        # if 'kggen' in arguments_main.models:
        if curr_model == 'kggen':
            cfg = KGGenConfig(**arguments_to_pass)
            output_dir = os.path.join(arguments_main.output_base_dir, cfg.output_path)
            os.makedirs(output_dir, exist_ok=True)
            runner.run(
                model_name='kggen',
                args=[
                    '--input', arguments_main.input_dataset_base_path,
                    # '--output', os.path.join(arguments_main.output_base_dir, 'kggen'),
                    '--output', output_dir,
                    '--model', cfg.model,
                    '--temperature', str(cfg.temperature),
                    '--max-workers', str(cfg.max_workers),
                    '--chunk-size', str(cfg.chunk_size),
                    '--base-url', str(cfg.base_url),
                    '--batch-size', str(cfg.batch_size),
                    '--cache-path', str(cfg.cache_path),
                    '--max-tokens', str(cfg.max_tokens)
                ],
            )
        elif curr_model == 'rebel':
            cfg = REBELConfig(**arguments_to_pass)
            output_dir = os.path.join(arguments_main.output_base_dir, cfg.output_path)
            os.makedirs(output_dir, exist_ok=True)
            runner.run(
                model_name='rebel',
                args=[
                    '--input', arguments_main.input_dataset_base_path,
                    '--output', output_dir,
                    '--model', cfg.model,
                    '--batch-size', str(cfg.batch_size),
                    '--cache-path', str(cfg.cache_path),
                    '--device', str(cfg.device),
                    '--max-length', str(cfg.max_length),
                    '--num-beams', str(cfg.num_beams),
                    '--max-workers', str(cfg.max_workers)
                ],
            )
        elif curr_model == 'rakg':
            cfg = RAKGConfig(**arguments_to_pass)
            output_dir = os.path.join(arguments_main.output_base_dir, cfg.output_path)
            os.makedirs(output_dir, exist_ok=True)
            args_to_pass = [
                '--input', arguments_main.input_dataset_base_path,
                '--output', output_dir,
                '--batch-size', str(cfg.batch_size),
                '--max-workers', str(cfg.max_workers),
                '--cache-path', str(cfg.cache_path),
                '--openai-model', str(cfg.openai_model),
                '--openai-embedding-model', str(cfg.openai_embedding_model),
                '--openai-similarity-model', str(cfg.openai_similarity_model),
                '--base-url', str(cfg.base_url),
                '--base-url-embedding-model', str(cfg.base_url_embedding_model),
            ]
            if cfg.use_similarity:
                args_to_pass.append('--use-similarity')

            runner.run(
                model_name='rakg',
                args=args_to_pass,
            )
        elif curr_model in ('relik', 'relik-cie', 'relik-oie'):
            # ReLiK uses in-process execution with per-snapshot indices.
            # The config must include 'relik_config' pointing to the ReLiK config JSON.
            relik_args = [
                '--config', arguments_to_pass['relik_config'],
            ]
            if 'device' in arguments_to_pass:
                relik_args.extend(['--device', str(arguments_to_pass['device'])])

            runner.run(
                model_name='relik',
                args=relik_args,
            )
        elif curr_model in ('edc_plus', 'edc'):
            # EDC/EDC+ uses its own config format (s01_run_v3.py).
            # The experiment config must include 'edc_config' pointing to the
            # EDC config JSON, and optionally 'postprocess_config'.
            edc_args = [
                '--edc-config', arguments_to_pass['edc_config'],
            ]
            if 'batch_size' in arguments_to_pass:
                edc_args.extend(['--batch-size', str(arguments_to_pass['batch_size'])])
            if 'max_workers' in arguments_to_pass:
                edc_args.extend(['--max-workers', str(arguments_to_pass['max_workers'])])
            if 'postprocess_config' in arguments_to_pass:
                edc_args.extend(['--postprocess-config', arguments_to_pass['postprocess_config']])

            runner.run(
                model_name='edc_plus',
                args=edc_args,
            )
        else:
            raise NotImplementedError(f'Unknown model: {curr_model}')


if __name__ == '__main__':
    main()
