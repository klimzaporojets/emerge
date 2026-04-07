"""
EDC / EDC+ wrapper for the benchmark pipeline.

This wrapper bridges the ModelRunner CLI interface to the edc-tt2kg repo's
s01_run_v3.py entry point. Unlike KG-GEN/RAKG/REBEL wrappers which process
instances directly, EDC+ has its own config-driven execution with:
  - 3-stage LLM pipeline (OIE → Schema Definition → Schema Canonicalization)
  - Its own batch processing and checkpoint/resume logic
  - Per-snapshot processing with separate target schemas

The wrapper receives a single --edc-config argument pointing to the EDC config
JSON file, and forwards it to s01_run_v3.py. The postprocessing step
(s02_postprocess_results_v2.py) is run separately after predictions are done.

Variants controlled by config:
  - EDC+ ICL (non-canonicalized): run_canonicalizer=false, prompt_templates_emerge/
  - EDC+ ICL (canonicalized):     run_canonicalizer=true,  prompt_templates_emerge/
  - EDC plain:                    run_canonicalizer=false, prompt_templates_emerge_original_edc/
  - EDC+ ZS:                     run_canonicalizer=false, prompt_templates_emerge_no_few_shot/

Usage (called by ModelRunner via run.sh):
  python wrapper.py --edc-config <path_to_edc_config.json> [--batch-size N] [--max-workers N]
"""
import argparse
import logging
import subprocess
import sys
import os

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('edc-plus-wrapper')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--edc-config', required=True,
                        help='Path to the EDC config JSON file (s01_run_v3 format)')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for EDC processing')
    parser.add_argument('--max-workers', type=int, default=32,
                        help='Concurrency for embeddings')
    parser.add_argument('--postprocess-config', type=str, default=None,
                        help='Path to s02_postprocess config JSON (optional, run after predictions)')

    args = parser.parse_args()

    logger.info('============ EDC+ WRAPPER PARAMETERS ============')
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')
    logger.info('=================================================')

    # Step 1: Run EDC prediction pipeline (s01_run_v3.py)
    edc_cmd = [
        sys.executable, '-u', 's01_run_v3.py',
        '--config_file', args.edc_config,
        '--batch_size', str(args.batch_size),
        '--max_workers', str(args.max_workers),
    ]

    logger.info(f'Running EDC prediction: {" ".join(edc_cmd)}')
    # Run from the edc-tt2kg repo directory (relative paths in config depend on it)
    edc_repo = os.environ.get('EDC_REPO_PATH')
    if edc_repo is None:
        # Fallback: assume edc-tt2kg is in PYTHONPATH
        edc_repo = None
        for p in sys.path:
            if os.path.exists(os.path.join(p, 's01_run_v3.py')):
                edc_repo = p
                break
    if edc_repo is None:
        raise RuntimeError('Cannot find edc-tt2kg repo. Set EDC_REPO_PATH env var '
                           'or ensure s01_run_v3.py is in PYTHONPATH.')

    subprocess.run(edc_cmd, check=True, cwd=edc_repo)
    logger.info('EDC prediction finished.')

    # Step 2: Run postprocessing (optional)
    if args.postprocess_config:
        postprocess_cmd = [
            sys.executable, '-u', 's02_postprocess_results_v2.py',
            '--config_file', args.postprocess_config,
        ]
        logger.info(f'Running EDC postprocessing: {" ".join(postprocess_cmd)}')
        subprocess.run(postprocess_cmd, check=True, cwd=edc_repo)
        logger.info('EDC postprocessing finished.')

    logger.info('EDC+ wrapper DONE.')


if __name__ == '__main__':
    main()
