"""
ReLiK wrapper for the benchmark pipeline.

Unlike KG-GEN/RAKG/REBEL which use external wrapper scripts, ReLiK runs
in-process using RelikBenchmarkV2 + ExperimentExecutor. This is because
ReLiK needs to load per-snapshot entity/relation indices into memory and
instantiate the model with custom index configurations.

Two modes:
  - relik-cie (task=BOTH): Closed IE with entity linking + relation extraction.
    Loads both entity and relation indices per snapshot.
  - relik-oie (task=TRIPLET): Open IE with relation extraction only.
    Loads only relation indices per snapshot.

Usage (called by ModelRunner via run.sh):
  python wrapper.py --config <path_to_relik_config.json> --device cuda
"""
import argparse
import json
import logging
import os
import sys
import traceback

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('relik-wrapper')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Path to the ReLiK config JSON file')
    parser.add_argument('--device', default='cuda',
                        help='Device for model execution (e.g. cuda, cuda:0, cpu)')
    args = parser.parse_args()

    logger.info('============ ReLiK WRAPPER PARAMETERS ============')
    logger.info(f'config: {args.config}')
    logger.info(f'device: {args.device}')
    logger.info('==================================================')

    config = json.load(open(args.config, 'rt'))

    from relik_benchmark_v2 import RelikBenchmarkV2
    from experiment_executor import ExperimentExecutor

    for curr_benchmark_model in config['benchmark_models']:
        try:
            curr_benchmark_model['device'] = args.device

            for curr_index in range(len(curr_benchmark_model['snapshots'])):
                logger.info(
                    f'Instantiating ReLiK on {args.device} for snapshot '
                    f'{curr_benchmark_model["snapshots"][curr_index]}'
                )

                relik_benchmark = RelikBenchmarkV2(
                    config=curr_benchmark_model,
                    index_to_use=curr_index
                )

                experiment_executor = ExperimentExecutor(
                    config=config,
                    config_benchmark=curr_benchmark_model,
                    benchmark_model=relik_benchmark,
                    snapshot_idx=curr_index
                )

                logger.info(f'Running ReLiK benchmark for snapshot {curr_index}')
                experiment_executor.run()
                logger.info(f'Finished ReLiK benchmark for snapshot {curr_index}')

        except Exception as e:
            logger.error(f'Error running ReLiK: {e}')
            traceback.print_exc()

    logger.info('ReLiK wrapper DONE.')


if __name__ == '__main__':
    main()
