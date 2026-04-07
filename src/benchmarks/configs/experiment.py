from tap import Tap
from typing import Dict, Any, List


class ExperimentConfig(Tap):
    """
    Global experiment configuration.
    """
    config_file: str

    # dataset: str = None
    # seed: int = 0
    input_dataset_base_path: str = None
    output_base_dir: str = None
    wrapper_root: str = None

    # model_name -> raw parameter dict
    models: List[Dict[str, Any]] = None
