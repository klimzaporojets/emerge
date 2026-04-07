import logging
import os
import subprocess
from pathlib import Path
from typing import List

_log_level_name = os.environ.get('LOGGING_LEVEL', '').strip()
_log_level = logging._nameToLevel.get(_log_level_name, logging.INFO)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=_log_level)
logger = logging.getLogger(__name__)


class ModelRunner:
    """Executes benchmark model wrappers by invoking their run.sh scripts."""

    def __init__(self, wrappers_root: Path):
        self.wrappers_root = wrappers_root

    def run(self, model_name: str, args: List[str]):
        """Run a model's wrapper script (run.sh) with the given arguments."""
        run_sh = self.wrappers_root / model_name / 'run.sh'

        if not run_sh.exists():
            raise FileNotFoundError(f"run.sh not found for model '{model_name}' "
                                    f"run-sh: {run_sh}")

        cmd = [str(run_sh)] + args
        logger.info(f'invoking_cmd: {cmd}')
        subprocess.run(cmd, check=True)
        logger.info(f'FINISHED_RUNNING {model_name}!!!')
