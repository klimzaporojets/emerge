from typing import Optional, Dict, Any, List

from tap import Tap


class ArgsEvaluation(Tap):
    """Configuration for the EMERGE evaluation pipeline (models, metrics, paths, scoring mode)."""
    config_file: Optional[str] = None

    # paths
    output_path: Optional[str] = None
    cache_path: Optional[str] = None
    # load_results_from_cache: when True, skip the entire preloader + evaluation pipeline
    # and load the cached WikiEvalResult from cache_path directly. Only use this after a
    # full run has already populated the cache. The pipeline also has incremental caching:
    # even with this set to False, already-computed instances are skipped automatically.
    load_results_from_cache: bool = False
    input_dataset_path: Optional[str] = None
    caches_dir_llm: Optional[str] = None
    kg_snapshots_path: Optional[str] = None

    # numeric
    llm_cache_flush_every: Optional[int] = None

    # mappings / structured config
    snapshot_year_to_kg_file: Optional[Dict[str, str]] = None
    metrics_to_calculate: Optional[Dict[str, Any]] = None
    llm_assessors: Optional[Dict[str, str]] = None

    # surface form control
    use_triple_labels_as_surface_forms: bool = False

    # scoring control: when True, instances where a model produces no predictions
    # are scored as P=0/R=0/F1=0 instead of being excluded from the average.
    # Default True (correct behavior). Set to False to reproduce legacy behavior.
    score_empty_predictions_as_zero: bool = True

    # lists
    models_to_evaluate: Optional[List[str]] = None
    models_to_report: Optional[List[str]] = None