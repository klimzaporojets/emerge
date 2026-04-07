from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


@dataclass
class WikiEvalResult:
    """Holds all evaluation state: preloaded batches, metric DataFrames, and predictions.

    This object is serialized to the cache .pkl file between runs, enabling incremental evaluation.
    """
    batch_factualness_gt: Optional[Any] = None
    batch_factualness_openie: Optional[Any] = None
    batch_completeness_gt: Optional[Any] = None
    batch_completeness_openie: Optional[Any] = None
    batch_cie_qid_triples: Optional[Any] = None

    df_predictions_cie_and_gt: Optional[pd.DataFrame] = None
    df_predictions_open_ie: Optional[pd.DataFrame] = None
    df_metrics_cie: Optional[pd.DataFrame] = None
    df_metrics_open_ie: Optional[pd.DataFrame] = None
    df_instances: Optional[pd.DataFrame] = None
    df_metrics_additional_triple_stats: Optional[pd.DataFrame] = None
