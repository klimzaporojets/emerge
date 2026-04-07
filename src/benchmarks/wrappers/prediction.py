from typing import Tuple, List, Set, Optional


class Prediction:
    """Container for a model's predicted triples on a single instance."""

    def __init__(self):
        self.predicted_triples = list()

    def add_predicted_triple(self, predicted_relation: Tuple[str, str, str],
                             predicted_triple_qids: Tuple[Optional[str], Optional[str], Optional[str]],
                             predicted_triple_labels: Tuple[Optional[str], Optional[str], Optional[str]]):
        self.predicted_triples.append({
            'extracted_relation': predicted_relation,
            'triple_qids': predicted_triple_qids,
            'triple_labels': predicted_triple_labels
        })

    def __str__(self):
        return f'Prediction: predicted_triples: {self.predicted_triples}'

    __repr__ = __str__
