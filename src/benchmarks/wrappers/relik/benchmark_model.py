from typing import List, Dict

from benchmarks.prediction import Prediction


class BenchmarkModel:
    def __init__(self, name: str):
        self.name = name
        self.entity_index_path = ''
        self.relation_index_path = ''

    def get_name(self) -> str:
        return self.name

    def run(self, texts: List[str], timestamps: List[int] = None,
            mentions_to_qids:List[Dict[str,str]]=None) -> List[Prediction]:
        pass
