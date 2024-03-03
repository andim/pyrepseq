__all__ = [
    "Levenshtein",
    "WeightedLevenshtein"
]

from numpy import ndarray
from pyrepseq.metric import Metric
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from scipy.spatial import distance
from typing import Iterable, Tuple


class WeightedLevenshtein(Metric):
    _edit_type_weights: Tuple[int]

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
    ) -> None:
        self._edit_type_weights = (insertion_weight, deletion_weight, substitution_weight)

    def calc_cdist_matrix(self, anchors: Iterable[str], comparisons: Iterable[str]) -> ndarray:
        return process.cdist(anchors, comparisons, scorer=self._levenshtein_scorer)

    def _levenshtein_scorer(self, *args, **kwargs) -> int:
        return Levenshtein.distance(
            *args, **kwargs, weights=self._edit_type_weights
        )

    def calc_pdist_vector(self, instances: Iterable[str]) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(instances, instances)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector
    

class Levenshtein(Metric):
    def __init__(self) -> None:
        self._weighted_levenshtein = WeightedLevenshtein()
    
    def calc_cdist_matrix(self, anchors: Iterable[str], comparisons: Iterable[str]) -> ndarray:
        return self._weighted_levenshtein.calc_cdist_matrix(anchors, comparisons)

    def calc_pdist_vector(self, instances: Iterable[str]) -> ndarray:
        return self._weighted_levenshtein.calc_pdist_vector(instances)