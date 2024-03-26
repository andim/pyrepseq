__all__ = [
    "Levenshtein",
    "WeightedLevenshtein"
]

from numpy import ndarray
from pyrepseq.metric import Metric
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein as RapidFuzzLevenshtein
from scipy.spatial import distance
from typing import Iterable, Tuple


class WeightedLevenshtein(Metric):
    """
    A generalised Levenshtein distance which supports different weights for insertions, deletions, and substitutions.

    Parameters
    ----------
    insertion_weight: int
        An integer multiplier for insertions
        Defaults to 1.
        
    deletion_weight: int
        An integer multiplier for deletions
        Defaults to 1.

    substitution_weight: int
        An integer multiplier for substitutions
        Defaults to 1.
    """

    name = "WeightedLevenshtein"
    _edit_type_weights: Tuple[int]

    def __init__(
        self,
        insertion_weight: int = 1,
        deletion_weight: int = 1,
        substitution_weight: int = 1,
    ) -> None:
        if insertion_weight == 1 and deletion_weight == 1 and substitution_weight == 1:
            self._scorer = RapidFuzzLevenshtein.distance
        
        else:
            self._scorer = lambda *args, **kwargs: RapidFuzzLevenshtein.distance(
                *args, **kwargs, weights=(insertion_weight, deletion_weight, substitution_weight)
            )

    def calc_cdist_matrix(self, anchors: Iterable[str], comparisons: Iterable[str]) -> ndarray:
        return process.cdist(anchors, comparisons, scorer=self._scorer, workers=-1)

    def calc_pdist_vector(self, instances: Iterable[str]) -> ndarray:
        pdist_matrix = self.calc_cdist_matrix(instances, instances)
        pdist_vector = distance.squareform(pdist_matrix, checks=False)
        return pdist_vector
    

class Levenshtein(Metric):
    """
    Levenshtein distance, also known as edit distance.
    """

    name = "Levenshtein"

    def __init__(self) -> None:
        self._weighted_levenshtein = WeightedLevenshtein()
    
    def calc_cdist_matrix(self, anchors: Iterable[str], comparisons: Iterable[str]) -> ndarray:
        return self._weighted_levenshtein.calc_cdist_matrix(anchors, comparisons)

    def calc_pdist_vector(self, instances: Iterable[str]) -> ndarray:
        return self._weighted_levenshtein.calc_pdist_vector(instances)