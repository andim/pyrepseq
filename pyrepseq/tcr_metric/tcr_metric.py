from abc import abstractmethod, ABC
from numpy import ndarray
from pandas import DataFrame
from typing import Iterable, Union


class TcrMetric(ABC):
    """
    TcrMetrics should expect DataFrames with each row representing a TCR.
    The DataFrames should have the following columns, in no particular order:

    TRAV
    CDR3A
    TRAJ
    TRBV
    CDR3B
    TRBJ

    Values in each column should be IMGT-standardized.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def calc_cdist_matrix(
        self, anchor_tcrs: DataFrame, comparison_tcrs: DataFrame
    ) -> ndarray:
        pass

    @abstractmethod
    def calc_pdist_vector(self, tcrs: DataFrame) -> ndarray:
        pass

    @property
    @abstractmethod
    def distance_bins(self) -> Iterable[Union[float, int]]:
        """
        Iterable containing the bin edges to use when generating tcr distance histograms.
        """
        pass
