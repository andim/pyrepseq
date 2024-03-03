__all__ = ["TcrMetric"]


from abc import abstractmethod, ABC
from numpy import ndarray
from pandas import DataFrame
from pyrepseq.metric import Metric


class TcrMetric(Metric):
    """
    Base abstract class for all metrics that operate on TCR .
    TcrMetrics should expect DataFrames with each row representing a TCR, in the standard pyrepseq format (see :py:func:`pyrepseq.io.standardize_dataframe`).
    The input DataFrames must also have at least one TCR-related column.
    Furthermore, if the input DataFrame(s) do not have the required column for the function of the specific metric, the metric will throw a ValueError explaining which columns are missing.
    All values in the table should be IMGT-standardized.
    """

    @abstractmethod
    def calc_cdist_matrix(
        self, anchors: DataFrame, comparisons: DataFrame
    ) -> ndarray:
        if not is_in_standard_format(anchors):
            raise ValueError("`anchors` must be a DataFrame in standard pyrepseq format (see https://pyrepseq.readthedocs.io/en/latest/api.html#pyrepseq.io.standardize_dataframe).")
        if not is_in_standard_format(comparisons):
            raise ValueError("`comparisons` must be a DataFrame in standard pyrepseq format (see https://pyrepseq.readthedocs.io/en/latest/api.html#pyrepseq.io.standardize_dataframe).")

    @abstractmethod
    def calc_pdist_vector(self, instances: DataFrame) -> ndarray:
        if not is_in_standard_format(instances):
            raise ValueError("`instances` must be a DataFrame in standard pyrepseq format (see https://pyrepseq.readthedocs.io/en/latest/api.html#pyrepseq.io.standardize_dataframe).")


def is_in_standard_format(input) -> bool:
    if not isinstance(input, DataFrame):
        return False
    
    tcr_columns = {"TRAV", "CDR3A", "TRAJ", "TRBV", "CDR3B", "TRBJ"}
    input_columns = set(input.columns)
    input_tcr_columns = tcr_columns.intersection(input_columns)

    if len(input_tcr_columns) == 0:
        return False
    
    return True