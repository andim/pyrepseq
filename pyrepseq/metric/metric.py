from abc import abstractmethod, ABC
from numpy import ndarray
from typing import Iterable


class Metric(ABC):
    """
    Base abstract class for all metrics in pyrepseq.
    This class outlines the interface that all metrics will implement.
    If a variable or function parameter can be any type of metric, then it should be typed to this class.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the metric as a string.
        """
        pass

    @abstractmethod
    def calc_cdist_matrix(
        self, anchors: Iterable, comparisons: Iterable
    ) -> ndarray:
        """
        Calculates a cdist matrix between two collections of objects.

        Parameters
        ----------

        anchors: Iterable
            A collections of objects to measure distances from.

        comparisons: Iterable
            A collection of objects to measure distances to.

        Returns
        -------
        numpy.ndarray
            A matrix of shape (N,M) where N is the number of elements in `anchors` and M is the number of elements in `comparisons`.
            The element in the ith row and jth column will contain the distance between the ith element of `anchors` and the jth element of `comparisons`.
        """
        pass

    @abstractmethod
    def calc_pdist_vector(self, instances: Iterable) -> ndarray:
        """
        Calculates a pdist vector given a collection of objects.

        Parameters
        ----------

        instances: Iterable
            A collection of objects to measure distances between.

        Returns
        -------
        numpy.ndarray
            A vector of shape (N*(N-1)/2,) where N is the number of elements in `instances`.
            The vector contains all distances that are possible between each possible pair of objects in `instances`.
        """
        pass
