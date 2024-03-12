from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from dvsb.registry import Registry


class Metric(ABC):
    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def compute(
        self,
        y_true: list[list[int]] | npt.NDArray[np.int_],
        y_pred: list[list[int]] | npt.NDArray[np.int_],
    ) -> float:
        ...


METRIC_REGISTRY = Registry[Metric]()
