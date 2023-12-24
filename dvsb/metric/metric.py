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
        y_true: list[list[int]],
        scores: npt.NDArray[np.float64] | None = None,
        y_pred: list[list[int]] | None = None,
    ) -> float:
        ...


METRIC_REGISTRY = Registry[Metric]()
