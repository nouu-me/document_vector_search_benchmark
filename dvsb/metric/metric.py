from abc import ABC, abstractmethod
from typing import Optional

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
        scores: Optional[npt.NDArray[np.float64]] = None,
        y_pred: Optional[list[list[int]]] = None,
    ) -> float:
        ...


METRIC_REGISTRY = Registry[Metric]()
