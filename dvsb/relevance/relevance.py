from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from dvsb.registry import Registry


class Relevance(ABC):
    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def compute(
        self, query_vectors: npt.NDArray[np.float64], context_vectors: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        ...


RELEVANCE_REGISTRY = Registry[Relevance]()
