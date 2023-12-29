from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from dvsb.registry import Registry


class Embedding(ABC):
    @abstractmethod
    def load(self, has_cuda: bool) -> None:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        ...


EMBEDDING_REGISTRY = Registry[Embedding]()
