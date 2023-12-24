from abc import ABC, abstractmethod
from typing import Any
from dvsb.registry import Registry


class LateInteractionRetriever(ABC):
    @abstractmethod
    def get_name(self) -> str:
        ...

    @abstractmethod
    def index(self, index_name: str, documents: list[str]) -> Any:
        ...

    @abstractmethod
    def query(
        self,
        queries: list[str],
        k: int,
    ) -> list[list[int]]:
        ...


LATE_INTERACTION_REGISTRY = Registry[LateInteractionRetriever]()
