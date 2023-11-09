from abc import ABC, abstractmethod

from dvsb.registry import Registry

__all__ = ["Dataset", "DATASET_REGISTRY"]


class Dataset(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Returns name of the dataset."""
        ...

    @abstractmethod
    def get_queries(self) -> list[str]:
        """Returns a list of all query strings."""
        ...

    @abstractmethod
    def get_contexts(self) -> list[str]:
        """Returns a list of all context strings."""
        ...

    @abstractmethod
    def get_related_context_locations(self) -> list[list[int]]:
        """Returns a list of all related context locations.
        The i-th returning value (say ans[i], a list of integers) is the related context locations for i-th query."""
        ...

    def get_stats(self) -> dict:
        locations = self.get_related_context_locations()
        return {
            "name": self.get_name(),
            "num_queries": len(self.get_queries()),
            "num_contexts": len(self.get_contexts()),
            "avg_num_related_contexts": sum([len(locs) for locs in locations], 0) / len(locations),
        }


DATASET_REGISTRY = Registry[Dataset]()
