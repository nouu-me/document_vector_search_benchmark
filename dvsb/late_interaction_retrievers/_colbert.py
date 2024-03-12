from typing import Union

from colbert import Indexer, IndexUpdater, Searcher
from colbert.data import Collection, Queries

# from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.infra import ColBERTConfig, Run, RunConfig

from .late_interaction_retriever import LATE_INTERACTION_REGISTRY, LateInteractionRetriever

__all__ = [
    "Indexer",
    "Searcher",
    "IndexUpdater",
    "Run",
    "RunConfig",
    "ColBERTConfig",
    "Queries",
    "Collection",
    "Indexer",
    "Searcher",
]


@LATE_INTERACTION_REGISTRY.register
class ColBERTRetriever(LateInteractionRetriever):
    def __init__(
        self,
        model_name: str,
        doc_maxlen: int = 300,
        kmeans_niters: int = 8,
        nbits: int = 4,
        n_gpu: int = 1,
        overwrite: Union[bool, str] = True,
    ) -> None:
        super().__init__(n_gpu=n_gpu)
        self.model_name = model_name
        self.checkpoint = model_name
        self.config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            checkpoint=self.checkpoint,
        )
        self.overwrite = overwrite

    def get_name(self) -> str:
        return f"ColBERTRetriever-{self.model_name}"

    def get_npgu(self) -> int:
        return self.n_gpu

    def index(self, index_name: str, documents: list[str]) -> None:
        self.index_name = index_name
        self.collection = documents
        indexer = Indexer(checkpoint=self.checkpoint, config=self.config)
        print("Indexing...", len(documents), "documents")
        indexer.index(name=self.index_name, collection=self.collection, overwrite=self.overwrite)

    def query(
        self,
        queries: list[str],
        k: int,
    ) -> list[list[int]]:
        searcher = Searcher(index=self.index_name, collection=self.collection)
        idx2queries = {i: x for i, x in enumerate(queries)}
        search_results = searcher.search_all(idx2queries, k=k)
        results = [[int(item[0]) for item in sublist] for sublist in search_results.todict().values()]
        return results
