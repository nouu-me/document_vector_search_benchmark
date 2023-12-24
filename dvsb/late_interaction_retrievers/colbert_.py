import sys

from colbert import Indexer, Searcher, IndexUpdater
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection

from .late_interaction_retriever import (
    LATE_INTERACTION_REGISTRY,
    LateInteractionRetriever,
)


@LATE_INTERACTION_REGISTRY.register
class ColBERTRetriever(LateInteractionRetriever):
    def __init__(
        self,
        model_name: str,
        doc_maxlen: int = 256,
        kmeans_niters: int = 4,
        nbits: int = 4,
        n_gpu: int = 1,
        query_token: str = "[質問]",
        doc_token: str = "[文書]",
    ) -> None:
        self.model_name = model_name
        self.checkpoint = model_name
        self.n_gpu = n_gpu
        self.config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
            checkpoint=self.checkpoint,
            query_token=query_token,
            doc_token=doc_token,
        )
        self.overwrite = True

    def get_name(self) -> str:
        return f"ColBERTRetriever-{self.model_name}"

    def index(self, index_name: str, documents: list[str]) -> None:
        self.index_name = index_name
        self.collection = documents
        indexer = Indexer(checkpoint=self.checkpoint, config=self.config)
        print("Indexing...", len(documents), "documents")
        indexer.index(
            name=self.index_name, collection=self.collection, overwrite=self.overwrite
        )

    def query(
        self,
        queries: list[str],
        k: int,
    ) -> list[list[int]]:
        searcher = Searcher(index=self.index_name, collection=self.collection)
        queries = {i: x for i, x in enumerate(queries)}
        results = searcher.search_all(queries, k=k)
        print(results)
        results = [
            [int(item[0]) for item in sublist] for sublist in results.todict().values()
        ]
        return results
