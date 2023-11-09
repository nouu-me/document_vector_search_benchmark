import numpy as np
import numpy.typing as npt
from dvsb.relevance.relevance import RELEVANCE_REGISTRY, Relevance


@RELEVANCE_REGISTRY.register
class Cosine(Relevance):
    def get_name(self) -> str:
        return "Cosine"

    def compute(
        self, query_vectors: npt.NDArray[np.float64], context_vectors: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
        context_vectors /= np.linalg.norm(context_vectors, axis=1, keepdims=True)
        cosine_similarity: npt.NDArray[np.float64] = query_vectors.dot(context_vectors.T)
        return cosine_similarity
