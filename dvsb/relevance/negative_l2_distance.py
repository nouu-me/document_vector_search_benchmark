import numpy as np
import numpy.typing as npt

from dvsb.relevance.relevance import Relevance, RELEVANCE_REGISTRY


@RELEVANCE_REGISTRY.register
class NegativeL2Distance(Relevance):
    def get_name(self) -> str:
        return "NegativeL2Distance"

    def compute(self, query_vectors: npt.NDArray[np.float64], context_vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        a: npt.NDArray[np.float64] = np.linalg.norm(query_vectors, axis=1)
        b: npt.NDArray[np.float64] = query_vectors.dot(context_vectors.T)
        c: npt.NDArray[np.float64] = np.linalg.norm(context_vectors, axis=1)
        squared_distances = a[:, np.newaxis] - 2 * b + c[np.newaxis, :]
        squared_distances[squared_distances < 0] = 0.
        neg_distances: npt.NDArray[np.float64] = -np.sqrt(squared_distances)
        return neg_distances
