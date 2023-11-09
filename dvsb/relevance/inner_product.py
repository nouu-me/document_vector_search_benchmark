import numpy as np
import numpy.typing as npt

from dvsb.relevance.relevance import Relevance, RELEVANCE_REGISTRY


@RELEVANCE_REGISTRY.register
class InnerProduct(Relevance):
    def get_name(self) -> str:
        return "InnerProduct"

    def compute(self, query_vectors: npt.NDArray[np.float64], context_vectors: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        inner_products: npt.NDArray[np.float64] = query_vectors.dot(context_vectors.T)
        return inner_products
