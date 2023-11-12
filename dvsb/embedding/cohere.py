import os

import cohere
import numpy as np
import numpy.typing as npt
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .embedding import EMBEDDING_REGISTRY, Embedding


@EMBEDDING_REGISTRY.register
class CohereEmbedding(Embedding):
    def __init__(self, model_name: str = "embed-multilingual-v3.0") -> None:
        self.model_name = model_name

    def load(self) -> None:
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])

    def get_name(self) -> str:
        return f"CohereEmbedding-{self.model_name}"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        embeddings = self.client.embed(texts, input_type="search_document", model=self.model_name).embeddings
        return np.asarray(embeddings)
