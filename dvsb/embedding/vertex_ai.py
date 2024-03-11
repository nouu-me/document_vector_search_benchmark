import time

import numpy as np
import numpy.typing as npt
from tenacity import retry, stop_after_attempt, wait_random_exponential
from vertexai.language_models import TextEmbeddingModel

from .embedding import EMBEDDING_REGISTRY, Embedding


@EMBEDDING_REGISTRY.register
class VertexAITextEmbedding(Embedding):
    def __init__(
        self, model_name: str = "textembedding-gecko-multilingual@001"
    ) -> None:
        self.model_name = model_name

    def load(self, has_cuda: bool = False) -> None:
        self.model = TextEmbeddingModel.from_pretrained(self.model_name)

    def get_name(self) -> str:
        return f"VertexAITextEmbedding-{self.model_name}"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        time.sleep(2)  # max limit: 30 call per second
        embeddings = self.model.get_embeddings(texts)
        return np.asarray([embedding.values for embedding in embeddings])
