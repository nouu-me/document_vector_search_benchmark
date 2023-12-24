import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from .embedding import EMBEDDING_REGISTRY, Embedding


@EMBEDDING_REGISTRY.register
class SentenceTransformerEmbedding(Embedding):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def load(self, has_cuda: bool = False) -> None:
        if has_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = SentenceTransformer(self.model_name).to(self.device)
        self.model.eval()

    def get_name(self) -> str:
        return f"SentenceTransformerEmbedding-{self.model_name}"

    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        embeddings = self.model.encode(texts)
        return np.asarray(embeddings)
