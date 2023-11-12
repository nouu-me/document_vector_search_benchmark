import numpy as np
import numpy.typing as npt
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .embedding import EMBEDDING_REGISTRY, Embedding


@EMBEDDING_REGISTRY.register
class OpenAIEmbedding(Embedding):
    def __init__(self, model_name: str = "text-embedding-ada-002") -> None:
        self.model_name = model_name

    def load(self) -> None:
        self.client = openai.OpenAI()

    def get_name(self) -> str:
        return f"OpenAIEmbedding-{self.model_name}"

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        texts = [text.replace("\n", " ") for text in texts]
        res = self.client.embeddings.create(input=texts, model=self.model_name)
        return np.asarray([d.embedding for d in res.data])
