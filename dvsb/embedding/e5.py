import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .embedding import EMBEDDING_REGISTRY, Embedding


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@EMBEDDING_REGISTRY.register
class E5Embedding(Embedding):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval()

    def get_name(self) -> str:
        return f"E5Embedding-{self.model_name}"

    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        if mode == "query":
            texts = ["query: " + text for text in texts]
        else:
            texts = ["passage: " + text for text in texts]
        batch_dict = self.tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        result: npt.NDArray[np.float64] = embeddings.numpy()
        return result
