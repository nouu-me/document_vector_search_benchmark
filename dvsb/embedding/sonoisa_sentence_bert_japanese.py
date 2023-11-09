import numpy as np
import numpy.typing as npt
from transformers import BertJapaneseTokenizer, BertModel
import torch

from .embedding import EMBEDDING_REGISTRY, Embedding


@EMBEDDING_REGISTRY.register
class SonoisaSentenceBertJapanese(Embedding):
    def __init__(self, model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2") -> None:
        self.model_name = model_name        
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def get_name(self) -> str:
        return f"SonoisaSentenceBertJapanese-{self.model_name}"

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        encoded_input = self.tokenizer.batch_encode_plus(texts, padding="longest", truncation=True, return_tensors="pt")
        with torch.inference_mode():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        result: npt.NDArray[np.float64] = embeddings.numpy()
        return result
