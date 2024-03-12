import numpy as np
import numpy.typing as npt
import torch
from transformers import LukeModel, MLukeTokenizer

from .embedding import EMBEDDING_REGISTRY, Embedding


@EMBEDDING_REGISTRY.register
class SonoisaSentenceLukeJapanese(Embedding):
    def __init__(self, model_name: str = "sonoisa/sentence-luke-japanese-base-lite") -> None:
        self.model_name = model_name

    def load(self, has_cuda: bool = False) -> None:
        if has_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.tokenizer = MLukeTokenizer.from_pretrained(self.model_name)
        self.model = LukeModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def get_name(self) -> str:
        return f"SonoisaSentenceLukeJapanese-{self.model_name}"

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        encoded_input = self.tokenizer.batch_encode_plus(
            texts, padding="longest", truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            model_output = self.model(**encoded_input)
            embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        if self.device == "cuda":
            embeddings = embeddings.cpu()
        result: npt.NDArray[np.float64] = embeddings.numpy()
        return result
