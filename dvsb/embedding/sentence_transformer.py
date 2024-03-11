from typing import Optional
import numpy as np
import numpy.typing as npt
from sentence_transformers import SentenceTransformer

from .embedding import EMBEDDING_REGISTRY, Embedding


@EMBEDDING_REGISTRY.register
class SentenceTransformerEmbedding(Embedding):
    def __init__(self, model_name: str, sequence_length: Optional[int] = None) -> None:
        self.model_name = model_name
        self.max_seq_length = sequence_length

    def load(self, has_cuda: bool = False) -> None:
        if has_cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.model = SentenceTransformer(self.model_name).to(self.device)


        if self.max_seq_length:
            if self.max_seq_length > self.model[0].auto_model.config.max_position_embeddings:
                print(f"You have specified a sequence length of {self.max_seq_length}",
                    f"but the model's maximum sequence length is {self.model[0].auto_model.config.max_position_embeddings}.")
                print(f"Proceeding with sequence length of {self.model[0].auto_model.config.max_position_embeddings}.")
                self.model.max_seq_length = self.model[0].auto_model.config.max_position_embeddings
        elif self.model.max_seq_length > 512:
            print(f"Model's maximum sequence length is {self.model.max_seq_length} and",
            "no sequence length was specified in the config...")
            print("Lowering sequence length to 512 to avoid CUDA memory issues.",
                  "Please specify sequence_length in the config if you would like to bypass this.")
            self.model.max_seq_length = 512
        
        self.model.eval()

    def get_name(self) -> str:
        return f"SentenceTransformerEmbedding-{self.model_name}"

    def get_embeddings(self, texts: list[str], mode: str) -> npt.NDArray[np.float64]:
        embeddings = self.model.encode(texts)
        return np.asarray(embeddings)
