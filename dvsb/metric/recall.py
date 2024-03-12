import numpy as np
import numpy.typing as npt
from dvsb.metric.metric import METRIC_REGISTRY, Metric


@METRIC_REGISTRY.register
class Recall(Metric):
    def __init__(self, k: int) -> None:
        self.k = k

    def get_name(self) -> str:
        return f"Recall@{self.k}"

    def compute(
        self,
        y_true: list[list[int]] | npt.NDArray[np.int_],
        y_pred: list[list[int]] | npt.NDArray[np.int_],
    ) -> float:
        y_pred_topk = [x[: self.k] for x in y_pred]
        accumulated = 0.0
        for y, y_ in zip(y_true, y_pred_topk):
            accumulated += len(set(y) & set(y_)) / len(y)
        return accumulated / len(y_true)
