from ctypes import ArgumentError
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
        y_true: list[list[int]],
        scores: npt.NDArray[np.float64] | None = None,
        y_pred: list[list[int]] | None = None,
    ) -> float:
        if y_pred is not None:
            if scores is not None:
                raise ArgumentError("You can only provide one of 'scores' or 'y_pred'.")
            y_pred = [x[: self.k] for x in y_pred]
        else:
            y_pred = np.argsort(scores, axis=1)[:, -self.k :][:, ::-1]

        accumulated = 0.0
        for y, y_ in zip(y_true, y_pred):
            accumulated += len(set(y) & set(y_)) / len(y)
        return accumulated / len(y_true)
