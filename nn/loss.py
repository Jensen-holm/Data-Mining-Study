from abc import ABC, abstractmethod
from nn.activation import SoftMax
import numpy as np


__all__ = ["Loss", "MSE", "CrossEntropy", "LOSSES"]


class Loss(ABC):
    @abstractmethod
    def forward(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass


class MSE(Loss):
    def forward(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return np.sum(np.square(y_hat - y_true)) / y_true.shape[0]

    def backward(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_hat - y_true) * (2 / y_true.shape[0])


class CrossEntropy(Loss):
    def forward(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_hat = np.asarray(y_hat)
        y_true = np.asarray(y_true)
        m = y_true.shape[0]
        p = self._softmax(y_hat)
        log_likelihood = -np.log(p[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self, y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        y_hat = np.asarray(y_hat)
        y_true = np.asarray(y_true)
        return (y_hat - y_true) / y_true.shape[0]

    @staticmethod
    def _softmax(X: np.ndarray) -> np.ndarray:
        return SoftMax().forward(X)


LOSSES: dict[str, Loss] = {
    "MSE": MSE(),
    "CrossEntropy": CrossEntropy(),
}
