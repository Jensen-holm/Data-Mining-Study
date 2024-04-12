from typing import Optional
from nn.activation import ACTIVATIONS, Activation
from nn.loss import LOSSES, Loss
import numpy as np

import gradio as gr


DTYPE = np.float32


class NN:
    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        hidden_size: int,
        input_size: int,
        output_size: int,
        activation_fn: str,
        loss_fn: str,
        seed: int,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed

        # try to get activation function and loss funciton
        act_fn = ACTIVATIONS.get(activation_fn, None)
        if act_fn is None:
            raise KeyError(f"Invalid Activation function '{activation_fn}'")
        loss_fn = LOSSES.get(loss_fn, None)
        if loss_fn is None:
            raise KeyError(f"Invalid Activation function '{activation_fn}'")
        self._activation_fn: Activation = act_fn
        self._loss_fn: Loss = loss_fn

        self._loss_history = list()
        self._weight_history = {
            "wo": [],
            "wh": [],
            "bo": [],
            "bh": [],
        }

        self._wo: Optional[np.ndarray] = None
        self._wh: Optional[np.ndarray] = None
        self._bo: Optional[np.ndarray] = None
        self._bh: Optional[np.ndarray] = None
        self._init_weights_and_biases()

    def _init_weights_and_biases(self) -> None:
        """
        NN._init_weights_and_biases(): Should only be ran once, right before training loop
            in order to initialize the weights and biases randomly.

        params:
            NN object with hidden layer size, output size, and input size
            defined.

        returns:
            self, modifies _bh, _bo, _wo, _wh NN attributes in place.
        """
        np.random.seed(self.seed)
        self._bh = np.zeros((1, self.hidden_size), dtype=DTYPE)
        self._bo = np.zeros((1, self.output_size), dtype=DTYPE)
        self._wh = np.asarray(
            np.random.randn(self.input_size, self.hidden_size)
            * np.sqrt(2 / self.input_size),
            dtype=DTYPE,
        )
        self._wo = np.asarray(
            np.random.randn(self.hidden_size, self.output_size)
            * np.sqrt(2 / self.hidden_size),
            dtype=DTYPE,
        )
        return

    def _forward(self, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        _forward(X_train): ran as the first step of each epoch during training.

        params:
            X_train: np.ndarray -> data that we are training the NN on.

        returns:
            output layer np array containing the predicted outputs calculated using
            the weights and biases of the current epoch.
        """
        assert self._activation_fn is not None

        # hidden layer
        hidden_layer_output = self._activation_fn.forward(
            np.dot(X_train, self._wh) + self._bh
        )
        # output layer (prediction layer)
        y_hat = self._activation_fn.forward(
            np.dot(hidden_layer_output, self._wo) + self._bo
        )
        return y_hat, hidden_layer_output

    def _backward(
        self,
        X_train: np.ndarray,
        y_hat: np.ndarray,
        y_train: np.ndarray,
        hidden_output: np.ndarray,
    ) -> None:
        assert self._activation_fn is not None
        assert self._wo is not None
        assert self._loss_fn is not None

        # Calculate the error at the output
        # This should be the derivative of the loss function with respect to the output of the network
        error_output = self._loss_fn.backward(
            y_hat, y_train
        ) * self._activation_fn.backward(y_hat)

        # Calculate gradients for output layer weights and biases
        wo_prime = np.dot(hidden_output.T, error_output) * self.learning_rate
        bo_prime = np.sum(error_output, axis=0, keepdims=True) * self.learning_rate

        # Propagate the error back to the hidden layer
        error_hidden = np.dot(error_output, self._wo.T) * self._activation_fn.backward(
            hidden_output
        )

        # Calculate gradients for hidden layer weights and biases
        wh_prime = np.dot(X_train.T, error_hidden) * self.learning_rate
        bh_prime = np.sum(error_hidden, axis=0, keepdims=True) * self.learning_rate

        # Update weights and biases
        self._wo -= wo_prime
        self._wh -= wh_prime
        self._bo -= bo_prime
        self._bh -= bh_prime

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "NN":
        assert self._loss_fn is not None

        for _ in gr.Progress().tqdm(range(self.epochs)):
            y_hat, hidden_output = self._forward(X_train=X_train)
            loss = self._loss_fn.forward(y_hat=y_hat, y_true=y_train)
            self._loss_history.append(loss)
            self._backward(
                X_train=X_train,
                y_hat=y_hat,
                y_train=y_train,
                hidden_output=hidden_output,
            )

            # keep track of weights an biases at each epoch for visualization
            self._weight_history["wo"].append(self._wo[0, 0])
            self._weight_history["wh"].append(self._wh[0, 0])
            self._weight_history["bo"].append(self._bo[0, 0])
            self._weight_history["bh"].append(self._bh[0, 0])
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._forward(X_train=X_test)[0]
