from dataclasses import dataclass, field
import gradio as gr
import numpy as np

from nn.activation import Activation
from nn.loss import Loss


DTYPE = np.float32


@dataclass
class NN:
    epochs: int
    learning_rate: float
    hidden_size: int
    input_size: int
    output_size: int
    hidden_activation_fn: Activation
    activation_fn: Activation
    loss_fn: Loss
    seed: int

    _loss_history: list = field(default_factory=lambda: [], init=False)
    _wo: np.ndarray = field(default_factory=lambda: np.ndarray([]), init=False)
    _wh: np.ndarray = field(default_factory=lambda: np.ndarray([]), init=False)
    _bo: np.ndarray = field(default_factory=lambda: np.ndarray([]), init=False)
    _bh: np.ndarray = field(default_factory=lambda: np.ndarray([]), init=False)
    _weight_history: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {
            "wo": [],
            "wh": [],
            "bo": [],
            "bh": [],
        },
        init=False,
    )

    def __post_init__(self) -> None:
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

    # def _forward(self, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    #     # Determine the activation function for the hidden layer
    #     if self._activation_fn.__class__.__name__ == "SoftMax":
    #         # Using ReLU for hidden layer when softmax is used in output layer
    #         hidden_layer_activation = Sigmoid()
    #     else:
    #         # Use the specified activation function if not using softmax
    #         hidden_layer_activation = self._activation_fn

    #     # Compute the hidden layer output
    #     hidden_layer_output = hidden_layer_activation.forward(
    #         np.dot(X_train, self._wh) + self._bh
    #     )

    #     # Compute the output layer (prediction layer) using the specified activation function
    #     y_hat = self._activation_fn.forward(
    #         np.dot(hidden_layer_output, self._wo) + self._bo
    #     )

    #     return y_hat, hidden_layer_output

    # TODO: make this forward function the main _forward function if
    # the loss function that the user selected is a "logits" loss. Call
    # The one above if it is not.
    def _forward(self, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hidden_layer_output = self.hidden_activation_fn.forward(
            np.dot(X_train, self._wh) + self._bh,
        )
        # Output layer does not apply softmax anymore, just return logits
        logits = np.dot(hidden_layer_output, self._wo) + self._bo
        return logits, hidden_layer_output

    def _backward(
        self,
        X_train: np.ndarray,
        y_hat: np.ndarray,
        y_train: np.ndarray,
        hidden_output: np.ndarray,
    ) -> None:
        assert self._wo is not None

        # Calculate the error at the output
        # This should be the derivative of the loss function with respect to the output of the network
        error_output = self.loss_fn.backward(y_hat, y_train)

        # Calculate gradients for output layer weights and biases
        wo_prime = np.dot(hidden_output.T, error_output) * self.learning_rate
        bo_prime = np.sum(error_output, axis=0, keepdims=True) * self.learning_rate

        # Propagate the error back to the hidden layer
        error_hidden = np.dot(error_output, self._wo.T) * self.activation_fn.backward(
            hidden_output
        )

        # Calculate gradients for hidden layer weights and biases
        wh_prime = np.dot(X_train.T, error_hidden) * self.learning_rate
        bh_prime = np.sum(error_hidden, axis=0, keepdims=True) * self.learning_rate

        # Gradient clipping to prevent overflow
        max_norm = 1.0  # You can adjust this threshold
        wo_prime = np.clip(wo_prime, -max_norm, max_norm)
        bo_prime = np.clip(bo_prime, -max_norm, max_norm)
        wh_prime = np.clip(wh_prime, -max_norm, max_norm)
        bh_prime = np.clip(bh_prime, -max_norm, max_norm)

        # Update weights and biases
        self._wo -= wo_prime
        self._wh -= wh_prime
        self._bo -= bo_prime
        self._bh -= bh_prime

    # TODO: implement batch size in training, this will speed up the training loop
    # quite a bit I believe
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> "NN":
        for _ in gr.Progress().tqdm(range(self.epochs)):
            y_hat, hidden_output = self._forward(X_train=X_train)
            loss = self.loss_fn.forward(y_hat=y_hat, y_true=y_train)
            self._loss_history.append(loss)
            self._backward(
                X_train=X_train,
                y_hat=y_hat,
                y_train=y_train,
                hidden_output=hidden_output,
            )

            # TODO: make a 3d visualization traversing loss plane. Might be too
            # expenzive to do though.
            # keep track of weights an biases at each epoch for visualization
            # self._weight_history["wo"].append(self._wo[0, 0])
            # self._weight_history["wh"].append(self._wh[0, 0])
            # self._weight_history["bo"].append(self._bo[0, 0])
            # self._weight_history["bh"].append(self._bh[0, 0])
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pred, _ = self._forward(X_test)
        return self.activation_fn.forward(pred)
