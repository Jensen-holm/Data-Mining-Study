from nn.nn import NN
import unittest

TEST_NN = NN(
    epochs=100,
    learning_rate=0.001,
    hidden_size=8,
    input_size=2,
    output_size=1,
    activation_fn="Sigmoid",
    loss_fn="MSE",
)


class TestNN(unittest.TestCase):
    def test_init_w_b(self) -> None:
        return

    def test_forward(self) -> None:
        return

    def test_backward(self) -> None:
        return

    def test_train(self) -> None:
        return


if __name__ == "__main__":
    unittest.main()
