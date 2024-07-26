import os
import sys
import pytest

sys.path.append(os.path.abspath(".."))

import random
import numpy as np
from numpyneuron import (
    TanH,
    Sigmoid,
    Relu,
    SoftMax,
    Sigmoid,
)

# these functions are meant to work with np.ndarray
# objects, but they will also work with numbers which
# makes testing a little bit simpler


def test_tanh() -> None:
    """
    tanh(1) =~ 0.76
    tanh'(1) =~ sech^2(1) =~ 0.419
    """
    tanh = TanH()
    assert tanh.forward(1) == pytest.approx(np.tanh(1))
    assert tanh.forward(1) == pytest.approx(0.7615941559557649)
    assert tanh.backward(1) == pytest.approx(0.41997434161402614)


def test_sigmoid() -> None:
    """
    sigmoid(1) =~ 0.73105
    sigmoid'(1) =~ 0.1966
    """
    sigmoid = Sigmoid()
    assert sigmoid.forward(1) == pytest.approx(0.7310585786300049)
    assert sigmoid.backward(1) == pytest.approx(0.4621171572600098)


def test_relu() -> None:
    """
    relu(n > 0) = n
    relu(n < 0) = 0
    relu'(n > 0) = 1
    relu'(n < 0) = 0
    """
    relu = Relu()
    random_n = random.randint(1, 100)
    assert relu.forward(random_n) == random_n
    assert relu.backward(random_n) == 1


def test_softmax() -> None:
    """
    softmax([1, 2, 3]) = [0.090031, 0.244728, 0.665241]
    """
    softmax = SoftMax()
    vec = np.array([1, 2, 3])
    assert np.allclose(
        softmax.forward(vec),
        np.array([0.090031, 0.244728, 0.665241]),
    )
    assert np.allclose(softmax.backward(vec), vec)
