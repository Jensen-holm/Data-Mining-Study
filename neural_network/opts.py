from neural_network.activation import *


activation = {
    "relu": {
        "main": relu,
        "prime": relu_prime,
    },

    "sigmoid": {
        "main": sigmoid,
        "prime": sigmoid_prime,
    },
}
