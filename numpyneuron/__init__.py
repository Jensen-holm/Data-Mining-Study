from .loss import *
from .activation import *
from .nn import *

ACTIVATIONS: dict[str, Activation] = {
    "Relu": Relu(),
    "Sigmoid": Sigmoid(),
    "TanH": TanH(),
    "SoftMax": SoftMax(),
}

LOSSES: dict[str, Loss] = {
    "MSE": MSE(),
    "CrossEntropy": CrossEntropy(),
    "CrossEntropyWithLogitsLoss": CrossEntropyWithLogits(),
}
