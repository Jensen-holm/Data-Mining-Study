# Neural Network Classification (from-scratch)

## Parameters
Think of epochs as rounds of training for your neural network. Each epoch means the network has gone through the entire dataset once, learning and adjusting its parameters. More epochs can lead to better accuracy, but too many can also overfit the model to your training data.

#### Activation functions 
introduce non-linearity to your neural network, allowing it to model complex relationships in data. The choice of activation function (like sigmoid, ReLU, or tanh) affects how the network processes and passes information between its layers.

#### Hidden Size
This refers to the number of neurons or units in the hidden layer(s) of your neural network. More hidden units can make the network more capable of learning complex patterns, but it can also make training slower and increase the risk of overfitting.

#### Learning Rate
Imagine this as the step size your neural network takes during training. It determines how much the network's parameters are updated based on the error it observes. A higher learning rate means bigger steps but can lead to overshooting the optimal values, while a smaller learning rate may take longer to converge or find the best values.

#### Test Size
When training a neural network, or any machine learning model for that matter, it is important to split the data into training and testing sets. The test size parameter specifys how to split up the data into these two sets. a test size of 0.2 will split it up so that 80% of the data is used for training, and 20% of the data is used for testing.


## Backprop Algorithm
Backpropagation, short for "backward propagation of errors," is the cornerstone of training artificial neural networks. It begins by initializing the network's weights and biases. During the forward pass, input data flows through the network's layers, undergoing weighted sum calculations and activation functions, eventually producing predictions. The algorithm then computes an error or loss by comparing these predictions to the actual target values. In the critical backward pass, starting from the output layer and moving in reverse, gradients of the loss with respect to each layer's outputs, weights, and biases are calculated using calculus and the chain rule. These gradients guide the adjustment of weights and biases in each layer, with the goal of minimizing the loss. This iterative process repeats for multiple epochs, refining the network's parameters until the error reaches an acceptable level or a fixed number of training iterations is completed, ultimately enabling the network to improve its predictions on new data.

## Implementation
Behind the scenes, my API implements the backprop algorithm. The main loop first initializes weights and biases randomly. The algorithm starts by iterating n times where n is the number of epochs you specify above. During each iteration, starting with the randomly initialized weights and biases, the activation function that you choose will be run inside of this compute node function below:

The activation function plays a crucial role in the behavior of your neural network. The compute node function, which we've discussed earlier, calculates the network's output. In each iteration of the training process, we compare this output to the actual data, which, in this case, represents the iris flower type. The difference between the predicted and actual values guides the algorithm in determining how much to adjust the network's weights and biases for better predictions. However, we must be careful to prevent the neural network from memorizing the training data, a problem in machine learning known as overfitting. To address this, we scale down the derivatives computed for weights and biases by the learning rate you specify, ensuring that the network learns in a controlled and meaningful manner. You'll notice that if you use 1 for the learning rate, the graph on loss/epoch is a lot choppier than it is if you have a lower learning rate like 0.01. The smoother the curve, the better. The process repeats for n epochs, then the final results are calculated, and our final weights and biases saved.

## Results

#### Log Loss

#### Accuracy Score



