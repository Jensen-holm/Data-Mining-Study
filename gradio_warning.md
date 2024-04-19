## What is this?

This is a no code platform for interacting with Numpy-Neuron, a neural network framework that I have built from scratch
using only [numpy](https://numpy.org/). Here, you can test different hyper parameters that will be fed to Numpy-Neuron and used to train a neural network for classification on the [MNIST](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset of 8x8 pixel images of hand drawn numbers.

Once training is done, the final model will be tested by making predictions on an unseen subset of the dataset called the validation set. There will be a plot of hits vs. misses, measuring the accuracy of the final model on images that did not see in training. There will also be a label at the bottom that shows the average confidence of the final model when it was making its predictions on unseen data across the different labels (digits 0-9).

## Local Development

The Numpy-Neuron package is [available on PyPI](https://pypi.org/project/numpyneuron/) and you can install it yourself with the command: `pip3 install numpyneuron`

## ⚠️ Warning ⚠️
This application is impossibly slow on the HuggingFace CPU instance that it is running on. It is advised to clone the 
repository and run it locally, or install the package using pip as mentioned above.

## Steps for running this GUI locally:

1. `git clone https://huggingface.co/spaces/Jensen-holm/Numpy-Neuron`

2. `pip3 install -r requirements.txt`

3. `python3 gradio_app.py`

