---
title: Numpy-Neuron
emoji: 🔙
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: mit
---

## What is this? <br>

The Numpy-Neuron is a GUI built around a neural network framework that I have built from scratch
in [numpy](https://numpy.org/). In this GUI, you can test different hyper parameters that will be fed to this framework and used
to train a neural network on the [MNIST](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) dataset of 8x8 pixel images.

## ⚠️ PLEASE READ ⚠️
This application is impossibly slow on the HuggingFace CPU instance that it is running on. It is advised to clone the 
repository and run it locally.

In order to get a decent classification score on the validation set of the MNIST data (hard coded to 20%), you will have to
do somewhere between 15,000 epochs and 50,000 epochs with a learning rate around 0.001, and a hidden layer size
over 10. (roughly the example that I have provided). Running this many epochs with a hidden layer of that size
is pretty expensive on 2 cpu cores that this space has. So if you are actually curious, you might want to clone
this and run it locally because it will be much much faster.

`git clone https://huggingface.co/spaces/Jensen-holm/Numpy-Neuron`

After cloning, you will have to install the dependencies from requirements.txt into your environment. (venv reccommended)

`pip3 install -r requirements.txt`

Then, you can run the application on local host with the following command.

`python3 app.py`


## Development

In order to push from this GitHub repo to the hugging face space:

`git push --force space main`
