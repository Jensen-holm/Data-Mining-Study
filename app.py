from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import gradio as gr

import nn  # custom neural network module
from vis import (  # classification visualization funcitons
    show_digits,
    hits_and_misses,
    loss_history_plt,
    make_confidence_label,
)


def _preprocess_digits(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    y = OneHotEncoder().fit_transform(digits.target.reshape(-1, 1)).toarray()
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        y,
        test_size=0.2,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = _preprocess_digits(seed=1)


def classification(
    Seed: int = 0,
    Hidden_Layer_Activation: str = "Relu",
    Activation_Func: str = "SoftMax",
    Loss_Func: str = "CrossEntropyWithLogitsLoss",
    Epochs: int = 100,
    Hidden_Size: int = 8,
    Learning_Rate: float = 0.001,
) -> tuple[gr.Plot, gr.Plot, gr.Label]:
    assert Activation_Func in nn.ACTIVATIONS
    assert Hidden_Layer_Activation in nn.ACTIVATIONS
    assert Loss_Func in nn.LOSSES

    classifier = nn.NN(
        epochs=Epochs,
        learning_rate=Learning_Rate,
        hidden_activation_fn=nn.ACTIVATIONS[Hidden_Layer_Activation],
        activation_fn=nn.ACTIVATIONS[Activation_Func],
        loss_fn=nn.LOSSES[Loss_Func],
        hidden_size=Hidden_Size,
        input_size=64,  # 8x8 image of pixels
        output_size=10,  # digits 0-9
        seed=Seed,
    )
    classifier.train(X_train=X_train, y_train=y_train)

    pred = classifier.predict(X_test=X_test)
    hits_and_misses_fig = hits_and_misses(y_pred=pred, y_true=y_test)
    loss_fig = loss_history_plt(
        loss_history=classifier._loss_history,
        loss_fn_name=classifier.loss_fn.__class__.__name__,
    )

    label_dict = make_confidence_label(y_pred=pred, y_test=y_test)
    return (
        gr.Plot(loss_fig, show_label=False),
        gr.Plot(hits_and_misses_fig, show_label=False),
        gr.Label(label_dict, label="Classification Confidence Rankings"),
    )


if __name__ == "__main__":
    with gr.Blocks() as interface:
        gr.Markdown("# Numpy Neuron")
        gr.Markdown(
            """
            ## What is this? <br>

            The Backpropagation Playground is a GUI built around a neural network framework that I have built from scratch
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

            Then, you can run the application on localhost with the following command.

            `python3 app.py`

            """
        )

        with gr.Tab("Classification"):
            with gr.Row():
                data_plt = show_digits()
                gr.Plot(data_plt)

            with gr.Row():
                seed_input = [gr.Number(minimum=0, label="Random Seed")]

            # inputs in the same row
            with gr.Row():
                with gr.Column():
                    numeric_inputs = [
                        gr.Slider(
                            minimum=100, maximum=100_000, step=50, label="Epochs"
                        ),
                        gr.Slider(
                            minimum=2, maximum=64, step=2, label="Hidden Network Size"
                        ),
                        gr.Number(minimum=0.00001, maximum=1.5, label="Learning Rate"),
                    ]

                with gr.Column():
                    fn_inputs = [
                        gr.Dropdown(
                            choices=["Relu", "Sigmoid", "TanH"],
                            label="Hidden Layer Activation",
                        ),
                        gr.Dropdown(choices=["SoftMax"], label="Output Activation"),
                        gr.Dropdown(
                            choices=["CrossEntropy", "CrossEntropyWithLogitsLoss"],
                            label="Loss Function",
                        ),
                    ]

            inputs = seed_input + fn_inputs + numeric_inputs
            with gr.Row():
                train_btn = gr.Button("Train", variant="primary")

            with gr.Row():
                gr.Examples(
                    examples=[
                        [
                            2,
                            "Relu",
                            "SoftMax",
                            "CrossEntropyWithLogitsLoss",
                            15_000,
                            14,
                            0.001,
                        ]
                    ],
                    inputs=inputs,
                )

            # outputs in row below inputs
            with gr.Row():
                plt_outputs = [
                    gr.Plot(label="Loss History / Epoch"),
                    gr.Plot(label="Hits & Misses"),
                ]

            with gr.Row():
                label_output = [gr.Label(label="Class Confidences")]

            train_btn.click(
                fn=classification,
                inputs=inputs,
                outputs=plt_outputs + label_output,
            )

        with gr.Tab("Regression"):
            gr.Markdown("### Coming Soon")

    interface.launch(show_error=True)
