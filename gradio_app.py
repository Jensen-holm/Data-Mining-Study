from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import gradio as gr

import numpyneuron as nn
from vis import (  # classification visualization funcitons
    show_digits,
    hits_and_misses,
    loss_history_plt,
    make_confidence_label,
)


def _preprocess_digits(
    seed: int,
) -> tuple[np.ndarray, ...]:
    digits = datasets.load_digits(as_frame=False)
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
    seed: int,
    hidden_layer_activation_fn_str: str,
    output_layer_activation_fn_str: str,
    loss_fn_str: str,
    epochs: int,
    hidden_size: int,
    batch_size: float,
    learning_rate: float,
) -> tuple[gr.Plot, gr.Plot, gr.Label]:
    assert hidden_layer_activation_fn_str in nn.ACTIVATIONS
    assert output_layer_activation_fn_str in nn.ACTIVATIONS
    assert loss_fn_str in nn.LOSSES

    loss_fn: nn.Loss = nn.LOSSES[loss_fn_str]
    h_act_fn: nn.Activation = nn.ACTIVATIONS[hidden_layer_activation_fn_str]
    o_act_fn: nn.Activation = nn.ACTIVATIONS[output_layer_activation_fn_str]

    nn_classifier = nn.NN(
        epochs=epochs,
        hidden_size=hidden_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        hidden_activation_fn=h_act_fn,
        output_activation_fn=o_act_fn,
        input_size=64,  # 8x8 pixel grid images
        output_size=10,  # digits 0-9
        seed=seed,
        _gradio_app=True,
    )

    nn_classifier.train(X_train=X_train, y_train=y_train)

    pred = nn_classifier.predict(X_test=X_test)
    hits_and_misses_fig = hits_and_misses(y_pred=pred, y_true=y_test)
    loss_fig = loss_history_plt(
        loss_history=nn_classifier._loss_history,
        loss_fn_name=nn_classifier.loss_fn.__class__.__name__,
    )

    label_dict = make_confidence_label(y_pred=pred, y_test=y_test)
    return (
        gr.Plot(loss_fig, show_label=False),
        gr.Plot(hits_and_misses_fig, show_label=False),
        gr.Label(label_dict, label="Classification Confidence Rankings"),
    )


if __name__ == "__main__":

    def _open_warning() -> str:
        with open("gradio_warning.md", "r") as f:
            return f.read()

    with gr.Blocks() as interface:
        gr.Markdown("# Numpy Neuron")
        gr.Markdown(_open_warning())

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
                        gr.Slider(minimum=100, maximum=10_000, step=50, label="Epochs"),
                        gr.Slider(
                            minimum=2, maximum=64, step=2, label="Hidden Network Size"
                        ),
                        gr.Slider(minimum=0.1, maximum=1, step=0.1, label="Batch Size"),
                        gr.Number(minimum=0.00001, maximum=1.5, label="Learning Rate"),
                    ]

                with gr.Column():
                    fn_inputs = [
                        gr.Dropdown(
                            choices=["Relu", "Sigmoid", "TanH"],
                            label="Hidden Layer Activation Function",
                        ),
                        gr.Dropdown(
                            choices=["SoftMax", "Sigmoid"],
                            label="Output Activation Function",
                        ),
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
                            "Sigmoid",
                            "CrossEntropyWithLogitsLoss",
                            2_000,
                            16,
                            1.0,
                            0.01,
                        ],
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

    interface.launch(show_error=True)
