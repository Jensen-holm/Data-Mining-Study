import plotly.express as px
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import gradio as gr
from vis import iris_3d_scatter
import nn  # custom neural network module


def _preprocess_iris_data(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = datasets.load_iris()
    X = iris["data"]
    y = iris["target"]
    # normalize the features
    X = StandardScaler().fit_transform(X)
    # one hot encode the target variables
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )


X_train, X_test, y_train, y_test = _preprocess_iris_data(seed=1)


def main(
    Seed: int = 0,
    Activation_Func: str = "SoftMax",
    Loss_Func: str = "CrossEntropy",
    Epochs: int = 100,
    Hidden_Size: int = 8,
    Learning_Rate: float = 0.01,
) -> gr.Plot:

    iris_classifier = nn.NN(
        epochs=Epochs,
        learning_rate=Learning_Rate,
        activation_fn=Activation_Func,
        loss_fn=Loss_Func,
        hidden_size=Hidden_Size,
        input_size=4,  # number of features in iris dataset
        output_size=3,  # three classes in iris dataset
        seed=Seed,
    )

    iris_classifier.train(X_train=X_train, y_train=y_train)
    loss_fig = px.line(
        x=[i for i in range(len(iris_classifier._loss_history))],
        y=iris_classifier._loss_history,
    )

    return gr.Plot(loss_fig)


if __name__ == "__main__":
    with gr.Blocks() as interface:
        gr.Markdown("# Backpropagation Playground")

        with gr.Tab("Classification"):

            with gr.Row():
                data_plt = iris_3d_scatter()
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
                        gr.Number(minimum=0.00001, maximum=1.5, label="Learning Rate"),
                    ]
                with gr.Column():
                    fn_inputs = [
                        gr.Dropdown(
                            choices=["SoftMax"], label="Activation Function"
                        ),
                        gr.Dropdown(choices=["CrossEntropy"], label="Loss Function"),
                    ]

            with gr.Row():
                train_btn = gr.Button("Train", variant="primary")

            # outputs in row below inputs
            with gr.Row():
                plt_outputs = [gr.Plot()]

            train_btn.click(
                fn=main,
                inputs=seed_input + fn_inputs + numeric_inputs,
                outputs=plt_outputs,
            )

        with gr.Tab("Regression"):
            ...

    interface.launch(show_error=True)
