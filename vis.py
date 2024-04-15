import matplotlib
from sklearn import datasets
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


def show_digits():
    digits = datasets.load_digits()
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    return fig


def loss_history_plt(loss_history: list[float], loss_fn_name: str):
    return px.line(
        x=[i for i in range(len(loss_history))],
        y=loss_history,
        title=f"{loss_fn_name} Loss vs. Training Epoch",
        labels={
            "x": "Epochs",
            "y": f"{loss_fn_name} Loss",
        },
    )


def hits_and_misses(y_pred: np.ndarray, y_true: np.ndarray):
    # decode the one hot encoded predictions
    y_pred_decoded = np.argmax(y_pred, axis=1)
    y_true_decoded = np.argmax(y_true, axis=1)

    hits = y_pred_decoded == y_true_decoded
    color = np.where(hits, "Hit", "Miss")
    hover_text = [
        "True: " + str(y_true_decoded[i]) + ", Pred: " + str(y_pred_decoded[i])
        for i in range(len(y_pred_decoded))
    ]

    return px.scatter(
        x=np.arange(len(y_pred_decoded)),
        y=y_true_decoded,
        color=color,
        title="Hits and Misses of Predictions",
        labels={
            "color": "Prediction Correctness",
            "x": "Sample Index",
            "y": "True Label",
        },
        color_discrete_map={"Hit": "blue", "Miss": "red"},
        hover_name=hover_text,
    )


def make_confidence_label(y_pred: np.ndarray, y_test: np.ndarray):
    # decode the one hot endoced predictions
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    confidence_dict: dict[str, float] = {}
    for idx, class_name in enumerate([str(i) for i in range(10)]):
        class_confidences_idxs = np.where(y_test_labels == idx)[0]
        class_confidences = y_pred[class_confidences_idxs, idx]
        confidence_dict[class_name] = float(np.mean(class_confidences))
    return confidence_dict
