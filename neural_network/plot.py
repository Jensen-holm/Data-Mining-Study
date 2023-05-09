import numpy as np
import base64
import io
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from neural_network.neural_network import NeuralNetwork

matplotlib.use("Agg")

def plot(model: NeuralNetwork) -> None:
    sns.set()
    fig, ax = plt.subplots()
    sns.lineplot(
        x=np.arange(len(model.loss_history)),
        y=model.loss_history,
        ax=ax,
    )
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss / Epoch")
    buf = io.BytesIO() 
    fig.savefig(buf, format="png")
    plt.close(fig)
    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    model.plot = plot_data
