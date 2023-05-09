import numpy as np
import base64
import io
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from neural_network.neural_network import NeuralNetwork

matplotlib.use("Agg")

def plot(model: NeuralNetwork) -> None:
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=np.arange(len(model.loss_history)),
        y=model.loss_history,
        ax=ax,
    )
    buf = io.BytesIO() 
    fig.savefig(buf, format="svg")
    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    model.plot = plot_data
