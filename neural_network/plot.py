import numpy as np
import base64
import io
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from neural_network.neural_network import NeuralNetwork

matplotlib.use("Agg")

def plot(model: NeuralNetwork) -> None:
    _ = sns.scatterplot(
        x=np.arange(len(model.loss_history)),
        y=model.loss_history,
    )
    buf = io.BytesIO() 
    plt.savefig(buf, format="svg")
    plt.clf()
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    model.plot = plot_data
