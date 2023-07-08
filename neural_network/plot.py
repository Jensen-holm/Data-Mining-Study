import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from plt_id import generate_image_key
import os

matplotlib.use("Agg")

UPLOAD_FOLDER = "/plots"


def plot(model) -> None:
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

    image_key = generate_image_key()

    plot_filename = os.path.join(UPLOAD_FOLDER, f"{image_key}.png")
    fig.savefig(plot_filename, format="png")
    plt.close(fig)

    model.plot_key = image_key
