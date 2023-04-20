import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def loss_history_plt(loss_history: np.array) -> None:
    sns.lineplot(
        x=np.arange(len(loss_history)),
        y=loss_history,
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.show()
