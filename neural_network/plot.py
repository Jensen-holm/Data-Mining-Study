import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sns.set()


def loss_history_plt(loss_history: list) -> None:
    fig, ax = plt.subplots()

    def animate(i):
        ax.clear()
        sns.lineplot(
            x=range(i),
            y=loss_history[:i],
            ax=ax,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Loss")

    _ = FuncAnimation(fig, animate, frames=len(loss_history), interval=100)
    plt.show()
