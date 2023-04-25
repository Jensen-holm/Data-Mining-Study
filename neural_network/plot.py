import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

sns.set()

"""
Save plots to the plots folder for when
we would like to show results on our little
flask application
"""

PLT_PATH: str = "static/assets/"


def loss_history_plt(loss_history: list) -> FuncAnimation:
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

    return FuncAnimation(fig, animate, frames=len(loss_history), interval=100)


def save_plt(plot, filename: str, animated: bool, fps=10):
    if not animated:
        plot.savefig(filename)
        return
    writer = FFMpegWriter(fps=fps)
    plot.save(PLT_PATH + filename, writer=writer)
