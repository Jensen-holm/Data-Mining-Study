import plotly.express as px
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import os


def iris_3d_scatter():
    df = px.data.iris()
    fig = px.scatter_3d(
        df,
        x="sepal_length",
        y="sepal_width",
        z="petal_width",
        color="species",
        size="petal_length",
        size_max=18,
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    return fig
