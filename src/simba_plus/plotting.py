from typing import Optional
import pandas as pd
from matplotlib.collections import PatchCollection

def plot_heatmap(df_c: pd.DataFrame, df_s: Optional[pd.DataFrame] = None):
    if df_s is None:
        df_s = df_c
    M, N = df_c.values.shape
    ylabels = df_c.columns.tolist()
    xlabels = df_c.index.tolist()

    x, y = np.meshgrid(np.arange(M), np.arange(N))
    s = df_s.T.values
    c = df_c.T.values

    fig, ax = plt.subplots(figsize=(20, 20))

    R = s/s.max()/2
    max_i = y.max()
    circles = [plt.Circle((j, max_i - i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap="Reds")
    ax.add_collection(col)

    ax.set(xticks=np.arange(M), yticks=np.arange(N),
        xticklabels=xlabels, yticklabels=ylabels[::-1])
    ax.set_xticks(np.arange(M+1)-0.5, minor=True)
    ax.set_yticks(np.arange(N+1)-0.5, minor=True)
    ax.grid(which='minor')
    plt.xticks(rotation=90)
    fig.colorbar(col)
    plt.show()

