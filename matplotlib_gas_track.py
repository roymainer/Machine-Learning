"""
https://towardsdatascience.com/linear-regression-in-real-life-4a78d7159f16
"""

import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# loading the datased
df = pd.read_csv('gas_track.csv')


def scatterplot(_df, x_dim, y_dim):
    x = _df[x_dim]  # get x values by using x label as key
    y = _df[y_dim]  # same for y

    fig, ax = plt.subplots(figsize=(10, 5))  # create a plot

    # plot data as scatter graph, set dots alpha to 0.7 for readability, use custom color map with the color sequence
    # this will color all dots matching to the category with the color map, two types (0,1) -> two colors
    ax.scatter(x, y, alpha=0.70, cmap=cm.brg)

    # add a title and axes labels
    ax.set_title("Total Miles Driven vs Total Paid for Gas")
    ax.set_xlabel('Total Driven (Miles)')
    ax.set_ylabel('Total Paid (Dollars)')

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    plt.show()  # show


scatterplot(df, 'Total Miles', 'Total Payed')
