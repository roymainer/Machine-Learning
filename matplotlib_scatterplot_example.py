"""
https://towardsdatascience.com/k-means-in-real-life-clustering-workout-sessions-119946f9e8dd
https://towardsdatascience.com/customizing-plots-with-python-matplotlib-bcf02691931f

K-Means Clustering is a unsupervised learning algorithm that can label/group similar data sets into distinct groups.

"""

import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# loading the datased
df = pd.read_csv('workout_log.csv')
# print(df)
# df.colums = ['date', 'distance_km', 'duration_min']


def scatterplot(_df, x_dim, y_dim, category):
    x = _df[x_dim]  # get x values by using x label as key
    y = _df[y_dim]  # same for y

    # converting original (numerical) lables into categorical labels
    categories = df[category].apply(lambda _x: 'weekday' if _x == 0 else 'weekend')  # 0 -> weekday, 1 -> weekend

    fig, ax = plt.subplots(figsize=(10, 5))  # create a plot

    # defining an array of colors
    colors = ['#2300A8', '#00A658']  # two color scheme for the dots  -- NOT WORKING!

    # iterates through the dataset plotting each data point and assigning it its corresponding color and label
    for i in range(len(df)):
        ax.scatter(x.ix[i], y.ix[i], alpha=0.70, color=colors[i % len(colors)], label=categories.ix[i])

    # # plot data as scatter graph, set dots alpha to 0.7 for readability, use custom color map with the color sequence
    # # this will color all dots matching to the category with the color map, two types (0,1) -> two colors
    # ax.scatter(x, y, alpha=0.70, c=df[category], cmap=cm.brg)

    # add a title and axes labels
    ax.set_title("Distance vs Workout Duration")
    ax.set_xlabel('Distance (Km)')
    ax.set_ylabel('Duration (min)')

    # removing top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # adds major gridlines
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    # adds legend
    ax.legend(categories.unique())

    plt.show()  # show


scatterplot(df, 'distance_km', 'duration_min', 'day_category')
