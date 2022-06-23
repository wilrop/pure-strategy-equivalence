import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.set()
sns.despine()
sns.set_context("paper", rc={"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16,
                             "legend.fontsize": 16})
sns.set_style('white', {'axes.edgecolor': "0.5", "pdf.fonttype": 42})
plt.gcf().subplots_adjust(bottom=0.15)


def save_plot(plot_name, filetype, dpi=300):
    """Save a generated plot with the requested name and filetype.

    Args:
        plot_name (str): The file to save the plot to.
        filetype (str): The filetype for the figure.
        dpi (int, optional): The resolution for the final figure. Used when saving as png. (Default value = 300)

    Returns:

    """
    if filetype == 'png':
        plt.savefig(plot_name + ".png", dpi=dpi)
    else:
        plt.savefig(plot_name + ".pdf")

    plt.clf()


def plot_points(filetype, name, df, equilibrium=None, min_x=0, max_x=5000, min_y=0, max_y=20, label1='$p_x$',
                label2='$p_y$', y_label='Price'):
    """Plot the learning curve of strategies over time.

    Args:
        filetype (str): The filetype to save the plots as. (Default value = 'pdf')
        name (str): The name to save the plot under.
        df (DataFrame): The data of the experiment.
        equilibrium (List[Tuple[float, str]], optional): A list of equilibrium points and their labels.
            (Default value = None)
        min_x (int, optional): The minimum value on the x-axis. (Default value = 0)
        max_x (int, optional): The maximum value on the x-axis. (Default value = 5000)
        min_y (int, optional): The minimum value on the y-axis. (Default value = 0)
        max_y (int, optional): The maximum value on the y-axis. (Default value = 20)
        label1 (str, optional): The label for the first player. (Default value = '$p_x$')
        label2 (str, optional): The label for the second player. (Default value = '$p_y$')
        y_label (str, optional): The label across the y-axis. (Default value = 'Price')

    Returns:

    """
    ax = sns.lineplot(x='iteration', y='player1', linewidth=2.0, data=df, ci='sd', label=label1)
    ax = sns.lineplot(x='iteration', y='player2', linewidth=2.0, data=df, ci='sd', label=label2)

    if equilibrium is None:
        equilibrium = []

    for point, label in equilibrium:
        x_data = np.arange(min_x, max_x)
        y_constant = np.full(len(x_data), point)
        ax = sns.lineplot(x=x_data, y=y_constant, linestyle='--', linewidth=1.0, ci='sd', label=label)

    ax.set(ylabel=y_label)
    ax.set(xlabel="Iteration")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plot_name = f"{name}"
    plt.tight_layout()
    save_plot(plot_name, filetype)


def plot_results(filetype='pdf'):
    """Plot the results of both experiments.

    Args:
        filetype (str, optional): The filetype to save the plots as. (Default value = 'pdf')

    Returns:

    """
    name1 = "polynomial_game"
    y_label1 = 'Strategy'
    df1 = pd.read_csv(f'{name1}.csv')
    label1, label2 = ('$x$', '$y$')
    equilibrium1 = [(0.39680, '$x^\\ast$'), (0.62996, '$y^\\ast$')]
    min_x, max_x = (0, df1['iteration'].max())
    min_y, max_y = (-1, 1)
    plot_points(filetype, name1, df1, equilibrium=equilibrium1, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                label1=label1, label2=label2, y_label=y_label1)

    name2 = "bertrand_price_game"
    y_label2 = 'Price'
    df2 = pd.read_csv(f'{name2}.csv')
    label1, label2 = ('$p_x$', '$p_y$')
    equilibrium2 = [(22.987, '$p^\\ast_x$ and $p^\\ast_y$')]
    min_x, max_x = (0, df2['iteration'].max())
    min_y, max_y = (10, 25)
    plot_points(filetype, name2, df2, equilibrium=equilibrium2, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                label1=label1, label2=label2, y_label=y_label2)


if __name__ == "__main__":
    plot_results()
