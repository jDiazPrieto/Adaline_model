from plotPerformance import plot_error
from plotPerformance import plot_scatter
import matplotlib.pyplot as plt


def plot_a_decision_boundary(axes, performance, epoch, wine_data):
    """
    This function plots the decision boundary for our wine_data

    :param axes: Matplotlib.Axes :
    :param performance: list of tuples: each tuple contains performance results for an epoch of our adaline training
    :param epoch: int
    :param wine_data: pandas Dataframe of our wine data
    :return: void
    """
    # get values for decision boundary
    w2, w1 = performance[epoch][2]
    slope = - w1 / w2
    b = performance[epoch][3] - 0.5

    # get min, max from x values
    xMin = wine_data['alcohol'].min()
    xMax = wine_data['alcohol'].max()

    # Calculate y values needed to draw the line
    pHmin = slope * xMin - b / w2
    pHmax = slope * xMax - b / w2

    axes[1].set_ylim([wine_data['pH'].min(), wine_data['pH'].max()])
    axes[1].set_xlim([xMin, xMax])
    axes[1].set_title('Decision boundary on epoch: {}'.format(epoch))

    # draw line
    axes[1].plot([xMin, xMax], [pHmin, pHmax], 'b--', label='Decision boundary')
    # fill good/bad regions for given epoch
    axes[1].fill_between([xMin, xMax], [pHmin, pHmax], wine_data['pH'].min(), color='#EAFBDF')
    axes[1].fill_between([xMin, xMax], [pHmin, pHmax], wine_data['pH'].max(), color='#F8CAD0')


def plot_adaline(performance, wine_data, good_thresh, bad_thresh, epoch= -1, save_plot=False):
    """
    Plot the performance of our adaline.
    This function will produce two plot figures:
    1) classification Errors vs Epochs
    2) Decision boundary for wine_data

    :param performance: list of tuples, each tuple contains performance information for an epoch
    :param wine_data: pandas.dataFrame
    :param good_thresh: int
    :param bad_thresh: int
    :param epoch: int that indicates the epoch that we want to display, if -1 we use the last epoch
    :param save_plot: boolean
    :return fig:
    """
    if epoch >= len(performance) or epoch < 0:
        epoch = len(performance) - 1
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    plot_error(axes, performance)
    plot_a_decision_boundary(axes, performance, epoch, wine_data)
    plot_scatter(axes, wine_data, good_thresh, bad_thresh)

    if save_plot:
        plt.savefig('./performance_plot.png')

    plt.show(fig)
    return fig