from enum import Enum

import matplotlib
from matplotlib import pyplot as plt

fontsize = 20


class Backends(str, Enum):
    none = 'agg'
    visible_tests = 'module://backend_interagg'  # pycharm sci editor
    inline_notebooks = 'module://matplotlib_inline.backend_inline'  # pycharm notebooks


def display_title(fig, title: str, show_title: bool = True):
    if show_title:
        title = title
        fig.suptitle(title, fontsize=fontsize)


def display_legend(ax, fig, show_legend: bool = True):
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(bbox_to_anchor=(1, 1), handles=handles, loc='upper left', fontsize=fontsize)
        plt.subplots_adjust(right=0.87)


def set_axis_label_font_size(ax):
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)


def reset_matplotlib(backend: str = Backends.none.value):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'figure.facecolor': 'white',
                         'axes.facecolor': 'white',
                         'figure.dpi': 300,
                         'axes.labelsize': fontsize,
                         'axes.titlesize': fontsize,
                         'xtick.labelsize': 'xx-large',
                         'ytick.labelsize': 'xx-large',
                         'legend.fontsize': fontsize,
                         'axes.grid': True,
                         'grid.color': "grey"
                         })

    matplotlib.rcParams['backend'] = backend


def use_latex_labels():
    plt.rc('text', usetex=True)


def up_block_size(block_size: int):
    matplotlib.rcParams['agg.path.chunksize'] = block_size
