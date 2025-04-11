import numpy as np
from matplotlib import pyplot as plt
from teneto import TemporalNetwork

from src.utils.plots.matplotlib_helper_functions import Backends, reset_matplotlib, set_axis_label_font_size, \
    display_title, fontsize


def from_list_of_times_mrf_to_node_node_time_3d_array(list_of_mrf_over_time: [np.array]):
    """
    Translates a list of mrf for different time points of format (node, node) to a 3d array of format (node, node,time)
    using the list index for time 0, 1, 2, 3
    """
    return np.stack(list_of_mrf_over_time, axis=2)


class TemporalNetworkAnalysis:
    def __init__(self, adjacency_matrices, node_names: [str], nettype: str = None,
                 backend: str = Backends.none.value):
        """
        :param node_names: [str] names for the nodes in the matrices
        :param adjacency_matrices: 3d nd.array of shape (node, node, time)
        :param nettype: teneto type
        """
        self.__adjacency_matrices = adjacency_matrices
        self.__node_names = node_names
        self.nettype = nettype if nettype else "wd"
        self.temporalNetwork: TemporalNetwork = TemporalNetwork(from_array=self.__adjacency_matrices,
                                                                nettype=self.nettype,
                                                                diagonal=True)
        self.__backend = backend

    def plot_slice_plot(self, time_labels: [str] = None, time_axis_name=None, title: str = "Temporal Network"):
        """
        Plots the slice plot of the network
        :param time_labels: if provided uses this for the time label otherwise numbers 1-tn
        :param time_axis_name: if provide uses this to label the x-axis
        """
        reset_matplotlib(self.__backend)
        fig_size = (10, 4)
        fig, axs = plt.subplots(nrows=1,
                                ncols=1,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)
        ax = axs[0, 0]
        self.temporalNetwork.plot('slice_plot', ax=ax, nodelabels=self.__node_names, timelabels=time_labels)

        if time_axis_name:
            ax.set_xlabel(time_axis_name, fontsize=fontsize)
        else:
            ax.set_xlabel('Time', fontsize=fontsize)

        display_title(fig, title=title)
        set_axis_label_font_size(ax)

        fig.tight_layout()
        plt.show()
        return fig

    def betweeness_centrality_pertime(self):
        """
        Calculates betweeneess centrality for each node in each of the time steps
        :returns nd.array of shape (nodes, times)
        """
        return self.temporalNetwork.calc_networkmeasure('temporal_betweenness_centrality', calc='pertime')

    def betweeness_centrality_overtime(self) -> float:
        """
        Calculates betweenness centrality for each node over all the time steps
        :returns nd.arrray of shape (nodes)
        """
        return self.temporalNetwork.calc_networkmeasure('temporal_betweenness_centrality', calc='overtime')

    def closeness_centrality(self):
        """
        Calculates temporal closeness centrality for each node over all the time steps
        :returns nd.arrray of shape (nodes)
        """
        return self.temporalNetwork.calc_networkmeasure('temporal_closeness_centrality')

    def degree_centrality(self):
        """
        Calculates degree centrality for each node over all the time steps
        :returns nd.arrray of shape (nodes)
        """
        return self.temporalNetwork.calc_networkmeasure('temporal_degree_centrality')
