from statistics import mean

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from src.data_generation.generate_synthetic_correlated_data import calculate_spearman_correlation
from src.data_generation.generate_synthetic_segmented_dataset import SyntheticDataSegmentCols
from src.use_case.ticc.TICC_helper import compute_bic
from src.use_case.temporal_network_analysis import TemporalNetworkAnalysis, \
    from_list_of_times_mrf_to_node_node_time_3d_array

from src.utils.plots.matplotlib_helper_functions import reset_matplotlib, display_title, set_axis_label_font_size, \
    fontsize, Backends


def check_if_two_matrices_are_the_same(m1, m2) -> bool:
    # without rounding they are not equal which might just be due to np's equality
    round_to = 8
    return (np.around(m1, decimals=round_to) == np.around(m2, decimals=round_to)).all()


class TICCResult:
    """
    Class to analyse the results from the TICC algorithm
    """

    def __init__(self, data: np.array, cluster_assignment: [], dictionary_of_mrf_for_clusters: {},
                 empirical_covariances: np.ndarray,
                 number_of_clusters: int, number_of_time_series: int, window_size: int, has_converged: bool,
                 assignments_over_time=None, time_series_names: [str] = None,
                 backend: str = Backends.none.value):
        """
        :param data: the actual data
        :param assignments_over_time:
        :param number_of_clusters: k provided to TICC
        :param cluster_assignment: list of assigned cluster number for each observation
        :param dictionary_of_mrf_for_clusters: markov random fields for each cluster, shape is a dictionary with a
        key for each of cluster and value being an array of w*n x w*n where w=window size, n = no variants
        :param empirical_covariances: empirical covariances for each cluster
        :param number_of_time_series: number of time series variates in data
        :param window_size: number of observations are in each window
        :param has_converged: indicates if the algorithm fully converged (no changes in mfr) between two runs
        :param assignments_over_time: list of length iterations of ndarrays of shape (observations,)
        :param time_series_names: [str] optional if provided these names will be used for the plots
        """
        self.data = data
        self.has_converged = has_converged
        self.number_of_time_series = number_of_time_series
        self.cluster_assignment = cluster_assignment
        self.number_of_observations = len(self.cluster_assignment)
        self.mrf = dictionary_of_mrf_for_clusters
        self.empirical_covariances = empirical_covariances
        self.number_of_clusters = number_of_clusters
        self.number_of_none_zero_clusters = len(set(self.cluster_assignment))
        self.window_size = window_size
        self.assignments_over_time = assignments_over_time
        self.times_series_names = time_series_names if time_series_names else [str(x) for x in
                                                                               range(self.number_of_time_series)]
        # cached/lazy calculated results
        self.__temporal_network_per_clusters = None
        self.__temporal_network_over_clusters = None
        self.__number_of_times_each_cluster_is_used = None
        self.__number_of_observations_per_segment = None
        self.__number_of_segments = None
        self.__segments = None
        self.__bic = None
        self.__segment_lengths = None
        self.__backend = backend

    def bic(self) -> float:
        """
        Calculates and caches bic for training results
        :return: float
        """
        if self.__bic is None:
            self.__bic = compute_bic(self.number_of_observations, self.cluster_assignment, self.mrf,
                                     self.empirical_covariances)
        return self.__bic

    def number_of_segments(self) -> int:
        """
        Restructures cluster_assignment into segments
        :return: int
        """
        if self.__number_of_segments is None:
            self.__number_of_segments = len(self.segments())

        return self.__number_of_segments

    def segments(self) -> [[]]:
        """
        Restructures cluster_assignments into all the segments
        :return:
        """
        if self.__segments is None:
            self.__segments = self.__calculate_segments()

        return self.__segments

    def number_of_times_each_cluster_is_used(self) -> []:
        """
        How many times each cluster has been used
        :return: list of number of times the cluster with the index of the list is used
        """
        if self.__number_of_times_each_cluster_is_used is None:
            repetition_of_cluster = self.__calculate_number_of_times_each_cluster_is_used()
            self.__number_of_times_each_cluster_is_used = repetition_of_cluster
        return self.__number_of_times_each_cluster_is_used

    def number_of_observations_per_segment(self) -> [()]:
        """
        :return: list of tuples with first element being segment number and second being number of observations
        """
        if self.__number_of_observations_per_segment is None:
            self.__number_of_observations_per_segment = self.__calculate_number_of_observations_in_each_segment()
        return self.__number_of_observations_per_segment

    def segment_lengths(self) -> [int]:

        """
        :returns: list of segment lengths
        """
        if self.__segment_lengths is None:
            self.__segment_lengths = self.__calculate_segment_lengths()
        return self.__segment_lengths

    def mean_segment_length(self, round_to: int = 3) -> float:
        """
        returns mean segment length
        """
        return round(mean(self.segment_lengths()), round_to)

    def min_segment_length(self) -> int:
        """
        returns min segment length
        """
        return min(self.segment_lengths())

    def max_segment_length(self) -> int:
        """
        returns max segment length
        """
        return max(self.segment_lengths())

    def check_diagonal_matrices_are_the_same(self) -> [bool]:
        """
        Checks if the resulting mrfs on the diagonal and lower off diagonal are all the same
        :return: dictionary with a list for each cluster of bool for each of the diagonal
        """
        # look at MRF for each cluster
        result = {}
        for cluster_id in self.mrf.keys():
            all_diags = []
            if self.window_size is 1:  # there is no off diag
                all_diags.append(True)
            else:
                for diag in range(self.window_size):
                    ms = self.adjacency_matrices_for_cluster(cluster_id=cluster_id, off_diagonal=diag)
                    are_same = []
                    for i in range(len(ms) - 1):
                        m1 = ms[i]
                        m2 = ms[i + 1]
                        are_same.append(check_if_two_matrices_are_the_same(m1, m2))
                    if are_same:
                        all_diags.append(are_same)

            result[cluster_id] = all_diags

        return result

    def __calculate_number_of_times_each_cluster_is_used(self) -> []:
        """
        Calculates how many times each cluster is used
        :return: list of number of times the cluster with the index of the list is used
        """
        repetition_of_cluster = [0] * self.number_of_clusters  # to allow for clusters to be zero inbetween
        for clusterx_list in self.segments():
            cluster_index = clusterx_list[0]
            repetition_of_cluster[int(cluster_index)] += 1
        return repetition_of_cluster

    def __calculate_segments(self) -> []:
        """
        Calculates a list of lists of segments
        :return:
        """
        list_of_segments = []
        current_cluster = int(self.cluster_assignment[0])
        segment = []

        for i in range(len(self.cluster_assignment)):
            current_observations_cluster = int(self.cluster_assignment[i])
            if current_observations_cluster != current_cluster:  # create new list
                current_cluster = current_observations_cluster
                list_of_segments.append(segment)
                segment = []
            segment.append(current_cluster)
        list_of_segments.append(segment)  # append the last one
        return list_of_segments

    def __calculate_number_of_observations_in_each_segment(self):
        """
        Calculates how many observations are in each segment
        :return: list of tuples with first element being cluster number and second element being number of observations
        """
        observations_in_segment = []
        for observations_for_segment in self.segments():
            cluster_index = observations_for_segment[0]
            observations_in_segment.append((cluster_index, len(observations_for_segment)))
        return observations_in_segment

    def clusters(self) -> [int]:
        """
        Returns list of clusters calculated from resulting MRFs
        """
        return list(self.mrf.keys())

    def get_mrf_for_cluster(self, cluster_id: int) -> np.ndarray:
        """
        Returns mrf matrix as ndarray for cluster with cluster_id
        :rtype: ndarray of dimensions w*n x w*n where w=window size, n = no variants
        """
        return self.mrf[cluster_id]

    def adjacency_matrices_for_cluster(self, cluster_id: int, off_diagonal: int = 0) -> [np.ndarray]:
        """
        Returns list of adjacency matrices for each t_x in window w. Each matrix (ndarray) is of dimensions w*n x w*n
        where w=window size, n = no variants. In case of TICC they should all be the same.
        :param cluster_id: which cluster
        :param off_diagonal: 0 is diagonal (time ti), 1 is lower one off diagonal (times inter relations), ...
        """
        return access_lower_block_diagonals(self.mrf[cluster_id], self.window_size, off_diagonal)

    def adjacency_matrices_for_each_time_relationship(self, cluster_id: int) -> []:
        """
        Returns all block nxn block matrices in the first column of the mrf. Matrix at index 0 is diagonal
        relationships between ts at time t, 1 is one off diagonal (relationship between t & t+1), etc.
        :param cluster_id: cluster identifier
        :rtype []: list of all the adjacency matrices for all the relationships, len = window size
        """
        result = []
        # move down the first column of block matrices
        for i in range(self.window_size):
            number_of_run = i + 1
            start_index = i * self.number_of_time_series
            end_index = (number_of_run * self.number_of_time_series)
            result.append(self.mrf[cluster_id][start_index:end_index, 0:self.number_of_time_series])
        return result

    def plot_mrf_for_cluster_as_heatmap(self, cluster_id, show_title: bool = True):
        """
        Plots heatmaps for the mrf for cluster with id cluster_id
        :param cluster_id: id for cluster
        :param show_title: bool if title is shown
        """
        # get min and max over all mrf for comparative colouring
        vmin, vmax = self.__get_min_max_mrf_value()

        graphs = self.adjacency_matrices_for_each_time_relationship(cluster_id)
        x_labels = []
        for time in range(self.window_size):
            if time == 0:
                if self.window_size == 1:
                    x_labels.append('time $t_0$\n')
                else:
                    x_labels.append('times $t_0, ..., t_' + str(self.window_size) + '$\n')
            else:
                x_labels.append("between times $t_x$ and $t_{x+" + str(time) + "}$\n")

        # setup figure
        reset_matplotlib(self.__backend)
        sns.set(font_scale=1.2)
        number_of_plot_columns = self.window_size + 1
        width_ratios = self.window_size * [10] + [0.2]  # 10 for the heatmaps, 0.5 for the colour bar
        fig, axs = plt.subplots(nrows=1,
                                ncols=number_of_plot_columns,
                                sharey=False,
                                sharex=False,
                                squeeze=0,
                                figsize=(25, 8),
                                width_ratios=width_ratios
                                )

        # plot data, cols are the different inter time relationship
        for col in range(self.window_size):
            ax = axs[0, col]
            data = np.around(graphs[col], decimals=2)
            sns.heatmap(data,
                        linewidth=0.0,
                        square=True,
                        ax=ax,
                        annot=True,
                        mask=data == 0,  # only show non-zero values
                        xticklabels=self.times_series_names,
                        yticklabels=self.times_series_names,
                        vmin=vmin,
                        vmax=vmax,
                        center=0,
                        cbar=True,
                        cmap="RdBu",
                        cbar_ax=axs[0, number_of_plot_columns - 1],
                        cbar_kws={"shrink": 0.9},
                        annot_kws={"fontsize": fontsize}
                        )
            ax.set_xlabel(x_labels[col], fontsize=fontsize)
            set_axis_label_font_size(ax)

        display_title(fig, 'Adjacency Matrices of TICC MRF for cluster ' + str(cluster_id), show_title)

        fig.tight_layout()
        plt.show()
        return fig

    def verify_all_paper_assumptions_for_mfr(self) -> bool:
        check_diag = self.check_diagonal_matrices_are_the_same()
        clusters_in_mrf = list(self.mrf.keys())
        all_diag_are_the_same = [all(check_diag[cluster]) for cluster in clusters_in_mrf]

        undirected_graphs = self.check_adjacency_matrices_are_undirected_graphs()
        all_adj_are_undirected = [all(undirected_graphs[cluster]) for cluster in clusters_in_mrf]

        differences_in_keys = self.find_differences_in_cluster_keys_assigned_and_between_mrf()

        if not all(all_diag_are_the_same):
            print("Not all clusters have Toeplitz Matrices")
            print(check_diag)
            print('\n')

        if not all(all_adj_are_undirected):
            print("Not all clusters have undirected graphs")
            print(all_adj_are_undirected)
            print('\n')

        if len(differences_in_keys) is not 0:
            print("Cluster ids used in assignment are different to MRF")
            print(differences_in_keys)
            print('\n')

        if all(all_diag_are_the_same) and all(all_adj_are_undirected) and len(differences_in_keys) is 0:
            print("Yes they hold.")

        return all(all_diag_are_the_same) and all(all_adj_are_undirected) and (len(differences_in_keys) is 0)

    def check_adjacency_matrices_are_undirected_graphs(self) -> {int: [bool]}:
        """
        Checks that adjacency matrices are symmetric. Only then are they undirected graphs.
        It only checks the first column. So if the result is a proper Toeplitz matrix then the rest does not need to
        be checked.
        """
        tol = 0.000001  # tolerance how much a value can be off
        result = {}
        for cluster in self.mrf.keys():
            ms = self.adjacency_matrices_for_each_time_relationship(cluster)
            ad_results = []
            for m in ms:
                # check if symmetric
                ad_results.append((np.abs(m - m.T) <= tol).all())
            result[cluster] = ad_results
        return result

    def find_differences_in_cluster_keys_assigned_and_between_mrf(self) -> []:
        """
         Returns the differences in keys found in assignment and the mrf
        """
        clusters_assigned = set(self.cluster_assignment)
        mrf = set(self.mrf.keys())
        return list(clusters_assigned - mrf)

    def print_info(self):
        """
        Prints some results - useful for notebooks
        """
        print("Number of non zero clusters")
        print(self.number_of_none_zero_clusters)
        print("Number of time each cluster is used")
        print(self.number_of_times_each_cluster_is_used())
        print("Number of segments")
        print(self.number_of_segments())
        print("Number of observation per segment (cluster_id, no_observation)")
        print(self.number_of_observations_per_segment())
        print("Do paper assumptions hold? (MFR related)")
        print(self.verify_all_paper_assumptions_for_mfr())
        print("Has converge before max iterations")
        print(self.has_converged)

    def get_mrf_for_cluster_as_3d_array(self, cluster: int):
        """
        Translates mrf for times $t_i$ of cluster into 3d array of shape [node, node, time]
        """
        mrfs = self.adjacency_matrices_for_cluster(cluster, 0)
        return np.stack(mrfs, axis=2)

    def betweenness_centrality_for_all_cluster(self):
        """
        Calculates betweenness centrality for all clusters

        (quicker to calculate per cluster than use the clusters as time points!)

        :return: nd.array of shape (cluster,nodes)
        """
        result = None
        for ta in self.temporal_network_per_clusters():
            betweenness_cent_for_cluster = ta.betweeness_centrality_overtime()
            if result is None:
                result = betweenness_cent_for_cluster
            else:
                result = np.vstack([result, betweenness_cent_for_cluster])

        return self.__expand_dimension_to_2d_array(result)

    def degree_centrality_for_all_clusters(self):
        """
        Calculates degree centrality for all clusters

        :return: nd.array of shape (cluster,nodes)
        """
        result = None
        for ta in self.temporal_network_per_clusters():
            degree_cent_for_cluster = ta.degree_centrality()
            if result is None:
                result = degree_cent_for_cluster
            else:
                result = np.vstack([result, degree_cent_for_cluster])

        return self.__expand_dimension_to_2d_array(result)

    def closeness_centrality_for_all_clusters(self):
        """
        Calculates closeness centrality for all clusters

        :return: nd.array of shape (cluster,nodes)
        """
        result = None
        for ta in self.temporal_network_per_clusters():
            degree_cent_for_cluster = ta.closeness_centrality()
            if result is None:
                result = degree_cent_for_cluster
            else:
                result = np.vstack([result, degree_cent_for_cluster])

        return self.__expand_dimension_to_2d_array(result)

    def plot_network_slice_plot_over_all_clusters(self):
        return self.temporal_network_over_clusters().plot_slice_plot(time_axis_name="Clusters")

    def temporal_network_over_clusters(self) -> TemporalNetworkAnalysis:
        """
        Calculates the temporal network using only the first diagonal block matrix for each cluster and using
        the different clusters quasi as time steps.
        (This is due to the graph at each time step being the same for TICC)
        """
        # lazy loading
        if self.__temporal_network_over_clusters is None:
            self.__temporal_network_over_clusters = self.__calculate_temporal_network_over_clusters()
        return self.__temporal_network_over_clusters

    def temporal_network_per_clusters(self) -> [TemporalNetworkAnalysis]:
        """
        Calculates the temporal network using only the first diagonal block matrix for each cluster.
        (This is due to the graph at each time step being the same for TICC)
        """
        # lazy loading
        if self.__temporal_network_per_clusters is None:
            self.__temporal_network_per_clusters = self.__calculate_temporal_network_per_clusters()
        return self.__temporal_network_per_clusters

    def __calculate_temporal_network_over_clusters(self) -> TemporalNetworkAnalysis:
        """
        Calculates Temporal Network for t0 of each cluster with times being the different clusters

        Given TICC forces the graph at each time t to be the same we can just use the first diagonal matrix to calculate
        this measure. We therefore use the clusters as time steps.
        """
        # for each cluster get the first adjacency matrix on the diagonal which is the graph at t0
        diagonals = [self.adjacency_matrices_for_cluster(cluster)[0] for cluster in
                     self.mrf.keys()]
        # stack them as 3d array as if each cluster were a point in time
        diagonals_3d_array = from_list_of_times_mrf_to_node_node_time_3d_array(diagonals)
        return TemporalNetworkAnalysis(adjacency_matrices=diagonals_3d_array, node_names=self.times_series_names,
                                       backend=self.__backend)

    def __calculate_temporal_network_per_clusters(self) -> [TemporalNetworkAnalysis]:
        """
        Calculates a Temporal Network for t0 of each cluster
        """
        result = []
        # for each cluster get the first adjacency matrix on the diagonal which is the graph at t0
        diagonals = [self.adjacency_matrices_for_cluster(cluster)[0] for cluster in
                     self.mrf.keys()]
        for diagonal in diagonals:
            as3darray = from_list_of_times_mrf_to_node_node_time_3d_array([diagonal])
            result.append(TemporalNetworkAnalysis(adjacency_matrices=as3darray, node_names=self.times_series_names,
                                                  backend=self.__backend))
        return result

    def __calculate_segment_lengths(self) -> [int]:
        return [tup[1] for tup in self.number_of_observations_per_segment()]

    @staticmethod
    def __expand_dimension_to_2d_array(result):
        """
            Translate nd.arrays of shape (x,) to 2d arrays of shape (1,x)
        """
        if result is None or result is []:  # empty list or none
            return result
        if result.ndim == 1:
            result = np.expand_dims(result, axis=0)
        return result

    def number_of_segments_with_x_observations(self, x: int):
        """
        :param x: count number of segments with x observations
        :return: number of segments with just one observation
        """
        no_obs_per_seg = self.number_of_observations_per_segment()
        number_of_obs = [tup[1] for tup in no_obs_per_seg]
        return number_of_obs.count(x)

    def __get_min_max_mrf_value(self):
        """ Finds the min and the max value in the mrf across all clusters"""
        mins = []
        maxs = []
        # get min and max for each cluster
        for cluster_id in self.mrf.keys():
            mins.append(self.mrf[cluster_id].min())
            maxs.append(self.mrf[cluster_id].max())
        # return overall min and max
        return min(mins), max(maxs)

    def to_labels_df(self, subject_id:str, round_to: int = 3):
        """
        Translates results into labels df so it fits the rest of the evaluation code we have
        :return:
        """
        segment_ids = []
        start_idxs = []
        end_idxs = []
        lengths = []
        cluster_ids = []
        empirical_correlations = []

        start_idx = 0
        for seg_id, segment in enumerate(self.segments()):
            assert len(set(segment)) == 1, "Found wrong segment with different cluster ids"
            segment_ids.append(seg_id)

            length = len(segment)
            end_idx = start_idx + length - 1  # used convention so far to select end_idx
            start_idxs.append(start_idx)
            end_idxs.append(end_idx)
            lengths.append(length)
            cluster_ids.append(segment[0])

            # calculate empirical correlation from data
            segment_data = self.data[start_idx:end_idx + 1, :]  # need to include end_idx in selection
            corrs = calculate_spearman_correlation(segment_data, round_to=round_to)
            empirical_correlations.append(corrs)

            # update for next segment
            start_idx = end_idx + 1

        result = {
            SyntheticDataSegmentCols.segment_id: segment_ids,
            SyntheticDataSegmentCols.start_idx: start_idxs,
            SyntheticDataSegmentCols.end_idx: end_idxs,
            SyntheticDataSegmentCols.length: lengths,
            SyntheticDataSegmentCols.pattern_id: cluster_ids,
            SyntheticDataSegmentCols.actual_correlation: empirical_correlations
        }
        df = pd.DataFrame(result)
        df.insert(0, SyntheticDataSegmentCols.subject_id, subject_id)
        return df


def access_lower_block_diagonals(a: np.array, number_of_blocks: int, sub_diagonal: int = 0):
    if a.ndim != 2:
        raise ValueError("Only 2-D arrays handled")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Needs to be square matrix")
    if sub_diagonal > number_of_blocks:
        raise ValueError("Sub diagonal needs to be <= number_of_blocks in matrix")

    result = []
    size = a.shape[0]
    size_of_block = int(size / number_of_blocks)

    # get the block matrices on the diagonal, 0 is diagonal, 1 is one below the diagonal, ...
    for i in range(number_of_blocks - sub_diagonal):
        start_index = i * size_of_block  # 0, size_of_block, 2 * size_of_block
        end_index = start_index + size_of_block  # size_of_block, 2 * size_of_block
        if sub_diagonal == 0:
            result.append(a[start_index:end_index, start_index:end_index])
        else:
            # selects lower half blocks of the off diagonals
            start_index_row = start_index + (sub_diagonal * size_of_block)
            end_index_row = start_index_row + size_of_block
            m = a[start_index_row:end_index_row, start_index:end_index]
            result.append(m)
    return result
