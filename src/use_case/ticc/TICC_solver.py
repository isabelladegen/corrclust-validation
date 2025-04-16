import numpy as np
from sklearn import mixture
import pandas as pd

from src.use_case.ticc.TICC_helper import getTrainTestSplit, update_clusters, upperToFull
from src.use_case.ticc.admm_solver import ADMMSolver

from src.use_case.ticc_result import TICCResult
from src.utils.plots.matplotlib_helper_functions import Backends


class TICC:
    def __init__(self, window_size=10, number_of_clusters=5, lambda_parameter=11e-2, beta=400, max_iters=100,
                 threshold=2e-5, cluster_reassignment=20, biased=False, allow_zero_cluster_inbetween: bool = True,
                 do_training_split: bool = False, keep_track_of_assignments: bool = False,
                 backend: str = Backends.none.value):
        """
        Parameters:
            - window_size: size of the sliding window
            - number_of_clusters: number of clusters
            - lambda_parameter: sparsity parameter
            - switch_penalty: temporal consistency parameter
            - maxIters: number of iterations
            - threshold: convergence threshold
            - cluster_reassignment: number of points to reassign to a 0 cluster
            - biased: Using the biased or the unbiased covariance
            - do_training_split: this was in the original algorithm although this should really happen outside the alg
            however to reproduce the original result we make it possible to provide this
            - keep_track_of_assignments: if put to true the cluster assignments over time for each iterations are kept
        """
        self.keep_track_of_assignments = keep_track_of_assignments
        self.do_training_split = do_training_split
        self.window_size = window_size
        self.number_of_clusters = number_of_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.maxIters = max_iters
        self.threshold = threshold
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.biased = biased
        self.allow_zero_cluster_inbetween = allow_zero_cluster_inbetween
        # training data (left over design): 2d ndarray of dimension (no_train-observations, w*n), example (19605, 30)
        self.complete_d_train = None  # gets assigned in fit

        # this is a weird design, but it is how it's been done. Predict uses this data, fit sets it
        self.trained_model = {'computed_covariance': None,
                              'cluster_mean_stacked_info': None,
                              'time_series_col_size': None}
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(102)

        self.cluster_assignments_over_time = []
        self.has_converged = False
        self.__backend = backend

    def fit(self, data, reassign_points_to_zero_clusters: bool = True,
            use_gmm_initialisation: bool = True) -> TICCResult:
        """
        Main method for TICC solver. (Ajdusted from original to return TICC result and print some updates)
        Parameters:
            :param data: numpy array with rows being observations and columns being the different time series
            :param use_gmm_initialisation: bool -> original True, but it's not mentioned in the paper and feels a bit like cheating
            :param reassign_points_to_zero_clusters: bool -> original True, if points should be assigned to the empty clusters or not
        """
        assert self.maxIters > 0  # must have at least one iteration
        self.log_parameters()

        has_converged = False
        times_series_arr = data
        time_series_rows_size = data.shape[0]
        time_series_col_size = data.shape[1]

        # DOESN'T BELONG HERE SHOULD BE DONE OUTSIDE THE ALGORITHM
        if self.do_training_split:
            # Train test split
            training_indices = getTrainTestSplit(time_series_rows_size, self.num_blocks, self.window_size)
            num_samples = len(training_indices)
            self.complete_d_train = self.stack_training_data(times_series_arr, num_samples, training_indices)
        else:
            # Stack the training data
            # 2d ndarray of dimension (no_train-observations, w*n), example (19605, 30)
            num_samples = time_series_rows_size
            self.complete_d_train = self.stack_training_data(times_series_arr, None, None)

        # Initialization
        # Gaussian Mixture
        if use_gmm_initialisation:
            gmm = mixture.GaussianMixture(n_components=self.number_of_clusters, covariance_type="full")
            gmm.fit(self.complete_d_train)
            clustered_points = gmm.predict(self.complete_d_train)  # -> this is the original starting point
        else:
            # it seems to make more sense to start with all points in the same cluster then gmm assigned clusters
            clustered_points = np.zeros(num_samples)

        old_clustered_points = None  # points from last iteration

        # store cluster assignements over time
        if self.keep_track_of_assignments:
            self.cluster_assignments_over_time.append(clustered_points)

        # PERFORM TRAINING ITERATIONS
        for iters in range(self.maxIters):
            print("\n\n\nITERATION ###", iters)
            # Get the train and test points
            # {cluster_num: [indices of observation in that cluster]}
            train_clusters_arr = self.transform_into_dictionary_of_clusters_and_indices_of_points(clustered_points)

            # dictionary of key=cluster index and value=total number of observations in that cluster
            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.number_of_clusters)}

            # opt_res is list of k (8) ndarrays of length 465
            # this is the M part
            opt_res, cluster_mean_stacked_info, empirical_covariances = self.train_clusters(self.complete_d_train,
                                                                                            len_train_clusters,
                                                                                            time_series_col_size,
                                                                                            train_clusters_arr)

            computed_covariance, train_cluster_inverse = self.optimize_clusters(opt_res)

            print("Length of clusters from train_cluster_array:")
            for cluster in range(self.number_of_clusters):
                print("length of the cluster ", cluster, "------>", len_train_clusters[cluster])

            # update old computed covariance
            old_computed_covariance = computed_covariance

            # this is the E part self.smoothen and update clusters
            # SMOOTHENING
            lle_all_points_clusters = self.smoothen_clusters(computed_covariance, cluster_mean_stacked_info,
                                                             self.complete_d_train, time_series_col_size)
            # Update cluster points - using NEW smoothening
            clustered_points = update_clusters(lle_all_points_clusters, switch_penalty=self.switch_penalty,
                                               allow_zero_cluster_in_between_none_zero=self.allow_zero_cluster_inbetween)

            # save training result
            self.trained_model = {'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'time_series_col_size': time_series_col_size}

            # recalculate lengths
            new_train_clusters = self.transform_into_dictionary_of_clusters_and_indices_of_points(clustered_points)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.number_of_clusters)}

            # missing from the paper but this code is assigning points to empty cluster, not quite sure why this
            # is needed and how it's done
            before_empty_cluster_assign = clustered_points.copy()
            print("Length of clusters from clustered_points BEFORE EMPTY ASSIGN:")
            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            if reassign_points_to_zero_clusters:
                if iters != 0:
                    old_cluster_num__used = [x[1] for x in list(old_computed_covariance.keys())]
                    cluster_norms = [(np.linalg.norm(old_computed_covariance[self.number_of_clusters, i]), i) for i in
                                     old_cluster_num__used]
                    norms_sorted = sorted(cluster_norms, reverse=True)
                    # clusters that are not 0 as sorted by norm
                    valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]
                    non_zero_clusters = len(old_cluster_num__used)
                    zero_clusters = self.number_of_clusters - non_zero_clusters
                    if zero_clusters > non_zero_clusters:
                        print("WARNING!! Assigning points to empty clusters is not designed to "
                              "work when there are more empty than used clusters.")
                        print("Number of empty clusters: " + str(zero_clusters) +
                              ". Number of used clusters: " + str(non_zero_clusters))
                        print("\n")
                    print("Valid clusters norms: " + str(valid_clusters))
                    print("\n")

                    # Add a point to the empty clusters
                    # assuming more non empty clusters than empty ones
                    counter = 0
                    for cluster_num in range(self.number_of_clusters):
                        if len_new_train_clusters[cluster_num] == 0:
                            cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                            counter = (counter + 1) % len(valid_clusters)
                            print("cluster that is zero is:", cluster_num, "selected cluster instead is:",
                                  cluster_selected)
                            start_point = np.random.choice(
                                new_train_clusters[cluster_selected])  # random point number from that cluster
                            for i in range(0, self.cluster_reassignment):
                                # put cluster_reassignment points from point_num in this cluster
                                point_to_move = start_point + i
                                if point_to_move >= len(clustered_points):
                                    break
                                clustered_points[point_to_move] = cluster_num
                                computed_covariance[self.number_of_clusters, cluster_num] = old_computed_covariance[
                                    self.number_of_clusters, cluster_selected]
                                cluster_mean_stacked_info[self.number_of_clusters, cluster_num] = self.complete_d_train[
                                                                                                  point_to_move, :]

            print("Length of clusters from clustered_points:")
            for cluster_num in range(self.number_of_clusters):
                print("length of cluster #", cluster_num, "-------->",
                      sum([x == cluster_num for x in clustered_points]))

            if self.keep_track_of_assignments:
                self.cluster_assignments_over_time.append(clustered_points)

            if np.array_equal(old_clustered_points, clustered_points):  # stop if nothing changes
                has_converged = True
                break

            old_clustered_points = before_empty_cluster_assign
            # end of training

        # train cluster inverse is np.linalg.inv(computed_covariance), it's not used in training!!!
        self.has_converged = has_converged
        return TICCResult(data, clustered_points, train_cluster_inverse, empirical_covariances, self.number_of_clusters,
                          time_series_col_size, self.window_size, has_converged, self.cluster_assignments_over_time,
                          backend=self.__backend)

    def transform_into_dictionary_of_clusters_and_indices_of_points(self, clustered_points):
        """
        Transforms ndarray of clustered points into dictionary of clusters and lists of indices for points in that cluster
        :param clustered_points: ndarray of length TS length
        :return: {cluster_id:[indices of points in cluster]}

        """
        train_clusters_arr = {k: [] for k in range(self.number_of_clusters)}  # {cluster: [point indices]}
        for point_idx, cluster_num in enumerate(clustered_points):
            train_clusters_arr[cluster_num].append(point_idx)
        return train_clusters_arr

    def compute_f_score(self, matching_em, matching_gmm, matching_kmeans, train_confusion_matrix_em,
                        train_confusion_matrix_gmm, train_confusion_matrix_kmeans):
        f1_EM_tr = -1  # computeF1_macro(train_confusion_matrix_EM,matching_EM,num_clusters)
        f1_GMM_tr = -1  # computeF1_macro(train_confusion_matrix_GMM,matching_GMM,num_clusters)
        f1_kmeans_tr = -1  # computeF1_macro(train_confusion_matrix_kmeans,matching_Kmeans,num_clusters)
        print("\n\n")
        print("TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.number_of_clusters):
            matched_cluster__e_m = matching_em[cluster]
            matched_cluster__g_m_m = matching_gmm[cluster]
            matched_cluster__k_means = matching_kmeans[cluster]

            correct_e_m += train_confusion_matrix_em[cluster, matched_cluster__e_m]
            correct_g_m_m += train_confusion_matrix_gmm[cluster, matched_cluster__g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster__k_means]

    def compute_matches(self, train_confusion_matrix_EM, train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        print("Not implemented")
        # matching_Kmeans = find_matching(train_confusion_matrix_kmeans)
        # # matching_GMM = find_matching(train_confusion_matrix_GMM)
        # matching_EM = find_matching(train_confusion_matrix_EM)
        # correct_e_m = 0
        # correct_g_m_m = 0
        # correct_k_means = 0
        # for cluster in range(self.number_of_clusters):
        #     matched_cluster_e_m = matching_EM[cluster]
        #     matched_cluster_g_m_m = matching_GMM[cluster]
        #     matched_cluster_k_means = matching_Kmeans[cluster]
        #
        #     correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster_e_m]
        #     correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster_g_m_m]
        #     correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster_k_means]
        # return matching_EM, matching_GMM, matching_Kmeans

    def smoothen_clusters(self, computed_covariance, cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        print("Smoothen cluster computed covariance keys:")
        print(computed_covariance.keys())
        cluster_num_currently_used = [x[1] for x in list(computed_covariance.keys())]
        for cluster in cluster_num_currently_used:
            cov_matrix = computed_covariance[self.number_of_clusters, cluster][0:(self.num_blocks - 1) * n,
                         0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        print("beginning the smoothening ALGORITHM")
        lle_all_points_clusters = np.zeros([clustered_points_len, self.number_of_clusters])
        for point in range(clustered_points_len):
            if point + self.window_size - 1 < complete_D_train.shape[0]:
                for cluster in cluster_num_currently_used:
                    cluster_mean_stacked = cluster_mean_stacked_info[self.number_of_clusters, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    lle_all_points_clusters[point, cluster] = lle

        return lle_all_points_clusters

    def optimize_clusters(self, opt_res):
        # initialise
        # dictionary with key=cluster index and value 2d ndarray of Toeplitz Matrices for each cluster
        train_cluster_inverse = {}
        # dictionary with key=(k,cluster index) and value covariance matrix, if you calculate np.linalg.inv(computed_covariance) you get train_cluster_inverse, this is sparse
        computed_covariance = {}

        for cluster in range(self.number_of_clusters):
            if opt_res[cluster] is None:
                continue
            opt_res_for_cluster = opt_res[cluster]
            print("OPTIMIZATION for Cluster #", cluster, "DONE!!!")
            # THIS IS THE SOLUTION
            S_est = upperToFull(opt_res_for_cluster, 0)
            X2 = S_est
            # the results of the statement below are never used so why is it calculated? Leaving it in as perhaps
            # important for quality of S_est
            # u, _ = np.linalg.eig(S_est)

            # added this to help with stability
            epsilon = 1e-6
            X2_reg = X2 + epsilon * np.eye(X2.shape[0])

            # Handle potential NaNs
            if np.isnan(X2_reg).any():
                X2_reg = np.nan_to_num(X2_reg, nan=epsilon)

            cov_out = np.linalg.inv(X2_reg)

            computed_covariance[self.number_of_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2_reg

        return computed_covariance, train_cluster_inverse

    def train_clusters(self, complete_D_train, len_train_clusters, n, train_clusters_arr):
        # initialise
        optRes = [None for i in range(self.number_of_clusters)]
        # dictionary with key=(k,cluster index) and value a list of length w*n not sure what the values mean
        cluster_mean_stacked_info = {}
        empirical_covariances = [None for i in range(self.number_of_clusters)]

        for cluster in range(self.number_of_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, self.window_size * n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]  # all observation currently in that cluster

                cluster_mean_stacked_info[self.number_of_clusters, cluster] = np.mean(D_train, axis=0)
                # Fit a model - OPTIMIZATION
                probSize = self.window_size * size_blocks
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                empirical_covariance = np.cov(np.transpose(D_train), bias=self.biased)

                rho = 1
                solver = ADMMSolver(lamb, self.window_size, size_blocks, 1, empirical_covariance)  # M step
                verbose = False
                optRes[cluster] = solver(1000, 1e-6, 1e-6, verbose)
                empirical_covariances[cluster] = empirical_covariance
        return optRes, cluster_mean_stacked_info, empirical_covariances

    def stack_training_data(self, data, num_train_points=None, training_indices=None):
        n = data.shape[1]
        if num_train_points is None:
            num_train_points = data.shape[0]
        if training_indices is None:
            training_indices = list(range(num_train_points))
        complete_d_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_d_train[i][k * n:(k + 1) * n] = data[idx_k][0:n]
        return complete_d_train

    def load_data(self, input_file):
        Data = np.loadtxt(input_file, delimiter=",")
        (m, n) = Data.shape  # m: num of observations, n: size of observation vector
        print("completed getting the data")
        return Data, m, n

    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("switch_penalty", self.switch_penalty)
        print("num_cluster", self.number_of_clusters)
        print("num stacked", self.window_size)

    def predict_clusters(self, test_data):
        """
        Given the current trained model, predict clusters.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        """
        if not isinstance(test_data, np.ndarray):
            raise TypeError(
                "test data must be a numpy array with rows observation at time i and columns different variates!")

        number_of_time_series = self.trained_model['time_series_col_size']
        test_data_number_of_ts = test_data.shape[1]
        if test_data_number_of_ts is not number_of_time_series:
            raise TypeError(
                "test data must have the same number of time series like the training data. Expected: " + str(
                    number_of_time_series) + " got " + str(test_data_number_of_ts))

        # stack data same as in training
        stacked_test_data = self.stack_training_data(test_data)

        if stacked_test_data.shape[1] is not number_of_time_series * self.window_size:
            raise TypeError(
                "Test data has wrong dimensions")

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         stacked_test_data,
                                                         number_of_time_series)

        # Update cluster points - using NEW smoothening
        clustered_points = update_clusters(lle_all_points_clusters, switch_penalty=self.switch_penalty,
                                           allow_zero_cluster_in_between_none_zero=self.allow_zero_cluster_inbetween)

        return TICCResult(test_data, clustered_points, None, None,
                          self.number_of_clusters, number_of_time_series, self.window_size, self.has_converged, [],
                          backend=self.__backend)


def analyse_segments(cluster_assignment, number_of_clusters):
    list_of_lists = []
    current_cluster = int(cluster_assignment[0])
    new_list = []
    # some more useful output than cluster assignment
    for i in range(len(cluster_assignment)):
        current_observations_cluster = int(cluster_assignment[i])
        if current_observations_cluster != current_cluster:  # create new list
            current_cluster = current_observations_cluster
            list_of_lists.append(new_list)
            new_list = []
        new_list.append(current_cluster)
    list_of_lists.append(new_list)  # the last one
    print("Cluster assignment")
    print("Number of segments: " + str(len(list_of_lists)))
    repetition_of_cluster = [0] * number_of_clusters
    for clusterx_list in list_of_lists:
        cluster_index = clusterx_list[0]
        repetition_of_cluster[int(cluster_index)] += 1
        print(
            "Segment for cluster: " + str(cluster_index) + ", with number of observations: " + str(len(clusterx_list)))
    print("\n")
    print("Number of times cluster repeats (list is in order of cluster):")
    print(repetition_of_cluster)
