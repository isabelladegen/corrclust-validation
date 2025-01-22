import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns
from src.evaluation.describe_synthetic_dataset import DescribeSyntheticDataset
from src.utils.configurations import SYNTHETIC_DATA_DIR
from src.utils.distance_measures import distance_calculation_method_for
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


class KNNForSyntheticWrapper:
    """This is assessing knn for the synthetic dataset
    See also paper on The UCR Time Series Archive for TS evaluation
    """

    def __init__(self, measure: str, n_neighbors: int = 1, data_dir: str = SYNTHETIC_DATA_DIR,
                 backend: str = Backends.none.value):
        # will store most recent predictions
        self.ds_name = None
        self.y_pred = None
        self.y_true = None
        self.x_test = None

        self.measure = measure
        self.n_neighbors = n_neighbors
        self.__backend = backend
        self.__data_dir = data_dir

        # train on ideal ground truth
        self.x_train, self.y_train = ModelCorrelationPatterns().x_and_y_of_patterns_to_model()

        distance_calc = distance_calculation_method_for(measure)

        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=distance_calc, n_jobs=-1)

        self.knn.fit(self.x_train, self.y_train)

    def load_data_and_predict(self, ds_name: str, data_type: str = SyntheticDataType.non_normal_correlated,
                              value_range: (float, float) = None, bad_partition_name: str = ""):
        """returns x_test, y_true and y_pred for the given synthetic dataset"""
        ds = DescribeSyntheticDataset(run_name=ds_name, data_type=data_type, value_range=value_range,
                                      data_dir=self.__data_dir,
                                      backend=self.__backend, bad_partition_name=bad_partition_name)
        self.ds_name = ds_name
        x_test, y_true = ds.x_and_y_of_patterns_modelled()
        return self.predict(x_test, y_true)

    def predict(self, x_test: np.ndarray, y_true: np.ndarray):
        self.x_test = x_test
        self.y_true = y_true
        self.y_pred = self.knn.predict(self.x_test)
        return self.x_test, self.y_true, self.y_pred

    def evaluate_scores(self, average, round_to: int = 2):
        if self.y_pred is None:
            assert False, "Call load_data_and_predict before evaluation"
        accuracy = accuracy_score(self.y_true, self.y_pred),
        precision = precision_score(self.y_true, self.y_pred, average=average, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, average=average)
        f1 = f1_score(self.y_true, self.y_pred, average=average)
        return round(accuracy[0], round_to), round(precision, round_to), round(recall, round_to), round(f1, round_to)

    def plot_confusion_matrix(self):
        matplotlib.rcParams['backend'] = self.__backend
        accuracy, precision, recall, f1 = self.evaluate_scores(average='macro')
        display = ConfusionMatrixDisplay.from_predictions(self.y_true, self.y_pred, xticks_rotation='vertical')
        fig = display.figure_
        fig.suptitle(self.ds_name + ": " + self.measure + "\nPrecision:" + str(precision) + ", Recall:" + str(
            recall) + ", F1:" + str(f1))
        plt.show()
        return fig

    def error_information(self, round_to: int = 2):
        """
        :return: error sample (correlation given), error correlations (ideal corr pattern of the wrong class),
        correct correlations (ideal corr pattern of the correct class), nn class' for error (wrong pattern id),
        correct class' for error (correct pattern id),
        distance to nn (distance to wrong pattern), distance to correct class (distance to correct pattern)
        """
        error_idx = [idx for idx, true_class in enumerate(self.y_true) if true_class != self.y_pred[idx]]
        if len(error_idx) == 0:
            return [], [], [], [], [], [], []
        error_sample = [self.x_test[idx] for idx in error_idx]
        correct_classes_for_errors = [self.y_true[idx] for idx in error_idx]
        nn_class_for_error = [self.y_pred[idx] for idx in error_idx]
        # find index in training for the correct class correlations
        idx_training_correct = [np.where(self.y_train == true_class)[0][0] for true_class in correct_classes_for_errors]
        idx_training_error = [np.where(self.y_train == true_class)[0][0] for true_class in nn_class_for_error]
        # actual training sample = correct classes
        correct_correlations = [self.x_train[idx] for idx in idx_training_correct]
        error_correlations = [self.x_train[idx] for idx in idx_training_error]

        # get distance to nearest neighbour
        d_to_nn, _ = self.knn.kneighbors(error_sample, n_neighbors=self.n_neighbors,
                                         return_distance=True)

        # calculate distance between error and true class
        d_to_correct_class = [round(self.knn.metric(error, correct), round_to) for error, correct in
                              zip(error_sample, correct_correlations)]

        return error_sample, error_correlations, correct_correlations, nn_class_for_error, correct_classes_for_errors, d_to_nn.ravel().round(
            2).tolist(), d_to_correct_class
