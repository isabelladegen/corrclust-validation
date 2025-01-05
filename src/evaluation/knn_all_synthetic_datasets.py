from dataclasses import dataclass
from os import path
from pathlib import Path

import pandas as pd

from src.evaluation.knn_for_synthetic_wrapper import KNNForSyntheticWrapper
from src.utils.configurations import SYNTHETIC_DATA_DIR, GENERATED_DATASETS_FILE_PATH, \
    distance_measure_assessment_dir_for, ROOT_RESULTS_DIR
from src.utils.load_synthetic_data import SyntheticDataType
from src.utils.plots.matplotlib_helper_functions import Backends


@dataclass
class AssessSynthCols:
    d_correct = "d correct"
    d_error = "d error"
    nn_pattern_id = "NN pattern id"
    correct_pattern_id = "Correct pattern id"
    correct_cor = "Correct cor"
    sample_cor = "Sample cor"
    error_cor = "Error cor"
    misclassification = "misclassification"
    errors = "errors"  # list of tuples in order (true label, misclassified with label)
    name: str = "name"
    measure: str = "measure"
    accuracy: str = "accuracy"
    precision: str = "precision"
    recall: str = "recall"
    f1: str = "f1"


class KnnAllSyntheticDatasets:
    def __init__(self, measures: [], n_neighbours=1, overall_ds_name: str = "n30", run_csv=GENERATED_DATASETS_FILE_PATH,
                 save_confusion_matrix: bool = False, root_results_dir: str = ROOT_RESULTS_DIR, add_ds_prefix: str = '',
                 data_type: str = SyntheticDataType.non_normal_correlated, data_dir: str = SYNTHETIC_DATA_DIR,
                 value_range: (float, float) = None, backend: str = Backends.none.value):
        self.measures = measures
        self.n_neighbours = n_neighbours
        self.save_confusion_matrix = save_confusion_matrix
        self.data_type = data_type
        self.value_range = value_range
        self.data_dir = data_dir
        self.backend = backend
        self.__csv_file = run_csv
        self.__root_results_dir = root_results_dir
        self.__overall_ds_name = overall_ds_name

        # load 30 ds
        self.ds_names = pd.read_csv(self.__csv_file)['Name'].tolist()

        # calculate knn for all ds and measures
        # scores_for_all_measures has columns name, measure, misclassification (number),
        # errors ([(predicted class, actual class)], accuracy, precision, recall, f1, n gen success (information from ds
        # generation), n gen failure (information from ds generation)
        # error_case_analysis has columns name, measure, error class (false prediction), correct class, NN class, d nn,
        # d correct class
        self.scores_for_all_measures, self.error_case_analysis = self.__calculate_knn_for_each_dataset()

    def __calculate_knn_for_each_dataset(self):
        # for scores_for_all_measures
        names = []
        measures = []
        misclassifications = []
        errorss = []
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        # for error_case_analysis
        ec_names = []
        ec_measures = []
        ec_sample_cor = []
        ec_error_cor = []
        ec_correct_cor = []
        ec_nn_class = []
        ec_correct_class = []
        ec_d_error = []
        ec_d_correct = []

        for measure in self.measures:
            results_folder = distance_measure_assessment_dir_for(self.__overall_ds_name, self.data_type,
                                                                 self.__root_results_dir, self.data_dir, measure)

            knn_for_measure = KNNForSyntheticWrapper(measure=measure, n_neighbors=self.n_neighbours,
                                                     data_dir=self.data_dir, backend=self.backend)

            for ds_name in self.ds_names:
                x_test, y_true, y_pred = knn_for_measure.predict(ds_name, data_type=self.data_type,
                                                                 value_range=self.value_range)
                errors = [(true_label, y_pred[idx]) for idx, true_label in enumerate(y_true) if
                          true_label != y_pred[idx]]
                if self.save_confusion_matrix:
                    fig = knn_for_measure.plot_confusion_matrix()
                    cm_path = path.join(results_folder, 'confusion-matrices')
                    Path(cm_path).mkdir(parents=True, exist_ok=True)
                    fig.savefig(path.join(cm_path, ds_name + '.png'))
                accuracy, precision, recall, f1 = knn_for_measure.evaluate_scores(average="macro")

                print("f1: " + str(f1.round(2)))

                # update results
                names.append(ds_name)
                measures.append(measure)
                misclassifications.append(len(errors))
                errorss.append(errors)
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

                # error case analysis
                error_sample, error_cor, correct_cor, nn_classes, correct_classes, d_nn, d_correct = knn_for_measure.error_information(
                    2)
                n_errors = len(error_sample)  # number of errors
                assert n_errors == len(errors), "Different number of errors should not happen"
                ec_names.extend([ds_name] * n_errors)
                ec_measures.extend([measure] * n_errors)
                ec_sample_cor.extend(error_sample)
                ec_error_cor.extend(error_cor)
                ec_correct_cor.extend(correct_cor)
                ec_nn_class.extend(nn_classes)
                ec_correct_class.extend(correct_classes)
                ec_d_error.extend(d_nn)
                ec_d_correct.extend(d_correct)

        score_for_all_measures = pd.DataFrame({AssessSynthCols.name: names, AssessSynthCols.measure: measures,
                                               AssessSynthCols.misclassification: misclassifications,
                                               AssessSynthCols.errors: errorss,
                                               AssessSynthCols.accuracy: accuracies,
                                               AssessSynthCols.precision: precisions,
                                               AssessSynthCols.recall: recalls, AssessSynthCols.f1: f1s,
                                               })

        error_case_analysis = pd.DataFrame({
            AssessSynthCols.name: ec_names,
            AssessSynthCols.measure: ec_measures,
            AssessSynthCols.sample_cor: ec_sample_cor,
            AssessSynthCols.error_cor: ec_error_cor,
            AssessSynthCols.correct_cor: ec_correct_cor,
            AssessSynthCols.nn_pattern_id: ec_nn_class,
            AssessSynthCols.correct_pattern_id: ec_correct_class,
            AssessSynthCols.d_error: ec_d_error,
            AssessSynthCols.d_correct: ec_d_correct,
        })

        return score_for_all_measures, error_case_analysis

    def perfect_f1_scores(self):
        df = self.scores_for_all_measures
        return df[df[AssessSynthCols.f1] == 1.0]

    def with_f1_less_than(self, less_than: float):
        df = self.scores_for_all_measures
        return df[df[AssessSynthCols.f1] < less_than]
