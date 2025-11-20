import os
from os import path

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.validity.visualise_raw_values import dist_labels
from src.utils.configurations import ResultsType, VALID_ROOT_RESULTS_DIR
from src.utils.distance_measures import DistanceMeasures

# TODO: load from results construct test 1 and 3 instead of hardcode here: Data for 23 clusters (original data)
data_23 = {
    'count': '23',
    'Normal 100%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.98, 0.00),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.98, 0.00),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.97, 0.00),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.97, 0.00)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.05, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.05, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.04, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (13475.54, 3038.84),
                dist_labels[DistanceMeasures.l2_cor_dist]: (12762.34, 2825.69),
                dist_labels[DistanceMeasures.l3_cor_dist]: (12487.38, 2748.09),
                dist_labels[DistanceMeasures.l5_cor_dist]: (12061.59, 2656.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (13324.21, 2958.72),
                dist_labels[DistanceMeasures.dot_transform_l2]: (12693.24, 2783.29)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (15.14, 3.05),
                dist_labels[DistanceMeasures.l2_cor_dist]: (6.63, 1.33),
                dist_labels[DistanceMeasures.l3_cor_dist]: (5.28, 1.06),
                dist_labels[DistanceMeasures.l5_cor_dist]: (4.31, 0.86),
                dist_labels[DistanceMeasures.dot_transform_l1]: (25.47, 4.99),
                dist_labels[DistanceMeasures.dot_transform_l2]: (7.71, 1.52)}
    },
    'Normal 70%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.97, 0.00),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.97, 0.00)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.06, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.05, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.05, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.05, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.06, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.05, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (9313.93, 2351.48),
                dist_labels[DistanceMeasures.l2_cor_dist]: (8849.07, 2300.49),
                dist_labels[DistanceMeasures.l3_cor_dist]: (8678.76, 2264.80),
                dist_labels[DistanceMeasures.l5_cor_dist]: (8397.54, 2186.52),
                dist_labels[DistanceMeasures.dot_transform_l1]: (9288.92, 2396.67),
                dist_labels[DistanceMeasures.dot_transform_l2]: (8832.04, 2308.71)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (10.56, 2.09),
                dist_labels[DistanceMeasures.l2_cor_dist]: (4.62, 0.94),
                dist_labels[DistanceMeasures.l3_cor_dist]: (3.69, 0.75),
                dist_labels[DistanceMeasures.l5_cor_dist]: (3.01, 0.61),
                dist_labels[DistanceMeasures.dot_transform_l1]: (17.84, 3.60),
                dist_labels[DistanceMeasures.dot_transform_l2]: (5.39, 1.11)}
    },
    'Normal 10%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.91, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.91, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.92, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.92, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.91, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.91, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.16, 0.02),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.14, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.14, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.14, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.16, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.14, 0.02)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (1257.43, 388.43),
                dist_labels[DistanceMeasures.l2_cor_dist]: (1186.00, 371.46),
                dist_labels[DistanceMeasures.l3_cor_dist]: (1162.61, 364.33),
                dist_labels[DistanceMeasures.l5_cor_dist]: (1126.41, 352.88),
                dist_labels[DistanceMeasures.dot_transform_l1]: (1273.07, 390.40),
                dist_labels[DistanceMeasures.dot_transform_l2]: (1189.52, 370.66)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (1.48, 0.33),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.64, 0.14),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.51, 0.11),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.41, 0.09),
                dist_labels[DistanceMeasures.dot_transform_l1]: (2.52, 0.55),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.74, 0.17)}
    },
    'Non-normal 100%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.97, 0.00),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.98, 0.00),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.98, 0.00),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.97, 0.00),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.97, 0.00)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.05, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.05, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.04, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (13391.78, 3178.89),
                dist_labels[DistanceMeasures.l2_cor_dist]: (12632.33, 2912.86),
                dist_labels[DistanceMeasures.l3_cor_dist]: (12343.12, 2822.84),
                dist_labels[DistanceMeasures.l5_cor_dist]: (11902.30, 2719.74),
                dist_labels[DistanceMeasures.dot_transform_l1]: (13250.29, 3064.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (12573.30, 2861.58)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (15.27, 3.32),
                dist_labels[DistanceMeasures.l2_cor_dist]: (6.62, 1.40),
                dist_labels[DistanceMeasures.l3_cor_dist]: (5.25, 1.11),
                dist_labels[DistanceMeasures.l5_cor_dist]: (4.27, 0.90),
                dist_labels[DistanceMeasures.dot_transform_l1]: (25.68, 5.44),
                dist_labels[DistanceMeasures.dot_transform_l2]: (7.70, 1.61)}
    },
    'Non-normal 10%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.90, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.91, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.91, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.92, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.90, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.91, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.16, 0.03),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.15, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.14, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.14, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.16, 0.03),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.15, 0.02)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (1237.54, 389.21),
                dist_labels[DistanceMeasures.l2_cor_dist]: (1169.73, 373.71),
                dist_labels[DistanceMeasures.l3_cor_dist]: (1146.87, 367.16),
                dist_labels[DistanceMeasures.l5_cor_dist]: (1110.89, 356.29),
                dist_labels[DistanceMeasures.dot_transform_l1]: (1253.41, 393.34),
                dist_labels[DistanceMeasures.dot_transform_l2]: (1173.81, 373.38)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (1.44, 0.33),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.62, 0.14),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.49, 0.11),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.40, 0.09),
                dist_labels[DistanceMeasures.dot_transform_l1]: (2.45, 0.57),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.73, 0.17)}
    }
}

# Data for 11 clusters
data_11 = {
    'count': '11',
    'Normal 100%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.98, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.04, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (17196.30, 10420.54),
                dist_labels[DistanceMeasures.l2_cor_dist]: (15271.93, 8375.50),
                dist_labels[DistanceMeasures.l3_cor_dist]: (14859.39, 7894.11),
                dist_labels[DistanceMeasures.l5_cor_dist]: (14612.00, 7594.39),
                dist_labels[DistanceMeasures.dot_transform_l1]: (17227.56, 10269.43),
                dist_labels[DistanceMeasures.dot_transform_l2]: (15240.01, 8287.83)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (86.27, 61.30),
                dist_labels[DistanceMeasures.l2_cor_dist]: (33.97, 23.91),
                dist_labels[DistanceMeasures.l3_cor_dist]: (26.58, 18.80),
                dist_labels[DistanceMeasures.l5_cor_dist]: (22.10, 15.37),
                dist_labels[DistanceMeasures.dot_transform_l1]: (129.00, 93.07),
                dist_labels[DistanceMeasures.dot_transform_l2]: (36.80, 25.30)}
    },
    'Normal 70%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.97, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.97, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.05, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.05, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.04, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (11589.44, 6665.89),
                dist_labels[DistanceMeasures.l2_cor_dist]: (10509.94, 5793.45),
                dist_labels[DistanceMeasures.l3_cor_dist]: (10270.58, 5541.95),
                dist_labels[DistanceMeasures.l5_cor_dist]: (10118.74, 5375.40),
                dist_labels[DistanceMeasures.dot_transform_l1]: (11549.30, 6240.22),
                dist_labels[DistanceMeasures.dot_transform_l2]: (10474.45, 5617.37)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (56.61, 34.42),
                dist_labels[DistanceMeasures.l2_cor_dist]: (22.49, 13.81),
                dist_labels[DistanceMeasures.l3_cor_dist]: (17.60, 10.84),
                dist_labels[DistanceMeasures.l5_cor_dist]: (14.62, 8.80),
                dist_labels[DistanceMeasures.dot_transform_l1]: (84.34, 52.71),
                dist_labels[DistanceMeasures.dot_transform_l2]: (24.30, 14.43)}
    },
    'Normal 10%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.92, 0.02),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.92, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.92, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.92, 0.02)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.13, 0.03),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.12, 0.03),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.12, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.11, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.13, 0.03),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.12, 0.03)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (1601.65, 905.04),
                dist_labels[DistanceMeasures.l2_cor_dist]: (1441.93, 780.04),
                dist_labels[DistanceMeasures.l3_cor_dist]: (1405.88, 734.72),
                dist_labels[DistanceMeasures.l5_cor_dist]: (1381.76, 698.59),
                dist_labels[DistanceMeasures.dot_transform_l1]: (1614.32, 865.77),
                dist_labels[DistanceMeasures.dot_transform_l2]: (1441.72, 764.50)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (8.47, 5.05),
                dist_labels[DistanceMeasures.l2_cor_dist]: (3.33, 1.99),
                dist_labels[DistanceMeasures.l3_cor_dist]: (2.59, 1.54),
                dist_labels[DistanceMeasures.l5_cor_dist]: (2.14, 1.24),
                dist_labels[DistanceMeasures.dot_transform_l1]: (12.61, 7.76),
                dist_labels[DistanceMeasures.dot_transform_l2]: (3.59, 2.07)}
    },
    'Non-normal 100%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.98, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.04, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (16826.88, 9730.84),
                dist_labels[DistanceMeasures.l2_cor_dist]: (14998.71, 7987.20),
                dist_labels[DistanceMeasures.l3_cor_dist]: (14596.36, 7496.38),
                dist_labels[DistanceMeasures.l5_cor_dist]: (14341.51, 7145.45),
                dist_labels[DistanceMeasures.dot_transform_l1]: (16912.04, 9614.67),
                dist_labels[DistanceMeasures.dot_transform_l2]: (14983.99, 7891.72)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (87.62, 60.79),
                dist_labels[DistanceMeasures.l2_cor_dist]: (34.13, 23.49),
                dist_labels[DistanceMeasures.l3_cor_dist]: (26.56, 18.21),
                dist_labels[DistanceMeasures.l5_cor_dist]: (21.97, 14.66),
                dist_labels[DistanceMeasures.dot_transform_l1]: (131.14, 93.41),
                dist_labels[DistanceMeasures.dot_transform_l2]: (37.03, 25.07)}
    },
    'Non-normal 10%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.92, 0.02),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.92, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.92, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.92, 0.02)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.13, 0.03),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.12, 0.03),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.12, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.12, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.13, 0.03),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.12, 0.03)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (1571.66, 893.23),
                dist_labels[DistanceMeasures.l2_cor_dist]: (1418.67, 771.66),
                dist_labels[DistanceMeasures.l3_cor_dist]: (1383.10, 727.58),
                dist_labels[DistanceMeasures.l5_cor_dist]: (1358.72, 692.56),
                dist_labels[DistanceMeasures.dot_transform_l1]: (1584.02, 855.69),
                dist_labels[DistanceMeasures.dot_transform_l2]: (1419.27, 756.71)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (8.22, 4.95),
                dist_labels[DistanceMeasures.l2_cor_dist]: (3.23, 1.94),
                dist_labels[DistanceMeasures.l3_cor_dist]: (2.51, 1.50),
                dist_labels[DistanceMeasures.l5_cor_dist]: (2.07, 1.20),
                dist_labels[DistanceMeasures.dot_transform_l1]: (12.25, 7.66),
                dist_labels[DistanceMeasures.dot_transform_l2]: (3.49, 2.03)}
    }
}

# Data for 6 clusters
data_6 = {
    'count': '6',
    'Normal 100%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.98, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.03, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.03, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.03, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.03, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (20985.09, 16753.32),
                dist_labels[DistanceMeasures.l2_cor_dist]: (18768.77, 14388.12),
                dist_labels[DistanceMeasures.l3_cor_dist]: (18333.11, 14114.06),
                dist_labels[DistanceMeasures.l5_cor_dist]: (18178.36, 14224.07),
                dist_labels[DistanceMeasures.dot_transform_l1]: (19493.51, 14707.92),
                dist_labels[DistanceMeasures.dot_transform_l2]: (18181.85, 13650.63)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (282.65, 254.99),
                dist_labels[DistanceMeasures.l2_cor_dist]: (106.98, 96.18),
                dist_labels[DistanceMeasures.l3_cor_dist]: (82.78, 75.45),
                dist_labels[DistanceMeasures.l5_cor_dist]: (69.70, 63.04),
                dist_labels[DistanceMeasures.dot_transform_l1]: (356.84, 304.95),
                dist_labels[DistanceMeasures.dot_transform_l2]: (109.45, 95.51)}
    },
    'Normal 70%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.97, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.97, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.97, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.04, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (13617.85, 9699.38),
                dist_labels[DistanceMeasures.l2_cor_dist]: (12145.79, 8057.84),
                dist_labels[DistanceMeasures.l3_cor_dist]: (11860.22, 7783.90),
                dist_labels[DistanceMeasures.l5_cor_dist]: (11738.74, 7692.56),
                dist_labels[DistanceMeasures.dot_transform_l1]: (12608.27, 8100.56),
                dist_labels[DistanceMeasures.dot_transform_l2]: (11755.38, 7487.11)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (189.92, 161.09),
                dist_labels[DistanceMeasures.l2_cor_dist]: (71.07, 59.30),
                dist_labels[DistanceMeasures.l3_cor_dist]: (54.85, 45.86),
                dist_labels[DistanceMeasures.l5_cor_dist]: (46.09, 37.52),
                dist_labels[DistanceMeasures.dot_transform_l1]: (240.86, 197.23),
                dist_labels[DistanceMeasures.dot_transform_l2]: (72.65, 58.14)}
    },
    'Normal 10%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.93, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.93, 0.02)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.11, 0.03),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.10, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.10, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.10, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.11, 0.03),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.10, 0.02)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (2029.33, 1193.11),
                dist_labels[DistanceMeasures.l2_cor_dist]: (1827.38, 1048.20),
                dist_labels[DistanceMeasures.l3_cor_dist]: (1791.24, 1036.65),
                dist_labels[DistanceMeasures.l5_cor_dist]: (1780.47, 1051.63),
                dist_labels[DistanceMeasures.dot_transform_l1]: (1912.60, 1108.73),
                dist_labels[DistanceMeasures.dot_transform_l2]: (1785.08, 1022.95)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (28.93, 21.46),
                dist_labels[DistanceMeasures.l2_cor_dist]: (10.87, 7.95),
                dist_labels[DistanceMeasures.l3_cor_dist]: (8.39, 6.14),
                dist_labels[DistanceMeasures.l5_cor_dist]: (7.05, 5.05),
                dist_labels[DistanceMeasures.dot_transform_l1]: (37.68, 29.30),
                dist_labels[DistanceMeasures.dot_transform_l2]: (11.23, 8.27)}
    },
    'Non-normal 100%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.98, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.98, 0.01)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.04, 0.01),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.03, 0.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.03, 0.01),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.03, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.04, 0.01),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.03, 0.01)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (21330.96, 17349.78),
                dist_labels[DistanceMeasures.l2_cor_dist]: (18824.80, 14464.88),
                dist_labels[DistanceMeasures.l3_cor_dist]: (18325.05, 14124.61),
                dist_labels[DistanceMeasures.l5_cor_dist]: (18100.46, 14165.92),
                dist_labels[DistanceMeasures.dot_transform_l1]: (19734.62, 15070.40),
                dist_labels[DistanceMeasures.dot_transform_l2]: (18226.76, 13693.02)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (302.63, 295.31),
                dist_labels[DistanceMeasures.l2_cor_dist]: (111.87, 107.09),
                dist_labels[DistanceMeasures.l3_cor_dist]: (85.96, 83.26),
                dist_labels[DistanceMeasures.l5_cor_dist]: (71.80, 68.86),
                dist_labels[DistanceMeasures.dot_transform_l1]: (377.94, 345.97),
                dist_labels[DistanceMeasures.dot_transform_l2]: (114.04, 105.28)}
    },
    'Non-normal 10%': {
        'SWC': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.93, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.93, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.93, 0.02)},
        'DBI': {dist_labels[DistanceMeasures.l1_cor_dist]: (0.11, 0.03),
                dist_labels[DistanceMeasures.l2_cor_dist]: (0.10, 0.02),
                dist_labels[DistanceMeasures.l3_cor_dist]: (0.10, 0.02),
                dist_labels[DistanceMeasures.l5_cor_dist]: (0.10, 0.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (0.11, 0.03),
                dist_labels[DistanceMeasures.dot_transform_l2]: (0.10, 0.02)},
        'VRC': {dist_labels[DistanceMeasures.l1_cor_dist]: (2021.20, 1211.60),
                dist_labels[DistanceMeasures.l2_cor_dist]: (1829.14, 1079.51),
                dist_labels[DistanceMeasures.l3_cor_dist]: (1792.50, 1067.12),
                dist_labels[DistanceMeasures.l5_cor_dist]: (1779.54, 1079.72),
                dist_labels[DistanceMeasures.dot_transform_l1]: (1905.71, 1136.85),
                dist_labels[DistanceMeasures.dot_transform_l2]: (1788.35, 1057.95)},
        'PBM': {dist_labels[DistanceMeasures.l1_cor_dist]: (28.90, 21.80),
                dist_labels[DistanceMeasures.l2_cor_dist]: (10.83, 8.01),
                dist_labels[DistanceMeasures.l3_cor_dist]: (8.33, 6.15),
                dist_labels[DistanceMeasures.l5_cor_dist]: (6.97, 5.02),
                dist_labels[DistanceMeasures.dot_transform_l1]: (37.67, 30.15),
                dist_labels[DistanceMeasures.dot_transform_l2]: (11.21, 8.45)}
    }
}


def plot_errorbars_for_data(data_1, data_2, data_3, title):
    global width, index
    # Thresholds
    thresholds = {'SWC': 0.9, 'DBI': 0.15, 'VRC': 1000, 'PBM': 10}
    y_limits = {'SWC': {'min': 0.3, 'max': 1},
                'DBI': {'min': 0, 'max': 1},
                'VRC': {'min': 0, 'max': 40000},
                'PBM': {'min': 0, 'max': 800}}
    threshold_direction = {
        'SWC': 'below',  # valid when > 0.9
        'DBI': 'above',  # valid when < 0.15
        'VRC': 'below',  # valid when > 1000
        'PBM': 'below',  # valid when > 100
    }
    # Setup
    conditions = ['Normal 100%', 'Normal 70%', 'Normal 10%', 'Non-normal 100%', 'Non-normal 10%']
    indices = ['SWC', 'DBI', 'VRC', 'PBM']
    distances = [dist_labels[DistanceMeasures.l1_cor_dist], dist_labels[DistanceMeasures.l2_cor_dist],
                 dist_labels[DistanceMeasures.l3_cor_dist], dist_labels[DistanceMeasures.l5_cor_dist],
                 dist_labels[DistanceMeasures.dot_transform_l1], dist_labels[DistanceMeasures.dot_transform_l2]]
    # colours = ['#008080','#008000', '#ca69ca']
    colours = ['#8cbed8', '#629318', '#ca69ca']
    marker_size = 6
    cap_size = 2
    fontsize = 12
    fig, axes = plt.subplots(5, 4, figsize=(16, 12))
    x_positions = np.arange(len(distances))
    width = 0.25
    for row, condition in enumerate(conditions):
        for col, index in enumerate(indices):
            ax = axes[row, col]

            # Extract data for all cluster counts
            means_23 = [data_1[condition][index][d][0] for d in distances]
            sds_23 = [data_1[condition][index][d][1] for d in distances]

            means_11 = [data_2[condition][index][d][0] for d in distances]
            sds_11 = [data_2[condition][index][d][1] for d in distances]

            means_6 = [data_3[condition][index][d][0] for d in distances]
            sds_6 = [data_3[condition][index][d][1] for d in distances]

            # Plot with offset positions
            ax.errorbar(x_positions - width, means_23, yerr=sds_23, fmt='o', capsize=cap_size,
                        color=colours[0], ecolor=colours[0], markersize=marker_size,
                        label=data_1['count'] if row == 0 and col == 0 else '')

            ax.errorbar(x_positions, means_11, yerr=sds_11, fmt='s', capsize=cap_size,
                        color=colours[1], ecolor=colours[1], markersize=marker_size,
                        label=data_2['count'] if row == 0 and col == 0 else '')

            ax.errorbar(x_positions + width, means_6, yerr=sds_6, fmt='^', capsize=cap_size,
                        color=colours[2], ecolor=colours[2], markersize=marker_size,
                        label=data_3['count'] if row == 0 and col == 0 else '')

            # Set consistent y-axis per column
            ymax = y_limits[index]['max']
            ymin = y_limits[index]['min']
            padding = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - padding, ymax + padding)

            # Threshold line
            threshold = thresholds[index]
            ax.axhline(y=threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

            # grey out failed sone
            ylim = ax.get_ylim()
            alpha = 0.08
            if threshold_direction[index] == 'above':
                ax.axhspan(threshold, ylim[1], alpha=alpha, color='grey', zorder=0)
                ax.axhspan(ylim[0], threshold, alpha=alpha, color='green', zorder=0)
            elif threshold_direction[index] == 'below':
                ax.axhspan(threshold, ylim[1], alpha=alpha, color='green', zorder=0)
                ax.axhspan(ylim[0], threshold, alpha=alpha, color='grey', zorder=0)

            # Formatting
            ax.tick_params(axis='y', labelsize=fontsize)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(distances, fontsize=fontsize)
            ax.grid(True, alpha=0.3, linestyle=':')

            if row == 0:
                ax.set_title(index, fontweight='bold', fontsize=fontsize)
            if col == 0:
                ax.set_ylabel(condition, fontweight='bold', fontsize=fontsize)

            ax.margins(y=0.15)
    # Add legend
    axes[0, 0].legend(loc='lower right', fontsize=fontsize, framealpha=0.9, title=title, title_fontsize=fontsize)
    plt.tight_layout()


if __name__ == "__main__":
    # legend title
    legend_t = 'Clusters'

    plot_errorbars_for_data(data_23, data_11, data_6, legend_t)
    results_folder = path.join(VALID_ROOT_RESULTS_DIR, ResultsType.internal_measure_evaluation, 'images')
    os.makedirs(results_folder, exist_ok=True)
    plt.savefig(path.join(results_folder, 'structural_test_1_3_clusters.png'), dpi=300, bbox_inches='tight')
    plt.show()
