from dataclasses import dataclass
from os import path

import pandas as pd

from src.utils.configurations import DISTRIBUTION_PARAMS_TO_MODEL_PATH


@dataclass
class DistParamsCols:
    c_iob: str = "c_iob"
    loc_iob: str = "loc_iob"
    scale_iob: str = "scale_iob"
    n_cob: str = "n_cob"
    c_ig: str = "c_ig"
    loc_ig: str = "loc_ig"
    scale_ig: str = "scale_ig"


class ModelDistributionParams:
    def __init__(self):
        file = DISTRIBUTION_PARAMS_TO_MODEL_PATH
        assert (path.exists(file))
        df = pd.read_csv(file)
        self.df = df

    def get_params_for(self, param: str):
        return self.df[param].to_list()
