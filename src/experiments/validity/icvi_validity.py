from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ICVIValCriteria:
    jaccard_corr: str = "Reference Measure: corr with Jaccard index"

    _thresholds: ClassVar[dict] = {
        jaccard_corr: 0.5,
    }

    @staticmethod
    def get_threshold_for(criteria: str) -> float:
        return ICVIValCriteria._thresholds[criteria]