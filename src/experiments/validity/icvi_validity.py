from dataclasses import dataclass


@dataclass
class ICVIValCriteria:
    jaccard_corr: str = "Reference Measure: corr with Jaccard index"

ICVI_THRESHOLDS = {
    ICVIValCriteria.jaccard_corr: 0.5,
}

# criteria for which lower values are better
icvi_inverse_criteria = []