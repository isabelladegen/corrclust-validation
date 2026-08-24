from dataclasses import dataclass

import pandas as pd

from src.evaluation.distance_metric_evaluation import EvaluationCriteria, criteria_short_names
from src.utils.configurations import Aggregators


@dataclass
class ValidityResultColumns:
    """Single source of the column names DistanceMeasureValidity writes. Anything
    reading its output (tests, table/text generation) should reference this,
    never a literal string."""
    structural: str = "Structural"
    criterion: str = "Criterion"
    convergent: str = "Convergent"
    discriminant: str = "Discriminant"
    discriminant_no_pattern: str = "Discriminant_raw"
    discriminant_degradation: str = "Discriminant_ds"
    external_normal_70: str = "External_normal_70"
    external_normal_10: str = "External_normal_10"
    external_non_normal_100: str = "External_non_normal_100"
    external_non_normal_10: str = "External_non_normal_10"
    external: str = "External"
    overall: str = "Overall"

    @staticmethod
    def mean_value_for(criterion: str) -> str:
        return f"{criteria_short_names[criterion]}_mean"

    @staticmethod
    def result_for(criterion: str) -> str:
        return f"{criteria_short_names[criterion]}_pass"


class DistanceMeasureValidity:
    """Decides validity of distance measures per Table 1 (methodology sec 4.2.1). Constructed
    from mean-value tables calculated over all subjects of a data variant (index=distance_measure,
    columns=EvaluationCriteria) for each of the  7 data variants."""

    # maps evaluation criteria to validity type
    _structural_criteria = [EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii,
                            EvaluationCriteria.disc_i, EvaluationCriteria.disc_ii]
    _criterion_predictive = EvaluationCriteria.disc_iii
    _structural_pass_minimum = 4

    # avg d (inter_ii) excluded: floored/ceilinged at 0.0 or 1.0, can't show gradual "worse"
    _degradation_criteria = [EvaluationCriteria.inter_i, EvaluationCriteria.inter_iii,
                             EvaluationCriteria.disc_i, EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii]

    # criterion -> (passes predicate, lower_is_better) -- single source of truth for thresholds
    _rules = {
        EvaluationCriteria.inter_i: (lambda x: x <= 0.1, True),
        EvaluationCriteria.inter_ii: (lambda x: x == 1.0, False),
        EvaluationCriteria.inter_iii: (lambda x: x > 0.7, False),
        EvaluationCriteria.disc_i: (lambda x: x > 4, False),
        EvaluationCriteria.disc_ii: (lambda x: x < 3, True),
        EvaluationCriteria.disc_iii: (lambda x: x > 0.98, False),
    }

    def __init__(self, normal_100: pd.DataFrame, normal_70: pd.DataFrame, normal_10: pd.DataFrame,
                 non_normal_100: pd.DataFrame, non_normal_10: pd.DataFrame,
                 raw_100: pd.DataFrame, downsampled_100: pd.DataFrame):
        """Each argument: mean values, index=distance_measure, columns=EvaluationCriteria,
        for that data condition (output of calculate_mean_sd's mean_df, one per condition)."""
        # used for construct-structural
        self._normal_100 = normal_100
        # used for external validity
        self._normal_70 = normal_70
        self._normal_10 = normal_10
        self._non_normal_100 = non_normal_100
        self._non_normal_10 = non_normal_10
        # used for discriminant validity
        self._raw_100 = raw_100
        self._downsampled_100 = downsampled_100

    def structural_validity(self) -> pd.DataFrame:
        return self._structural_result(self._normal_100)

    def criterion_validity(self) -> pd.DataFrame:
        return pd.DataFrame({
            ValidityResultColumns.mean_value_for(self._criterion_predictive): self._normal_100[
                (self._criterion_predictive, Aggregators.mean)],
            ValidityResultColumns.criterion: self._passes(self._normal_100, self._criterion_predictive),
        })

    def convergent_validity(self) -> bool:
        return bool(self.structural_validity()[ValidityResultColumns.structural].sum() >= 2)

    def discriminant_validity(self) -> pd.DataFrame:
        no_pattern_pass = self._structural_pass_count(self._raw_100) <= (
                len(self._structural_criteria) - self._structural_pass_minimum)

        degradation_count = sum(self._is_worse(self._downsampled_100, self._normal_100, c).astype(int)
                                for c in self._degradation_criteria)
        degrades_pass = degradation_count >= self._structural_pass_minimum
        return pd.DataFrame({
            ValidityResultColumns.discriminant_no_pattern: no_pattern_pass,
            ValidityResultColumns.discriminant_degradation: degrades_pass,
            ValidityResultColumns.discriminant: no_pattern_pass & degrades_pass,
        })

    def external_validity(self) -> pd.DataFrame:
        result = pd.DataFrame(index=self._normal_100.index)

        result[ValidityResultColumns.external_normal_70] = self._structural_result(self._normal_70)[
            ValidityResultColumns.structural]
        result[ValidityResultColumns.external_normal_10] = self._structural_result(self._normal_10)[
            ValidityResultColumns.structural]
        result[ValidityResultColumns.external_non_normal_100] = self._structural_result(self._non_normal_100)[
            ValidityResultColumns.structural]
        result[ValidityResultColumns.external_non_normal_10] = self._structural_result(self._non_normal_10)[
            ValidityResultColumns.structural]

        cols = [ValidityResultColumns.external_normal_70, ValidityResultColumns.external_normal_10,
                ValidityResultColumns.external_non_normal_100, ValidityResultColumns.external_non_normal_10]
        result[ValidityResultColumns.external] = result[cols].all(axis=1)
        return result

    def overall_validity(self) -> pd.DataFrame:
        overall = pd.concat([self.structural_validity(), self.criterion_validity(),
                             self.discriminant_validity(), self.external_validity()], axis=1)
        overall[ValidityResultColumns.convergent] = self.convergent_validity()
        overall[ValidityResultColumns.overall] = (
                overall[ValidityResultColumns.structural] & overall[ValidityResultColumns.criterion] & overall[
            ValidityResultColumns.convergent]
                & overall[ValidityResultColumns.discriminant] & overall[ValidityResultColumns.external])
        return overall

    def mean_sd_valid_summary_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validity summary table for one data variant: 'mean (SD sd)*' per distance measure per
        criterion, star = passes its threshold. Assumes mean/sd are already rounded upstream.
        Self-contained per criterion, so works on any of the 7 condition tables without
        needing to know which one it is. Criteria without a rule (e.g. stability) are skipped."""
        result = pd.DataFrame(index=df.index)

        for criterion in df.columns.get_level_values(0).unique():
            if criterion not in self._rules:
                continue
            mean = df[(criterion, Aggregators.mean)]
            sd = df[(criterion, Aggregators.std)]
            star = self._passes(df, criterion).map({True: "*", False: ""})
            column = criteria_short_names[criterion]
            result[column] = mean.astype(str) + " (SD " + sd.astype(str) + ")" + star

        return result

    def _structural_result(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for criterion in self._structural_criteria:
            result[ValidityResultColumns.mean_value_for(criterion)] = df[(criterion, Aggregators.mean)]
            result[ValidityResultColumns.result_for(criterion)] = self._passes(df, criterion)
        result[ValidityResultColumns.structural] = self._structural_pass_count(df) >= self._structural_pass_minimum
        return result

    def _structural_pass_count(self, df: pd.DataFrame) -> pd.Series:
        return sum(self._passes(df, c).astype(int) for c in self._structural_criteria)

    def _passes(self, df: pd.DataFrame, criterion: str) -> pd.Series:
        passes_fn, _ = self._rules[criterion]
        return df[(criterion, Aggregators.mean)].apply(passes_fn)

    def _is_worse(self, df: pd.DataFrame, reference_df: pd.DataFrame, criterion: str) -> pd.Series:
        _, lower_is_better = self._rules[criterion]
        return (df[(criterion, Aggregators.mean)] > reference_df[
            (criterion, Aggregators.mean)]) if lower_is_better else (
                df[(criterion, Aggregators.mean)] < reference_df[(criterion, Aggregators.mean)])
