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


DM_THRESHOLDS = {
    EvaluationCriteria.inter_i: 0.1,
    EvaluationCriteria.inter_ii: 1.0,
    EvaluationCriteria.inter_iii: 0.7,
    EvaluationCriteria.scale_free_inter_iii: 0.4,
    EvaluationCriteria.disc_i: 4,
    EvaluationCriteria.disc_ii: 3,
    EvaluationCriteria.disc_iii: 0.98,
}


class ValidityRule:
    """One full config (Table 1) for a DistanceMeasureValidity run. Structural is reused as-is
    for external validity (same determination, replicated across data variants).
    Structural criteria are a list of EvaluationCriteria required to establish structural construct and external validity
    Structural_must_hold is a list of criteria that must pass
    Structural minimum is the number of the left over criteria that must pass
    Dito for discriminant_no_pattern and discriminant_degradation
    """

    def __init__(self,
                 structural_criteria: list[str],
                 structural_minimum: int,
                 discriminant_no_pattern_criteria: list[str],
                 discriminant_no_pattern_minimum: int,
                 discriminant_degradation_criteria: list[str],
                 discriminant_degradation_minimum: int,
                 criterion_predictive: str,
                 structural_must_hold: list[str] = None,
                 discriminant_no_pattern_must_hold: list[str] = None,
                 discriminant_degradation_must_hold: list[str] = None):
        self.structural = CriteriaRule(structural_criteria, structural_minimum, structural_must_hold)
        self.no_pattern = CriteriaRule(discriminant_no_pattern_criteria, discriminant_no_pattern_minimum,
                                       discriminant_no_pattern_must_hold)
        self.degradation = CriteriaRule(discriminant_degradation_criteria, discriminant_degradation_minimum,
                                        discriminant_degradation_must_hold)
        self.criterion_predictive = criterion_predictive


class CriteriaRule:
    """
    Evaluate criteria. Every criterion in must_hold has to satisfy the condition. Separately, at least `minimum` of
    the criteria not in must_hold must also satisfy it.
    """

    def __init__(self, criteria: list[str], minimum: int, must_hold: list[str] = None):
        self.criteria = criteria
        self.minimum = minimum
        self.must_hold = must_hold or []

    def evaluate(self, satisfied: dict) -> pd.Series:
        index = satisfied[self.criteria[0]].index
        mandatory_ok = pd.Series(True, index=index)
        for c in self.must_hold:
            mandatory_ok &= satisfied[c]
        remaining = [c for c in self.criteria if c not in self.must_hold]
        count_ok = (sum(satisfied[c].astype(int) for c in remaining) >= self.minimum
                    if remaining else True)
        return mandatory_ok & count_ok


# various rules
INITIAL_PAPER_RULE = ValidityRule(
    structural_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii,
                         EvaluationCriteria.disc_i, EvaluationCriteria.disc_ii],
    structural_minimum=4,
    discriminant_no_pattern_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                      EvaluationCriteria.inter_iii, EvaluationCriteria.disc_i,
                                      EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii],
    discriminant_no_pattern_minimum=4,
    discriminant_degradation_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_i,
                                       EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii],
    discriminant_degradation_minimum=4,
    criterion_predictive=EvaluationCriteria.disc_iii,
)

# removing entropy based rules since not derivable from theory
STRICT_MUST_PASS_RULES = ValidityRule(
    structural_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii,
                         EvaluationCriteria.disc_i, EvaluationCriteria.disc_ii],
    structural_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii],
    structural_minimum=0, # entropy can fail
    discriminant_no_pattern_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                      EvaluationCriteria.inter_iii, EvaluationCriteria.disc_i,
                                      EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii],
    discriminant_no_pattern_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_iii],
    discriminant_no_pattern_minimum=0, # entropy can fail
    discriminant_degradation_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_i,
                                       EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii],
    discriminant_degradation_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_iii,
                                        EvaluationCriteria.disc_iii], #allwoing the levels set still to pass structural 2
    discriminant_degradation_minimum=0, # entropy can fail
    criterion_predictive=EvaluationCriteria.disc_iii,
)

# strictly must pass for rule 1-3 plus rule 3 replaced with cliffs delta
REVIEWED_RULES = ValidityRule(
    structural_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.scale_free_inter_iii,
                         EvaluationCriteria.disc_i, EvaluationCriteria.disc_ii],
    structural_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.scale_free_inter_iii],
    structural_minimum=0, # entropy can fail
    discriminant_no_pattern_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                      EvaluationCriteria.scale_free_inter_iii, EvaluationCriteria.disc_i,
                                      EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii],
    discriminant_no_pattern_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.scale_free_inter_iii, EvaluationCriteria.disc_iii],
    discriminant_no_pattern_minimum=0, # entropy can fail
    discriminant_degradation_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.scale_free_inter_iii, EvaluationCriteria.disc_i,
                                       EvaluationCriteria.disc_ii, EvaluationCriteria.disc_iii],
    discriminant_degradation_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.scale_free_inter_iii,
                                        EvaluationCriteria.disc_iii], #allwoing the levels set still to pass structural 2
    discriminant_degradation_minimum=0, # entropy can fail
    criterion_predictive=EvaluationCriteria.disc_iii,
)

DROPPING_OVERALL_ENTROPY = ValidityRule(
    structural_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii,
                         EvaluationCriteria.disc_ii],
    structural_minimum=4,
    discriminant_no_pattern_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                      EvaluationCriteria.inter_iii, EvaluationCriteria.disc_ii,
                                      EvaluationCriteria.disc_iii],
    discriminant_no_pattern_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_iii],
    discriminant_no_pattern_minimum=0,  # level set entropy can pass by coincidence, everything else must fail
    discriminant_degradation_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_ii,
                                       EvaluationCriteria.disc_iii],
    discriminant_degradation_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                        EvaluationCriteria.inter_iii, EvaluationCriteria.disc_iii],
    discriminant_degradation_minimum=0,  # level set entropy can stay flat, everything else must worsen
    criterion_predictive=EvaluationCriteria.disc_iii,
)


class DistanceMeasureValidity:
    """Decides validity of distance measures per Table 1 (methodology sec 4.2.1). Constructed
    from mean-value tables calculated over all subjects of a data variant (index=distance_measure,
    columns=EvaluationCriteria) for each of the  7 data variants."""

    # criterion -> (passes predicate, lower_is_better) -- single source of truth for thresholds
    _rules = {
        EvaluationCriteria.inter_i: (lambda x: x <= DM_THRESHOLDS[EvaluationCriteria.inter_i], True),
        EvaluationCriteria.inter_ii: (lambda x: x == DM_THRESHOLDS[EvaluationCriteria.inter_ii], False),
        EvaluationCriteria.inter_iii: (lambda x: x > DM_THRESHOLDS[EvaluationCriteria.inter_iii], False),
        EvaluationCriteria.scale_free_inter_iii: (lambda x: x > DM_THRESHOLDS[EvaluationCriteria.scale_free_inter_iii], False),
        EvaluationCriteria.disc_i: (lambda x: x > DM_THRESHOLDS[EvaluationCriteria.disc_i], False),
        EvaluationCriteria.disc_ii: (lambda x: x < DM_THRESHOLDS[EvaluationCriteria.disc_ii], True),
        EvaluationCriteria.disc_iii: (lambda x: x > DM_THRESHOLDS[EvaluationCriteria.disc_iii], False),
    }

    def __init__(self, validity_rule: ValidityRule, normal_100: pd.DataFrame,
                 normal_70: pd.DataFrame, normal_10: pd.DataFrame, non_normal_100: pd.DataFrame,
                 non_normal_10: pd.DataFrame, raw_100: pd.DataFrame, downsampled_100: pd.DataFrame):
        """Each argument: mean values, index=distance_measure, columns=EvaluationCriteria,
        for that data condition (output of calculate_mean_sd's mean_df, one per condition)."""
        # setup criteria to evaluate for each aspect
        self._validity_rule = validity_rule
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
        cp = self._validity_rule.criterion_predictive
        return pd.DataFrame({
            ValidityResultColumns.mean_value_for(cp): self._normal_100[(cp, Aggregators.mean)],
            ValidityResultColumns.criterion: self._passes(self._normal_100, cp),
        })

    def convergent_validity(self) -> bool:
        return bool(self.structural_validity()[ValidityResultColumns.structural].sum() >= 2)

    def discriminant_validity(self) -> pd.DataFrame:
        no_pattern_satisfied = {c: ~self._passes(self._raw_100, c) for c in self._validity_rule.no_pattern.criteria}
        no_pattern_pass = self._validity_rule.no_pattern.evaluate(no_pattern_satisfied)

        degradation_satisfied = {c: self._is_worse(self._downsampled_100, self._normal_100, c)
                                 for c in self._validity_rule.degradation.criteria}
        degrades_pass = self._validity_rule.degradation.evaluate(degradation_satisfied)

        return pd.DataFrame({
            ValidityResultColumns.discriminant_no_pattern: no_pattern_pass,
            ValidityResultColumns.discriminant_degradation: degrades_pass,
            ValidityResultColumns.discriminant: no_pattern_pass & degrades_pass,
        })

    def external_validity(self) -> pd.DataFrame:
        # unchanged below: each call to _structural_result already goes through
        # self._rule.structural, so external reuses the same rule automatically
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
        satisfied = {}
        for criterion in self._validity_rule.structural.criteria:
            passes = self._passes(df, criterion)
            result[ValidityResultColumns.mean_value_for(criterion)] = df[(criterion, Aggregators.mean)]
            result[ValidityResultColumns.result_for(criterion)] = passes
            satisfied[criterion] = passes
        result[ValidityResultColumns.structural] = self._validity_rule.structural.evaluate(satisfied)
        return result

    def _passes(self, df: pd.DataFrame, criterion: str) -> pd.Series:
        passes_fn, _ = self._rules[criterion]
        return df[(criterion, Aggregators.mean)].apply(passes_fn)

    def _is_worse(self, df: pd.DataFrame, reference_df: pd.DataFrame, criterion: str) -> pd.Series:
        _, lower_is_better = self._rules[criterion]
        return (df[(criterion, Aggregators.mean)] > reference_df[
            (criterion, Aggregators.mean)]) if lower_is_better else (
                df[(criterion, Aggregators.mean)] < reference_df[(criterion, Aggregators.mean)])
