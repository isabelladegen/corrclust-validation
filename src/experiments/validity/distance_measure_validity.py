from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from src.evaluation.distance_metric_evaluation import EvaluationCriteria, criteria_short_names
from src.utils.configurations import Aggregators
from src.utils.load_synthetic_data import SyntheticDataType


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
    EvaluationCriteria.scale_free_inter_i: -0.8,  # relates to AUC >=0.9 (2*AUC-1=delta)
    EvaluationCriteria.inter_ii: 1.0,
    EvaluationCriteria.inter_iii: 0.7,
    EvaluationCriteria.scale_free_inter_iii: 0.4,  # relates to AUC > 0.7 for being in the acceptable range
    EvaluationCriteria.disc_i: 4,
    EvaluationCriteria.disc_ii: 3,
    EvaluationCriteria.disc_iii: 0.9,
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
                                      EvaluationCriteria.disc_ii],
    discriminant_no_pattern_minimum=4,
    discriminant_degradation_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_i,
                                       EvaluationCriteria.disc_ii],
    discriminant_degradation_minimum=4,
    criterion_predictive=EvaluationCriteria.disc_iii,
)

# strictly must pass for rule 1-3, 1 and 3 now scale free with cliff's delta, 2 corrected for independence issue
REVIEWED_RULES = ValidityRule(
    structural_criteria=[EvaluationCriteria.scale_free_inter_i, EvaluationCriteria.inter_ii,
                         EvaluationCriteria.scale_free_inter_iii],
    structural_must_hold=[EvaluationCriteria.scale_free_inter_i, EvaluationCriteria.inter_ii,
                          EvaluationCriteria.scale_free_inter_iii],
    structural_minimum=0,
    discriminant_no_pattern_criteria=[EvaluationCriteria.scale_free_inter_i, EvaluationCriteria.inter_ii,
                                      EvaluationCriteria.scale_free_inter_iii],
    discriminant_no_pattern_must_hold=[EvaluationCriteria.scale_free_inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.scale_free_inter_iii],
    discriminant_no_pattern_minimum=0,
    discriminant_degradation_criteria=[EvaluationCriteria.scale_free_inter_i,
                                       EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.scale_free_inter_iii],
    discriminant_degradation_must_hold=[EvaluationCriteria.scale_free_inter_i, EvaluationCriteria.scale_free_inter_iii],
    # allowing the levels set still to pass structural 2
    discriminant_degradation_minimum=0,
    criterion_predictive=EvaluationCriteria.disc_iii,  # run for construct, discriminant and external!
)

# removing entropy based rules since not derivable from theory
STRICT_MUST_PASS_RULES = ValidityRule(
    structural_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii,
                         EvaluationCriteria.disc_i, EvaluationCriteria.disc_ii],
    structural_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii],
    structural_minimum=0,  # entropy can fail
    discriminant_no_pattern_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                      EvaluationCriteria.inter_iii, EvaluationCriteria.disc_i,
                                      EvaluationCriteria.disc_ii],
    discriminant_no_pattern_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii],
    discriminant_no_pattern_minimum=0,  # entropy can fail
    discriminant_degradation_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_i,
                                       EvaluationCriteria.disc_ii],
    discriminant_degradation_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_iii],
    # allwoing the levels set still to pass structural 2
    discriminant_degradation_minimum=0,  # entropy can fail
    criterion_predictive=EvaluationCriteria.disc_iii,
)

DROPPING_OVERALL_ENTROPY = ValidityRule(
    structural_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii, EvaluationCriteria.inter_iii,
                         EvaluationCriteria.disc_ii],
    structural_minimum=4,
    discriminant_no_pattern_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                      EvaluationCriteria.inter_iii, EvaluationCriteria.disc_ii],
    discriminant_no_pattern_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii],
    discriminant_no_pattern_minimum=0,  # level set entropy can pass by coincidence, everything else must fail
    discriminant_degradation_criteria=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                       EvaluationCriteria.inter_iii, EvaluationCriteria.disc_ii, ],
    discriminant_degradation_must_hold=[EvaluationCriteria.inter_i, EvaluationCriteria.inter_ii,
                                        EvaluationCriteria.inter_iii],
    discriminant_degradation_minimum=0,  # level set entropy can stay flat, everything else must worsen
    criterion_predictive=EvaluationCriteria.disc_iii,
)


@dataclass
class Comparison:
    """Single source of truth per EvaluationCriteria: the operator token. Predicate and
    lower_is_better derive from it. Tokens are plain strings, not LaTeX -- LaTeX rendering of a
    token happens only in the presentation layer (run_distance_measure_validity.py), this class
    has no business knowing about LaTeX."""
    le: ClassVar[str] = '<='
    lt: ClassVar[str] = '<'
    eq: ClassVar[str] = '='
    gt: ClassVar[str] = '>'
    ge: ClassVar[str] = '>='
    neq: ClassVar[str] = '!='

    _operator_fn: ClassVar[dict] = {
        le: lambda x, t: x <= t, lt: lambda x, t: x < t,
        eq: lambda x, t: x == t, gt: lambda x, t: x > t,
    }
    _inverted: ClassVar[dict] = {le: gt, lt: ge, eq: neq, gt: le}
    _lower_is_better: ClassVar[set] = {le, lt}

    @staticmethod
    def passes(operator: str, value: float, threshold: float) -> bool:
        return Comparison._operator_fn[operator](value, threshold)

    @staticmethod
    def lower_is_better(operator: str) -> bool:
        return operator in Comparison._lower_is_better

    @staticmethod
    def inverted(operator: str) -> str:
        return Comparison._inverted[operator]


class DistanceMeasureValidity:
    """Decides validity of distance measures per Table 1 (methodology sec 4.2.1). Constructed
    from mean-value tables calculated over all subjects of a data variant (index=distance_measure,
    columns=EvaluationCriteria) for each of the  7 data variants."""

    # criterion -> (passes predicate, lower_is_better) -- single source of truth for thresholds
    _rules = {
        EvaluationCriteria.inter_i: Comparison.le,
        EvaluationCriteria.scale_free_inter_i: Comparison.le,
        EvaluationCriteria.inter_ii: Comparison.eq,
        EvaluationCriteria.inter_iii: Comparison.gt,
        EvaluationCriteria.scale_free_inter_iii: Comparison.gt,
        EvaluationCriteria.disc_i: Comparison.gt,
        EvaluationCriteria.disc_ii: Comparison.lt,
        EvaluationCriteria.disc_iii: Comparison.gt,
    }

    absolute_value_criteria = [EvaluationCriteria.scale_free_inter_iii]

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
            ValidityResultColumns.criterion: self.passes(self._normal_100, cp),
        })

    def convergent_validity(self) -> bool:
        return bool(self.structural_validity()[ValidityResultColumns.structural].sum() >= 2)

    def discriminant_validity_details(self) -> pd.DataFrame:
        """Per-criterion detail for both discriminant checks, now built by calling
        _structural_result with the matching data_type instead of hand-rolling satisfied dicts --
        same pass logic as everywhere else in the class."""
        cp = self._validity_rule.criterion_predictive
        cp_short = criteria_short_names[cp]

        raw = self._structural_result(self._raw_100, self._validity_rule.no_pattern,
                                      data_type=SyntheticDataType.raw)
        raw = raw.rename(columns=self._discriminant_rename_map(
            self._validity_rule.no_pattern.criteria, 'raw', ValidityResultColumns.discriminant_no_pattern))
        cp_no_pattern = self.passes(self._raw_100, cp, data_type=SyntheticDataType.raw)
        raw[f"{cp_short}_raw_mean"] = self._raw_100[(cp, Aggregators.mean)]
        raw[f"{cp_short}_raw_pass"] = cp_no_pattern
        raw[ValidityResultColumns.discriminant_no_pattern] = (
                raw[ValidityResultColumns.discriminant_no_pattern] & cp_no_pattern)

        ds = self._structural_result(self._downsampled_100, self._validity_rule.degradation,
                                     data_type=SyntheticDataType.rs_1min, reference_df=self._normal_100)
        ds = ds.rename(columns=self._discriminant_rename_map(
            self._validity_rule.degradation.criteria, 'ds', ValidityResultColumns.discriminant_degradation))
        cp_degradation = self.passes(self._downsampled_100, cp, data_type=SyntheticDataType.rs_1min,
                                     reference_df=self._normal_100)
        ds[f"{cp_short}_ds_mean"] = self._downsampled_100[(cp, Aggregators.mean)]
        ds[f"{cp_short}_ds_pass"] = cp_degradation
        ds[ValidityResultColumns.discriminant_degradation] = (
                ds[ValidityResultColumns.discriminant_degradation] & cp_degradation)

        result = raw.join(ds)
        result[ValidityResultColumns.discriminant] = (
                result[ValidityResultColumns.discriminant_no_pattern]
                & result[ValidityResultColumns.discriminant_degradation])
        return result

    @staticmethod
    def _discriminant_rename_map(criteria: list, suffix: str, structural_column_name: str) -> dict:
        """Renames _structural_result's generic {short}_mean/{short}_pass/Structural columns to
        the discriminant table's {short}_raw_mean / {short}_ds_mean style, preserving the existing
        CSV column names exactly."""
        rename_map = {ValidityResultColumns.structural: structural_column_name}
        for criterion in criteria:
            short = criteria_short_names[criterion]
            rename_map[ValidityResultColumns.mean_value_for(criterion)] = f"{short}_{suffix}_mean"
            rename_map[ValidityResultColumns.result_for(criterion)] = f"{short}_{suffix}_pass"
        return rename_map

    def discriminant_validity(self) -> pd.DataFrame:
        details = self.discriminant_validity_details()
        cols = [ValidityResultColumns.discriminant_no_pattern, ValidityResultColumns.discriminant_degradation,
                ValidityResultColumns.discriminant]
        return details[cols]

    def external_validity_details(self) -> pd.DataFrame:
        """Per-criterion structural detail for each of the four external conditions (same structural
        rule reused as-is, see class docstring). Ends with External, not Overall."""
        result = pd.DataFrame(index=self._normal_100.index)
        cp = self._validity_rule.criterion_predictive

        conditions = {
            ValidityResultColumns.external_normal_70: self._normal_70,
            ValidityResultColumns.external_normal_10: self._normal_10,
            ValidityResultColumns.external_non_normal_100: self._non_normal_100,
            ValidityResultColumns.external_non_normal_10: self._non_normal_10,
        }
        for label, df in conditions.items():
            detail = self._structural_result(df).add_prefix(f"{label}_")
            detail = detail.rename(columns={f"{label}_{ValidityResultColumns.structural}": label})
            cp_pass = self.passes(df, cp)
            detail[f"{label}_{ValidityResultColumns.mean_value_for(cp)}"] = df[(cp, Aggregators.mean)]
            detail[f"{label}_{ValidityResultColumns.result_for(cp)}"] = cp_pass
            detail[label] = detail[label] & cp_pass
            result = result.join(detail)
        result[ValidityResultColumns.external] = result[list(conditions.keys())].all(axis=1)
        return result

    def external_validity(self) -> pd.DataFrame:
        details = self.external_validity_details()
        cols = [ValidityResultColumns.external_normal_70, ValidityResultColumns.external_normal_10,
                ValidityResultColumns.external_non_normal_100, ValidityResultColumns.external_non_normal_10,
                ValidityResultColumns.external]
        return details[cols]

    def overall_validity(self) -> pd.DataFrame:
        overall = pd.concat([self.structural_validity(), self.criterion_validity(),
                             self.discriminant_validity(), self.external_validity()], axis=1)
        overall[ValidityResultColumns.convergent] = self.convergent_validity()
        overall[ValidityResultColumns.overall] = (
                overall[ValidityResultColumns.structural] & overall[ValidityResultColumns.criterion] & overall[
            ValidityResultColumns.convergent]
                & overall[ValidityResultColumns.discriminant] & overall[ValidityResultColumns.external])
        return overall

    def default_criteria_order(self) -> list:
        """Structural criteria (Table 1 order) followed by criterion_predictive -- the columns
        mean_sd_valid_summary_table and the LaTeX tables use by default."""
        return self._validity_rule.structural.criteria + [self._validity_rule.criterion_predictive]

    def mean_sd_valid_summary_table(self, df: pd.DataFrame, criteria: list = None, data_type: str = None,
                                    reference_df: pd.DataFrame = None) -> pd.DataFrame:
        """Validity summary table for one data variant: 'mean (SD sd)*' per distance measure per
        criterion, star = passes its threshold. Assumes mean/sd are already rounded upstream.
        Self-contained per criterion, so works on any of the 7 condition tables without
        needing to know which one it is. Criteria without a rule (e.g. stability) are skipped."""
        criteria = criteria if criteria is not None else df.columns.get_level_values(0).unique()
        result = pd.DataFrame(index=df.index)
        for criterion in criteria:
            if criterion not in self._rules:
                continue
            mean = df[(criterion, Aggregators.mean)]
            sd = df[(criterion, Aggregators.std)]
            star = self.passes(df, criterion, data_type, reference_df).map({True: "*", False: ""})
            column = criteria_short_names[criterion]
            result[column] = mean.astype(str) + " (SD " + sd.astype(str) + ")" + star
        return result

    def _structural_result(self, df: pd.DataFrame, rule: CriteriaRule = None,
                           data_type: str = None,
                           reference_df: pd.DataFrame = None) -> pd.DataFrame:
        """Per-criterion mean+pass detail for `rule` (defaults to the validity rule's structural
        rule), pass/fail decided by _passes() under data_type. Used directly for construct/external
        validity (data_type=normal_correlated) and reused by discriminant_validity_details for the
        raw/downsampled conditions -- one implementation, not two."""
        rule = rule or self._validity_rule.structural
        result = pd.DataFrame(index=df.index)
        satisfied = {}
        for criterion in rule.criteria:
            passes = self.passes(df, criterion, data_type, reference_df)
            result[ValidityResultColumns.mean_value_for(criterion)] = df[(criterion, Aggregators.mean)]
            result[ValidityResultColumns.result_for(criterion)] = passes
            satisfied[criterion] = passes
        result[ValidityResultColumns.structural] = rule.evaluate(satisfied)
        return result

    def passes(self, df: pd.DataFrame, criterion: str, data_type: str = None,
               reference_df: pd.DataFrame = None) -> pd.Series:
        """Pass/fail for one criterion. data_type only changes anything for the two conditions
        where 'valid' doesn't mean 'meets its own threshold': SyntheticDataType.raw (correctly
        invalid -- no canonical pattern to detect) and SyntheticDataType.rs_1min/downsampled
        (worse than reference_df, i.e. normal_100). Every other value, including the default
        None and non_normal_correlated, is the plain threshold check -- there was never a branch
        for normal_correlated vs non_normal_correlated, they were always the same case. Sparsity
        (70%/10%) never reaches this method: it only changes which df was computed from, not how
        that df's values get judged."""
        operator = self._rules[criterion]
        threshold = DM_THRESHOLDS[criterion]
        take_abs = criterion in self.absolute_value_criteria
        check = (lambda x: Comparison.passes(operator, abs(x), threshold)) if take_abs \
            else (lambda x: Comparison.passes(operator, x, threshold))

        if data_type == SyntheticDataType.raw:
            return ~df[(criterion, Aggregators.mean)].apply(check)
        if data_type == SyntheticDataType.rs_1min:
            if reference_df is None:
                raise ValueError("reference_df is required when data_type is rs_1min")
            return self._is_worse(df, reference_df, criterion)
        return df[(criterion, Aggregators.mean)].apply(check)

    def _is_worse(self, df: pd.DataFrame, reference_df: pd.DataFrame, criterion: str) -> pd.Series:
        if criterion in self.absolute_value_criteria:
            return df[(criterion, Aggregators.mean)].abs() < reference_df[(criterion, Aggregators.mean)].abs()
        lower_is_better = Comparison.lower_is_better(self._rules[criterion])
        return (df[(criterion, Aggregators.mean)] > reference_df[(criterion, Aggregators.mean)]) if lower_is_better \
            else (df[(criterion, Aggregators.mean)] < reference_df[(criterion, Aggregators.mean)])

    @staticmethod
    def comparison_symbol_for(criterion: str, data_type: str) -> str:
        operator = DistanceMeasureValidity._rules[criterion]
        return Comparison.inverted(operator) if data_type == SyntheticDataType.raw else operator
