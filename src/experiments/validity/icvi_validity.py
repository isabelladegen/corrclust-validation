from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import pandas as pd

from src.experiments.validity.distance_measure_validity import CriteriaRule
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import Aggregators
from src.utils.load_synthetic_data import SyntheticDataType


class ICVITier(str, Enum):
    excellent = "excellent"
    poor = "poor"
    no_structure = "no_structure"
    between_poor_and_excellent = "between_poor_and_excellent"


@dataclass
class ICVIValCriteria:
    criterion: str = "Jaccard_corr"
    structural_1: str = "gt"
    structural_2: str = "worst"
    structural_3: str = "gt_k_invariance"
    structural_4: str = "gt_m_invariance"

    _thresholds: ClassVar[dict] = {criterion: 0.5}

    _tier_thresholds: ClassVar[dict] = {
        ClusteringQualityMeasures.silhouette_score: {
            ICVITier.excellent: 0.9, ICVITier.poor: 0.26, ICVITier.no_structure: 0},
        ClusteringQualityMeasures.dbi: {
            ICVITier.excellent: 0.15, ICVITier.poor: 1, ICVITier.no_structure: 2},
        ClusteringQualityMeasures.vrc: {
            ICVITier.excellent: 350, ICVITier.poor: 3.5, ICVITier.no_structure: 1},
        ClusteringQualityMeasures.pmb: {
            ICVITier.excellent: 130, ICVITier.poor: 0.05, ICVITier.no_structure: 0.02},
    }

    _normal_tier_for_test: ClassVar[dict] = {
        structural_1: ICVITier.excellent, structural_2: ICVITier.poor,
        structural_3: ICVITier.excellent, structural_4: ICVITier.excellent,
    }
    _ds_tier_for_test: ClassVar[dict] = {
        structural_1: ICVITier.between_poor_and_excellent, structural_2: ICVITier.poor,
        structural_3: ICVITier.between_poor_and_excellent, structural_4: ICVITier.between_poor_and_excellent,
    }

    _display_names: ClassVar[dict] = {
        criterion: "Criterion", structural_1: "Structural 1", structural_2: "Structural 2",
        structural_3: "Structural 3", structural_4: "Structural 4",
    }

    @staticmethod
    def criterion_threshold() -> float:
        return ICVIValCriteria._thresholds[ICVIValCriteria.criterion]

    @staticmethod
    def _tier_for(criteria: str, data_type: str) -> "ICVITier":
        if data_type == SyntheticDataType.raw:
            return ICVITier.no_structure
        elif data_type == SyntheticDataType.rs_1min:
            return ICVIValCriteria._ds_tier_for_test[criteria]
        return ICVIValCriteria._normal_tier_for_test[criteria]

    @staticmethod
    def _is_floor(tier: "ICVITier", measure: str) -> bool:
        return (tier == ICVITier.excellent) == ClusteringQualityMeasures.is_higher_better(measure)

    @staticmethod
    def passes(criteria: str, value: float, measure: str = None,
               data_type: str = SyntheticDataType.normal_correlated) -> bool:
        if criteria == ICVIValCriteria.criterion:
            return abs(value) > ICVIValCriteria._thresholds[criteria]
        tier = ICVIValCriteria._tier_for(criteria, data_type)
        return ICVIValCriteria._passes_tier(tier, value, measure)

    @staticmethod
    def _passes_tier(tier: "ICVITier", value: float, measure: str) -> bool:
        if tier == ICVITier.between_poor_and_excellent:
            excellent_t = ICVIValCriteria._tier_thresholds[measure][ICVITier.excellent]
            poor_t = ICVIValCriteria._tier_thresholds[measure][ICVITier.poor]
            lo, hi = sorted([excellent_t, poor_t])
            return lo < value < hi
        t = ICVIValCriteria._tier_thresholds[measure][tier]
        return (value > t) if ICVIValCriteria._is_floor(tier, measure) else (value < t)

    @staticmethod
    def comparison_for(criteria: str, measure: str,
                       data_type: str = SyntheticDataType.normal_correlated) -> tuple:
        """('>' or '<', threshold) describing this criteria's pass condition for `measure`.
        Raises for between_poor_and_excellent, which is a band, not a single-sided threshold."""
        tier = ICVIValCriteria._tier_for(criteria, data_type)
        if tier == ICVITier.between_poor_and_excellent:
            raise ValueError(f"{criteria} is a two-sided band, not a single threshold")
        symbol = '>' if ICVIValCriteria._is_floor(tier, measure) else '<'
        return symbol, ICVIValCriteria._tier_thresholds[measure][tier]

    @staticmethod
    def footnote_text_for(criteria: str, measures: list) -> str:
        parts = []
        for m in measures:
            symbol, threshold = ICVIValCriteria.comparison_for(criteria, m)
            parts.append(f'{ClusteringQualityMeasures.get_display_name_for_measure(m)} {symbol} {threshold}')
        return ', '.join(parts)

    @staticmethod
    def display_name_for(criteria: str) -> str:
        return ICVIValCriteria._display_names[criteria]


@dataclass
class CriteriaForVariant:
    normal_100: str = 'normal_100'
    normal_70: str = 'normal_70'
    normal_10: str = 'normal_10'
    non_normal_100: str = 'non_normal_100'
    non_normal_10: str = 'non_normal_10'
    raw_100: str = 'raw_100'
    ds_100: str = 'downsampled_100'

    _all_criteria = [ICVIValCriteria.criterion, ICVIValCriteria.structural_1, ICVIValCriteria.structural_2,
                     ICVIValCriteria.structural_3,
                     ICVIValCriteria.structural_4]

    _tests_for_variant: ClassVar[dict] = {
        normal_100: _all_criteria,
        normal_70: _all_criteria,
        normal_10: _all_criteria,
        non_normal_100: _all_criteria,
        non_normal_10: _all_criteria,
        raw_100: [ICVIValCriteria.criterion, ICVIValCriteria.structural_1],
        ds_100: [ICVIValCriteria.criterion, ICVIValCriteria.structural_1],
    }

    _display_names: ClassVar[dict] = {
        normal_100: 'Normal 100\\%',
        normal_70: 'Normal 70\\%',
        normal_10: 'Normal 10\\%',
        non_normal_100: 'Non-normal 100\\%',
        non_normal_10: 'Non-normal 10\\%',
    }

    @staticmethod
    def criteria_for(variant: str) -> list:
        return CriteriaForVariant._tests_for_variant[variant]

    @staticmethod
    def display_name_for(variant: str) -> str:
        return CriteriaForVariant._display_names[variant]


@dataclass
class ICVIValidityResultColumns:
    structural: str = "Structural"
    criterion: str = "Criterion"
    discriminant_raw: str = "Discriminant_raw"
    discriminant_ds: str = "Discriminant_ds"
    discriminant: str = "Discriminant"
    external_normal_70: str = "External_normal_70"
    external_normal_10: str = "External_normal_10"
    external_non_normal_100: str = "External_non_normal_100"
    external_non_normal_10: str = "External_non_normal_10"
    external: str = "External"
    overall: str = "Overall"


class ICVIValidity:
    """
    Criterion: normal_100 only.
    Structural 1-4: normal_100 only.
    Discriminant: raw_100 and downsampled_100. Which criteria apply to each variant comes from
    CriteriaForVariant — currently only {criterion, structural_1}
    External: normal_70, normal_10, non_normal_100, non_normal_10. Each re-runs Criterion +
    Structural 1-4 (the full CriteriaForVariant set) at that condition.
    """

    def __init__(self, internal_measures: list, normal_100: dict, normal_70: dict, normal_10: dict,
                 non_normal_100: dict, non_normal_10: dict, raw_100: dict, downsampled_100: dict, round_to: int = 2):
        """Each data variant is a dict, keyed by ICVIValCriteria fields based on which criteria applies to the variant:
                 {structural_1: gt_df, structural_2: worst_df, structural_3: [gt_k11_df, gt_k6_df],
                 structural_4: [gt_m50_df, gt_m25_df], criterion: criterion_df}."""
        self._internal_measures = internal_measures
        self._normal_100 = normal_100
        self._normal_70 = normal_70
        self._normal_10 = normal_10
        self._non_normal_100 = non_normal_100
        self._non_normal_10 = non_normal_10
        self._raw_100 = raw_100
        self._downsampled_100 = downsampled_100
        self._round_to = round_to

    def structural_validity(self) -> pd.DataFrame:
        return self._stacked(lambda m: self._structural_detail_for_measure(
            m, CriteriaForVariant.normal_100, self._normal_100))

    def criterion_validity(self) -> pd.DataFrame:
        return self._stacked(self._criterion_detail_for_measure)

    def discriminant_validity_details(self) -> pd.DataFrame:
        return self._stacked(self._discriminant_detail_for_measure)

    def discriminant_validity(self) -> pd.DataFrame:
        d = self.discriminant_validity_details()
        return d[[ICVIValidityResultColumns.discriminant_raw, ICVIValidityResultColumns.discriminant_ds,
                  ICVIValidityResultColumns.discriminant]]

    def external_validity_details(self) -> pd.DataFrame:
        return self._stacked(self._external_detail_for_measure)

    def external_validity(self) -> pd.DataFrame:
        d = self.external_validity_details()
        cols = [ICVIValidityResultColumns.external_normal_70, ICVIValidityResultColumns.external_normal_10,
                ICVIValidityResultColumns.external_non_normal_100, ICVIValidityResultColumns.external_non_normal_10,
                ICVIValidityResultColumns.external]
        return d[cols]

    def overall_validity(self) -> pd.DataFrame:
        """Two-level (measure, distance_measure) index, mirroring DM's overall_validity_results.csv:
        Structural's full per-criterion mean/pass detail baked in, Criterion mean+pass,
        Discriminant and External collapsed to their combined booleans (full detail lives in
        discriminant_validity_details()/external_validity_details()), then Overall = AND of all
        four. No Convergent — left out deliberately, it only becomes apparent in hindsight
        whether 2+ ICVIs survive everything else."""
        overall = pd.concat([self.structural_validity(), self.criterion_validity(),
                             self.discriminant_validity(), self.external_validity()], axis=1)
        overall[ICVIValidityResultColumns.overall] = (
                overall[ICVIValidityResultColumns.structural] & overall[ICVIValidityResultColumns.criterion]
                & overall[ICVIValidityResultColumns.discriminant] & overall[ICVIValidityResultColumns.external])
        return overall

    def mean_sd_valid_summary_table(self, df: pd.DataFrame, criteria: str,
                                    data_type: str = SyntheticDataType.normal_correlated) -> pd.DataFrame:
        """The paper's display table: distance_measure rows, one column per internal measure,
        'mean (SD sd)*' with a star where it passes. Different axis orientation from the audit
        tables above (which are per-measure, two-level indexed) — this is for publication, those
        are for auditing every (measure, distance_measure) combination independently."""
        result = pd.DataFrame(index=df.index)
        for measure in self._internal_measures:
            mean = df[(measure, Aggregators.mean)]
            sd = df[(measure, Aggregators.std)]
            star = mean.apply(lambda x, m=measure: ICVIValCriteria.passes(criteria, x, m, data_type)).map(
                {True: "*", False: ""})
            mean_str = mean.map(lambda x: f"{x:.{self._round_to}f}")
            sd_str = sd.map(lambda x: f"{x:.{self._round_to}f}")
            result[ClusteringQualityMeasures.get_display_name_for_measure(measure)] = (
                    mean_str + " (SD " + sd_str + ")" + star)
        return result

    def mean_sd_valid_summary_table_for_subcriteria(self, dfs: list, criteria: str, sub_labels: tuple,
                                                    measures: list = None,
                                                    data_type: str = SyntheticDataType.normal_correlated) -> pd.DataFrame:
        """dfs: [df_for_first_sub_condition, df_for_second_sub_condition], e.g. [k11_df, k6_df].
        Star logic is identical to mean_sd_valid_summary_table, applied independently per cell."""
        measures = measures if measures is not None else self._internal_measures
        result = pd.DataFrame(index=dfs[0].index)
        for measure in measures:
            display = ClusteringQualityMeasures.get_display_name_for_measure(measure)
            for df, lbl in zip(dfs, sub_labels):
                mean = df[(measure, Aggregators.mean)]
                sd = df[(measure, Aggregators.std)]
                star = mean.apply(lambda x, m=measure: ICVIValCriteria.passes(criteria, x, m, data_type)).map(
                    {True: "*", False: ""})
                mean_str = mean.map(lambda x: f"{x:.{self._round_to}f}")
                sd_str = sd.map(lambda x: f"{x:.{self._round_to}f}")
                result[(display, lbl)] = mean_str + " (SD " + sd_str + ")" + star
        result.columns = pd.MultiIndex.from_tuples(result.columns)
        return result

    def _stacked(self, per_measure_fn) -> pd.DataFrame:
        """Runs per_measure_fn(measure) for each internal measure, each returning a df indexed
        by distance_measure, and stacks into one (measure, distance_measure) two-level index."""
        return pd.concat({m: per_measure_fn(m) for m in self._internal_measures},
                         names=['measure', 'distance_measure'])

    def _structural_detail_for_measure(self, measure: str, variant: str, cond: dict,
                                       data_type: str = SyntheticDataType.normal_correlated) -> pd.DataFrame:
        """Full per-criterion mean+pass detail for one measure, evaluated against whichever
        criteria CriteriaForVariant says apply to this variant. must_hold = structural_1/2,
        whichever are present. The rest (structural_3/4, when present) need at least one to
        pass — with none remaining (e.g. raw/downsampled), that's trivially satisfied, so the
        rule reduces to just checking structural_1."""
        structural_criteria = [c for c in CriteriaForVariant.criteria_for(variant) if c != ICVIValCriteria.criterion]
        must_hold = [c for c in structural_criteria
                     if c in (ICVIValCriteria.structural_1, ICVIValCriteria.structural_2)]
        remaining = [c for c in structural_criteria if c not in must_hold]
        rule = CriteriaRule(criteria=structural_criteria, must_hold=must_hold, minimum=1 if remaining else 0)

        result = None
        satisfied = {}
        for c in structural_criteria:
            short = ICVIValCriteria.display_name_for(c)
            if c in (ICVIValCriteria.structural_3, ICVIValCriteria.structural_4):
                sub_labels = ('K=11', 'K=6') if c == ICVIValCriteria.structural_3 else ('M=50', 'M=25')
                sub_passes = []
                for df, lbl in zip(cond[c], sub_labels):
                    mean = df[(measure, Aggregators.mean)]
                    passes = mean.apply(lambda x: ICVIValCriteria.passes(c, x, measure, data_type))
                    if result is None:
                        result = pd.DataFrame(index=mean.index)
                    result[f"{short} {lbl}_mean"] = mean
                    result[f"{short} {lbl}_pass"] = passes
                    sub_passes.append(passes)
                satisfied[c] = sub_passes[0] & sub_passes[1]
            else:
                mean = cond[c][(measure, Aggregators.mean)]
                passes = mean.apply(lambda x: ICVIValCriteria.passes(c, x, measure, data_type))
                if result is None:
                    result = pd.DataFrame(index=mean.index)
                result[f"{short}_mean"] = mean
                result[f"{short}_pass"] = passes
                satisfied[c] = passes
        result[ICVIValidityResultColumns.structural] = rule.evaluate(satisfied)
        return result

    def _criterion_detail_for_measure(self, measure: str) -> pd.DataFrame:
        df = self._normal_100[ICVIValCriteria.criterion]
        mean = df[(measure, Aggregators.mean)]
        passes = mean.apply(lambda x: ICVIValCriteria.passes(ICVIValCriteria.criterion, x))
        result = pd.DataFrame(index=df.index)
        result[f"{ICVIValCriteria.display_name_for(ICVIValCriteria.criterion)}_mean"] = mean
        result[ICVIValidityResultColumns.criterion] = passes
        return result

    def _discriminant_detail_for_measure(self, measure: str) -> pd.DataFrame:
        raw = self._structural_detail_for_measure(measure, CriteriaForVariant.raw_100, self._raw_100,
                                                  data_type=SyntheticDataType.raw)
        ds = self._structural_detail_for_measure(measure, CriteriaForVariant.ds_100, self._downsampled_100,
                                                 data_type=SyntheticDataType.rs_1min)
        raw = raw.add_suffix("_raw").rename(
            columns={f"{ICVIValidityResultColumns.structural}_raw": ICVIValidityResultColumns.discriminant_raw})
        ds = ds.add_suffix("_ds").rename(
            columns={f"{ICVIValidityResultColumns.structural}_ds": ICVIValidityResultColumns.discriminant_ds})
        result = raw.join(ds)
        result[ICVIValidityResultColumns.discriminant] = (
                result[ICVIValidityResultColumns.discriminant_raw] & result[ICVIValidityResultColumns.discriminant_ds])
        return result

    def _external_detail_for_measure(self, measure: str) -> pd.DataFrame:
        """Full per-criterion mean+pass detail per external condition, prefixed with the
        condition label (e.g. External_normal_70_Structural 1_mean), so it's visible exactly
        which test broke and at what value for each condition — not just the collapsed bool."""
        conditions = {
            ICVIValidityResultColumns.external_normal_70: self._normal_70,
            ICVIValidityResultColumns.external_normal_10: self._normal_10,
            ICVIValidityResultColumns.external_non_normal_100: self._non_normal_100,
            ICVIValidityResultColumns.external_non_normal_10: self._non_normal_10,
        }
        result = None
        for label, cond in conditions.items():
            structural_detail = self._structural_detail_for_measure(measure, CriteriaForVariant.normal_100, cond)
            structural_detail = structural_detail.add_prefix(f"{label}_").rename(
                columns={f"{label}_{ICVIValidityResultColumns.structural}": label})

            criterion_mean = cond[ICVIValCriteria.criterion][(measure, Aggregators.mean)]
            criterion_pass = criterion_mean.apply(lambda x: ICVIValCriteria.passes(ICVIValCriteria.criterion, x))
            criterion_name = ICVIValCriteria.display_name_for(ICVIValCriteria.criterion)
            structural_detail[f"{label}_{criterion_name}_mean"] = criterion_mean
            structural_detail[f"{label}_{criterion_name}_pass"] = criterion_pass
            structural_detail[label] = structural_detail[label] & criterion_pass

            result = structural_detail if result is None else result.join(structural_detail)
        result[ICVIValidityResultColumns.external] = result[list(conditions.keys())].all(axis=1)
        return result
