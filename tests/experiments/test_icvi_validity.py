import pandas as pd
from hamcrest import *

from src.experiments.validity.icvi_validity import ICVIValidity, ICVIValCriteria, ICVIValidityResultColumns
from src.evaluation.describe_bad_partitions import DescribeBadPartCols
from src.utils.clustering_quality_measures import ClusteringQualityMeasures
from src.utils.configurations import Aggregators
from src.utils.load_synthetic_data import SyntheticDataType

INTERNAL_MEASURES = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi,
                     ClusteringQualityMeasures.vrc, ClusteringQualityMeasures.pmb]

# Comfortably clears "excellent" for every measure.
_EXCELLENT_ROW = {
    ClusteringQualityMeasures.silhouette_score: 0.95,
    ClusteringQualityMeasures.dbi: 0.05,
    ClusteringQualityMeasures.vrc: 400,
    ClusteringQualityMeasures.pmb: 150,
}

# Comfortably clears "poor" for every measure.
_POOR_ROW = {
    ClusteringQualityMeasures.silhouette_score: 0.2,
    ClusteringQualityMeasures.dbi: 1.5,
    ClusteringQualityMeasures.vrc: 2.0,
    ClusteringQualityMeasures.pmb: 0.03,
}

# Comfortably clears "no structure" for every measure.
_NO_STRUCTURE_ROW = {
    ClusteringQualityMeasures.silhouette_score: -0.1,
    ClusteringQualityMeasures.dbi: 2.5,
    ClusteringQualityMeasures.vrc: 0.5,
    ClusteringQualityMeasures.pmb: 0.01,
}

# Strictly between poor and excellent for every measure.
_BETWEEN_ROW = {
    ClusteringQualityMeasures.silhouette_score: 0.5,
    ClusteringQualityMeasures.dbi: 0.5,
    ClusteringQualityMeasures.vrc: 50,
    ClusteringQualityMeasures.pmb: 1.0,
}


def _stats_df(values: dict, index=("DM 1",)):
    """Mean-only MultiIndex stats df: index=distance_measure, columns=(measure, Aggregators.mean).
    Extra measure columns beyond what an ICVIValidity instance's internal_measures covers are
    harmless — they're simply never indexed. Missing a column the instance DOES cover raises
    KeyError, so a df built for a single-measure test must include that measure's key."""
    df = pd.DataFrame([values for _ in index], index=list(index))
    df.columns = pd.MultiIndex.from_product([df.columns, [Aggregators.mean]])
    return df


def _default_normal_condition():
    """Passes Criterion + Structural 1-4 comfortably. Used to fill any condition slot a test
    isn't exercising, so it never drags down the result being asserted on."""
    return {
        ICVIValCriteria.structural_1: _stats_df(_EXCELLENT_ROW),
        ICVIValCriteria.structural_2: _stats_df(_POOR_ROW),
        ICVIValCriteria.structural_3: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.structural_4: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.criterion: _stats_df({m: 0.6 for m in INTERNAL_MEASURES}),
    }


def _default_raw_condition():
    """Passes Discriminant's raw check (no_structure on every test) comfortably."""
    return {
        ICVIValCriteria.structural_1: _stats_df(_NO_STRUCTURE_ROW),
        ICVIValCriteria.structural_2: _stats_df(_NO_STRUCTURE_ROW),
        ICVIValCriteria.structural_3: [_stats_df(_NO_STRUCTURE_ROW), _stats_df(_NO_STRUCTURE_ROW)],
        ICVIValCriteria.structural_4: [_stats_df(_NO_STRUCTURE_ROW), _stats_df(_NO_STRUCTURE_ROW)],
        ICVIValCriteria.criterion: _stats_df({m: 0.1 for m in INTERNAL_MEASURES}),
    }


def _default_ds_condition():
    """Passes Discriminant's downsampled check (between poor/excellent, worst stays poor)."""
    return {
        ICVIValCriteria.structural_1: _stats_df(_BETWEEN_ROW),
        ICVIValCriteria.structural_2: _stats_df(_POOR_ROW),
        ICVIValCriteria.structural_3: [_stats_df(_BETWEEN_ROW), _stats_df(_BETWEEN_ROW)],
        ICVIValCriteria.structural_4: [_stats_df(_BETWEEN_ROW), _stats_df(_BETWEEN_ROW)],
        ICVIValCriteria.criterion: _stats_df({m: 0.3 for m in INTERNAL_MEASURES}),
    }


def _validity(measures=INTERNAL_MEASURES, normal_100=None, normal_70=None, normal_10=None,
              non_normal_100=None, non_normal_10=None, raw_100=None, downsampled_100=None):
    return ICVIValidity(
        internal_measures=measures,
        normal_100=normal_100 if normal_100 is not None else _default_normal_condition(),
        normal_70=normal_70 if normal_70 is not None else _default_normal_condition(),
        normal_10=normal_10 if normal_10 is not None else _default_normal_condition(),
        non_normal_100=non_normal_100 if non_normal_100 is not None else _default_normal_condition(),
        non_normal_10=non_normal_10 if non_normal_10 is not None else _default_normal_condition(),
        raw_100=raw_100 if raw_100 is not None else _default_raw_condition(),
        downsampled_100=downsampled_100 if downsampled_100 is not None else _default_ds_condition(),
    )


# ---------------------------------------------------------------------------
# select_ground_truth_row / select_worst_row: row selection on one subject's
# 67-partition file. Ground truth = no wrong clusters, no shifted segments.
# Worst = maximum n_wrong_clusters among partitions with no segments shifted.
# ---------------------------------------------------------------------------

def _partition_df(rows: list):
    return pd.DataFrame(rows, columns=[DescribeBadPartCols.n_wrong_clusters, DescribeBadPartCols.n_obs_shifted,
                                       ClusteringQualityMeasures.silhouette_score])


# ---------------------------------------------------------------------------
# ICVIValCriteria.passes: threshold boundaries, higher-is-better (SWC) and
# lower-is-better (DBI), for direction handling
# ---------------------------------------------------------------------------

def test_structural_1_excellent_valid_just_above_threshold_higher_is_better():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_1, 0.91,
                                       ClusteringQualityMeasures.silhouette_score), is_(True))


def test_structural_1_excellent_invalid_at_threshold_higher_is_better():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_1, 0.9,
                                       ClusteringQualityMeasures.silhouette_score), is_(False))


def test_structural_1_excellent_valid_just_below_threshold_lower_is_better():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_1, 0.14, ClusteringQualityMeasures.dbi), is_(True))


def test_structural_1_excellent_invalid_at_threshold_lower_is_better():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_1, 0.15, ClusteringQualityMeasures.dbi), is_(False))


def test_structural_2_poor_valid_just_below_threshold_higher_is_better():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_2, 0.25,
                                       ClusteringQualityMeasures.silhouette_score), is_(True))


def test_structural_2_poor_invalid_at_threshold_higher_is_better():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_2, 0.26,
                                       ClusteringQualityMeasures.silhouette_score), is_(False))


def test_all_tests_use_no_structure_tier_on_raw_data():
    # structural_1 normally wants excellent; raw forces no_structure for every test, including
    # structural_2, which normally wants poor
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_1, -0.1,
                                       ClusteringQualityMeasures.silhouette_score, data_type=SyntheticDataType.raw),
                is_(True))
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_2, -0.1,
                                       ClusteringQualityMeasures.silhouette_score, data_type=SyntheticDataType.raw),
                is_(True))
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_2, 0.2,
                                       ClusteringQualityMeasures.silhouette_score, data_type=SyntheticDataType.raw),
                is_(False))  # "poor" quality no longer passes once raw forces no_structure


def test_structural_1_between_poor_and_excellent_on_downsampled_data():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_1, 0.5,
                                       ClusteringQualityMeasures.silhouette_score, data_type=SyntheticDataType.rs_1min),
                is_(True))
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_1, 0.95,
                                       ClusteringQualityMeasures.silhouette_score, data_type=SyntheticDataType.rs_1min),
                is_(False))  # excellent itself is outside the band


def test_structural_2_stays_poor_tier_on_downsampled_data():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_2, 0.2,
                                       ClusteringQualityMeasures.silhouette_score, data_type=SyntheticDataType.rs_1min),
                is_(True))
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.structural_2, 0.5,
                                       ClusteringQualityMeasures.silhouette_score, data_type=SyntheticDataType.rs_1min),
                is_(False))


def test_criterion_ignores_measure_and_data_type():
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.criterion, 0.51), is_(True))
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.criterion, -0.51), is_(True))
    assert_that(ICVIValCriteria.passes(ICVIValCriteria.criterion, 0.5), is_(False))


# ---------------------------------------------------------------------------
# structural_validity: structural_3/4 need ALL sub-conditions (K=11 AND K=6;
# M=50 AND M=25); structural_1/2 are must-hold; 3/4 need only one of the two
# ---------------------------------------------------------------------------

def test_structural_3_requires_both_k_conditions():
    normal_100 = {
        ICVIValCriteria.structural_1: _stats_df(_EXCELLENT_ROW),
        ICVIValCriteria.structural_2: _stats_df(_POOR_ROW),
        ICVIValCriteria.structural_3: [_stats_df(_EXCELLENT_ROW), _stats_df(_POOR_ROW)],  # K=6 fails
        ICVIValCriteria.structural_4: [_stats_df(_POOR_ROW), _stats_df(_POOR_ROW)],  # both fail, can't rescue
        ICVIValCriteria.criterion: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.6}),
    }
    result = _validity(normal_100=normal_100).structural_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score,"DM 1" ][ICVIValidityResultColumns.structural], is_(False))


def test_structural_valid_when_1_and_2_hold_and_only_3_passes():
    normal_100 = {
        ICVIValCriteria.structural_1: _stats_df(_EXCELLENT_ROW),
        ICVIValCriteria.structural_2: _stats_df(_POOR_ROW),
        ICVIValCriteria.structural_3: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.structural_4: [_stats_df(_POOR_ROW), _stats_df(_POOR_ROW)],
        ICVIValCriteria.criterion: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.6}),
    }
    result = _validity(normal_100=normal_100).structural_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.structural], is_(True))


def test_structural_valid_when_1_and_2_hold_and_only_4_passes():
    normal_100 = {
        ICVIValCriteria.structural_1: _stats_df(_EXCELLENT_ROW),
        ICVIValCriteria.structural_2: _stats_df(_POOR_ROW),
        ICVIValCriteria.structural_3: [_stats_df(_POOR_ROW), _stats_df(_POOR_ROW)],
        ICVIValCriteria.structural_4: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.criterion: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.6}),
    }
    result = _validity(normal_100=normal_100).structural_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.structural], is_(True))


def test_structural_invalid_when_1_fails_despite_3_and_4_passing():
    normal_100 = {
        ICVIValCriteria.structural_1: _stats_df(_POOR_ROW),  # fails excellent
        ICVIValCriteria.structural_2: _stats_df(_POOR_ROW),
        ICVIValCriteria.structural_3: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.structural_4: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.criterion: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.6}),
    }
    result = _validity(normal_100=normal_100).structural_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.structural], is_(False))


def test_structural_invalid_when_2_fails_despite_3_and_4_passing():
    normal_100 = {
        ICVIValCriteria.structural_1: _stats_df(_EXCELLENT_ROW),
        ICVIValCriteria.structural_2: _stats_df(_EXCELLENT_ROW),  # doesn't degrade to poor
        ICVIValCriteria.structural_3: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.structural_4: [_stats_df(_EXCELLENT_ROW), _stats_df(_EXCELLENT_ROW)],
        ICVIValCriteria.criterion: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.6}),
    }
    result = _validity(normal_100=normal_100).structural_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.structural], is_(False))


def test_structural_result_is_independent_per_measure():
    two_measures = [ClusteringQualityMeasures.silhouette_score, ClusteringQualityMeasures.dbi]
    normal_100 = {
        ICVIValCriteria.structural_1: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.95,  # passes
                                                 ClusteringQualityMeasures.dbi: 0.5}),  # fails
        ICVIValCriteria.structural_2: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.2,
                                                 ClusteringQualityMeasures.dbi: 1.5}),
        ICVIValCriteria.structural_3: [_stats_df({ClusteringQualityMeasures.silhouette_score: 0.95,
                                                  ClusteringQualityMeasures.dbi: 0.05})] * 2,
        ICVIValCriteria.structural_4: [_stats_df({ClusteringQualityMeasures.silhouette_score: 0.95,
                                                  ClusteringQualityMeasures.dbi: 0.05})] * 2,
        ICVIValCriteria.criterion: _stats_df({ClusteringQualityMeasures.silhouette_score: 0.6,
                                              ClusteringQualityMeasures.dbi: 0.6}),
    }
    result = _validity(measures=two_measures, normal_100=normal_100).structural_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.structural], is_(True))
    assert_that(result.loc[ClusteringQualityMeasures.dbi, "DM 1"][ICVIValidityResultColumns.structural], is_(False))


# ---------------------------------------------------------------------------
# criterion_validity: reads normal_100 only
# ---------------------------------------------------------------------------

def test_criterion_valid_above_threshold():
    normal_100 = _default_normal_condition()
    normal_100[ICVIValCriteria.criterion] = _stats_df({ClusteringQualityMeasures.silhouette_score: 0.51})
    result = _validity(measures=[ClusteringQualityMeasures.silhouette_score],
                       normal_100=normal_100).criterion_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.criterion], is_(True))


def test_criterion_invalid_at_threshold():
    normal_100 = _default_normal_condition()
    normal_100[ICVIValCriteria.criterion] = _stats_df({ClusteringQualityMeasures.silhouette_score: 0.5})
    result = _validity(measures=[ClusteringQualityMeasures.silhouette_score],
                       normal_100=normal_100).criterion_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.criterion], is_(False))


# ---------------------------------------------------------------------------
# discriminant_validity: raw must show no_structure on every test, downsampled
# must land between poor and excellent (structural_2 stays at poor)
# ---------------------------------------------------------------------------

def test_discriminant_valid_when_raw_and_downsampled_both_behave():
    result = _validity().discriminant_validity()  # both defaults are built to pass
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.discriminant], is_(True))


def test_discriminant_invalid_when_raw_shows_structure():
    raw_100 = _default_raw_condition()
    raw_100[ICVIValCriteria.structural_1] = _stats_df(_EXCELLENT_ROW)  # raw shouldn't score excellent
    result = _validity(raw_100=raw_100).discriminant_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.discriminant], is_(False))


def test_discriminant_invalid_when_downsampled_stays_excellent_instead_of_degrading():
    downsampled_100 = _default_ds_condition()
    downsampled_100[ICVIValCriteria.structural_1] = _stats_df(_EXCELLENT_ROW)  # doesn't degrade
    result = _validity(downsampled_100=downsampled_100).discriminant_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.discriminant], is_(False))


def test_discriminant_invalid_when_downsampled_worst_improves_past_poor():
    downsampled_100 = _default_ds_condition()
    downsampled_100[ICVIValCriteria.structural_1] = _stats_df(_EXCELLENT_ROW)  # worst no longer poor
    result = _validity(downsampled_100=downsampled_100).discriminant_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.discriminant], is_(False))


# ---------------------------------------------------------------------------
# external_validity: Criterion + Structural 1-4, per condition, ANDed across
# normal_70, normal_10, non_normal_100, non_normal_10. No discriminant.
# ---------------------------------------------------------------------------

def test_external_valid_when_all_four_conditions_pass():
    result = _validity().external_validity()  # defaults all pass
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.external], is_(True))


def test_external_invalid_when_one_condition_fails_criterion():
    normal_10 = _default_normal_condition()
    normal_10[ICVIValCriteria.criterion] = _stats_df({m: 0.4 for m in INTERNAL_MEASURES})
    result = _validity(normal_10=normal_10).external_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.external], is_(False))


def test_external_invalid_when_one_condition_fails_structural():
    non_normal_100 = _default_normal_condition()
    non_normal_100[ICVIValCriteria.structural_2] = _stats_df(_EXCELLENT_ROW)  # doesn't degrade to poor
    result = _validity(non_normal_100=non_normal_100).external_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.external], is_(False))


def test_external_details_isolates_which_condition_failed():
    normal_70 = _default_normal_condition()
    normal_70[ICVIValCriteria.criterion] = _stats_df({m: 0.4 for m in INTERNAL_MEASURES})
    details = _validity(normal_70=normal_70).external_validity_details()
    assert_that(details.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.external + "_normal_70"], is_(False))
    assert_that(details.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.external + "_normal_10"], is_(True))


# ---------------------------------------------------------------------------
# overall_validity: AND of all four dimensions
# ---------------------------------------------------------------------------

def test_overall_valid_when_every_dimension_passes():
    result = _validity().overall_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.external], is_(True))


def test_overall_invalid_when_only_discriminant_fails():
    raw_100 = _default_raw_condition()
    raw_100[ICVIValCriteria.structural_1] = _stats_df(_EXCELLENT_ROW)
    result = _validity(raw_100=raw_100).overall_validity()
    assert_that(result.loc[ClusteringQualityMeasures.silhouette_score, "DM 1"][ICVIValidityResultColumns.overall], is_(False))


# ---------------------------------------------------------------------------
# mean_sd_valid_summary_table
# ---------------------------------------------------------------------------

def test_summary_table_adds_star_when_criterion_passes():
    columns = pd.MultiIndex.from_tuples([
        (ClusteringQualityMeasures.silhouette_score, Aggregators.mean),
        (ClusteringQualityMeasures.silhouette_score, Aggregators.std),
    ])
    df = pd.DataFrame([[0.95, 0.01]], index=["DM 1"], columns=columns)
    result = _validity(measures=[ClusteringQualityMeasures.silhouette_score]).mean_sd_valid_summary_table(
        df, ICVIValCriteria.structural_1)
    assert_that(result.loc["DM 1", ClusteringQualityMeasures.get_display_name_for_measure(
        ClusteringQualityMeasures.silhouette_score)], is_("0.95 (SD 0.01)*"))


def test_summary_table_omits_star_when_criterion_fails():
    columns = pd.MultiIndex.from_tuples([
        (ClusteringQualityMeasures.silhouette_score, Aggregators.mean),
        (ClusteringQualityMeasures.silhouette_score, Aggregators.std),
    ])
    df = pd.DataFrame([[0.5, 0.01]], index=["DM 1"], columns=columns)
    result = _validity(measures=[ClusteringQualityMeasures.silhouette_score]).mean_sd_valid_summary_table(
        df, ICVIValCriteria.structural_1)
    assert_that(result.loc["DM 1", ClusteringQualityMeasures.get_display_name_for_measure(
        ClusteringQualityMeasures.silhouette_score)], is_("0.5 (SD 0.01)"))
