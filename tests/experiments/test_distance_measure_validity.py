import pandas as pd
from hamcrest import *

from src.evaluation.distance_metric_evaluation import EvaluationCriteria, criteria_short_names
from src.experiments.validity.distance_measure_validity import DistanceMeasureValidity, ValidityResultColumns
from src.utils.configurations import Aggregators

# Comfortably passes every structural criterion and predictive criterion validity.
# Used to fill constructor slots that a given test isn't exercising.
_VALID_ROW = {
    EvaluationCriteria.inter_i: 0.05,
    EvaluationCriteria.inter_ii: 1.0,
    EvaluationCriteria.inter_iii: 0.9,
    EvaluationCriteria.disc_i: 4.5,
    EvaluationCriteria.disc_ii: 2.0,
    EvaluationCriteria.disc_iii: 0.99,
}
_UNUSED_TABLE = pd.DataFrame([_VALID_ROW], index=["DM 1"])
_UNUSED_TABLE.columns = pd.MultiIndex.from_product([_UNUSED_TABLE.columns, [Aggregators.mean]])


# helper function to create validity class with minimal data given to test conditions
def _create_validity_class(normal_100=None, normal_70=None, normal_10=None,
                           non_normal_100=None, non_normal_10=None, raw_100=None, downsampled_100=None):
    return DistanceMeasureValidity(
        normal_100=normal_100 if normal_100 is not None else _UNUSED_TABLE,
        normal_70=normal_70 if normal_70 is not None else _UNUSED_TABLE,
        normal_10=normal_10 if normal_10 is not None else _UNUSED_TABLE,
        non_normal_100=non_normal_100 if non_normal_100 is not None else _UNUSED_TABLE,
        non_normal_10=non_normal_10 if non_normal_10 is not None else _UNUSED_TABLE,
        raw_100=raw_100 if raw_100 is not None else _UNUSED_TABLE,
        downsampled_100=downsampled_100 if downsampled_100 is not None else _UNUSED_TABLE,
    )


# ---------------------------------------------------------------------------
# Structural validity (Table 1, evaluations 1-5), computed from normal_100
# ---------------------------------------------------------------------------

def test_identity_preservation_valid_at_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_i: 0.1}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.inter_i)], is_(True))


def test_identity_preservation_invalid_just_past_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_i: 0.11}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.inter_i)], is_(False))


def test_ordinal_structure_preservation_valid_when_all_subjects_agree():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_ii: 1.0}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.inter_ii)], is_(True))


def test_ordinal_structure_preservation_invalid_when_not_all_subjects_agree():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_ii: 0.99}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.inter_ii)], is_(False))


def test_average_sensitivity_valid_just_above_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_iii: 0.71}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.inter_iii)], is_(True))


def test_average_sensitivity_invalid_at_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_iii: 0.7}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.inter_iii)], is_(False))


def test_discrimination_valid_just_above_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.disc_i: 4.01}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.disc_i)], is_(True))


def test_discrimination_invalid_at_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.disc_i: 4.0}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.disc_i)], is_(False))


def test_consistency_valid_just_below_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.disc_ii: 2.99}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.disc_ii)], is_(True))


def test_consistency_invalid_at_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.disc_ii: 3.0}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.result_for(EvaluationCriteria.disc_ii)], is_(False))


def test_structural_validity_valid_when_exactly_four_of_five_pass():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_i: 0.2}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.structural], is_(True))


def test_structural_validity_invalid_when_only_three_of_five_pass():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.inter_i: 0.2,
                                EvaluationCriteria.inter_iii: 0.3}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).structural_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.structural], is_(False))


# ---------------------------------------------------------------------------
# Predictive criterion validity (macro F1), computed from normal_100
# ---------------------------------------------------------------------------

def test_predictive_criterion_validity_valid_above_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.disc_iii: 0.99}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).criterion_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.criterion], is_(True))


def test_predictive_criterion_validity_invalid_at_threshold():
    normal_100 = pd.DataFrame([{**_VALID_ROW, EvaluationCriteria.disc_iii: 0.98}], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100).criterion_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.criterion], is_(False))


# ---------------------------------------------------------------------------
# Convergent validity: population-level, needs >=2 distance measures in normal_100
# ---------------------------------------------------------------------------

def test_convergent_validity_valid_when_two_distance_measures_are_structurally_valid():
    normal_100 = pd.DataFrame([_VALID_ROW, _VALID_ROW], index=["DM 1", "DM 2"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    assert_that(_create_validity_class(normal_100=normal_100).convergent_validity(), is_(True))


def test_convergent_validity_invalid_when_only_one_distance_measure_is_structurally_valid():
    invalid_row = {**_VALID_ROW, EvaluationCriteria.inter_i: 0.2, EvaluationCriteria.inter_iii: 0.3}
    normal_100 = pd.DataFrame([_VALID_ROW, invalid_row], index=["DM 1", "DM 2"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    assert_that(_create_validity_class(normal_100=normal_100).convergent_validity(), is_(False))


# ---------------------------------------------------------------------------
# Discriminant validity, no canonical pattern in raw data (structural tests
# must FAIL on raw_100: valid measure passes at most 1 of 5)
# ---------------------------------------------------------------------------

_NO_PATTERN_ROW = {
    EvaluationCriteria.inter_i: 3.0,  # fails, far from 0
    EvaluationCriteria.inter_ii: 0.0,  # fails
    EvaluationCriteria.inter_iii: 0.1,  # fails
    EvaluationCriteria.disc_i: 2.0,  # fails
    EvaluationCriteria.disc_ii: 2.5,  # passes by coincidence, low variance without structure
    EvaluationCriteria.disc_iii: 0.0,
}

_REFERENCE_ROW = {
    EvaluationCriteria.inter_i: 0.05,
    EvaluationCriteria.inter_ii: 1.0,
    EvaluationCriteria.inter_iii: 0.9,
    EvaluationCriteria.disc_i: 4.5,
    EvaluationCriteria.disc_ii: 2.0,
    EvaluationCriteria.disc_iii: 0.99,
}

# worse on every degradation criterion, still individually valid
_ALL_WORSE_ROW = {
    EvaluationCriteria.inter_i: 0.08,
    EvaluationCriteria.inter_ii: 1.0,
    EvaluationCriteria.inter_iii: 0.75,
    EvaluationCriteria.disc_i: 4.2,
    EvaluationCriteria.disc_ii: 2.5,
    EvaluationCriteria.disc_iii: 0.985,
}

# worse on 4 of 5 (predictive criterion validity held at the reference value)
_FOUR_OF_FIVE_WORSE_ROW = {**_ALL_WORSE_ROW, EvaluationCriteria.disc_iii: 0.99}

# worse on only 3 of 5 (identity preservation also held at the reference value)
_THREE_OF_FIVE_WORSE_ROW = {**_FOUR_OF_FIVE_WORSE_ROW, EvaluationCriteria.inter_i: 0.05}


def test_discriminant_valid_when_all_five_degradation_criteria_worsen():
    normal_100 = pd.DataFrame([_REFERENCE_ROW], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    downsampled_100 = pd.DataFrame([_ALL_WORSE_ROW], index=["DM 1"])
    downsampled_100.columns = pd.MultiIndex.from_product([downsampled_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100, downsampled_100=downsampled_100).discriminant_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.discriminant_degradation], is_(True))


def test_discriminant_valid_when_exactly_four_of_five_degradation_criteria_worsen():
    normal_100 = pd.DataFrame([_REFERENCE_ROW], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    downsampled_100 = pd.DataFrame([_FOUR_OF_FIVE_WORSE_ROW], index=["DM 1"])
    downsampled_100.columns = pd.MultiIndex.from_product([downsampled_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100, downsampled_100=downsampled_100).discriminant_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.discriminant_degradation], is_(True))


def test_discriminant_invalid_when_only_three_of_five_degradation_criteria_worsen():
    normal_100 = pd.DataFrame([_REFERENCE_ROW], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    downsampled_100 = pd.DataFrame([_THREE_OF_FIVE_WORSE_ROW], index=["DM 1"])
    downsampled_100.columns = pd.MultiIndex.from_product([downsampled_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100, downsampled_100=downsampled_100).discriminant_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.discriminant_degradation], is_(False))


def test_discriminant_invalid_when_degradation_condition_fails_but_no_pattern_condition_holds():
    normal_100 = pd.DataFrame([_REFERENCE_ROW], index=["DM 1"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    raw_100 = pd.DataFrame([_NO_PATTERN_ROW], index=["DM 1"])
    downsampled_100 = pd.DataFrame([_THREE_OF_FIVE_WORSE_ROW], index=["DM 1"])  # only 3 of 5 worsen
    raw_100.columns = pd.MultiIndex.from_product([raw_100.columns, [Aggregators.mean]])
    downsampled_100.columns = pd.MultiIndex.from_product([downsampled_100.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_100=normal_100, raw_100=raw_100,
                                    downsampled_100=downsampled_100).discriminant_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.discriminant], is_(False))


# ---------------------------------------------------------------------------
# External validity: same structural rule, generalised across four conditions
# ---------------------------------------------------------------------------

_INVALID_ROW = {**_VALID_ROW, EvaluationCriteria.inter_i: 0.2, EvaluationCriteria.inter_iii: 0.3}  # 3 of 5


def test_external_validity_valid_when_all_four_conditions_hold():
    valid_table = pd.DataFrame([_VALID_ROW], index=["DM 1"])
    valid_table.columns = pd.MultiIndex.from_product([valid_table.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_70=valid_table, normal_10=valid_table,
                                    non_normal_100=valid_table, non_normal_10=valid_table).external_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.external], is_(True))


def test_external_validity_invalid_when_normal_70_condition_fails():
    valid_table = pd.DataFrame([_VALID_ROW], index=["DM 1"])
    valid_table.columns = pd.MultiIndex.from_product([valid_table.columns, [Aggregators.mean]])
    invalid_table = pd.DataFrame([_INVALID_ROW], index=["DM 1"])
    invalid_table.columns = pd.MultiIndex.from_product([invalid_table.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_70=invalid_table, normal_10=valid_table,
                                    non_normal_100=valid_table, non_normal_10=valid_table).external_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.external], is_(False))


def test_external_validity_invalid_when_normal_10_condition_fails():
    valid_table = pd.DataFrame([_VALID_ROW], index=["DM 1"])
    valid_table.columns = pd.MultiIndex.from_product([valid_table.columns, [Aggregators.mean]])
    invalid_table = pd.DataFrame([_INVALID_ROW], index=["DM 1"])
    invalid_table.columns = pd.MultiIndex.from_product([invalid_table.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_70=valid_table, normal_10=invalid_table,
                                    non_normal_100=valid_table, non_normal_10=valid_table).external_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.external], is_(False))


def test_external_validity_invalid_when_non_normal_100_condition_fails():
    valid_table = pd.DataFrame([_VALID_ROW], index=["DM 1"])
    invalid_table = pd.DataFrame([_INVALID_ROW], index=["DM 1"])
    valid_table.columns = pd.MultiIndex.from_product([valid_table.columns, [Aggregators.mean]])
    invalid_table.columns = pd.MultiIndex.from_product([invalid_table.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_70=valid_table, normal_10=valid_table,
                                    non_normal_100=invalid_table, non_normal_10=valid_table).external_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.external], is_(False))


def test_external_validity_invalid_when_non_normal_10_condition_fails():
    valid_table = pd.DataFrame([_VALID_ROW], index=["DM 1"])
    invalid_table = pd.DataFrame([_INVALID_ROW], index=["DM 1"])
    valid_table.columns = pd.MultiIndex.from_product([valid_table.columns, [Aggregators.mean]])
    invalid_table.columns = pd.MultiIndex.from_product([invalid_table.columns, [Aggregators.mean]])
    result = _create_validity_class(normal_70=valid_table, normal_10=valid_table,
                                    non_normal_100=valid_table, non_normal_10=invalid_table).external_validity()
    assert_that(result.loc["DM 1", ValidityResultColumns.external], is_(False))


# ---------------------------------------------------------------------------
# Overall validity: the AND of every dimension. Needs >=2 distance measures
# throughout, since Convergent is population-level, and every one of the 7
# tables needs the same index or the AND across dfs misaligns to NaN.
# ---------------------------------------------------------------------------

def test_overall_validity_valid_when_every_dimension_passes():
    normal_100 = pd.DataFrame([_VALID_ROW, _VALID_ROW], index=["DM 1", "DM 2"])
    raw_100 = pd.DataFrame([_NO_PATTERN_ROW, _NO_PATTERN_ROW], index=["DM 1", "DM 2"])
    downsampled_100 = pd.DataFrame([_ALL_WORSE_ROW, _ALL_WORSE_ROW], index=["DM 1", "DM 2"])
    external_table = pd.DataFrame([_VALID_ROW, _VALID_ROW], index=["DM 1", "DM 2"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    raw_100.columns = pd.MultiIndex.from_product([raw_100.columns, [Aggregators.mean]])
    downsampled_100.columns = pd.MultiIndex.from_product([downsampled_100.columns, [Aggregators.mean]])
    external_table.columns = pd.MultiIndex.from_product([external_table.columns, [Aggregators.mean]])

    result = _create_validity_class(normal_100=normal_100, raw_100=raw_100, downsampled_100=downsampled_100,
                                    normal_70=external_table, normal_10=external_table,
                                    non_normal_100=external_table, non_normal_10=external_table).overall_validity()

    assert_that(result.loc["DM 1", ValidityResultColumns.overall], is_(True))
    assert_that(result.loc["DM 2", ValidityResultColumns.overall], is_(True))


def test_overall_validity_invalid_for_the_one_distance_measure_that_fails_external_validity():
    normal_100 = pd.DataFrame([_VALID_ROW, _VALID_ROW], index=["DM 1", "DM 2"])
    raw_100 = pd.DataFrame([_NO_PATTERN_ROW, _NO_PATTERN_ROW], index=["DM 1", "DM 2"])
    downsampled_100 = pd.DataFrame([_ALL_WORSE_ROW, _ALL_WORSE_ROW], index=["DM 1", "DM 2"])
    external_table = pd.DataFrame([_VALID_ROW, _VALID_ROW], index=["DM 1", "DM 2"])
    normal_100.columns = pd.MultiIndex.from_product([normal_100.columns, [Aggregators.mean]])
    raw_100.columns = pd.MultiIndex.from_product([raw_100.columns, [Aggregators.mean]])
    downsampled_100.columns = pd.MultiIndex.from_product([downsampled_100.columns, [Aggregators.mean]])
    external_table.columns = pd.MultiIndex.from_product([external_table.columns, [Aggregators.mean]])

    # DM 1 fails structurally on normal_70 specifically, DM 2 is unaffected
    normal_70 = pd.DataFrame([_INVALID_ROW, _VALID_ROW], index=["DM 1", "DM 2"])
    normal_70.columns = pd.MultiIndex.from_product([normal_70.columns, [Aggregators.mean]])

    result = _create_validity_class(normal_100=normal_100, raw_100=raw_100, downsampled_100=downsampled_100,
                                    normal_70=normal_70, normal_10=external_table,
                                    non_normal_100=external_table, non_normal_10=external_table).overall_validity()

    assert_that(result.loc["DM 1", ValidityResultColumns.overall], is_(False))
    # Convergent is population-level (>=2 structurally valid DMs in normal_100 alone),
    # so DM 1's external failure doesn't drag DM 2 down with it
    assert_that(result.loc["DM 2", ValidityResultColumns.overall], is_(True))


def test_formatted_results_adds_star_when_criterion_passes_threshold():
    columns = pd.MultiIndex.from_tuples([
        (EvaluationCriteria.inter_i, Aggregators.mean),
        (EvaluationCriteria.inter_i, Aggregators.std),
    ])
    df = pd.DataFrame([[0.05, 0.01]], index=["DM 1"], columns=columns)

    result = _create_validity_class().mean_sd_valid_summary_table(df)

    assert_that(result.loc["DM 1", criteria_short_names[EvaluationCriteria.inter_i]], is_("0.05 (SD 0.01)*"))


def test_formatted_results_omits_star_when_criterion_fails_threshold():
    columns = pd.MultiIndex.from_tuples([
        (EvaluationCriteria.inter_i, Aggregators.mean),
        (EvaluationCriteria.inter_i, Aggregators.std),
    ])
    df = pd.DataFrame([[0.11, 0.01]], index=["DM 1"], columns=columns)

    result = _create_validity_class().mean_sd_valid_summary_table(df)

    assert_that(result.loc["DM 1", criteria_short_names[EvaluationCriteria.inter_i]], is_("0.11 (SD 0.01)"))
