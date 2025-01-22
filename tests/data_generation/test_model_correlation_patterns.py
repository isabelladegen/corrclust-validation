import numpy as np
import pytest
from hamcrest import *
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns, PatternCols, check_if_psd

patterns = ModelCorrelationPatterns()
df = patterns.df


def test_reads_the_model_pattern_csv_file_into_pandas_dataframe():
    assert_that(df.shape[0], is_(27))

    # test a few different patterns
    assert_that(df.iloc[0][PatternCols.id], is_(0))
    assert_that(df.iloc[0][PatternCols.canonical_patterns], instance_of(tuple))
    assert_that(df.iloc[0][PatternCols.canonical_patterns], is_((0, 0, 0)))
    assert_that(df.iloc[0][PatternCols.relaxed_patterns], instance_of(tuple))
    assert_that(df.iloc[0][PatternCols.relaxed_patterns], is_((0, 0, 0)))
    assert_that(df.iloc[0][PatternCols.reg_term], is_(0))
    assert_that(df.iloc[0][PatternCols.is_ideal], is_(True))

    assert_that(df.iloc[23][PatternCols.id], is_(23))
    assert_that(df.iloc[23][PatternCols.canonical_patterns], is_((-1, 1, -1)))
    assert_that(df.iloc[23][PatternCols.relaxed_patterns], is_((-1, 1, -1)))
    assert_that(df.iloc[23][PatternCols.reg_term], is_(0.0001))
    assert_that(df.iloc[23][PatternCols.is_ideal], is_(True))

    assert_that(df.iloc[26][PatternCols.id], is_(26))
    assert_that(df.iloc[26][PatternCols.canonical_patterns], is_((-1, -1, -1)))
    assert_that(np.isnan(df.iloc[26][PatternCols.relaxed_patterns]))
    assert_that(df.iloc[26][PatternCols.reg_term], is_(0.1))
    assert_that(df.iloc[26][PatternCols.is_ideal], is_(False))

    assert_that(df.iloc[4][PatternCols.id], is_(4))
    assert_that(df.iloc[4][PatternCols.canonical_patterns], is_((0, 1, 1)))
    assert_that(df.iloc[4][PatternCols.relaxed_patterns], is_((0.0, 0.71, 0.7)))
    assert_that(df.iloc[4][PatternCols.reg_term], is_(0.1))
    assert_that(df.iloc[4][PatternCols.is_ideal], is_(False))


def test_returns_ids_for_patterns_to_model_for_data_generation():
    ids = patterns.ids_of_patterns_to_model()

    # four of the patterns cannot be sensibly modeled
    assert_that(len(ids), is_(23))
    assert_that(14 in ids, is_(False))
    assert_that(16 in ids, is_(False))
    assert_that(22 in ids, is_(False))
    assert_that(26 in ids, is_(False))


def test_returns_dictionary_of_ideal_correlations_for_data_generation():
    correlations = patterns.canonical_patterns()

    assert_that(len(correlations), is_(23))
    assert_that(correlations[0], is_([0, 0, 0]))
    assert_that(correlations[4], is_([0, 1, 1]))
    assert_that(correlations[13], is_([1, 1, 1]))


def test_return_patterns_to_model_as_x_and_y():
    x, y = patterns.x_and_y_of_patterns_to_model()

    assert_that(x.shape[0], is_(23))
    assert_that(x.shape[1], is_(3))
    assert_that(y.shape[0], is_(23))

    assert_that(all(np.equal(x[0, :], np.array([0, 0, 0]))))
    assert_that(y[0], is_(0))

    # uses relaxed patterns
    assert_that(all(np.equal(x[4, :], np.array([0, 0.71, 0.7]))))
    assert_that(y[4], is_(4))

    assert_that(all(np.equal(x[22, :], np.array([-1, -1, 1]))))
    assert_that(y[22], is_(25))


def test_which_patterns_are_unadjusted_valid_correlation_matrices():
    result = patterns.perfect_valid_correlation_patterns()

    assert_that(len(result), is_(11))

    values = list(result.values())
    assert_that(values[0], contains_exactly(0, 0, 0))
    assert_that(values[1], contains_exactly(0, 0, 1))
    assert_that(values[2], contains_exactly(0, 0, -1))
    assert_that(values[3], contains_exactly(0, 1, 0))
    assert_that(values[4], contains_exactly(0, -1, 0))
    assert_that(values[5], contains_exactly(1, 0, 0))
    assert_that(values[6], contains_exactly(1, 1, 1))
    assert_that(values[7], contains_exactly(1, -1, -1))
    assert_that(values[8], contains_exactly(-1, 0, 0))
    assert_that(values[9], contains_exactly(-1, 1, -1))
    assert_that(values[10], contains_exactly(-1, -1, 1))


def test_relax_patterns_with_two_strong_correlations():
    result = patterns.calculate_relaxed_patterns()

    assert_that(len(result), is_(23))
    assert_that(result[4], contains_exactly(0, 0.71, 0.7))
    assert_that(result[5], contains_exactly(0, 0.71, -0.70))


def test_relaxed_patterns_calculation_matches_spreadsheet():
    calculated_relaxed_patterns = patterns.calculate_relaxed_patterns()
    relaxed_patterns = patterns.relaxed_patterns()
    assert_that(calculated_relaxed_patterns == relaxed_patterns)


@pytest.mark.skip(reason="Just for testing patterns are valid not actually testing anythings")
def test_eigenvalues_of_patterns():
    p_adjusted = [0.7, 0.7, -0.01]
    # p_adjusted = [0.6, 0.4, -0.6]
    res = check_if_psd(p_adjusted)
    assert_that(res)
