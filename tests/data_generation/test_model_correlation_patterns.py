import numpy as np
from hamcrest import *
from src.data_generation.model_correlation_patterns import ModelCorrelationPatterns, PatternCols

patterns = ModelCorrelationPatterns()
df = patterns.df


def test_reads_the_model_pattern_csv_file_into_pandas_dataframe():
    assert_that(df.shape[0], is_(27))

    # test a few different patterns
    assert_that(df.iloc[0][PatternCols.id], is_(0))
    assert_that(df.iloc[0][PatternCols.ideal_cors], instance_of(tuple))
    assert_that(df.iloc[0][PatternCols.ideal_cors], is_((0, 0, 0)))
    assert_that(df.iloc[0][PatternCols.modelable_cors], instance_of(tuple))
    assert_that(df.iloc[0][PatternCols.modelable_cors], is_((0, 0, 0)))
    assert_that(df.iloc[0][PatternCols.reg_term], is_(0))
    assert_that(df.iloc[0][PatternCols.should_model], is_(True))
    assert_that(df.iloc[0][PatternCols.is_ideal], is_(True))

    assert_that(df.iloc[23][PatternCols.id], is_(23))
    assert_that(df.iloc[23][PatternCols.ideal_cors], is_((-1, 1, -1)))
    assert_that(df.iloc[23][PatternCols.modelable_cors], is_((-1, 1, -1)))
    assert_that(df.iloc[23][PatternCols.reg_term], is_(0.0001))
    assert_that(df.iloc[23][PatternCols.should_model], is_(True))
    assert_that(df.iloc[23][PatternCols.is_ideal], is_(True))

    assert_that(df.iloc[26][PatternCols.id], is_(26))
    assert_that(df.iloc[26][PatternCols.ideal_cors], is_((-1, -1, -1)))
    assert_that(df.iloc[26][PatternCols.modelable_cors], is_(('nan', 'nan', 'nan')))
    assert_that(df.iloc[26][PatternCols.should_model], is_(False))
    assert_that(df.iloc[26][PatternCols.reg_term], is_(0.1))
    assert_that(df.iloc[26][PatternCols.is_ideal], is_(False))

    assert_that(df.iloc[4][PatternCols.id], is_(4))
    assert_that(df.iloc[4][PatternCols.ideal_cors], is_((0, 1, 1)))
    assert_that(df.iloc[4][PatternCols.modelable_cors], is_((0.1, 0.8, 0.8)))
    assert_that(df.iloc[4][PatternCols.reg_term], is_(0.1))
    assert_that(df.iloc[4][PatternCols.should_model], is_(True))
    assert_that(df.iloc[4][PatternCols.is_ideal], is_(False))


def test_returns_ids_for_patterns_to_model_for_data_generation():
    ids = patterns.ids_of_patterns_to_model()

    # four of the patterns cannot be sensibly modeled
    assert_that(len(ids), is_(23))
    assert_that(14 in ids, is_(False))
    assert_that(16 in ids, is_(False))
    assert_that(22 in ids, is_(False))
    assert_that(26 in ids, is_(False))


def test_returns_dictionary_of_patterns_to_model_for_data_generation():
    patterns_to_model = patterns.patterns_to_model()

    assert_that(len(patterns_to_model), is_(23))
    assert_that(patterns_to_model[0], is_(([0, 0, 0], 0)))
    assert_that(patterns_to_model[4], is_(([0.1, 0.8, 0.8], 0.1)))
    assert_that(patterns_to_model[13], is_(([1, 1, 1], 0.0001)))


def test_returns_dictionary_of_ideal_correlations_for_data_generation():
    correlations = patterns.ideal_correlations()

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

    assert_that(all(np.equal(x[22, :], np.array([-1, -1, 1]))))
    assert_that(y[22], is_(25))
