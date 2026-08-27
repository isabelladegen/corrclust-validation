import pandas as pd
from hamcrest import *

from src.evaluation.boundary_pattern_generator import ModifiedPatternGenerator, ModifiedCols, save_modified_patterns, \
    read_modified_patterns


def test_creates_23_patterns_all_boundary_inwards_of_the_elliptope():
    modify_by = 0.1
    gen = ModifiedPatternGenerator(mae=modify_by)
    df = gen.generate()

    assert_that(df.shape, is_((23,4)))
    assert_that(df[ModifiedCols.mae].unique(), is_(modify_by)) # all
    assert_that(df.loc[0][ModifiedCols.relaxed_pattern], contains_exactly(0, 0, 0))
    assert_that(df.loc[0][ModifiedCols.modified_pattern], contains_exactly(0.1, 0.1, 0.1))

    assert_that(df.loc[1][ModifiedCols.relaxed_pattern], contains_exactly(0, 0, 1))
    assert_that(df.loc[1][ModifiedCols.modified_pattern], contains_exactly(-0.1, -0.1, 0.9))

    assert_that(df.loc[2][ModifiedCols.relaxed_pattern], contains_exactly(0, 0, -1))
    assert_that(df.loc[2][ModifiedCols.modified_pattern], contains_exactly(0.1, 0.1, -0.9))


def test_saves_and_reads_modified_patterns(tmp_path):
    file_path =  str(tmp_path / "fake_patterns.csv")
    gen = ModifiedPatternGenerator(mae=0.1)
    df = gen.generate()

    save_modified_patterns(df, file_path)

    saved_version = read_modified_patterns(file_path)
    pd.testing.assert_frame_equal(df, saved_version)
