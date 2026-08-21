from hamcrest import *

from src.utils.level_sets import LevelSets


def test_assigns_pattern_pairs_to_level_sets():
    ls = LevelSets()
    pairs_for_level = ls.pairs_for_level_lookup
    levels_for_pairs = ls.level_for_pair_lookup

    # right number of levels sets, total pattern pairs (*2 since each pair is in the lookup either way round other
    # than the identical pairs of which we have 23)
    assert_that(len(pairs_for_level.keys()), is_(5)) # for relaxed patterns level set 5 is not reachable
    n_pairs = 23 + 253 * 2
    assert_that(sum(len(v) for v in pairs_for_level.values()), is_(n_pairs))

    assert_that(len(levels_for_pairs.keys()), is_(n_pairs))

    # level set 0 all self pair
    assert_that(len(pairs_for_level[0]), is_(23))

    # test two pairs from each level set that do not change whether relaxed or canonical
    assert_that(pairs_for_level[1], has_item((0, 1)))
    assert_that(levels_for_pairs[(0, 1)], is_(1))
    assert_that(pairs_for_level[1], has_item((1, 0)))
    assert_that(levels_for_pairs[(1, 0)], is_(1))
    assert_that(pairs_for_level[2], has_item((1, 2)))
    assert_that(pairs_for_level[2], has_item((2, 1)))
    assert_that(pairs_for_level[3], has_item((13, 15)))
    assert_that(pairs_for_level[3], has_item((15, 13)))
    assert_that(pairs_for_level[4], has_item((17, 18)))
    assert_that(pairs_for_level[4], has_item((18, 17)))
    assert_that(levels_for_pairs[(18, 17)], is_(4))

    # those that change if relaxed
    assert_that(pairs_for_level[1], has_item((10, 11)))  # becomes 1
    assert_that(pairs_for_level[1], has_item((11, 10)))  # becomes 1
    assert_that(pairs_for_level[2], has_item((12, 13)))  # becomes 2
    assert_that(pairs_for_level[2], has_item((13, 12)))  # becomes 2
    assert_that(pairs_for_level[2], has_item((2, 4)))  # becomes 2
    assert_that(pairs_for_level[2], has_item((4, 2)))  # becomes 2
    assert_that(pairs_for_level[3], has_item((5, 7)))  # 3
    assert_that(pairs_for_level[3], has_item((7, 5)))  # 3
    assert_that(pairs_for_level[4], has_item((12, 25)))  # 4
    assert_that(pairs_for_level[4], has_item((25, 12)))  # 4

    assert_that(levels_for_pairs[(13, 12)], is_(2))
    assert_that(levels_for_pairs[(25, 12)], is_(4))
    assert_that(levels_for_pairs[(12, 25)], is_(4))


def test_returns_levels_in_level_sets():
    ls = LevelSets()

    assert_that(ls.levels, contains_exactly(0, 1,2,3,4))

def test_adjacent_level_set_indices():
    ls = LevelSets()

    assert_that(ls.adjacent_indices, contains_exactly((0, 1), (1, 2), (2, 3), (3, 4)))