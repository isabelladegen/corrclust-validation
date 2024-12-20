from hamcrest import *

from src.data_generation.model_distribution_params import ModelDistributionParams, DistParamsCols


def test_can_load_distribution_params_to_model():
    mpam = ModelDistributionParams()

    c_iob = mpam.get_params_for(DistParamsCols.c_iob)
    loc_iob = mpam.get_params_for(DistParamsCols.loc_iob)
    scale_iob = mpam.get_params_for(DistParamsCols.scale_iob)
    n_cob = mpam.get_params_for(DistParamsCols.n_cob)
    p_cob = mpam.get_params_for(DistParamsCols.p_cob)
    c_ig = mpam.get_params_for(DistParamsCols.c_ig)
    loc_ig = mpam.get_params_for(DistParamsCols.loc_ig)
    scale_ig = mpam.get_params_for(DistParamsCols.scale_ig)

    # check we have 30 of each
    assert_that(len(c_iob), is_(30))
    assert_that(len(loc_iob), is_(30))
    assert_that(len(scale_iob), is_(30))
    assert_that(len(n_cob), is_(30))
    assert_that(len(p_cob), is_(30))
    assert_that(len(c_ig), is_(30))
    assert_that(len(loc_ig), is_(30))
    assert_that(len(scale_ig), is_(30))

    # check for uniqueness
    assert_that(len(set(c_iob)), is_(30))
    assert_that(len(set(loc_iob)), is_(29))
    assert_that(len(set(scale_iob)), is_(30))
    assert_that(len(set(n_cob)), is_(1))
    assert_that(len(set(p_cob)), is_(30))
    assert_that(len(set(c_ig)), is_(13))
    assert_that(len(set(loc_ig)), is_(30))
    assert_that(len(set(scale_ig)), is_(30))




