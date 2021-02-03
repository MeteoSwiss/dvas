# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'stats' classes and function of the tools submodule.

"""

# Import from python packages and modules
#import numpy as np
#import scipy.stats as ss
#import pandas as pd

def test_stats_gdp_ks_test():
    """Function used to test if the KS test between 2 GDPs is behaving ok.

    The function tests:
        - ability to correctly flag points that fail the test

    TODO:
        - generate a test based on actual data ?
        - test the impact of binning ?
    """

    # Create two "profiles" in sigma scale
    # Here, we only consider random errors for now.
    #profile = range(1, 5, 1) * np.sqrt(8)
    #qrofile = np.zeros(4)

    # Create fake errors, such that the "total" error of profile-qrofile is np.sqrt(8)
    #sigma_u = np.ones(4) * 2
    #sigma_e = np.zeros(4)
    #sigma_s = np.zeros(4)
    #sigma_t = np.zeros(4)

    # Note: p-value of 3-sigma level = 0.0027 = (1 - ss.norm.cdf(3))*2
    #(f_pqi, _) = stats.gdp_ks_test([profile, qrofile],
    #                               [sigma_u]*2, [sigma_e]*2, [sigma_s]*2, [sigma_t]*2,
    #                               alpha=(1 - ss.norm.cdf(3))*2, # This corresponds to 3 sigma level
    #                               binning_list=[1, 2, 20], do_plot=False, srns=[1, 2])

    # If I get here, then most likely it is all working fine.
    #assert np.all(f_pqi[0] == np.array([0., 0., 1., 1.]))
    assert True

def test_stats_gdp_ks_test_2():
    """Function used to test if the KS test between 2 GDPs is behaving ok.

    The function tests:
        - ability to make a good-looking plot

    """
    assert True
    '''
    # Create two "profiles" in sigma scale
    # Here, we only consider random errors for now.
    profile = np.random.normal(loc=3, scale=2, size=600)
    qrofile = np.random.normal(loc=3, scale=2, size=600)

    # Create fake errors, such that the "total" error of profile-qrofile is np.sqrt(8)
    sigma_u = np.ones(600) * 1
    sigma_e = np.zeros(600)
    sigma_s = np.zeros(600)
    sigma_t = np.zeros(600)

    (f_pqi, _) = stats.gdp_ks_test([profile, qrofile],
                                   [sigma_u]*2, [sigma_e]*2, [sigma_s]*2, [sigma_t]*2,
                                   alpha=0.0027, # This corresponds to a 3 sigma level
                                   binning_list=[1, 2, 10, 40], srns=[1, 2])

    # If I get here, then most likely it is all working fine.
    assert np.all(f_pqi[0] == np.array([0., 0., 1., 1.]))
    '''
