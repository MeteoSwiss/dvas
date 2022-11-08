# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'gruan' classes and function of the tools submodule.

"""

# Import from python packages and modules
import numpy as np
import pytest
import pandas as pd

from dvas.data.strategy.data import GDPProfile
from dvas.data.data import MultiGDPProfile
from dvas.database.database import InfoManager
from dvas.hardcoded import FLG_INCOMPATIBLE

# Functions to test
from dvas.tools.gdps.gdps import combine

# Define db_data. This is some black magic that is directly related to conftest.py
# This is a temporary db, that is required for get_info() to work properly with mdl_id.
# It all relies on the db config files located in the processing arena
db_data = {
    'sub_dir': 'test_tool_gdps',
    'data': [{'mdl_name': 'AR-GDP_001',
              'srn': 'AR1',
              'pid': '0'},
             {'mdl_name': 'AR-GDP_001',
              'srn': 'BR1',
              'pid': '1'},
             {'mdl_name': 'AR-GDP_001',
              'srn': 'CR1',
              'pid': '2'},
             ]}


@pytest.fixture
def gdp_1_prfs(db_init):
    """ Return a MultiGDPProfile with 1 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [0, 1, 2], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1)])

    return multiprf


@pytest.fixture
def gdp_2_prfs_real(db_init):
    """ Returns a MultiGDPProfile with 2 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [16168.318359375, 16172.6962890625,   16177.47265625,
                                   16183.2431640625,  16189.013671875,   16196.27734375,
                                   16198.16796875, 16202.8447265625, 16206.1279296875,
                                   16211.5009765625,  16216.873046875,   16222.64453125,
                                   16224.6337890625, 16228.6142578125,  16236.076171875,
                                   16238.5634765625, 16243.9365234375, 16248.9111328125,
                                   16254.6826171875, 16259.8564453125,  16261.646484375,
                                   16266.0244140625, 16272.7900390625, 16276.6708984375,
                                   16280.55078125,  16282.541015625, 16287.1181640625,
                                   16292.490234375, 16297.0673828125,  16302.837890625,
                                   16305.921875,       16307.8125],
                           'val': [215.16178894, 215.21238708, 215.27020264, 215.33743286,
                                   215.41479492, 215.50041199, 215.58969116, 215.67655945,
                                   215.75535583, 215.82229614, 215.87620544, 215.91833496,
                                   215.95120239, 215.97753906, 215.99987793, 216.01992798,
                                   216.03862,    216.05606079, 216.07185364, 216.08599854,
                                   216.09867859, 216.10978699, 216.11889648, 216.12561035,
                                   216.12918091, 216.1293335,  216.12643433, 216.12104797,
                                   216.11351013, 216.10374451, 216.09164429, 216.07739258],
                           'flg': [0] * 32, 'tdt': np.arange(0, 32, 1),
                           'ucr': [0.07555172, 0.08199992, 0.08165803, 0.08122376, 0.08155127,
                                   0.08151338, 0.08151397, 0.08127866, 0.08158639, 0.08061193,
                                   0.08002041, 0.07791159, 0.07304052, 0.0691337,  0.06931446,
                                   0.0652895,  0.06085247, 0.06019598, 0.05996601, 0.05956999,
                                   0.05957349, 0.05995082, 0.06036155, 0.06025105, 0.06082179,
                                   0.0606791,  0.06057125, 0.0612908,  0.06411498, 0.06740434,
                                   0.06809566, 0.06971079],
                           'ucs': [0.03440255, 0.03444686, 0.0344841,  0.0345178,  0.03454728,
                                   0.03458127, 0.03460733, 0.0346407,  0.03468493, 0.03471906,
                                   0.03475836, 0.03479092, 0.03482113, 0.03488062, 0.03494934,
                                   0.03498489, 0.03503325, 0.03508255, 0.03516109, 0.03522544,
                                   0.03528746, 0.03535303, 0.03541872, 0.03547047, 0.03551596,
                                   0.03556027, 0.03558585, 0.03562085, 0.03564566, 0.03566941,
                                   0.03568577, 0.03568037],
                           'uct': [0.2031993,  0.2032079,  0.20321515, 0.20322174, 0.20322751,
                                   0.20323423, 0.20323917, 0.20324568, 0.2032543,  0.20326099,
                                   0.20326872, 0.20327514, 0.20328094, 0.20329265, 0.20330639,
                                   0.20331331, 0.20332292, 0.20333269, 0.20334841, 0.20336127,
                                   0.20337355, 0.20338666, 0.20339996, 0.2034103,  0.20341939,
                                   0.20342818, 0.20343328, 0.20344032, 0.20344527, 0.20345007,
                                   0.20345323, 0.20345189],
                           'ucu': [np.nan]*32})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [16170.109375,  16173.392578125, 16178.1689453125,
                                   16182.8447265625, 16188.6162109375, 16193.6904296875,
                                   16198.068359375, 16202.0478515625,  16206.724609375,
                                   16211.4013671875,    16215.6796875, 16219.9580078125,
                                   16224.5341796875, 16228.4150390625, 16233.3896484375,
                                   16239.359375,   16243.33984375,     16248.015625,
                                   16252.3935546875,  16256.771484375, 16261.1494140625,
                                   16265.12890625, 16269.4072265625,  16274.283203125,
                                   16277.7646484375, 16281.5458984375,   16285.42578125,
                                   16289.7041015625, 16294.9775390625,   16299.75390625,
                                   16303.4345703125, 16306.9169921875],
                           'val': [215.04988098, 215.08650208, 215.13090515, 215.18603516,
                                   215.25404358, 215.33486938, 215.4251709,  215.51831055,
                                   215.60653687, 215.68391418, 215.74784851, 215.79933167,
                                   215.84141541, 215.87727356, 215.90872192, 215.93695068,
                                   215.96278381, 215.98667908, 216.00883484, 216.02935791,
                                   216.0483551,  216.06585693, 216.08174133, 216.09567261,
                                   216.10705566, 216.11524963, 216.11997986, 216.12156677,
                                   216.12059021, 216.11734009, 216.11131287, 216.10166931],
                           'flg': [0] * 32, 'tdt': np.arange(0, 32, 1),
                           'ucr': [0.07256918, 0.0871015,  0.0944924,  0.09441814, 0.09380143,
                                   0.09393378, 0.09405857, 0.09433935, 0.09445141, 0.09447317,
                                   0.09403224, 0.09246008, 0.08812037, 0.07778805, 0.07471242,
                                   0.07461127, 0.06599253, 0.05912118, 0.0582149,  0.0581477,
                                   0.05765999, 0.05778417, 0.05820568, 0.05815959, 0.05915051,
                                   0.06003633, 0.06001211, 0.05997865, 0.0612901,  0.06316715,
                                   0.06527633, 0.06582592],
                           'ucs': [0.034509,  0.03456827, 0.03463086, 0.03468378, 0.03473072,
                                   0.03477457, 0.034821,  0.03485432, 0.03489641, 0.03494982,
                                   0.0349712,  0.03500867, 0.03506324, 0.03510539, 0.03514243,
                                   0.03517632, 0.03518511, 0.03521434, 0.03525354, 0.03527968,
                                   0.03530406, 0.03531879, 0.03532578, 0.03532924, 0.03532648,
                                   0.03531347, 0.0352973,  0.03528114, 0.03526764, 0.03525002,
                                   0.0352381,  0.03522123],
                           'uct': [0.20321359, 0.20322511, 0.20323737, 0.20324776, 0.20325699,
                                   0.20326562, 0.20327473, 0.20328124, 0.20328951, 0.20330006,
                                   0.20330422, 0.20331158, 0.20332238, 0.20333068, 0.20333803,
                                   0.20334478, 0.20334642, 0.20335218, 0.20335995, 0.20336509,
                                   0.20336989, 0.20337273, 0.20337403, 0.20337464, 0.20337392,
                                   0.20337117, 0.20336778, 0.20336442, 0.20336163, 0.20335798,
                                   0.20335545, 0.2033519],
                           'ucu': [np.nan]*32})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2)])

    return multiprf


@pytest.fixture
def gdp_3_prfs(db_init):
    """ Return a MultiGDPProfile with 3 GDPprofiles in it. """

    # Get the oids from the profiles
    oids = [item['oid'] for item in db_init.data]

    # Prepare some datasets to play with
    info_1 = InfoManager('20210302T0000Z', oids[0], tags=['e:1', 'r:1'])
    data_1 = pd.DataFrame({'alt': [10., 15., 20.], 'val': [10., 20., 30.], 'flg': [1, 1, 1],
                           'tdt': [1, 2, 3], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})
    info_2 = InfoManager('20210302T0000Z', oids[1], tags=['e:1', 'r:1'])
    data_2 = pd.DataFrame({'alt': [11., 16., 20.1], 'val': [10.5, 21., np.nan], 'flg': [1, 1, 1],
                           'tdt': [1, 2, 3], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})
    info_3 = InfoManager('20210302T0000Z', oids[2], tags=['e:1', 'r:1'])
    data_3 = pd.DataFrame({'alt': [10.1, 17., 20.], 'val': [11., 21.1, np.nan], 'flg': [1, 1, 1],
                           'tdt': [1, 2, 3], 'ucr': [1, 1, 1], 'ucs': [1, 1, 1],
                           'uct': [1, 1, 1], 'ucu': [1, 1, 1]})

    # Let's build a multiprofile so I can test things out.
    multiprf = MultiGDPProfile()
    multiprf.update({'val': None, 'tdt': None, 'alt': None, 'flg': None, 'ucr': None, 'ucs': None,
                     'uct': None, 'ucu': None},
                    [GDPProfile(info_1, data_1), GDPProfile(info_2, data_2),
                     GDPProfile(info_3, data_3)])

    return multiprf


# Let us test a series of conditions for the different types of uncertainty types
def test_combine(gdp_1_prfs, gdp_2_prfs_real, gdp_3_prfs):
    """Function used to test if the routine combining GDP profiles is ok.

    The function tests:
        - correct propagation of errors when rebining a single profile

    """
    # 0) Ultra-basic test: the mean of a single profile with binning = 1 should return the
    # same thing
    for method in ['arithmetic mean', 'weighted arithmetic mean', 'weighted circular mean']:
        out, _ = combine(gdp_1_prfs, binning=1, method=method, chunk_size=200, n_cpus=1)

        for key in ['val', 'ucr', 'ucs', 'uct', 'ucu']:
            assert out.profiles[0].data[key].round(10).equals(gdp_1_prfs.profiles[0].data[key])
        for key in ['tdt', 'alt']:
            assert np.array_equal(out.profiles[0].data.index.get_level_values(key),
                                  gdp_1_prfs.profiles[0].data.index.get_level_values(key))

    # 1) Basic test: does it work with multiprocessing ? Also check proper tagging
    out, _ = combine(gdp_1_prfs, binning=2, method='arithmetic mean', chunk_size=200, n_cpus='max')
    assert np.all(out.profiles[0].data.loc[0, 'val'] == 15.)
    assert out.get_info('eid')[0] == 'e:1'
    assert out.get_info('rid')[0] == 'r:1'

    # 2) Check the weighted mean errors ...
    out, _ = combine(gdp_3_prfs, binning=1, method='weighted arithmetic mean', chunk_size=200,
                     n_cpus='max')
    assert np.all(out.profiles[0].data.loc[0, 'ucu'] == np.sqrt(1/3))
    assert np.all(out.profiles[0].data.loc[0, 'uct'] == 1.)

    # 3) Test for #166, and the fact that chunk_size should have no impact on the outcome
    out_a, _ = combine(gdp_2_prfs_real, binning=3, method='arithmetic delta', n_cpus=1,
                       mask_flgs=FLG_INCOMPATIBLE, chunk_size=3)
    out_b, _ = combine(gdp_2_prfs_real, binning=3, method='arithmetic delta', n_cpus=16,
                       mask_flgs=FLG_INCOMPATIBLE, chunk_size=4)
    assert out_a[0].data.equals(out_b[0].data)
