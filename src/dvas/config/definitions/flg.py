"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for `.config.ConfigManager.Flag class`.

"""

# Import current packages modules
from ...database.model import Flg as TableFlg
from ...hardcoded import FLG_NOPRF, FLG_NOPRF_DESC
from ...hardcoded import FLG_INVALID, FLG_INVALID_DESC
from ...hardcoded import FLG_INTERP, FLG_INTERP_DESC
from ...hardcoded import FLG_INCOMPATIBLE, FLG_INCOMPATIBLE_DESC
from ...hardcoded import FLG_HASCWS, FLG_HASCWS_DESC
from ...hardcoded import FLG_PRELAUNCH, FLG_PRELAUNCH_DESC
from ...hardcoded import FLG_ASCENT, FLG_ASCENT_DESC
from ...hardcoded import FLG_DESCENT, FLG_DESCENT_DESC
from ...hardcoded import FLG_PBL, FLG_PBL_DESC
from ...hardcoded import FLG_TROPO, FLG_TROPO_DESC
from ...hardcoded import FLG_FREETROPO, FLG_FREETROPO_DESC
from ...hardcoded import FLG_STRATO, FLG_STRATO_DESC
from ...hardcoded import FLG_UTLS, FLG_UTLS_DESC


#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{TableFlg.bit_pos.name}$": {
        "type": "integer",
        "minimum": 0,
        "maximum": 62,  # Cannot go up to 63, because we need one bit for the sign
    },
    rf"^{TableFlg.flg_name.name}$": {
        "type": "string"
    },
    rf"^{TableFlg.flg_desc.name}$": {
        "type": "string"
    }
}

#: list: Constant labels
CONST_LABELS = [{TableFlg.bit_pos.name: i,
                 TableFlg.flg_name.name: arg[0],
                 TableFlg.flg_desc.name: arg[1]
                 } for i, arg in enumerate(((FLG_NOPRF, FLG_NOPRF_DESC),
                                            (FLG_INVALID, FLG_INVALID_DESC),
                                            (FLG_INTERP, FLG_INTERP_DESC),
                                            (FLG_INCOMPATIBLE, FLG_INCOMPATIBLE_DESC),
                                            (FLG_HASCWS, FLG_HASCWS_DESC),
                                            (FLG_PRELAUNCH, FLG_PRELAUNCH_DESC),
                                            (FLG_ASCENT, FLG_ASCENT_DESC),
                                            (FLG_DESCENT, FLG_DESCENT_DESC),
                                            (FLG_PBL, FLG_PBL_DESC),
                                            (FLG_TROPO, FLG_TROPO_DESC),
                                            (FLG_FREETROPO, FLG_FREETROPO_DESC),
                                            (FLG_STRATO, FLG_STRATO_DESC),
                                            (FLG_UTLS, FLG_UTLS_DESC)))]

#: str: Config manager key name
KEY = TableFlg.__name__

#: str: Node name able to be generated by regexp
NODE_GEN = ''
