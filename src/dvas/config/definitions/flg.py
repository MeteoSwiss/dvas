"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for `.config.ConfigManager.Flag class`.

"""

# Import current packages modules
from ...database.model import Flg as TableFlg
from ...hardcoded import FLG_EMPTY_NAME, FLG_EMPTY_DESC
from ...hardcoded import FLG_INCOMPATIBLE_NAME, FLG_INCOMPATIBLE_DESC
from ...hardcoded import FLG_INTERP_NAME, FLG_INTERP_DESC
from ...hardcoded import FLG_DESCENT_NAME, FLG_DESCENT_DESC

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{TableFlg.bit_pos.name}$": {
        "type": "integer",
        "minimum": 0,
        "maximum": 63,
    },
    rf"^{TableFlg.flg_name.name}$": {
        "type": "string"
    },
    rf"^{TableFlg.flg_desc.name}$": {
        "type": "string"
    }
}

#: list: Constant labels
CONST_LABELS = [
    {
        TableFlg.bit_pos.name: i,
        TableFlg.flg_name.name: arg[0],
        TableFlg.flg_desc.name: arg[1]
    } for i, arg in enumerate(
        (
            (FLG_EMPTY_NAME, FLG_EMPTY_DESC),
            (FLG_INTERP_NAME, FLG_INTERP_DESC),
            (FLG_INCOMPATIBLE_NAME, FLG_INCOMPATIBLE_DESC),
            (FLG_DESCENT_NAME, FLG_DESCENT_DESC)
        )
    )
]

#: str: Config manager key name
KEY = TableFlg.__name__

#: str: Node name able to be generated by regexp
NODE_GEN = ''
