"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for
.config.ConfigManager.Model class.

"""

# Import current packages modules
from ...hardcoded import MODEL_PAT
from ...database.model import Model as TableModel

#: dict: Default values of labels
LABEL_VAL_DEF = {
    TableModel.mid.name: '',
}

#: list: Constant node values
CONST_NODES = [
    {
        TableModel.mdl_name.name: '',
        TableModel.mdl_desc.name: 'Null instrument type',
    }
]

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{TableModel.mdl_name.name}$": {
        "type": "string",
        "pattern": MODEL_PAT,
    },
    rf"^{TableModel.mdl_desc.name}$": {
        "type": "string"
    },
    rf"^{TableModel.mid.name}$": {
        "type": "string"
    }
}

#: str: Config manager key name
KEY = TableModel.__name__

#: str: Node name able to be generated by regexp
NODE_GEN = ''
