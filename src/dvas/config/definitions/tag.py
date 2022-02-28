"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Required attributes definition for
.config.ConfigManager.Tag class.

"""

# Note:
# Although the tag config file is no longer necessary because the tags are generated
# if they are missing, this module has been kept for the sake of clarity because
# the default tags are used in other packages and modules.


# Import current packages modules
from ...database.model import Tag as TableTag
from ...hardcoded import TAG_NONE_NAME, TAG_NONE_DESC
from ...hardcoded import TAG_RAW_NAME, TAG_RAW_DESC
from ...hardcoded import TAG_GDP_NAME, TAG_GDP_DESC
from ...hardcoded import TAG_1S_NAME, TAG_1S_DESC
from ...hardcoded import TAG_CWS_NAME, TAG_CWS_DESC
from ...hardcoded import TAG_DTA_NAME, TAG_DTA_DESC
from ...hardcoded import TAG_SYNC_NAME, TAG_SYNC_DESC
from ...hardcoded import TAG_EMPTY_NAME, TAG_EMPTY_DESC

#: dict: Parameter pattern properties (JSON_SCHEMA)
PARAMETER_PATTERN_PROP = {
    rf"^{TableTag.tag_name.name}$": {
        "type": "string",
    },
    rf"^{TableTag.tag_desc.name}$": {
        "type": "string"
    },
}

#: list: Constant node values
CONST_LABELS = [
    {
        TableTag.tag_name.name: TAG_NONE_NAME,
        TableTag.tag_desc.name: TAG_NONE_DESC
    },
    {
        TableTag.tag_name.name: TAG_RAW_NAME,
        TableTag.tag_desc.name: TAG_RAW_DESC
    },
    {
        TableTag.tag_name.name: TAG_GDP_NAME,
        TableTag.tag_desc.name: TAG_GDP_DESC
    },
    {
        TableTag.tag_name.name: TAG_1S_NAME,
        TableTag.tag_desc.name: TAG_1S_DESC
    },
    {
        TableTag.tag_name.name: TAG_CWS_NAME,
        TableTag.tag_desc.name: TAG_CWS_DESC
    },
    {
        TableTag.tag_name.name: TAG_DTA_NAME,
        TableTag.tag_desc.name: TAG_DTA_DESC
    },
    {
        TableTag.tag_name.name: TAG_SYNC_NAME,
        TableTag.tag_desc.name: TAG_SYNC_DESC
    },
    {
        TableTag.tag_name.name: TAG_EMPTY_NAME,
        TableTag.tag_desc.name: TAG_EMPTY_DESC
    },

]

#: str: Config manager key name
KEY = TableTag.__name__

#: str: Node name able to be generated by regexp
NODE_GEN = TableTag.tag_name.name
