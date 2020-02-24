
# Define node order
NODE_PATTERN = [r'instr_type_\d{1,2}$']

# Define default root parameters
ROOT_PARAMS_DEF = {
    'name': '',
    'desc': '',
}

# Define parameter JSON_SCHEMA
PARAMETER_PATTERN_PROP = {
    r"^name$": {
        "type": "string",
        "pattern": r"(^[a-z0-9]{3,}\-[a-z0-9]+$)|($^)",
    },
    r"^desc$": {
        "type": "string"
    }
}
