"""Testing module for .config

"""

# Import python packages and modules
from pathlib import Path
from jsonschema import validate, exceptions
import pytest

from mdtpyhelper.check import CheckfuncAttributeError

from uaii2021.config import ConfigManager


class TestConfigManager:

    document = {
        'raw_data': """ 
            master:
                idx_unit: ms
                dt_format:
                delimiter: ;
                usecols: [0, 1, 2, 3, 4, 5, 6]
                namecols: ['idx', 'T', 'RH', 'P', 'A', 'WF', 'WD']
                x_dec: -3
                x_a: 1.0
                x_b: 0.0
                type_name: None
                skiprows: 0
            '00':
                idx_unit: dt
                dt_format: YY
            """
    }

    def test_validate_document(self):
        assert ConfigManager.validate_document([1, 2], {"maxItems": 2}) is None

        with pytest.raises(exceptions.ValidationError):
            ConfigManager.validate_document([1, 2, 3], {"maxItems": 2})

        with pytest.raises(exceptions.SchemaError):
            ConfigManager.validate_document([1, 2], {"maxItems": '2'})

    def test_instanciate_all_childs(self, tmp_path):
        tmp_path = Path(tmp_path.as_posix())
        cfg_mngrs = ConfigManager.instantiate_all_childs(Path(tmp_path))

        return cfg_mngrs

    def test_jsonschema(self, tmp_path):

        for key, cfg_mngr in self.test_instanciate_all_childs(tmp_path).items():
            assert hasattr(cfg_mngr, 'JSONSCHEMA') is True

    def test_get_document(self, tmp_path):

        for key, cfg_mngr in self.test_instanciate_all_childs(tmp_path).items():
            file_path = tmp_path / cfg_mngr.config_file_name

            with file_path.open('w') as fid:
                fid.write(self.document[key])

            assert type(cfg_mngr.get_document()) is dict
