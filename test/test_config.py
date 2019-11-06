"""Testing module for .config

"""

# Import python packages and modules
from pathlib import Path
from jsonschema import validate
import pytest


from mdtpyhelper.check import CheckfuncAttributeError

from uaii2021.config import ConfigManager
from uaii2021.config import ConfigReadError, ConfigItemError


class TestConfigManager:

    document = {
        'raw_data_ok': """
            '00':
                idx_unit: dt
                dt_format: YY
                x_a: 2.0
            '99':
                usecols: [0, 1, 3]
                namecols: ['idx', 'T', 'RH']
            '10':
                delimiter: ','
                # Convert knots to m/s
                WF_a: 0.51546
            """,
        'raw_data_valid_err': """
            '0':
                idx_unit: dt
            """
    }

    def test_instanciate_all_childs(self, tmp_path):
        tmp_path = Path(tmp_path.as_posix())
        cfg_mngrs = ConfigManager.instantiate_all_childs(tmp_path)

        return cfg_mngrs

    def test_jsonschema(self, tmp_path):

        for key, cfg_mngr in self.test_instanciate_all_childs(tmp_path).items():
            assert hasattr(cfg_mngr, 'JSONSCHEMA') is True

    def test_master(self, tmp_path):

        for key, cfg_mngr in self.test_instanciate_all_childs(tmp_path).items():
            assert hasattr(cfg_mngr, 'MASTER') is True
            assert type(cfg_mngr.MASTER) is dict
            assert (len(cfg_mngr.MASTER.keys()) == 1)
            assert 'master' in cfg_mngr.MASTER.keys()
            assert validate(instance=cfg_mngr.MASTER, schema=cfg_mngr.JSONSCHEMA) is None

    def test_get_document(self, tmp_path):

        for key, cfg_mngr in self.test_instanciate_all_childs(tmp_path).items():
            file_path = tmp_path / cfg_mngr.config_file_name

            with file_path.open('w') as fid:
                fid.write(self.document[key + '_ok'])

            cfg_mngr.read()

            assert type(cfg_mngr.data) is dict

            with file_path.open('w') as fid:
                fid.write(self.document[key + '_valid_err'])

            with pytest.raises(ConfigReadError):
                cfg_mngr.update()

    def test_getitem(self, tmp_path):

        for key, cfg_mngr in self.test_instanciate_all_childs(tmp_path).items():
            file_path = tmp_path / cfg_mngr.config_file_name

            with file_path.open('w') as fid:
                fid.write(self.document[key + '_ok'])

            cfg_mngr.read()

            assert cfg_mngr['T_a'] == 1.0
            assert cfg_mngr['00.T_a'] == 2.0
            assert cfg_mngr[['99', 'usecols']] == [0, 1, 3]

            with pytest.raises(ConfigItemError):
                cfg_mngr['99']

            with pytest.raises(ConfigItemError):
                cfg_mngr['usecol']

            with pytest.raises(CheckfuncAttributeError):
                cfg_mngr[0]

            with pytest.raises(CheckfuncAttributeError):
                cfg_mngr['']