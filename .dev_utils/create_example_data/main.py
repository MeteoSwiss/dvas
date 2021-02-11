"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# TODO
#  Create a final "examples and tests" dataset from real data and delete this file.


import os
from shutil import copyfile
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
from dvas import expl_path
import re

expl_path = Path('.').resolve().parents[1] / 'test' / 'processing_arena'
N = 200

expl_path = Path('.').resolve().parents[1] / 'examples'
N = 100000


def sin_rand(n, scale=1):
    return np.sin(np.arange(n)*np.pi/(2*n) + np.random.rand()*2*np.pi) / scale


def from_rs92():

    new_gdp_type = 'BR'
    test_type = 'YT'
    meassite = 'XXX'

    # Search RS41 type
    gdp_files = list((Path('.').resolve().glob('*RS92*')))

    new_gdp_files = []
    for i, gdp_file in enumerate(gdp_files):

        # Create new gdp
        time_str = re.search(r'\_(\d+T\d+)\_', gdp_file.as_posix()).group(1)
        new_gdp_file = expl_path / 'data' / 'gdps' / (meassite + '-RS-01_2_' + new_gdp_type + '-GDP_001_' + time_str + '_1-000-001.nc')
        new_gdp_files.append(new_gdp_file)

        with nc.Dataset(gdp_file) as fid:

            with nc.Dataset(new_gdp_file, 'w', format="NETCDF4") as new_fid:

                # Text attributes can be of variable length
                new_fid.set_ncstring_attrs(True)

                # Set metadata
                new_fid.setncattr('description', 'Dummy ' + new_gdp_type + ' GDP - version 001 - modified data from real Gruan Data Product')
                new_fid.setncattr(
                    'g.Measurement.StandardTime', fid.getncattr('g.Ascent.StandardTime')
                )
                new_fid.setncattr('g.Measurement.BalloonNumber', '1')
                new_fid.setncattr(
                    'g.Measurement.Id',
                    '1' if re.search(r'2018\-01\-10', new_fid.getncattr('g.Measurement.StandardTime')) else '2'
                )
                new_fid.setncattr('g.Product.Code', f"{new_gdp_type}-GDP_001")
                new_fid.setncattr('g.MainSonde.SerialNumber', new_gdp_type + f'-00{i}')
                new_fid.setncattr('g.Product.Version', '0')
                new_fid.setncattr('g.Site.Key', meassite)
                new_fid.sensor_temp_u_enlarged = 0.1

                # Set data
                # Include a typical dimension
                new_fid.createDimension("time", None)

                # Add time
                time = new_fid.createVariable("time", "f4", ("time",), zlib=True, least_significant_digit=3)
                time.units = 's'
                time[:] = fid['time'][:N]

                n_data = len(fid['time'][:N])

                # Add temp
                temp = new_fid.createVariable("temp", "f4", ("time",), zlib=True, least_significant_digit=3)
                temp.units = 'K'
                temp.long_name = 'Temperature'
                temp[:] = fid['temp'][:N] + sin_rand(n_data, 50)

                # Add temp uncertainty environmental correlated
                temp_uec = new_fid.createVariable("temp_uec", "f4", ("time",), zlib=True, least_significant_digit=3)
                temp_uec.units = 'K'
                temp_uec.long_name = 'Environmental correlated uncertainty of temperature'
                temp_uec[:] = fid['u_std_temp'][:N] + sin_rand(n_data, 50)

                # Add temp temporal environmental correlated
                temp_utc = new_fid.createVariable("temp_utc", "f4", ("time",), zlib=True, least_significant_digit=3)
                temp_utc.units = 'K'
                temp_utc.long_name = 'Temporal correlated uncertainty of temperature'
                temp_utc[:] = fid['u_cor_temp'][:N] + sin_rand(n_data, 50)

                # Add alt
                alt = new_fid.createVariable("alt", "f4", ("time",), zlib=True, least_significant_digit=3)
                alt.units = 'm'
                alt[:] = fid['alt'][:N] + sin_rand(n_data, 10)

                # Add pressure
                press = new_fid.createVariable("press", "f4", ("time",), zlib=True, least_significant_digit=3)
                press.units = 'hPa'
                press[:] = fid['press'][:N] + sin_rand(n_data, 10)

        time_str = re.search(r'\_(\d+T\d+)\_', new_gdp_file.as_posix()).group(1)
        test_file = new_gdp_file.parents[1] / 'data_test' / (test_type + '.' + meassite + '_' + time_str + '.csv')
        meta_test_file = test_file.parent / (test_file.stem + '.yml')


        # Create test files
        with nc.Dataset(new_gdp_file) as new_gdp_fid:

            print(new_gdp_fid)
            print(new_gdp_fid.variables)

            # Create metadata
            metadata = []
            for key, arg in {
                'event_dt': 'g.Measurement.StandardTime',
                'rig': 'g.Measurement.BalloonNumber',
                'event_id': 'g.Measurement.Id',
                'station': 'g.Site.Key'
            }.items():
                metadata.append(f"{key}: '{new_gdp_fid.getncattr(arg)}'")

            metadata.append(f"instr_type: '{test_type}'")
            metadata.append(f"sn: '{test_type + f'-10{i}'}'")
            metadata.append(f"product_id: '0'")

            if re.search('T12', new_gdp_fid.getncattr('g.Measurement.StandardTime')) is not None:
                metadata.append(f"day_night: 'day'")
            else:
                metadata.append(f"day_night: 'night'")

            meta_test_file.write_text(f'{os.linesep}'.join(metadata), encoding='UTF-8')

            # Create data
            laps = round(5 * np.random.rand())
            n_data = len(time := new_gdp_fid['time'][laps:])
            data = pd.DataFrame(
                {
                    'time': time,
                    'temp (°C)': new_gdp_fid['temp'][laps:] - 273.15 + sin_rand(n_data, 10),
                    'alt (m)': new_gdp_fid['alt'][laps:] + sin_rand(n_data, 10),
                    'press (m)': new_gdp_fid['press'][laps:] + sin_rand(n_data, 10)
                }
            )
            data.to_csv(test_file, sep=',', index=False, float_format='    %.2f')



def from_rs41():

    new_gdp_type = 'AR'
    test_type = 'ZT'
    meassite = 'XXX'

    # Search RS41 type
    gdp_files = list((Path('.').resolve().glob('*RS41*')))

    new_gdp_files = []
    for i, gdp_file in enumerate(gdp_files):

        # Create new gdp
        time_str = re.search(r'\_(\d+T\d+)\_', gdp_file.as_posix()).group(1)
        new_gdp_file = expl_path / 'data' / 'gdps' / (meassite + '-RS-01_2_' + new_gdp_type + '-GDP_001_' + time_str + '_1-000-001.nc')
        new_gdp_files.append(new_gdp_file)

        with nc.Dataset(gdp_file) as fid:

            with nc.Dataset(new_gdp_file, 'w', format="NETCDF4") as new_fid:

                # Text attributes can be of variable length
                new_fid.set_ncstring_attrs(True)

                # Set metadata
                new_fid.setncattr('description', 'Dummy ' + new_gdp_type + ' GDP - version 001 - modified data from real Gruan Data Product')
                new_fid.setncattr(
                    'g.Measurement.StandardTime', fid.getncattr('g.Measurement.StandardTime')
                )
                new_fid.setncattr('g.Measurement.BalloonNumber', '1')
                new_fid.setncattr(
                    'g.Measurement.Id',
                    '1' if re.search(r'2018\-01\-10', new_fid.getncattr('g.Measurement.StandardTime')) else '2'
                )
                new_fid.setncattr('g.Product.Code', f"{new_gdp_type}-GDP_001")
                new_fid.setncattr('g.MainSonde.SerialNumber', new_gdp_type + f'-00{i}')
                new_fid.setncattr('g.Product.Version', '0')
                new_fid.setncattr('g.Site.Key', meassite)
                new_fid.sensor_temp_u_enlarged = 0.1

                # Set data
                # Include a typical dimension
                new_fid.createDimension("time", None)

                # Add time
                time = new_fid.createVariable("time", "f4", ("time",), zlib=True, least_significant_digit=3)
                time.units = 's'
                time[:] = fid['time'][:N]

                n_data = len(fid['time'][:N])

                # Add temp
                temp = new_fid.createVariable("temp", "f4", ("time",), zlib=True, least_significant_digit=3)
                temp.units = 'K'
                temp.long_name = 'Temperature'
                temp[:] = fid['temp'][:N] + sin_rand(n_data, 50)

                # Add temp uncertainty environmental correlated
                temp_uec = new_fid.createVariable("temp_uec", "f4", ("time",), zlib=True, least_significant_digit=3)
                temp_uec.units = 'K'
                temp_uec.long_name = 'Environmental correlated uncertainty of temperature'
                temp_uec[:] = fid['temp_uc_ncor'][:N] + sin_rand(n_data, 50)

                # Add temp spacial environmental correlated
                temp_usc = new_fid.createVariable("temp_usc", "f4", ("time",), zlib=True, least_significant_digit=3)
                temp_usc.units = 'K'
                temp_usc.long_name = fid['temp_uc_scor'].long_name
                temp_usc[:] = fid['temp_uc_scor'][:N] + sin_rand(n_data, 50)

                # Add temp temporal environmental correlated
                temp_utc = new_fid.createVariable("temp_utc", "f4", ("time",), zlib=True, least_significant_digit=3)
                temp_utc.units = 'K'
                temp_utc.long_name = fid['temp_uc_tcor'].long_name
                temp_utc[:] = fid['temp_uc_tcor'][:N] + sin_rand(n_data, 50)

                # Add alt
                alt = new_fid.createVariable("alt", "f4", ("time",), zlib=True, least_significant_digit=3)
                alt.units = 'm'
                alt[:] = fid['alt'][:N] + sin_rand(n_data, 10)

                # Add pressure
                press = new_fid.createVariable("press", "f4", ("time",), zlib=True, least_significant_digit=3)
                press.units = 'hPa'
                press[:] = fid['press'][:N] + sin_rand(n_data, 10)

        time_str = re.search(r'\_(\d+T\d+)\_', new_gdp_file.as_posix()).group(1)
        test_file = new_gdp_file.parents[1] / 'data_test' / (test_type + '.' + meassite + '_' + time_str + '.csv')
        meta_test_file = test_file.parent / (test_file.stem + '.yml')

        # Create test data
        with nc.Dataset(new_gdp_file) as new_gdp_fid:

            print(new_gdp_fid)
            print(new_gdp_fid.variables)

            # Create metadata
            metadata = []
            for key, arg in {
                'event_dt': 'g.Measurement.StandardTime',
                'rig': 'g.Measurement.BalloonNumber',
                'event_id': 'g.Measurement.Id',
                'station': 'g.Site.Key'
            }.items():
                metadata.append(f"{key}: '{new_gdp_fid.getncattr(arg)}'")

            metadata.append(f"instr_type: '{test_type}'")
            metadata.append(f"sn: '{test_type + f'-10{i}'}'")
            metadata.append(f"product_id: '0'")

            if re.search('T12', new_gdp_fid.getncattr('g.Measurement.StandardTime')) is not None:
                metadata.append(f"day_night: 'day'")
            else:
                metadata.append(f"day_night: 'night'")

            meta_test_file.write_text(f'{os.linesep}'.join(metadata), encoding='UTF-8')

            # Create data
            laps = round(5 * np.random.rand())
            n_data = len(time := new_gdp_fid['time'][laps:])
            data = pd.DataFrame(
                {
                    'time': time,
                    'temp (°C)': new_gdp_fid['temp'][laps:] - 273.15 + sin_rand(n_data, 10),
                    'alt (m)': new_gdp_fid['alt'][laps:] + sin_rand(n_data, 10),
                    'press (m)': new_gdp_fid['press'][laps:] + sin_rand(n_data, 10)
                }
            )
            data.to_csv(test_file, sep=';', index=False, line_terminator=os.linesep + os.linesep, float_format='%.2f')


if __name__ == '__main__':

    from_rs41()
    from_rs92()
