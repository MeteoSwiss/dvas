"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

from pathlib import Path
from setuptools import setup, find_packages  # Always prefer setuptools over distutils

# Run the version file
with open(Path('.') / 'src' / 'dvas' / 'dvas_version.py') as fid:
    version = next(
        line.split("'")[1] for line in fid.readlines() if 'VERSION' in line
    )

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    dependency_links=[],
    name="dvas",
    version=version,
    license='GNU General Public License v3 or later (GPLv3+)',

    # Include all packages under src
    packages=find_packages("src"),

    # Tell setuptools packages are under src
    package_dir={"": "src"},

    url="https://github.com/MeteoSwiss-MDA/dvas",
    author="MeteoSwiss",
    author_email="",
    description="Data Visualisation and Analysis Software for meteorological radiosounding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.8.0',
    install_requires=[
        "jsonschema",
        "matplotlib",
        "netCDF4",
        'numpy',
        "pampy",
        "pandas",
        "peewee",
        "pytest",
        "pytest-env",
        "pytest-datafiles",
        "ruamel-yaml",
        "scipy",
        "sre_yield",
    ],

    # Setup entry points to use DVAS from a terminal
    entry_points={'console_scripts': ['dvas=dvas.__main__:main']},

    classifiers=[

        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Meteorology',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',

    ],

    include_package_data=True,  # So that non .py files make it onto pypi, and then back !

)
