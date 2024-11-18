"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import from python packages
from pathlib import Path
from setuptools import setup, find_packages  # Always prefer setuptools over distutils

# Run the version file
with open(Path('.') / 'src' / 'dvas' / 'version.py', encoding='utf-8') as fid:
    version = next(line.split("'")[1] for line in fid.readlines() if 'VERSION' in line)

with open("README.md", "r", encoding='utf-8') as fh:
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

    url="https://meteoswiss.github.io/dvas",
    project_urls={
        'Source': 'https://github.com/MeteoSwiss/dvas/',
        'Changelog': 'https://meteoswiss.github.io/dvas/changelog.html',
        'Issues': 'https://github.com/MeteoSwiss/dvas/issues'
    },
    author="MeteoSwiss",
    author_email="",
    description="Data Visualisation and Analysis Software for the UAII 2022 field campaign",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.10.0, <3.12',
    install_requires=[
        "astroid>=2.13",
        "jsonschema",
        "matplotlib",
        "netcdf4",
        'numpy<2.0',
        "pandas>=2.0.0, <2.2",
        "peewee",
        "pytz",
        "ruamel-yaml",
        "scipy",
        "sre_yield",
    ],
    extras_require={
        'dev': ['sphinx', 'sphinx-rtd-theme', 'plantweb', 'pylint', 'pytest', 'pytest-data']
    },
    # Setup entry points to use dvas from a terminal
    entry_points={
        'console_scripts': ['dvas_init_arena=dvas_recipes.__main__:dvas_init_arena',
                            'dvas_optimize=dvas_recipes.__main__:dvas_optimize',
                            'dvas_run_recipe=dvas_recipes.__main__:dvas_run_recipe']
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.10',
    ],

    include_package_data=True,  # So that non .py files make it onto pypi, and then back !
)
