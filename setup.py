# No need for a module docstring. Disable the pylint warning.
# pylint: disable=C0114

from pathlib import Path
from setuptools import setup, find_packages  # Always prefer setuptools over distutils

# Run the version file
with open(Path('.') / 'src' / 'dvas' / 'dvas_version.py') as fid:
    version = next(
        line.split("'")[1] for line in fid.readlines() if 'VERSION' in line
    )

setup(
    dependency_links=[],
    name="dvas",
    version=version,

    # Include all packages under src
    packages=find_packages("src"),

    # Tell setuptools packages are under src
    package_dir={"": "src"},

    url="",
    license="MIT",
    author="MDA",
    author_email="",
    description="Data Visualisation and Analysis Software for meteorological radiosounding",
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

    # Use DVAS from a terminal
    #entry_points={'console_scripts': ['dvas=dvas.__main__:main']},

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
        'License :: OSI Approved :: BSD 3-Clause "New" or "Revised" License (BSD-3-Clause)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',

    ],

    include_package_data=True,  # So that non .py files make it onto pypi, and then back !
    package_data={
        'pytest_cfg': ['pytest.ini'],
        'test_files': ['./test/*'],
        #'docs': ['../docs/build']
    }

)
