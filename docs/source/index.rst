
.. include:: ./substitutions.rst

dvas |version| |stars| |watch|
==============================

|copyright| |license| |github| |doi| |pypi| |last-commit| |issues|

**Welcome to the dvas documentation !**

This Python package has been written to handle the data analysis of the `2022 Upper-Air
Instrument Intercomparison (UAII) <https://www.gruan.org/community/campaigns/uaii2022>`_ field
campaign, organized under the auspice of `World Meteorological
Organization (WMO) <https://community.wmo.int/en/activity-areas/imop/intercomparisons>`_.
The following pages (should) contain all that you need to know to install and
run dvas.

.. note::
  These pages focus on the technical aspects of dvas. For an exhaustive description of the
  its physics/statistics capabilities and performances, please refer to the Final Report of the
  UAII 2022.

dvas is being developed on Github, where you can submit all your
`questions <https://github.com/MeteoSwiss/dvas/discussions>`_ and
`bug reports <https://github.com/MeteoSwiss/dvas/issues>`_.

.. important::
  dvas will be actively maintained up to 2023-11-30, at which point the use of dvas will be
  supported on a best-effort basis only.

Within the scope of the UAII 2022 field campaign, dvas is responsible for: the ingestion of
Manufacturer Data Products (MDPs) and GRUAN Data Products (GDPs) in a dedicated database,
the cleanup and synchronization of these radiosonde profiles on a flight-by-flight basis,
the assembly of Combined Working measurement Standards (CWSs) from GDPs, and the assembly of the
so-called :math:`\Lambda_{C,L}` profiles.

The dvas code is composed of two modules:

  - ``dvas``: the core module that contains all the low-level classes and routines, and
  - ``dvas_recipes``: the higher-level module that contains all the *analysis recipes* specific to a
    given scenario: for example, the UAII 2022 campaign.

dvas is meant to be used as a standalone tool. Users interested to reproduce the UAII 2022
analysis need only to :ref:`run the corresponding recipe <running>`.

.. note::
    The scientific processing routines inside dvas are intimately linked to the specific
    ``dvas.data.strategy.data.Profile`` & ``dvas.data.data.MultiProfile`` classes
    (and their children). Instances of theses classes are best initialized from the dvas database.
    **This implies that using the scientific processing routines while by-passing the use of the
    dvas database (e.g. for custom applications) is neither straightforward, nor supported.**


**Table of contents:**

.. toctree::
    :maxdepth: 1

    Home <self>
    installation
    running_recipes
    troubleshooting
    acknowledge
    license
    changelog
    Github repository <https://github.com/MeteoSwiss-MDA/dvas>
    modules
    doc_todo
