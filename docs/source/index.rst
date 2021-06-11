
.. include:: ./substitutions.rst

dvas |version| |stars| |watch|
==============================

|copyright| |license| |github| |pypi| |last-commit| |issues|

.. todo::

    Tags for the latest pypi release and associated DOI should be added when releasing the code
    for the first time. These should also be added to the :ref:`acknowledge:Acknowledging dvas` page.

**Welcome to the dvas documentation !**

This Python package has been written, primarily, to handle the data analysis of the 2022 Upper-Air
Instrument Intercomparison (UAII) campaign, organized under the auspice of World Meteorological
Organization (WMO). The following pages (should) contain all that you need to know to install and
run dvas.

The code is composed of two sub-packages:

  - ``dvas``: the core package that contains all the low-level modules and routines, and
  - ``dvas_recipes``: the high-level package that contains all the *analysis recipes* specific to a
    given scenario: for example, the UAII 2022 campaign.

There are therefore two distinct ways to use dvas:

  1. :ref:`running_recipes:Running dvas recipes`: users interested to reproduce the UAII 2022
     analysis (either for the original dataset, or for a different-but-similar one) need only to use
     the ``dvas_recipes`` sub-package to do so.

  2. :ref:`creating_recipes:Creating dvas recipes`: users interesting to adjust/alter/expand the
     default analysis recipes, or even to write their own recipe from scratch, must use the core
     ``dvas`` package to do so.

.. note::
    The processing routines inside dvas are intimately linked the specific `Profile` dvas classes.
    These, in turn, are best populated from the dvas database. This implies that using the
    processing routines while by-passing the use of the dvas database (e.g. for custom applications)
    is neither straightforward, nor supported.

dvas is being developed on Github, where you can submit all your
`questions <https://github.com/MeteoSwiss/dvas/discussions>`_ and
`bug reports <https://github.com/MeteoSwiss/dvas/issues>`_. Contributions via pull requests are also
welcome. See :ref:`troubleshooting:Troubleshooting` for more details.


.. toctree::
    :caption: Table of contents
    :maxdepth: 1

    Home <self>
    installation
    running_recipes
    creating_recipes
    troubleshooting
    acknowledge
    license
    changelog
    Contributing <https://github.com/MeteoSwiss-MDA/dvas/blob/develop/CONTRIBUTING.md>
    Github repository <https://github.com/MeteoSwiss-MDA/dvas>
    modules
    doc_todo

.. note::

    These pages are very much a work in progress. If you feel like helping, we have a list of
    missing items on the :ref:`doc_todo:Documentation todos` page.
