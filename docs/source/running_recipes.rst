.. _running:

Running dvas recipes
====================

dvas recipes are pre-packaged, pre-configured analysis pipelines, designed to perform
**specific tasks in a specific order**. Their primary purpose is to allow any official dvas analysis
to be reproduced by anyone, anywhere.

Setup of a dvas processing arena
--------------------------------

We will here use the case of the UAII 2022 field campaign, for which a dedicated dvas recipe was
created. Once dvas has been :ref:`installed <install>`, follow these steps to reproduce the entire
UAII 2022 field campaign analysis cascade:

  1. Initialize a new dvas *processing arena*. In a terminal, in a location of your choice, type:

    .. code-block:: none

       dvas_init_arena

    This will create a new folder ``dvas_proc_arena``.

    .. hint::

       Use ``dvas_init_arena --help`` to see what options exist:

       .. literalinclude:: dvas_help_msg.txt
          :language: none

    The processing arena you just created got pre-filled with:

      a) a series of configuration files for the dvas database (more on this below),
      b) a series of so-called *fid-eid*  ``.csv`` files with lists of flights, and
      c) the official dvas recipe file for the UAII 2022 field campaign: ``uaii22.rcp``.


  2. Fetch the flight data to be processed from the UAII 2022 Supplementary Material,
     and unpack it not too far away. To be specific, you need to download and extract the
     *organized_flight_data_for_dvas_input.tar.gz* element of the
     `dataset on Zenodo <https://doi.org/10.5281/zenodo.10160683>`_.

  3. Verify how well dvas runs on your machine with the dedicated entry point:

    .. code-block:: none

      cd dvas_proc_arena
      dvas_optimize

    This command will process some mock data repeatedly, to explore which ``chunk_size`` value
    provides the best performance. For certain costly operations, dvas can break profiles into chunks
    to reduce the memory consumption (by keeping the size of the correlation matrices small). If the
    chunks are too small, however, the performances will degrade because there aren't enough cores to
    process them all efficiently.

    For example, on a 2021 MacBook Pro (16-inch) with 64 GB of RAM and an Apple M1 Max CPU with
    10-cores, we find that a chunk size of ~150-200 works best, with a processing time (reported by
    ``dvas_optimize``) of ~1.1 seconds. By comparison, on a 2019 MacBook Pro (16-inch) with 32 GB
    of RAM and a 2.3 GHz 8-core Intel Core i9 CPU, we find the same ideal chunk size of ~150, but with
    a processing time of ~2.4 seconds.

You now have all the elements required to run the dvas recipe for the UAII 2022 field camapign.
Time to take a closer look at the ...

Anatomy of the dvas processing arena
------------------------------------

There are 3 main components to a dvas processing arena: the
:ref:`original data <original_data>`,
the :ref:`database config files <config_data>`,
and the :ref:`recipe file <recipe_file>`. Let's take a closer look at each of these.

.. _original_data:

The original data folder
........................

This folder contains all the original data to be processed. The actual structuring of subfolders
inside it does not actually matter to dvas. Note, however, the following restriction:

  - for non-GDP radiosondes: a ``.yml`` text file **with the same name** as the original data file
    must be specified for all datasets. These ``.yml`` files must contain all the metadata not
    otherwise present in the original data files.

The data folder included in the UAII 2022 Supplementary Material already contains all the necessary
metadata files - no need to change anything there.

.. _config_data:

The config folder
.................

This folder contains all the information required to setup the dvas database, and have it ingest all
the original data correctly. Modifying these files is only required if one wishes to include
datasets that differ from those already supported by dvas.

.. _recipe_file:

The recipe file
...............

The dvas recipes are described in YAML files with the ``.rcp`` extension. In there, you will find
general recipe parameters, including the list of variable names to process, together with the list
of all the processing steps and their associated parameters. All these steps refer to high-level
routines and modules from the ``dvas_recipes`` sub-package, that themselves rely on core ``dvas``
modules and functions.

The ``uaii2022.rcp`` contains all the instructions required the reproduce the official data analysis
of the UAII 2022 field campaign described in the Final Report. This file also contains the different
recipe parameters, some of which must be changed to reflect your specific setup:

  1. Set ``rcp_paths:orig_data_path:sub_path`` to point to the location where you unpacked the
  campaign data. If you followed the instructions above, the line should read:

  .. code-block:: YAML

    sub_path: ./original_data/day_flights

  .. hint::

    We **strongly** recommand to process night and day flights separately to limit the memory use.

  2. Set the name of the person/institution running the recipe under
  ``rcp_params:general:institution``, that will appear in the global attribute ``institution``
  in the NetCDF files created by dvas:

  .. code-block:: YAML

    institution: &inst_name 'J. Doe, Sirius Cybernetics'

  3. [If warranted] Adjust the ``rcp_params:general:chunk_size:`` to the value reported by the
  ``dvas_optimize`` command.

  4. Uncomment the appropriate time-of-day (``tods``) line under step 10:

  .. code-block:: YAML

    tods:
      - daytime
      #- [nighttime, twilight]


Execution of a dvas recipe
--------------------------

With a dedicated dvas processing arena in place, and the parameters of the UAII 2022 recipe adjusted
to your specific system, you should now be able to launch the data processing.

To do so, use the ``dvas_run_recipe`` entry point from the command line:

  .. code-block:: none

    cd dvas_proc_arena
    dvas_run_recipe uaii22.rcp uaii2022_fid-eid-rid-edt_day.csv -s '00' -e '00'


This will trigger the ``uaii2022.rcp`` recipe, for the flights specified in the
``uaii2022_fid-eid-rid-edt_day.csv`` file [#f1]_, starting with step ``00`` and ending with step ``10``.

.. warning::

  Be aware that running the entire UAII 2022 field campaign analysis takes a long time. On a
  2021 MacBook Pro (16-inch) with 64 GB RAM and an Apple M1 Max CPU with 10 cores,
  **the processing of daytime flights takes 62.5 hours** (51.7 hours for the nighttime flights).


.. rubric:: Footnotes

.. [#f1] See `this issue <https://github.com/MeteoSwiss/dvas/issues/243>`_ for a discussion of why this file is required.

