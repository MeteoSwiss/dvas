.. _running:

Running dvas recipes
====================

dvas recipes can be seen as pre-packaged, pre-configured analysis pipelines, designed to perform
**specific tasks in a specific order**. Their primary purpose is to allow any official dvas analysis
to be reproduced by anyone anywhere for any comaptible dataset.

As an example, we will use the case of the UAII 2022 campaign, for which a dedicated dvas recipe was
created. With dvas :ref:`installed <install>`, follow these 2 steps to reproduce the entire
UAII 2022 analysis cascade:

  1. Initialize a new dvas *processing arena*. In a terminal, in a location of your choice, type:

    .. code-block:: none

       dvas_init_arena

    This will create a new folder ``dvas_proc_arena``.

    .. hint::

       Use ``dvas_init_arena --help`` to see what options exist:

       .. literalinclude:: dvas_help_msg.txt
          :language: none

  2. The processing arena you just created got pre-filled with 1) a series of configuration files
  for the dvas database (more on this below) and 2) the official dvas recipe files, including
  `uaii22.rcp`. To run the recipe, use the `dvas_run_recipe` entry point from the command line:

     .. code-block:: none

       cd dvas_proc_arena
       dvas_run_recipe uaii22.rcp

Launching the recipe was the easy part. But what is actually happening ?  Time to take a
closer look at the ...

Anatomy of a dvas processing arena
----------------------------------

There are 3 main components to a dvas processing arena: the
:ref:`raw data <raw_data>`,
the :ref:`database config files <config_data>`,
and the :ref:`recipe file <recipe_file>`. Let's take a closer look at each of these.

.. _raw_data:

The data folder
...............
This folder contains all the raw data to be processed. The actual structuring of subfolders inside
`./data` does not actually matter to dvas, so feel free to organize your datasets as you please.
Note, however, the following restrictions:

  - for non-GDP radiosondes: a `.yml` text file **with the same name** as the raw data file must be
    specified for all datasets. These must `.yml` contain all the metadata not otherwise present in
    the raw data files.

    .. todo:: Specify what these metadata should be

.. _config_data:

The config folder
.................

This folder contains all the information required to setup the dvas database, and have it ingest all
the raw data correctly. Modifying these files is only required if one wishes to include datasets
that differ from those already supported by dvas.

.. _recipe_file:

The recipe file
...............

The dvas recipes are described in YAML file with the `.rcp` extension. In there, you will find
general recipe parameters, including the list of variable names to process, together with the list
of all the processing steps and their associated parameters. All these steps refer to high-level
routines and modules from the ``dvas_recipes`` sub-package, that themselves rely on core ``dvas``
modules and functions.

Specifc recipes
----------------

demo
....

Used for test and development purposes. It contains dummy routines to illustrate how recipe
processing steps can be assembled.

uaii22
......
Performs the official UAII 2022 analysis for the field campaign data.
