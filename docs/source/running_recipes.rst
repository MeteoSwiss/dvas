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

       dvas_init uaii22

    This will create a new folder ``dvas_uaii22_arena``.

    .. hint::

       You can see all the available recipes using ``dvas_init --help``:

       .. literalinclude:: dvas_help_msg.txt
          :language: none

  2. To run the recipe, simply move into the processing arena, launch an ipython session, and launch
     the recipe:

     .. code-block:: none

       cd dvas_uaii22_arena
       ipython --pylab
       In [1]: run main_script.py

Launching the `uaii22` recipe was the easy part. But what is actually happening ?  Time to take a
closer look at the ...

Anatomy of a dvas processing arena
----------------------------------

There are 3 main components to a dvas processing arena: the
:ref:`raw data <raw_data>`,
the :ref:`database config files <config_data>`,
and the :ref:`core script <core_script>` of the recipe. Let's take a look at each of these.

.. _raw_data:

The data folder
...............
This folder contains all the raw data to be processed. THe actual structuring of subfolders inside
`./data` does not actually matter to dvas, so feel free to organize your datasets as you please.
Note, however, the following restrictions:

  - for non-GDP radiosondes: a `.yml` text file **with the same name** as the raw data file must be
    be specified for all datasets. These `.yml` contain all the metadata not otherwise present in
    the raw data files.

.. _config_data:

The config folder
.................

This folder contains all the information required to setup the dvas database, and have it ingest all
all the data. Modifying these files is only required if one wishes to include datasets that differ
from those already supported by the recipe.

.. _core_script:

The recipe core script
......................

This Python script contains the actual processing steps of the recipe. It is build upon high-level
routines and modules from the ``dvas_recipes`` sub-package, that themselves rely on core ``dvas``
modules and functions.



Specifc recipes
----------------

demo
....

Illustrates many of the core ``dvas`` tools and how to use them. This is a good starting point for
users interested to modify existing recipes, or create their own.


uaii22
......
Performs the official UAII 2022 analysis for the field campaign data.
