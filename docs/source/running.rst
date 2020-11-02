.. _running:

Running dvas
============

This page is very much a WIP. Feel free to add/restructure this as you see fit.

Naming conventions
------------------

The following applies throughout dvas:

    * all radiosonde profiles must be associated to a unique **serial number**.
    * all radiosonde profiles must be associated to a specific **site**, **event**, and **rig**,
      which respectively encode the spatial, temporal, and payload configurations of the launch.
      This information must be provided to dvas via `tags` in the metadata associated to each dataset.

      .. hint::
          If two radiosondes are launched from the same location and at the same time, but with two
          distinct balloons, they have an identical **site** and **event** value, but distinct **rig**
          and **serial number** values.

    * GDPs are automatically associated to a specific **GDP model**.

Filtering syntax
----------------

The following are valid filtering syntax for querying the dvas database:

.. code-block:: python3

    filter = "tag('e1')" # Extracts all the profiles associated to the event tagged "1".
