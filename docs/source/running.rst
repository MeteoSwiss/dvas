.. _running:

Running dvas
============

1. Initialize a new dvas *processing arena*. In a terminal, in a location of your choice, type:

.. code-block:: none

    dvas --init


Naming conventions
------------------

The following applies throughout dvas:

    * all radiosonde profiles must be associated to a unique **serial number** ``sn``.
    * all radiosonde profiles must be associated to a specific ``site``, ``event``, and ``rig``,
      which respectively encode the spatial, temporal, and payload configurations of the launch.
      This information must be provided to dvas via `tags` in the metadata associated to each dataset.

      .. hint::
          If two radiosondes are launched from the same location and at the same time, but with two
          distinct balloons, they have an identical **site** and **event** value, but distinct **rig**
          and **serial number** values.

    * GDPs are automatically associated to a specific **GDP model**.

Filtering syntax
----------------

The following grammatic rules allow to filter and extract specific datasets from the database:

.. code-block:: none

    - all(): Select all
    - [datetime|dt]('<ISO datetime>', ['=='(default)|'>='|'>'|'<='|'<'|'!=']): Select by datetime
    - [serialnumber|sn]('<Serial number>'): Select by serial number
    - [tag](['<Tag>'|('<Tag 1>', ...,'<Tag n>')]): Select by tag
    - and_(<expr 1>, ..., <expr n>): Intersection
    - or_(<expr 1>, ..., <expr n>): Union
    - not_(<expr>): All - <expr>

.. rubric:: Examples

.. code-block:: python3

    # Extract all the profiles associated to the event tagged "1":
    filt_evt = "tag('e1')"

    # Extract all the profiles from a specifc launch time:
    filt_dt = "dt('20160715T120000Z', '==')"

    # Two or more conditions can also be combined, for example:
    filt_comb = "and_(%s,%s)" % (filt_evt, filt_dt)

    # With the suitable filter, the data can then be extracted from the database:
    data_t = time_mngr.load(filt, 'trepros1')
