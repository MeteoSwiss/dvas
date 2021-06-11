.. _creating:

Creating dvas recipes
=====================

This page contains information for advanced dvas users that would like to alter existing
processing recipes, or create their own.

.. hint::

    If you plan on coding with dvas, we strongly suggest that you begin by taking a look at the
    `contributing guidelines <https://github.com/MeteoSwiss-MDA/dvas/blob/develop/CONTRIBUTING.md>`_
    on Github. Even if you do not want to share your modifications (which is totally fine !), these
    guidelines contain critical information regarding the structure and spirit of the code.



Naming conventions
------------------

Protected tag names
...................

The following tag names are used by dvas, and thus protected. They are defined in
`dvas/config/definitions/tag.py`:

    * ``raw``:
    * ``gdp``:
    * ``derived``:
    * ``empty``:
    * ``sync``:
    *  ``1s``:
    * ``resampled``:

Default database variable names
...............................

The following names can be altered by the users in the different config files. These are the ones
implemented by defaults. They follow MeteoSwiss conventions:

    * ``time``: time since launch
    * ``gph``: geopotential height
    * ``temp``: temperature
    * ``rh``: relative humidty
    * ``pres``: pressure
    * ``wdir``: wind direction
    * ``wspeed``: wind speed
    * ``xxx_ucr``: rig-uncorrelated uncertainty of xxx
    * ``xxx_ucs``: spatial-correlated uncertainty of xxx
    * ``xxx_uct``: temporal-correlated uncertainty of xxx
    * ``xxx_ucu``: true uncorrelated uncertainty of xxx

Parameter names
...............

For instances of `Profile`, `MultiProfile`, and their children, the following parameter (`prm`)
names are applicable (as defined in `dvas.hardcoded`):

   * ``val``: primary Profile value
   * ``alt``: altitude
   * ``tdt``: time delta
   * ``ucr``: rig-uncorrelated uncertainty
   * ``ucs``: spatial-correlated uncertainty
   * ``uct``: temporal-correlated uncertainty
   * ``ucu``: true uncorrelated uncertainty
   * ``flg``: flags

The following parameters are applicable to their event metadata:

 * ``oid``: object ID, unique for profiles with the same radiosonde serial number and product ID
 * ``eid``: measurement event ID
 * ``rid``: rig ID (in case of multiple flights)
 * ``mid``: the GDP model ID

All radiosonde profiles must be associated to a specific ``event`` and ``rig``,
which (respectively) encode the spatial+temporal and payload configurations of the launch.
This information must be provided to dvas via `tags` in the metadata associated to each dataset.

.. hint::
    If two radiosondes are launched from the same location and at the same time, but with two
    distinct balloons, they have an identical **event** value, but distinct **rig** and
    **object ID** values.

Event filtering syntax
----------------------

The following grammatic rules allow to filter and extract specific event datasets from the database:

.. code-block::

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

 # Extract all the profiles from a specific launch time:
 filt_dt = "dt('20160715T120000Z', '==')"

 # Two or more conditions can also be combined, for example:
 filt_comb = "and_(%s,%s)" % (filt_evt, filt_dt)

 # With the suitable filter, the data can then be extracted from the database:
 data_t = time_mngr.load(filt, 'trepros1')

Plotting tools
--------------

A lot of efforts were invested in the dvas plots, with the aim to produce high quality material that
is "publication ready" out of the box. For examples, dvas plots come in two fixed widths (in inches)
of  ``dvas.plots.utils.WIDTH_ONECOL = 6.92`` and ``dvas.plots.utils.WIDTH_ONECOL = 14.16``.
When scaled by 50%, these can be directly used as 1-column and 2-column plots (respectively) in
scientific articles.

Here's a few things you can do as a dvas user to control the general plotting behavior of the code.

.. code-block:: python3

    # Let us import the required sub-module
    import dvas.plots.utils as dpu

    # You can drastically improve the look of the dvas plots by using your system-wide LaTeX
    # distribution (which must evidently be installed properly). Use it at your own risk.
    dpu.set_mplstyle('latex')

    # If you want to go back to the default matplotlib LaTeX, run
    #dpu.set_mplstyle('nolatex')

    # You can alter the default formats the plots will be saved in via dpu.PLOT_FMTS.
    # The defaults formats is 'png'
    dpu.PLOT_FMTS = ['png', 'pdf']
    # If you do not want to save anything, set:
    #dpu.PLOT_FMTS = []


Each plotting function can also be fed a series of ``**kwargs`` keywords arguments. The following
three will let you better control the filenames and formats of the plots generated by dvas:

    * ``fn_prefix (str)``: a str to which the nominal plot filename gets appended.
    * ``fn_suffix (str)``: a str that gets appended to the plot filename.
    * ``fmts (str or list of str)``: will override ``dpu.PLOT_FMTS`` for this one plot only.
    * ``show_plt (bool)``: will override ``dpu.PLOT_SHOW`` to show the plot on screen (or not).
