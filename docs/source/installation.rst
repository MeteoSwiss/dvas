
Installation
============

.. todo::

    Include a link to the pypi page in the very next sentence.

dvas is available on pypi, which should make its installation straighforward. In a terminal, type:

.. code-block:: python

   pip install dvas

And that should take care of things. dvas uses `semantic versioning <https://semver.org/>`_. 

The most recent release of dvas is also available for download from its 
`Github repository <https://github.com/MeteoSwiss-MDA/dvas/releases/latest/>`_.

Requirements
------------
dvas is compatible with the following python versions:

.. literalinclude:: ../../setup.py
    :language: python
    :lines: 26

Furthermore, dvas relies on a few external modules, which will be automatically installed by ``pip``
if required:

.. literalinclude:: ../../setup.py
    :language: python
    :lines: 27-40

Testing the installation
------------------------

.. todo::

    Describe how to test that dvas was properly installed.
 