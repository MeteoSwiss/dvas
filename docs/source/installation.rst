.. include:: ./substitutions.rst

.. _install:

Installation
============

.. todo::

    Include a link to the pypi page in the very next sentence.

dvas will be available on pypi, which should make its installation straightforward.
In a terminal, type:

.. code-block:: python

   pip install dvas

And that will take care of things. dvas uses `semantic versioning <https://semver.org/>`_.
The latest stable version is |version|.

The most recent release of dvas is also available for download from its
`Github repository <https://github.com/MeteoSwiss-MDA/dvas/releases/latest/>`_.

Requirements
------------
dvas is compatible with the following python versions:

.. literalinclude:: ../../setup.py
    :language: python
    :lines: 40

Furthermore, dvas relies on a few external modules, which will be automatically installed by ``pip``
if required:

.. literalinclude:: ../../setup.py
    :language: python
    :lines: 41-56

Testing the installation
------------------------

To ensure that dvas was installed properly, try to call the help of its high-level entry point. In a
terminal, type:

.. code-block:: none

   dvas_init -h

This should return the following information:

.. literalinclude:: dvas_help_msg.txt
    :language: none
