.. _installation:

Installation
============

*species* is compatible with Python 3.7/3.8/3.9 and is available in the `PyPI repository <https://pypi.org/project/species/>`_ and on `Github <https://github.com/tomasstolker/species>`_.

Installation from PyPI
----------------------

The Python package can be installed with the `pip package manager <https://packaging.python.org/tutorials/installing-packages/>`_:

.. code-block:: console

    $ pip install species

Or, to update to the most recent version:

.. code-block:: console

   $ pip install --upgrade species

Please check for any errors and warnings during the installation to make sure that all dependencies are correctly installed.

.. important::
   Before installing ``species``, it is required to separately install ``cython``:

   .. code-block:: console

       $ pip install cython

.. important::
   The ``PyMultiNest`` package requires the manual installation of ``MultiNest``. Please follow the `instructions <https://johannesbuchner.github.io/PyMultiNest/install.html>`_ for the building the library and make sure that the ``LD_LIBRARY_PATH`` (on Linux) or ``DYLD_LIBRARY_PATH`` (on macOS) environment variable is set. It is also possible to use ``species`` without installing ``MultiNest`` (but a warning will appear), apart from the functionalities that rely on ``PyMultiNest``.   

Installation from Github
------------------------

Installation from Github is done by cloning the repository:

.. code-block:: console

    $ git clone git@github.com:tomasstolker/species.git

And running the setup script to install the package and its dependencies:

.. code-block:: console

    $ python setup.py install

.. important::
   If an error occurs when running ``setup.py`` then possibly ``pip`` needs to be updated to the latest version:

   .. code-block:: console

       $ pip install --upgrade pip

Alternatively to running the ``setup.py`` file, the folder where ``species`` is located can also be added to the ``PYTHONPATH`` environment variable such that the package is found by Python. The command may depend on the OS that is used, but is typically something like:

.. code-block:: console

    $ export PYTHONPATH=$PYTHONPATH:/path/to/species

New commits can be pulled from Github once a local copy of the repository exists:

.. code-block:: console

    $ git pull origin main

Do you want to make changes to the code? Please fork the `species` repository on the Github page and clone your own fork instead of the main repository. Contributions and pull requests are very welcome (see :ref:`contributing` section).

Testing `species`
-----------------

The installation can now be tested, for example by starting Python in interactive mode and printing the version number of the installed package:

.. code-block:: python

    >>> import species
    >>> species.__version__
