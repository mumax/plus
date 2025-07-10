:nosearch:

Installation
============

Dependencies
------------

You should install these yourself

* CUDA Toolkit 10.0 or later
* A C++ compiler which supports C++17, such as GCC
* On Windows (good luck): MSVC 2019

These will be installed automatically within the conda environment

* cmake 4.0.0
* Python 3.13
* pybind11 v2.13.6
* NumPy
* matplotlib
* SciPy
* Sphinx

Linux
-----

Make sure that the following applications and build tools are installed:

* C++ compiler which supports c++17, such as GCC
* CPython (version ≥ 3.8 recommended) and pip
* CUDA Toolkit (version 10.0 or later)
* git
* miniconda or anaconda

Make especially sure that everything CUDA-related (like ``nvcc``) can be found
inside your path. This can be done by editing your ``~/.bashrc`` file and adding
the following lines.

.. code-block:: bash

    # add CUDA
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

The paths might differ if CUDA Toolkit has been installed in a different location.
If successful, a command such as ``nvcc --version`` should work.

Clone the mumax\ :sup:`+` git repository. The ``--recursive flag`` is used here to get
the pybind11 submodule which is needed to build mumax\ :sup:`+`.

.. code-block:: bash

    git clone --recursive https://github.com/mumax/plus.git mumaxplus && cd mumaxplus

We recommend to install mumax\ :sup:`+` in a clean conda environment. You could also skip
this step and use your own conda environment instead if preferred.

.. code-block:: bash

    conda env create -f environment.yml
    conda activate mumaxplus

Then build and install mumax\ :sup:`+` using pip.

.. code-block:: bash

    pip install -v .

If changes are made to the code, then ``pip install -v .`` can be used to
rebuild mumax\ :sup:`+`. If you want to change the Python code without needing
to reinstall, you can use ``pip install -ve .``.

You could also compile the source code with double precision, by changing
``FP_PRECISION`` in ``CMakeLists.txt`` from ``SINGLE`` to ``DOUBLE`` before
rebuilding.

.. code-block:: bash

    add_definitions(-DFP_PRECISION=DOUBLE) # FP_PRECISION should be SINGLE or DOUBLE

Windows
-------

.. warning::

    These instructions are old and worked at some point (2021), but not today.
    If you are brave enough to try Windows and you manage to get it working,
    please let us know!

#. Install Visual Studio 2019 and the desktop development with C++ workload

#. Install CUDA Toolkit 10.x

#. Install cmake

#. Download the pybind11 submodule with git

   .. code-block:: bash

      git submodule init
      git submodule update

#. Install Python packages using conda

   .. code-block:: bash

      conda env create -f environment.yml

#. Build ``mumaxplus`` using ``setuptools``

   .. code-block:: bash

      activate mumaxplus
      python setup.py develop

   or ``conda``

   .. code-block:: bash

      conda activate mumaxplus
      conda develop -b .