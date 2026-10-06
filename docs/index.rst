:nosearch:

mumax\ :sup:`+`
=========================

Welcome to the documentation!
-----------------------------

`mumax⁺ <https://github.com/mumax/plus>`_ is a versatile and extensible
GPU-accelerated micromagnetic simulator written in C++ and CUDA with a Python
interface. This project is in development alongside
`mumax³ <https://github.com/mumax/3>`_. If you have any questions, feel free to 
use the `mumax⁺ GitHub Discussions <https://github.com/mumax/plus/discussions>`_.

Citations
---------

mumax\ :sup:`+` is described in the following paper:

    mumax+: extensible GPU-accelerated micromagnetics and beyond

    https://www.nature.com/articles/s41524-025-01893-y

Please cite this paper if you would like to cite mumax\ :sup:`+`.
All demonstrations in the paper were simulated using version `v1.1.0 <https://github.com/mumax/plus/tree/v1.1.0>`_ of the code. The scripts used to generate the data can be found in the `paper2025 directory <https://github.com/mumax/plus/tree/paper2025/paper2025>`_ under the `paper2025 tag <https://github.com/mumax/plus/tree/paper2025>`_.

GPU
---

mumax\ :sup:`+` is cross-platform and runs on Linux and Windows platforms. You need an 
NVIDIA GPU with compute capability 5.2 or higher, as listed `here <https://developer.nvidia.com/cuda/gpus>`_. You also need to use 
NVIDIA's proprietary graphics driver, which may already be installed on your system. 
The benchmark below may guide your GPU choice.

  .. raw:: html
     :file: _static/bench.html

If you want to contribute your GPU benchmark to this figure you can run
:file:`examples/bench.py` and send us the :file:`bench.txt` output file.

Contents
--------

.. toctree::
    :maxdepth: 1

    install

.. toctree::
    :maxdepth: 3
    :titlesonly:

    api

.. toctree::
    :maxdepth: 3
    :titlesonly:

    tutorial

.. toctree::
    :maxdepth: 2

    examples

.. toctree::
    :maxdepth: 2

    class_diagrams