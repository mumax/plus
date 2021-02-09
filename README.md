# mumax5
GPU accelerated micromagnetic simulator.

# Dependencies

* MSVC 2019 (Windows)
* CUDA 10.x
* cmake>=3.18
* CPython 3
* pybind11
* NumPy
* matplotlib

# Installation from Source (Linux)

Make sure that the following applications and build tools are installed:
* c++ compiler which supports c++17
* CPython *(version 3.x recommended)* and pip 
* CUDA Toolkit *(version 10.0 or later)*
* cmake *(version 3.18 or later)*. This can be installed using pip.
* git

Clone the mumax5 git repository. The `--recursive` flag is used here to get the pybind11 submodule which is needed to build mumax5.
```
git clone --recursive https://github.ugent.be/mumax/mumax5 && cd mumax5
```
Build and install mumax5 using pip
```
pip install .
```
or, if you are planning to contribute to the development of mumax5, then we recommend to install miniconda or anaconda to install mumax5 in a clean conda environment, and to put the pre-commit hooks in place as follows
```
conda env create -f environment.yml
conda activate mumax5
pip install -e .
pre-commit install
```
If changes are made to the c++ code, then `pip install -ve .` can be used to rebuild mumax5.

# Installation from Source (Windows)

1. Install Visual Studio 2019 and the desktop development with C++ workload
2. Install CUDA Toolkit 10.x
3. Install cmake
4. Download the pybind11 submodule with git
```
git submodule init
git submodule update
```
5. Install Python packages using conda
```
conda env create -f environment.yml
```
6. Build `mumax5` using `setuptools`
```
activate mumax5
python setup.py develop
```
or `conda`
```
conda activate mumax5
conda develop -b .
```

# Debugging (Windows)

Visual Studio (VS) can be used for debugging of the C++ host and device code. For this purpose it is important to have [NVIDIA Nsight Visual Studio Edition](https://docs.nvidia.com/gameworks/index.html#developertools/desktop/nsight/nvidia_nsight.htm), which is a VS extension. It should be automatically installed with NVIDIA CUDA Toolkit.

To start debugging session:

1. Launch `mumax5.sln` in VS, it is created automatically by cmake and placed in `mumax5/build`
2. When project is loaded, in *Solution Configurations* select *Debug*, *x64*
3. In *Solution Explorer* set `cmd` as startup project
4. For the `core` and `physics` projects set *CUDA C/C++->Device->Generate GPU Debug Information* to *Yes* using [project properties](https://docs.nvidia.com/nsight-visual-studio-edition/2020.2/cuda-build-run/index.html)
5. Rebuild the mumax5.sln solution with VS
6. Launch a debugging session via *Extensions->NSight->Start CUDA Debugging*

# Building the documentation

Documentation for mumax5 can be generated using [Sphinx](https://www.sphinx-doc.org). Run the following command in the docs directory to let Sphinx build the HTML documentation pages:
```
make html
```
The documentation can now be found at `docs/_build/html/index.html`.

# Contributing
Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
