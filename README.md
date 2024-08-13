# MuMaxPlus
GPU accelerated micromagnetic simulator.

# Dependencies

* MSVC 2019 (Windows)
* CUDA 10.x
* cmake>=3.24
* CPython 3
* pybind11
* NumPy
* matplotlib

# Installation from Source (Linux)

Make sure that the following applications and build tools are installed:
* c++ compiler which supports c++17
* CPython *(version 3.x recommended)* and pip 
* CUDA Toolkit *(version 10.0 or later)*
* cmake *(version 3.24 or later)*. This can be installed using pip.
* git

Make especially sure that everything CUDA-related (like `nvcc`) can be found inside your path. This can be done by editing your `~/.bashrc` file and adding the following lines.
```bash
# add CUDA
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
The paths might differ if CUDA Toolkit has been installed in a different location. If successful, a command such as `nvcc --version` should work.

Clone the mumaxplus git repository. The `--recursive` flag is used here to get the pybind11 submodule which is needed to build mumaxplus.
```
git clone --recursive https://github.ugent.be/mumax/mumaxplus && cd mumaxplus
```
Build and install mumaxplus using pip
```
pip install .
```
or, if you are planning to contribute to the development of mumaxplus, then we recommend to install miniconda or anaconda to install mumaxplus in a clean conda environment, and to put the pre-commit hooks in place as follows
```
conda env create -f environment.yml
conda activate mumaxplus
pip install -e .
pre-commit install
```
If changes are made to the c++ code, then `pip install -ve .` can be used to rebuild mumaxplus.

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
6. Build `mumaxplus` using `setuptools`
```
activate mumaxplus
python setup.py develop
```
or `conda`
```
conda activate mumaxplus
conda develop -b .
```

# Debugging (Windows)

Visual Studio (VS) can be used for debugging of the C++ host and device code. For this purpose it is important to have [NVIDIA Nsight Visual Studio Edition](https://docs.nvidia.com/gameworks/index.html#developertools/desktop/nsight/nvidia_nsight.htm), which is a VS extension. It should be automatically installed with NVIDIA CUDA Toolkit.

To start debugging session:

1. Launch `mumaxplus.sln` in VS, it is created automatically by cmake and placed in `mumaxplus/build`
2. When project is loaded, in *Solution Configurations* select *Debug*, *x64*
3. In *Solution Explorer* set `cmd` as startup project
4. For the `core` and `physics` projects set *CUDA C/C++->Device->Generate GPU Debug Information* to *Yes* using [project properties](https://docs.nvidia.com/nsight-visual-studio-edition/2020.2/cuda-build-run/index.html)
5. Rebuild the mumaxplus.sln solution with VS
6. Launch a debugging session via *Extensions->NSight->Start CUDA Debugging*


# Testing

Several automated tests are located inside the `test/` directory. Type `pytest` inside the terminal to run them. Some are marked as `slow`, such as `test_mumax3_standardproblem5.py`. You can deselect those by running `pytest -m "not slow"`. Tests inside the `test/mumax3/` directory require external installation of MuMax3. They are marked by `mumax3` and can be deselected in the same way.

# Building the documentation

Documentation for mumaxplus follows the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) and can be generated using [Sphinx](https://www.sphinx-doc.org). Run the following command in the docs directory to let Sphinx build the HTML documentation pages:
```
make html
```
The documentation can now be found at `docs/_build/html/index.html`.

# Contributing
Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
