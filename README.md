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
or, if you are planning to contribute to the development of mumax5, then we recommend to install miniconda or anaconda to install mumax5 in a clean conda environment
```
conda env create -f environment.yml
conda activate mumax5
pip install -e .
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

# Contributing
Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
