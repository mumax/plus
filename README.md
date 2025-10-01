<!-- markdownlint-disable MD028 MD033 MD034 -->

# mumax⁺

A versatile and extensible GPU-accelerated micromagnetic simulator written in C++ and CUDA with a Python interface. This project is in development alongside its popular predecessor [mumax³](https://github.com/mumax/3).
If you have any questions, feel free to use the [mumax mailing list](https://groups.google.com/g/mumax2).

**Documentation, tutorials and examples can be found on the [mumax⁺ website](https://mumax.github.io/plus/).**

## Paper

mumax⁺ is described in the following paper:
> mumax+: extensible GPU-accelerated micromagnetics and beyond
>
> https://arxiv.org/abs/2411.18194

Please cite this paper if you would like to cite mumax⁺.
All demonstrations in the paper were simulated using version [v1.1.0](https://github.com/mumax/plus/tree/v1.1.0) of the code. The scripts used to generate the data can be found in the [paper2025 directory](https://github.com/mumax/plus/tree/paper2025/paper2025) under the [paper2025 tag](https://github.com/mumax/plus/tree/paper2025).

## Installation

### Dependencies

You should install the following tools yourself. Click the arrows for more details.

<details><summary>CUDA Toolkit <i>(version 10.0 or later)</i></summary>

* **Windows**: Download an installer from [the CUDA website](https://developer.nvidia.com/cuda-toolkit-archive).
* **Linux**: Use `sudo apt-get install nvidia-cuda-toolkit`, or [download an installer](https://developer.nvidia.com/cuda-toolkit-archive).

> ⚠️ Make especially sure that everything CUDA-related (like `nvcc`) can be found inside your PATH.
> On Linux, for instance, this can be done by editing your `~/.bashrc` file and adding the following lines:
>
> ```bash
> # add CUDA
> export PATH="/usr/local/cuda/bin:$PATH"
> export LD_LIBRARY_PATH="/usr/local/cuda/> lib64:$LD_LIBRARY_PATH"
> ```
>
> The paths may differ if the CUDA Toolkit was installed in a different location.

👉 *Check CUDA installation with: `nvcc --version`*

</details>

<details><summary>A C++ compiler which supports C++17</summary>

* **Linux:** `sudo apt-get install gcc`
  * ⚠️ each CUDA version has a maximum supported `gcc` version. [This StackOverflow answer](https://stackoverflow.com/a/46380601) lists the maximum supported `gcc` version for each CUDA version. If necessary, use `sudo apt-get install gcc-<min_version>` instead, with the appropriate `<min_version>`.
* **Windows:**
  * CUDA does not support the `gcc` compiler on Windows, so download and install [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/) with the "Desktop development with C++" workload. After installing, check if the path to `cl.exe` was added to your `PATH` environment variable (i.e., check whether `where cl.exe` returns an appropriate path like `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.29.30133\bin\HostX64\x64`). If not, add it manually.

👉 *Check C installation with: `gcc --version` on Linux and `where.exe cl.exe` on Windows.*

</details>

<details><summary>Git</summary>

* **Windows:** [Download](https://git-scm.com/downloads) and install.
* **Linux:** `sudo apt install git`

👉 *Check Git installation with: `git --version`*

</details>

<details><summary>CPython <i>(version ≥ 3.8 recommended)</i>, pip and miniconda/anaconda</summary>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;All these Python-related tools should be included in a standard installation of [Anaconda or Miniconda](https://www.anaconda.com/download/success).

👉 *Check installation with `python --version`, `pip --version` and `conda --version`.*

</details>

### Building mumax⁺

First, clone the mumax⁺ Git repository. The `--recursive` flag is used in the following command to get the pybind11 submodule, which is needed to build mumax⁺.

```bash
git clone --recursive https://github.com/mumax/plus.git mumaxplus
cd mumaxplus
```

We recommend to install mumax⁺ in a clean conda environment as follows. You could also skip this step and use your own conda environment instead if preferred.

<details><summary>Click to show tools automatically installed in the conda environment</summary>

* cmake 4.0.0
* Python 3.13
* pybind11 v2.13.6
* NumPy
* matplotlib
* SciPy
* Sphinx

</details>

```bash
conda env create -f environment.yml
conda activate mumaxplus
```

Finally, build and install mumax⁺ using pip.

```bash
pip install .
```

> [!TIP]
> If changes are made to the code, then ``pip install -v .`` can be used to rebuild mumax⁺, with the `-v` flag enabling verbose debug information.
>
> If you want to change only the Python code, without needing to reinstall after each change, ``pip install -ve .`` can also be used.

> [!TIP]
> The source code can also be compiled with double precision, by changing `FP_PRECISION` in `CMakeLists.txt` from `SINGLE` to `DOUBLE` before rebuilding.
>
> ```cmake
> add_definitions(-DFP_PRECISION=DOUBLE) # FP_PRECISION > should be SINGLE or DOUBLE
> ```

<details><summary><h3>Troubleshooting</h3></summary>

* (*Windows*) If you encounter the error `No CUDA toolset found`, try copying the files in `NVIDIA GPU Computing Toolkit/CUDA/<version>/extras/visual_studio_integration/MSBuildExtensions` to `Microsoft Visual Studio/<year>/<edition>/MSBuild/Microsoft/VC/<version>/BuildCustomizations`. See [these instructions](https://github.com/NVlabs/tiny-cuda-nn/issues/164#issuecomment-1280749170) for more details.

</details>

## Documentation

Documentation for mumax⁺ can be found at http://mumax.github.io/plus.
It follows the [NumPy style guide](https://numpydoc.readthedocs.io/en/latest/format.html) and is generated using [Sphinx](https://www.sphinx-doc.org). You can build it yourself by running the following command in the `docs/` directory:

```bash
make html
```

The documentation can then be found at `docs/_build/html/index.html`.

## Examples

Lots of example codes are located in the `examples/` directory. They are either simple Python scripts, which can be executed inside said directory like any Python script

```bash
python standardproblem4.py
```

or they are interactive notebooks (`.ipynb` files), which can be run using Jupyter.

## Testing

Several automated tests are located inside the `test/` directory. Type `pytest` inside the terminal to run them. Some are marked as `slow`, such as `test_mumax3_standardproblem5.py`. You can deselect those by running `pytest -m "not slow"`. Tests inside the `test/mumax3/` directory require external installation of mumax³. They are marked by `mumax3` and can be deselected in the same way.

## Contributing

Contributions are gratefully accepted. To contribute code, fork our repo on GitHub and send a pull request.
