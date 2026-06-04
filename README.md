<!-- markdownlint-disable MD028 MD033 MD034 -->

# mumax⁺

A versatile and extensible GPU-accelerated micromagnetic simulator written in C++ and CUDA with a Python interface. This project is in development alongside [mumax³](https://github.com/mumax/3).
If you have any questions, feel free to use the [mumax⁺ GitHub Discussions](https://github.com/mumax/plus/discussions).

**Installation instructions, documentation, tutorials and examples can be found on the [mumax⁺ website](https://mumax.github.io/plus/).**

## Paper

mumax⁺ is described in the following paper:
> mumax+: extensible GPU-accelerated micromagnetics and beyond
>
> https://www.nature.com/articles/s41524-025-01893-y

Please cite this paper if you would like to cite mumax⁺.
All demonstrations in the paper were simulated using version [v1.1.0](https://github.com/mumax/plus/tree/v1.1.0) of the code. The scripts used to generate the data can be found in the [paper2025 directory](https://github.com/mumax/plus/tree/paper2025/paper2025) under the [paper2025 tag](https://github.com/mumax/plus/tree/paper2025).

## Installation

See [INSTALL.md](INSTALL.md) or the [mumax⁺ website](https://mumax.github.io/plus/install.html) for full instructions, covering the 3 main ways you can run/install mumax⁺:

- online through [Google Colab](https://colab.research.google.com/github/mumax/plus/blob/master/examples/colab.ipynb),
- installing pre-built wheels from the [GitHub releases](https://github.com/mumax/plus/releases/latest),
- building from source.

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

## Floating-point precision

mumax⁺ can use either single or double floating-point precision.
This can be controlled by the command-line argument `--mumaxplus-fp-precision` and/or the environment variable `MUMAXPLUS_FP_PRECISION`.

See this [tutorial page](https://mumax.github.io/plus/tutorial/precision.html) or [example notebook](examples/precision.ipynb) for more details.

## Contributing

Contributions are gratefully accepted. To contribute code, fork our repo on GitHub and send a pull request.
