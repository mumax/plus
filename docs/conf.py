# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath("../mumaxplus"))

# -- Project information -----------------------------------------------------

project = "mumaxplus"
author = "Diego De Gusem, Oleh Kozynets, Ian Lateur, Lars Moreels, Jeroen Mulkers"
release = "1.0.2"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc", 
    "sphinx.ext.autosummary", 
    "numpydoc", 
]

# The default options for autodoc directives. They are applied to all autodoc directives
# automatically. It must be a dictionary which maps option names to the values.
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "special-members": "__call__, __getitem__",
}

# Boolean indicating whether to scan all found documents for autosummary directives, and
# to generate stub pages for each.
autosummary_generate = True

# If True, methods and attributes will be shown twice
numpydoc_show_class_members = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
