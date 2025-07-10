# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath("../mumaxplus"))

# -- Project information -----------------------------------------------------

project = "mumax⁺"
# "Diego De Gusem, Oleh Kozynets, Ian Lateur, Lars Moreels, Jeroen Mulkers"
author = "the DyNaMat group, Ghent University, Belgium."
release = "1.1.0"

date = datetime.datetime.now()
html_last_updated_fmt = f"{date.strftime("%Y")}-{date.strftime("%m")}-{date.strftime("%d")}, v{release}"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc", 
    "sphinx.ext.autosummary", 
    "sphinxcontrib.video",
    "sphinx_copybutton",
    "sphinx.ext.napoleon"
]

toc_object_entries_show_parents = 'hide'

# The default options for autodoc directives. They are applied to all autodoc directives
# automatically. It must be a dictionary which maps option names to the values.
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "special-members": "",
    "undoc-members": True,
    'show-inheritance': True,
}

autodoc_member_order = 'bysource'

# Boolean indicating whether to scan all found documents for autosummary directives, and
# to generate stub pages for each.
autosummary_generate = True

# If True, methods and attributes will be shown twice
numpydoc_show_class_members = False

# Suppress full names like mumaxplus.World → just World when False
add_module_names = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build"]

templates_path = ['_templates']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.

html_theme = "sphinx_book_theme"
html_logo = "_static/nimble-plus.png"
html_title = "mumax⁺"

html_static_path = ['_static']
html_css_files = ['logo.css', 'custom.css']

html_favicon = "_static/nimble-plus.png"

rst_epilog = """
.. |author| replace:: {}
""".format(author)

rst_epilog = """
.. |version| replace:: {}
""".format(release)

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

html_theme_options = {
    "show_toc_level": 2,
    "icon_links": [
        {
            "name": "DyNaMat research group",
            "url": "https://www.ugent.be/we/solidstatesciences/dynamat/en",
            "icon": "fas fa-university",  # FontAwesome university icon
            "type": "fontawesome",
        },
        {
            "name": "GitHub repository",
            "url": "https://github.com/mumax/plus",  # GitHub repo
            "icon": "fab fa-github",  #  
            "type": "fontawesome",
        },
        {
            "name": "Paper on arXiv",
            "url": "https://arxiv.org/abs/2411.18194",
            "icon": "fas fa-file-alt",  # generic document icon
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",  # screen reader label
    "use_download_button": False,
}