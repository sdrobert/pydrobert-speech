# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------

project = "pydrobert-speech"
copyright = "2022, Sean Robertson"
author = "Sean Robertson"
language = "en"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.programoutput",
    "sphinx_autodoc_typehints",
]

naploeon_numpy_docstring = True

intersphinx_mapping = {
    "scipy": ("http://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pydrobert.kaldi": ("https://pydrobert-kaldi.readthedocs.io/en/latest", None),
    "python": ("https://docs.python.org/", None),
    "soundfile": ("https://python-soundfile.readthedocs.io/en/latest/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
autodoc_mock_imports = [
    "numpy",
    "matplotlib.axes",
    "matplotlib.colors",
    "matplotlib.figure",
    "matplotlib",
]
autodoc_typehints = "none"
autodoc_type_aliases = napoleon_type_aliases = {"np.ndarray": "numpy.ndarray"}
autodoc_inherit_docstrings = False
napoleon_preprocess_types = True
typehints_document_rtype = False
napoleon_use_rtype = False


# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


highlight_language = "none"

master_doc = "index"
