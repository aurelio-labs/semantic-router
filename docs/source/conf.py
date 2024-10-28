from typing import List
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Semantic Router"
copyright = "2024, Aurelio AI"
author = "Aurelio AI"
release = "0.0.72"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinxawesome_theme"]

templates_path = ["_templates"]
exclude_patterns: List[str] = []
autosummary_generate = True
numpydoc_show_class_members = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = "<span>#</span>"
html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
