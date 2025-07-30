# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Paths -----------------------------------------------------
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('../../'))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'agentlib_flexquant'
copyright = '2025, Felix Stegemerten'
author = 'Felix Stegemerten'
release = '0.1.0'

# The full version, including alpha/beta/rc tags
with open(Path(__file__).parents[2].joinpath(project, "__init__.py"), "r") as file:
    for line in file.readlines():
        if line.startswith("__version__"):
            release = (
                line.replace("__version__", "")
                .split("=")[1]
                .strip()
                .replace("'", "")
                .replace('"', "")
            )

# The short X.Y version.
version = ".".join(release.split(".")[:2])
# The full version, including alpha/beta/rc tags.
release = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.coverage",
    # "m2r2",  # Enable .md files
    "myst_parser",
    "sphinx.ext.napoleon",  # Enable google docstrings
    "sphinxcontrib.autodoc_pydantic",
]

templates_path = ['_templates']
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config = False
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# Material theme options (see theme.conf for more information)
html_theme_options = {
    # If False, expand all TOC entries
    "globaltoc_collapse": True,
    # If True, show hidden TOC entries
    "globaltoc_includehidden": False,
}
html_static_path = []
