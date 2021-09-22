# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'species'
copyright = '2018--2021, Tomas Stolker'
author = 'Tomas Stolker'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx'
]

# Disable notebook timeout
nbsphinx_timeout = -1

# Allow errors from notebooks
nbsphinx_allow_errors = True

autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build',
                    'Thumbs.db',
                    '.DS_Store',
                    'tutorials/.ipynb_checkpoints/*']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

html_theme_options = {
    'path_to_docs': 'docs',
    'repository_url': 'https://github.com/tomasstolker/species',
    'repository_branch': 'main',
    'launch_buttons': {
        'binderhub_url': 'https://mybinder.org',
        'notebook_interface': 'jupyterlab',
    },
    'use_edit_page_button': True,
    'use_issues_button': True,
    'use_repository_button': True,
    'use_download_button': True,
    'logo_only': True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = '_static/species_logo.png'
html_search_language = 'en'

html_context = {'display_github': True,
                'github_user': 'tomasstolker',
                'github_repo': 'species',
                'github_version': 'main/docs/'}
