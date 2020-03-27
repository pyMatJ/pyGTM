# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import mock
sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'pyGTM'
copyright = '2020, Mathieu Jeannin'
author = 'Mathieu Jeannin'

# The full version, including alpha/beta/rc tags
release = '2.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 
              'sphinx.ext.autosummary',
#              	'sphinxcontrib.fulltoc',
#              'sphinx.ext.coverage', 
              'sphinx.ext.napoleon']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

## RTD expected the master file to be contents.rst
master_doc = 'index'

#### To mock the modules imported in the main project
MOCK_MODULES = ['numpy', 'scipy.optimize']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster' # or default

html_theme_options = {
	'fixed_sidebar': 'true',
}

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']