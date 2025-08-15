# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tropygal'
copyright = '205, Leandro Beraldo e Silva'
author = 'Leandro Beraldo e Silva'
release = '0.1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx.ext.viewcode",
              "nbsphinx"]

nbsphinx_prolog = ""
nbsphinx_epilog = ""

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinxdoc'
#html_theme = 'agogo'
html_static_path = ['_static']

master_doc = 'index'
