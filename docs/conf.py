# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import sys
import os

sys.path.insert(0, os.path.abspath("../src"))


html_use_index = True
html_domain_indices = True
html_baseurl = "https://jguerra-astro.github.io/dynamicall/"
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "dynamicAll"
copyright = "2024, Juan Guerra"
author = "Juan Guerra"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

mathjax_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.min.js"
)
utomodule = "dynamicAll"

extensions = [
    "sphinx.ext.autodoc",  # Automatically generates documentation from docstrings -- Required
    "sphinx.ext.viewcode",  # Adds links to the source code of documented Python objects -- Optional,but recommended
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings -- Optional, but highly recommended
    "sphinx.ext.autosummary",  # Generate autodoc summaries -- Optional
    "sphinx.ext.intersphinx",  # Link to other Sphinx documentation -- Optional
    "nbsphinx",
    "sphinx.ext.mathjax",
    "myst_nb",  # myst_nb is a plugin for MyST Markdown that adds support for executing and displaying Jupyter notebooks.
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    # "sphinxcontrib-fulltoc", requires pip install sphinxcontrib-fulltoc
    # "sphinx_panels",
    "sphinx.ext.todo",
    # "sphinx_new_tab_link",
]
nbsphinx_execute = "off"
nb_execution_mode = "off"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")
mathjax3_config = {
    "tex": {"tags": "ams", "useLabelIds": True},
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"  # requires pip install pydata-sphinx-theme
html_static_path = ["_static"]

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/jguerra-astro/dynamicall",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
    ]
}
autodoc_mock_imports = ["agama"]
