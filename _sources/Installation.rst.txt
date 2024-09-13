*************************
Installation Instructions
*************************

This guide provides instructions for installing `Dynamicall` using different Python environment management tools.
We recommend using Mamba for the fastest and most efficient installation, but instructions for Conda and venv are also provided.

Installation
============
The recommended way of using `Dynamicall` is to install it in a fresh virtual environment.


using Mamba
-----------

1. if you havent already, install [mamba](https://mamba.readthedocs.io/en/latest/)

.. code-block:: bash

    mamba create -n dynamicall python=3.10 # or whatever you want to call your environment
    mamba activate 

.. note::
    Installing agama will require a few "y (or n)" inputs from you, some of which may take a few minutes to complete.

.. code-block:: bash
    
    pip install jax jaxopt agama emcee corner

.. code-block:: bash

    mamba install scipy astropy numpyro matplotlib arviz astroquery scikit-learn
    


2. Clone the repository in the directory of your choice and install the dependencies

.. code-block:: bash

    git clone git@github.com:jguerra-astro/dynamicall.git
    cd dynamicall
    pip install -e .

using Conda
-----------
1. if you havent already, install [conda](https://docs.conda.io/en/latest/miniconda.html)

.. code-block:: bash

    conda create -n dynamicall python=3.10 # or whatever you want to call your environment
    conda activate dynamicall

.. note::
    Installing agama will require a few "y (or n)" inputs from you, some of which may take a few minutes to complete.

.. code-block:: bash
    
    pip install jax jaxopt agama emcee corner

.. code-block:: bash

    conda install scipy astropy numpyro matplotlib arviz astroquery scikit-learn

2. Clone the repository in the directory of your choice and install the dependencies

.. code-block:: bash

    git clone git@github.com:jguerra-astro/dynamicall.git
    cd dynamicall
    pip install -e .


testing
-------
To test the installation run the following commands:

.. code-block:: bash

    python

.. code-block:: python

    import dynamicAll

Python Dependencies
===================

We suggest you look at the `installation instructions for Jax <https://github.com/google/jax#installation>`_ to minimize errors.
Specially since installation instructions will vary depending on whether you have a gpu or not.
jax, jaxlib, jaxopt are required as well, but they should be installed as part of installing jax.

.. warning::
    Installing jax things incorrectly/out of order may lead to you having different version of jax and jaxlib, which are imcompatible with each other and will lead to errors.

.. note::
    We suggest that you **do not** attempt to install the *gpu* version of Jax if you are on a mac at this time.