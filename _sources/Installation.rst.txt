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
    mamba activate dynamicall

2. Clone the repository and install the dependencies

.. code-block:: bash

    git clone git@github.com:jguerra-astro/dynamicall.git
    cd dynamicall
    mamba env update -f dynamicall_env.yml
    pip install .

using Conda
-----------
1. if you havent already, install [conda](https://docs.conda.io/en/latest/miniconda.html)

.. code-block:: bash

    conda create -n dynamicall python=3.10 # or whatever you want to call your environment
    conda activate dynamicall

2. Clone the repository and install the dependencies

.. code-block:: bash

    git clone git@github.com:jguerra-astro/dynamicall.git
    cd dynamicall
    conda env update -f dynamicall_env.yml
    pip install. 

using venv
-----------
1. virtual environments are included in the standard library, so you don't need to install anything extra

.. code-block:: bash

    python -m venv dynamicall
    source dynamicall/bin/activates

2. Clone the repository and install the dependencies

.. code-block:: bash

    git clone git@github.com:jguerra-astro/dynamicall.git
    cd dynamicall
    pip install -r requirements.txt
    pip install .


If you're not installing in a new environment, you can skip the first step in each of the above methods.
The main python dependencies are `jax`, `jaxlib`, `jaxopt`, `numpyro`, `emcee`, `corner`, `arviz`, `astroquery`, `pynbody`, `colorcet`, and `agama`.


Python Dependencies
===================

We suggest you look at the `installation instructions for Jax <https://github.com/google/jax#installation>`_ to minimize errors.
Specially since installation instructions will vary depending on whether you have a gpu or not.
jax, jaxlib, jaxopt are required as well, but they should be installed as part of installing jax.

.. warning::
    Installing jax things incorrectly/out of order may lead to you having different version of jax and jaxlib, which are imcompatible with each other and will lead to errors.

.. note::
    We suggest that you **do not** attempt to install the *gpu* version of Jax if you are on a mac at this time.

Numpyro should be installable using pip, but if you have issues, they have more detailed installation instructions `here <https://num.pyro.ai/en/latest/getting_started.html#installation>`_.

Certain methods require Agama, which can be installed by cloning the GitHub repository e.g:

.. code-block:: bash

    git clone https://github.com/GalacticDynamics-Oxford/Agama
    cd Agama    
    pip install .

The installation of this is a bit messy so i'll eventually get rid of it, but for now it is required.

Other dependencies include:
emcee, corner, arviz, astroquery, pynbody, colorcet -- all of which can be installed with pip

Once you are done installing the dependencies, you're ready to install ``DynamicAll``.


Installing fron source
----------------------

``DynamicAll`` is currently not available on PyPI, but it can be installed by cloning the GitHub repository e.g:

.. code-block:: bash    
    
    $ git clone git@github.com:jguerra-astro/dynamicAll.git    
    $ cd dynamicAll        
    $ pip install .


