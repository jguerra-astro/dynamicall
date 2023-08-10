*************************
Installation Instructions
*************************

You'll have two options depending on where you are in your project.

1. If you are starting out knowing you'll need to use this package then you could use the the enviroment.yml file provided to create a conda enviroment with all the dependencies.

2. If you are in the middle of a project and already have a conda enviroment, you may just want to individually install the following dependencies.


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


