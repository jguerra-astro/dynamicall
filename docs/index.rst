Dynamicall
===================================================

The internal kinematics of dwarf galaxies is a powerful tool to understand their formation and evolution as well as put constraints on dark matter models.

Dynamicall is an open-source python package that provides a set of tools to analyze the internal kinematics of dwarf galaxies and other dynamical systems.


The author's use case is as a Jeans modeling tool, but it also provides a flexbile set of additional tools to analyze the internal kinematics of dwarf galaxies and other dynamical systems.
The main developments over other similar tools are the use of `Jax <https://jax.readthedocs.io/en/latest/#>`__ for all calculations leading to a significant speedup from just-in-time(JIT) compilation and gpu acceleraton, and the compatibility with`Numpyro <http://num.pyro.ai/en/stable/>`_ for the Bayesian inference.
The use of Jax also allows us to use automatic differentiation to calculate the gradients of the posterior probability distribution with respect to the parameters, which is necessary to use `Hamiltonian Monte Carlo (HMC) <https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo>`__ methods for the inference, and useful in order to Forecast uncertainties in the parameters using Fisher Matrix methods.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Installation
   notebooks/numerical_methods
   notebooks/tutorial_potential
   notebooks/tutorial_plummer
   notebooks/tutorial_jfactors
   notebooks/tutorial_dracoish
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Caveats
=======

As will be explained several times throughout the documentation, when using this in python scripts or notebooks, you'll want to include the following lines at the beginning of your script/notebook:

.. code-block:: python

   import jax
   from jax._src.config import config
   config.update("jax_enable_x64", True)

If that becomes annoying, you can also add the following lines to your .bashrc/.zshrc file (or whatever file is appropriate for your shell):

.. code-block:: bash 
   
   export JAX_ENABLE_X64=True


Units
-----

For Now, all units are in kpc, km/s, and Msun. This will be changed in the future to allow for more flexibility.



Contributors
============

**Main Author:** Juan Guerra