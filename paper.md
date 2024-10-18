---
title: 'Dynamicall: Tools for Modeling Resolved Galaxies Using Jax and Numpyro'
tags:
  - Python
  - Astronomy
  - Dynamics
  - Galactic Dynamics
  - Dwarf Galaxies
authors:
  - name: Juan Guerra
    orcid: 0000-0002-7600-5110
    # equal-contrib: true
    affiliation: '1' # (Multiple affiliations must be quoted)

  - name: Marla Geha
    orcid: 0000-0002-7007-9725
    affiliation: '1'

affiliations:
 - name: Yale University
   index: 1

date:  20 September 2024
bibliography: paper.bib

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Studying the underlying dark matter distribution of galaxies relies on single snapshots of kinematic tracers.
In the regime where individual stars can be resolved various methods have been developed to infer the underlying mass distribution of these galaxies [@Jeans1915-an; @Binney2011-ja; @Mamon2013-bi; @Cappellari2015-jt; @Read2017-nl; @Diakogiannis2019-yd].


We present a new tool, `Dynamicall`, a pure python package for modeling the dynamics of galaxies using `Jax` and `Numpyro`.
`Dynamicall` provides a suite of functions for modeling the dynamics of galaxies including functions for all basic dynamical quantities like density, mass, potential, distribution functions, actions, J-Factors and D-Factors. 
A variety of common models used in astronomy are implemented for ease of use such as Plummer, NFW, and Hernquist-Zhao profiles.
We provide a flexible and extensible framework for users to implement their own models with a minimal amount of code.
Leveraging `Jax` and `Numpyro` allows for the just-in-time (JIT) compilation, GPU acceleration and automatic differentiation, greatly speeding up the process of fitting models to data.

We showcase the use of `Dynamicall` by implementing spherical Jeans modeling and apply it to a mock data set of a dwarf galaxy.
We additionally show 1) how to use the Fisher information matrix to forecast the uncertainties on model parameters, 2) how to use built in functions to generate mock data sets to test the code, and 3) how to build additional models using the base classes provided.

# Statement of need

The use of GPU accelerated codes, automatic differentiation and Hamiltonian Monte Carlo has become increasily widely adopted in the field of astronomy.
As datasets become larger and more complex, the need for efficient and scalable tools to model the dynamics of galaxies has become more pressing.
Although there are many tools available to model the dynamics of galaxies, many of them are not built with the latest advances in computational tools in mind and require significant rewrites to take advantage of these tools.

Additionally the ability to calculate the gradients of likelihood functions leads to the ability to forecast uncertaints on model parameters via Fisher information matrix calculations [@Fisher1935-rf; @Rao1945-ep; @Guerra2021-ir].
This has  been tradionally done with finite difference methods which can require a lot of computational time, but can also be very sensitive to the choice of step size.
The use of automatic differentiation allows for the calculation of the gradients of the likelihood function with respect to the model parameters to be done with machine precision and in a fraction of the time as it would not require various call to the likelihood function with different step sizes.

We provide spherical mass modeling as a built in example of the capabilities of `Dynamicall` but the framework is also flexible enough to be extended to other models.

# software citations
`Dynamicall` uses the following software packages:

- `Jax` [@jax2018github]
- `Numpyro` [@phan2019composable; @bingham2019pyro]
- `matplotlib` [@Hunter:2007]
- `emcee` [@emcee]
- `corner` [@corner]
- `arviz` [@arviz_2019]
- `astropy` [@astropy:2013; @astropy:2018; @astropy:2022]
- `pytest` [@pytestx.y]
- `agama` [@Vasiliev2019-aa]

# Acknowledgements

# References