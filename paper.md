---
title: 'Dynamicall: Tools for Modeling Galaxies from individual stars Using Jax and Numpyro'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - dwarf galaxies
authors:
  - name: Juan Guerra
    orcid: 0000-0000-0000-0000
    # equal-contrib: true
    affiliation: '1' # (Multiple affiliations must be quoted)

  - name: Marla Geha
    orcid: 0000-0000-0000-0000
    affiliation: '1'

affiliations:
 - name: Yale University
   index: 1

date:  20 September 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
Studying the underlying dark matter distribution of dwarf galaxies relies on single snapshots of kinematic tracers (e.g. line-of-sight velocities of individual stars) to probe the velocity distribution of the galaxy
Various methods have been developed to infer the underlying mass distribution of these galaxies [@Jeans1915-an; @Binney2011-ja; @Mamon2013-bi; @Cappellari2015-jt; @Read2017-nl; @Diakogiannis2019-yd].


We present a new tool, `Dynamicall`; A pure python package for modeling the dynamics of galaxies using `Jax` and `Numpyro`.
`Dynamicall` provides a suite of functions for modeling the dynamics of galaxies including functions for all basic dynamical quantities like density, mass, potential, distribution functions, actions, J-Factors and D-Factors. 
A variety of common models used in astronomy are implemented for ease of use such as Plummer, NFW, and Hernquist-Zhao profiles.
We also provide a flexible and extensible framework for users to implement their own models with a minimal amount of code.
Leveraging `Jax` and `Numpyro` allows for the just-in-time (JIT) compilation, GPU acceleration and automatic differentiation,greatly speeding up the process of fitting models to data.

We showcase the use of `Dynamicall` by implementing spherical Jeans modeling and apply it to a mock data set of a dwarf galaxy
We additionally show 1) how to use the Fisher information matrix to forecast the uncertainties on model parameters, 2) how we can use the built in functions to generate mock data sets to test the code, and 3) how to build additional models using the base classes provided.





# Statement of need

In recent years the use of GPU accelerated codes as well as the use of automatic differentiation and Hamiltonian Monte Carlo has become increasily widely adopted in the field of astronomy.
As data sets become larger and more complex, the need for efficient and scalable tools to model the dynamics of galaxies has become more pressing.
Although there are many tools available to model the dynamics of galaxies, many of them are not built with the latest advances in computational tools in mind and would require significant rewrites to take advantage of these tools.

Additionally the ability to calculate the gradients of the likelihood functions built with these functions leads to the ability to forecast uncertaints on model parameters via Fisher information matrix calculations.
This has  been tradionally done with finite difference methods which besides not only require a lot of computational time, but can also be very sensitive to the choice of step size.
The use of automatic differentiation allows for the calculation of the gradients of the likelihood function with respect to the model parameters to be done with machine precision and in a fraction of the time as it would not require various call to the likelihood function with different step sizes.

We provide spherical mass modeling as a first example of the capabilities of `Dynamicall` but the framework is flexible enough to be extended to other models and other types of data sets.


# References