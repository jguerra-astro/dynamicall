---
title: 'Dynamicall: Tools for Modeling Galaxies Using Jax and Numpyro'
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
Studying the underlying dark matter distribution of dwarf galaxies relies on single kinematic snapshots of the stars in these galaxies i.e. the line-of-sight (usually) velocity distribution at one time.
Given these observations there have been various methods developed to infer the underlying mass distribution of these galaxies.

We present a new tool, `Dynamicall`, which is built on top of the `Jax` and intented to be use with MCMC samplers like `Numpyro` which are able to take advantage of all the benefits of Jax.
This tool provides a suite of functions for modeling the dynamics of galaxies.
We provide functions for all the basic dynamical functions like density,mass, potential, distribution functions, actions for a variety of common models used in astronomy.
We also provide a flexible and extensible framework for users to implement their own models with a minimal amount of code.
The ability to use `Jax` and `Numpyro` allows for the use of GPU acceleration and automatic differentiation which can greatly speed up the process of fitting models to data.

We showcase the use of `Dynamicall` by implementing the well know spherical Jeans modeling an apply it to a mock data set of a dwarf galaxy.
We additionally show how to use the Fisher information matrix to forecast the uncertainties on the model parameters.
How we can use the built in functions to generate mock data sets to test the code.
How to build additional models using the base classes provided.



# Statement of need

In recent years the use of GPU accelerated codes as well as the use of automatic differentiation and hamiltonian monte carlo has become increasily widely adopted in the field of astronomy.
As data sets become larger and more complex, the need for efficient and scalable tools to model the dynamics of galaxies has become more pressing.
Although there are many tools available to model the dynamics of galaxies, many of them are not built with the latest advances in computational tools in mind and would require significant rewrites to take advantage of these tools.

Additionally the ability to calculate the gradients of the likelihood functions built with this functions leads to the ability to forecast uncertaints on model parameters via Fisher information matrix calculations. This has  been tradionally done with finite difference methods which besides also requiring a lot of computational time, can be very sensitive to the choice of step size.
The use of automatic differentiation allows for the calculation of the gradients of the likelihood function with respect to the model parameters to be done with machine precision and in a fraction of the time as it would not require various call to the likelihood function with different step sizes.

We provide spherical mass modeling as a first example of the capabilities of `Dynamicall` but the framework is flexible enough to be extended to other models and other types of data sets.




# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)


# References