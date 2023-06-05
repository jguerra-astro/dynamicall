import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from jax._src.config import config
from jaxopt import Bisection
from scipy.optimize import root
from scipy.stats import binned_statistic
import scipy
import agama
from scipy.spatial.transform import Rotation
from astropy.stats import histogram
config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from typing import Callable
from numpyro.infer import MCMC, NUTS
from jax import random

import matplotlib.pyplot as plt
import corner
import arviz as az
from functools import partial
from numpyro.distributions import Distribution
from numpyro.distributions import constraints
from abc import ABC, abstractmethod
from jax import lax, vmap
from jax.lax import scan
from jax import random





# # Generate some random data
# data = jnp.concatenate([
#     dist.Normal(-3, 1).sample((100,)),
#     dist.Normal(3, 1).sample((100,))
# ])

# # Define the model
# def gaussian_mixture_model(data):
#     # Define the mixture weights
#     weights = numpyro.sample("weights", dist.Dirichlet(jnp.ones(2)))

#     # Define the means and standard deviations of the two Gaussians
#     means = numpyro.sample("means", dist.Normal(jnp.array([-5., 5.]), jnp.array([1., 1.])))
#     stddevs = numpyro.sample("stddevs", dist.HalfNormal(jnp.array([1., 1.])))

#     # Define the mixture distribution
#     component_dist = dist.Normal(means, stddevs)
#     mixture_dist = dist.MixtureSameFamily(dist.Categorical(weights), component_dist)

#     # Sample from the mixture distribution
#     numpyro.sample("obs", mixture_dist, obs=data)

# # Run inference
# nuts_kernel = NUTS(gaussian_mixture_model)
# mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
# mcmc.run(jnp.array(data))
# mcmc.print_summary()




# class MagJ(Distribution):
#     # arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
#     # support = constraints.real
#     # reparametrized_params = ["loc", "scale"]

#     def __init__(self,weights = 1,actions =1, scale=1, validate_args=None):
#         self.weights = weights
#         self.actions = actions
#         self.scale   = scale
#         batch_shape = lax.broadcast_shapes(jnp.shape(actions), jnp.shape(scale))
#         super().__init__(batch_shape = batch_shape,validate_args=validate_args, event_shape=())

        
#     def sample(self, key, sample_shape=()):
#         raise NotImplementedError

#     def log_prob(self, value):
#         normalization_term  = jnp.sqrt(2*jnp.pi)*self.scale
#         value = jnp.atleast_1d(value)
#         integral_term = jnp.exp(-.5*(self.actions- value[:,None])**2/self.scale**2)/normalization_term
#         return jnp.log(jnp.sum(self.weights*integral_term,axis=1))




# def MagModel(Jbox):
    
#     # Jk = 
#     pi  = numpyro.sample('pi',dist.Dirichlet())

    

class Plummer(Distribution):
    def __init__(self, a):
        self.a = a
        batch_shape = jnp.shape(self.a)
        super(Plummer, self).__init__(batch_shape=batch_shape)
    
    def log_prob(self, r):
        q = r / self.a
        log_coeff = jnp.log(3) - 3 * jnp.log(self.a)
        log_rho = -5 / 2 * jnp.log(1 + q ** 2)
        log_p = log_coeff + log_rho + 2 * jnp.log(r)
        return log_p