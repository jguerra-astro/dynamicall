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

class MagJ(Distribution):
    # arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    # support = constraints.real
    # reparametrized_params = ["loc", "scale"]

    def __init__(self,weights = 1,actions =1, scale=1, validate_args=None):
        self.weights = weights
        self.actions = actions
        self.scale   = scale
        batch_shape = lax.broadcast_shapes(jnp.shape(actions), jnp.shape(scale))
        super().__init__(batch_shape = batch_shape,validate_args=validate_args, event_shape=())

        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):
        normalization_term  = jnp.sqrt(2*jnp.pi)*self.scale
        value = jnp.atleast_1d(value)
        integral_term = jnp.exp(-.5*(self.actions- value[:,None])**2/self.scale**2)/normalization_term
        return jnp.log(jnp.sum(self.weights*integral_term,axis=1))




def MagModel(Jbox):
    
    # Jk = 
    pi  = numpyro.sample('pi',dist.Dirichlet())

    
