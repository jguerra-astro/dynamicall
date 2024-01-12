#third-party
from functools import partial
from typing import Callable
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sc
from jax._src.config import config
from jaxopt import Bisection
from scipy.integrate import quad
config.update("jax_enable_x64", True)
from jax import random

# project 
from . import abel
from .base import JaxPotential
from .models import *


#TODO: Write a sampler using jax to speed up calculations.

class ConditionalSampler:

    def __init__(self,
                dm_component,
                stellar_component,
                anisotropy_model,
                evolve: bool =False):
        
        self.dm_component      = dm_component
        self.stellar_component = stellar_component
        self.anisotropy_model  = anisotropy_model


        # first find drho/dphi

        r = jnp.logspace(-10,5,10_000)
        rho = self.dm_component.density(r)
        phi = self.dm_component.potential(r)

    def sample(self,N:int):

        # First sample from stellar component
        x,y,z = self.stellar_component.sample_xyz(N)
        r =  jnp.sqrt(x**2 + y**2 + z**2)

        # The sample velocities from conditional distribution


def sample_from_DF(model,N: int,method = 'rejection'):
    """
    Sample from the distribution function of a given model.
    """
    DF = model.DF