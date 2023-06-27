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

x, w = np.polynomial.legendre.leggauss(128) #TODO: Find a better way to set this.

x = jnp.array(x)
w = jnp.array(w)

#TODO: Write a sampler using jax to speed up calculations.



class ConditionalSampler:

    def __init__(self,
                dm_component,
                stellar_component,
                anisotropy_model,
                evolve=False):
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



def rejection_sample(model_class, samplesize, r_vr_vt=False, r_v=False, filename=None, brute=True):
    
    nX, nV, Xlim, Vlim, = model_class.sampler_input
    nA = 0

    if nX+nV > 6:
        print ('We only support a maximum of 6 variables')
        return
    if (r_vr_vt and r_v)==True:
        print ('You cannot have both r_vr_vt AND r_v set to True')
        return
    if (Xlim[1] <= Xlim[0] or Vlim[1] <= Vlim[0]):
        print ('ERROR: rmax <= rmin or vmax<=vmin, please double check the sample limits')

    ans = rejectsample(model_class.DF, Xlim, Vlim, nX, nV, nA,
                [], [], samplesize, r_vr_vt, r_v, z_vr_vt=False)

    if filename != None:
        if (r_vr_vt or r_v ):
            np.savetxt(filename, np.c_[ans])
        else:
            np.savetxt(filename, np.c_[np.transpose(ans)])

    return ans