from typing import Callable
import astropy.units as u
import astropy.constants as const
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
from typing import Callable
import emcee
from scipy.optimize import curve_fit
from .base import Data
from . import models


class Likelihood:

    def __init__(self, data: Data, model: Callable):
        self.data = data
        self.model = model


    def lnLikelihood(self,theta: jax.Array) -> float:
        '''
        log likelihood function for 2+1 dimensional data e.g (x,y,v_{los}) equivalenetly (R,v_{los})

        Parameters
        ----------
        theta : jax.Array
            Parameters we're trying to fit
            e.g. for basic models: (inner-slope, rho_d,r_d,stellar-anisotropy)


        Returns
        -------
        float
            _description_


        Notes
        -------

        '''

        q  = data[0]            # projected radii
        v  = data[1]            # line-of-sight/"radial" velocitiess
        N  = v.shape[0]         # number of observed stars
        sigmasq  = model(q,theta) + obs_error**2
            
        # log of likelihood
        term1 = - 0.5 * jnp.sum(np.log(sigmasq))
        term2 = - 0.5 * jnp.sum(v**2/sigmasq)   # This assumes that the velocities are gaussian centered at 0 -- could also fit for this
        term3 = - 0.5 * N *jnp.log(2*np.pi) # doesnt really need to be here

        return term1+term2+term3

    def lnLikelihood2(self,theta: jax.Array) -> float:
        '''
        Log likelihood assuming 3+1 dimenstional data (R,v_{los},v_{pmr},v_{pmt})

        Parameters
        ----------
        theta : jax.Array
            _description_

        Returns
        -------
        float
            _description_

        Notes
        -------
        Ok, so whats up? What does adding positions do

        '''
        q   = data['R']       # projected radii
        v1  = data['vlos']    # line-of-sight/"radial" velocitiess
        v2  = data['vpmr']
        v3  = data['vpmt']

        N  = v.shape[0]         # number of observed stars
        sigmasq  = model(q,theta) + obs_error**2
        sigmasq  = model(q,theta) + obs_error**2

        return -1