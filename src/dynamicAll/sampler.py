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
from scipy.integrate import fixed_quad, quad # I should get rid of these

# project 
from . import abel
from .base import JaxPotential
from .models import *
try:
    x,w  = np.loadtxt('/home/jjg57/DCJL/newjeans/src/data/gausleg_100',delimiter=',')
except:
    x,w  = np.loadtxt('/Users/juan/phd/projects/dynamicAll/src/data/gausleg_100',delimiter=',')

x = jnp.array(x)
w = jnp.array(w)

def integral(x,M,a):

    return jax.grad(Plummer._density(x,M,a))/jnp.sqrt(x - Plummer._potential(x,M,a))

def eddington(y: float,params):
    r'''
    .. math::
    f(\mathcal{E})
    =\frac{1}{\sqrt{8} \pi^2} \frac{\mathrm{d}}{\mathrm{d} \mathcal{E}} \int_0^{\mathcal{E}} \frac{\mathrm{d} \Psi}{\sqrt{\mathcal{E}-\Psi}} \frac{\mathrm{d} \nu}{\mathrm{d} \Psi} .

    Parameters
    ----------
    y : float
        relative energy: :math:`\mathcal{E} \equiv-H+\Phi_0=\Psi-\frac{1}{2} v^2`
    '''
    dnu = jax.grad(jax.potential.Plummer())

    coeff = jnp.sqrt(8)*jnp.pi**2

    return jax.grad(jnp.sum(w*integral(x,*params)))/coeff

