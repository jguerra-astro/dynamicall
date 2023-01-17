from typing import Callable, Union
import jax
import jax.numpy as jnp
from jax._src.config import config
config.update("jax_enable_x64", True)
from scipy import optimize
import numpy as np
from numpy import *


x,w  = np.loadtxt('/Users/juan/phd/projects/weird-jeans/src/data/gausleg_200',delimiter=',')
x = jnp.array(x)
w = jnp.array(w)

# @jax.jit #? Why can't i jit this one??
def gauss_quad1D(func : Callable,a:Union[float,jnp.ndarray],b:Union[float,jnp.ndarray],params=()) -> jnp.ndarray:
    '''
    1 dimensional integral using gaussian-Legengdre quadruture
    NOTE:
        x,w are numpy arrays that are calculated before hand -- currently 200 -- maybe I should go up in this?
        The way that it's written now should be pretty fast
    Parameters
    ----------
    func : Callable
        integrand
    a : float
        lower bound 
    b : tuple[float,np.ndarray]
        upper bound
    theta : dict
        dictionary of args for func

    Returns
    -------
    np.ndarray
        integral of func from a to b
    Examples:
    TODO: What accuracy to I want? Should test for mass profiles that are known analytically
    >>np.testing.assert_almost_equal(gauss_quad1D(lambda x,theta: 2*x,0,1,theta=None),1,decimal=5)
    '''
    xi=(b-a)*0.5*x[:,None] + .5*(b+a)
    wi=(b-a)*0.5*w[:,None]
    return jnp.sum(wi*func(xi,*params),axis=0)