from typing import Callable
from scipy import integrate
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.config import config
config.update("jax_enable_x64", True)


def project(func: Callable,R: np.ndarray,params) -> np.ndarray:
    '''
    project: F( R(x,y) ) -> f( r(x,y,z) ) using the abel transform

    Parameters
    ----------
    func : Callable
        function of projected (on-sky) radii, R
    R : jnp.ndarray
        Projected distance (R = x**2 +y**2)
    
    params : 
        parameters for func

    Returns
    -------
    jnp.ndarray
        projected positions

    Notes
    -----
    I had to increase epsabs on quad in order to get the test to pass.
    This will make these functions significantly slower.
    This doesnt exactly matter for now as, these methods are not being 
    used in the fitting functions, but they could in principle so
    I should try to make them as fast as possible.

    '''
    f_abel = lambda x,R: 2*func(x,*params)*x/np.sqrt(x**2-R**2)
    out= np.array([integrate.quad(f_abel,i,np.inf,args=(i,),limit=1000,epsrel=1e-12,epsabs=1.49e-11)[0] for i in R])   
    return out

def deproject(func: Callable,R: np.ndarray,params) -> np.ndarray:
    '''
    f( r(x,y,z) ) -> F(R(x,y))

    Project 

    Parameters
    ----------
    func : Callable
        _description_
    R : np.ndarray
        _description_
    params : _type_
        _description_

    Returns
    -------
    np.ndarray
        _description_
    '''

    dfunc = jax.grad(func)
    f_abel = lambda x,R: -1*dfunc(x,*params)/jnp.sqrt(x**2-R**2)/jnp.pi
    out= np.array([integrate.quad(f_abel,i,np.inf,args=(i,),limit=1000,epsrel=1e-12,epsabs=1.49e-11)[0] for i in R])   
    return out