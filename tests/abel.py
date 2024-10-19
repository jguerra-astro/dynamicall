from typing import Callable
from scipy import integrate
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.config import config

config.update("jax_enable_x64", True)

# DEPRECATED -- Don't need these anymroe


def project(func: Callable, R: np.ndarray, params) -> np.ndarray:
    r"""
    Forward Abel transform : Project an optically thin, spherically symmetric function onto a plane.
    :math:`f(r(x,y,z))\rightarrow F(R(x,y))`

    .. math::
        F(R)=2 \int_R^{\infty} \frac{f(r) r}{\sqrt{r^2-R^2}} d r

    Parameters
    ----------
    func : Callable
        function of projected (on-sky) radii, R
    R : jnp.ndarray
        Projected distance :math:`(R = \sqrt{x^2 +y^2})`

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

    TODO: Get rid of quad in both functions

    """
    f_abel = lambda x, R: 2 * func(x, *params) * x / np.sqrt(x**2 - R**2)
    out = np.array(
        [
            integrate.quad(
                f_abel, i, np.inf, args=(i,), limit=1000, epsrel=1e-12, epsabs=1.49e-11
            )[0]
            for i in R
        ]
    )
    return out


def deproject(func: Callable, R: jnp.ndarray, params) -> np.ndarray:
    r"""
    Inverse Abel transform is used to calculate the emission function given a projection, i.e.
    :math:`F(R(x,y)) \rightarrow f( r(x,y,z) )`

    .. math::
        f(r)=-\frac{1}{\pi} \int_r^{\infty} \frac{d F}{d y} \frac{d y}{\sqrt{y^2-r^2}}

    Parameters
    ----------
    func : Callable
        function of :math:`r = \sqrt{x^2+y^2+z^2}.` This must be written in a jax friendly way
    R : np.ndarray
        _description_
    params : _type_
        _description_

    Returns
    -------
    np.ndarray
        _description_
    """

    dfunc = jax.grad(func)
    f_abel = lambda x, R: -1 * dfunc(x, *params) / jnp.sqrt(x**2 - R**2) / jnp.pi
    out = np.array(
        [
            integrate.quad(
                f_abel, i, np.inf, args=(i,), limit=1000, epsrel=1e-12, epsabs=1.49e-11
            )[0]
            for i in R
        ]
    )
    return out
