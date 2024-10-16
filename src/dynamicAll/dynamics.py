# jax
import jax
import jax.numpy as jnp
from jaxopt import Bisection
from jax import random
from jax._src.config import config
from functools import partial

config.update("jax_enable_x64", True)


# numpyro
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

# project
from .base import JaxPotential


def Wolf(yerr, y=None, rhalf_dist=None):
    rng_key = random.PRNGKey(0)
    rng_key, rng_key_ = random.split(rng_key)
    # Run NUTS.
    kernel = NUTS(Dynamics.dispersion)
    num_samples = 3000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(rng_key_, y=y, error=yerr, rhalf_dist=rhalf_dist)
    mcmc.print_summary()
    samples_1 = mcmc.get_samples()
    # corner.corner(samples_1,var_names=['mu','sigma','mwolf'],quantiles=[0.16, 0.5, 0.84],show_titles=True,labels=[r'$\mu$',r'$\sigma$',r'$M_{\rm wolf}$'],title_fmt='2.f');
    return samples_1


class Dynamics:
    @staticmethod
    def dispersion(error, y=None, rhalf_dist=None):
        r"""
        simple numpyro model function for finding the mean velocity of stars

        .. math::
            P(v|\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{(v-\mu)^{2}}{2\sigma^{2}}}

        Parameters
        ----------
        data : _type_
            presumably velocities
        error : _type_
            error on velocities
        rhalf : _type_, optional
            If you want to calculate the Wolf mass, you can supply an r_{1/2}, by default None

        NOTES
        -----f
        Where do  the r_{1/2}'s come from
        It'd be nice if i could use r_{1/2}'s with an error
        e.g.
        rh       = numpyro.sample('rhalf',dist.Normal(rhalf,drhalf))
        mwolf    = numpyro.deterministic('mwolf',(4*rh*sigma**2)/(3*G))
        should add a with numpyro plate

        """
        G = 4.30091731e-6  # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        mu = numpyro.sample("mu", dist.Uniform(jnp.min(y), jnp.max(y)))
        sigma = numpyro.sample("sigma", dist.Uniform(0.01, 100))

        if rhalf_dist != None:
            rhalf = numpyro.sample("rhalf", rhalf_dist)
            mwolf = numpyro.deterministic("mwolf", (4 * rhalf * sigma**2) / (3 * G))

        s = jnp.sqrt(sigma**2 + error**2)
        numpyro.sample("obs", dist.Normal(mu, s), obs=y)


class Anisotropy:
    def __init__(self, model):
        self.model = model


class Priors:
    def __init__(self, dictionary):
        self.dictionary = dictionary


# def model_flat(data,error):
#     # set up priors on parameters
#     m_rhos = numpyro.sample("m_rhos", dist.Uniform(5,30)   )  # \ln{scale density} -- dont want negatives
#     m_rs   = numpyro.sample("m_rs"  , dist.Uniform(-10,10) )  # \ln{scale density} -- dont want negatives
#     m_a    = numpyro.sample("m_a"   , dist.Uniform(-1,5)   )  #inner-slope
