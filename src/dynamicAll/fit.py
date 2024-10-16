import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import random
from jax._src.config import config

config.update("jax_enable_x64", True)

import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
import warnings
import numpyro.distributions as dist
from . import models, base, data
import astropy.units as u
import astropy.constants as const
from functools import partial
import inspect

from .base import Data


class Priors:
    """
    Class for defining priors for the model
    """

    def __init__(self):
        self.priors = {}

    def add_prior(self, name, prior_dist, base="one"):
        self.priors[name] = prior_dist

    def uniform(self, name, lower=0, upper=1):
        self.priors[name] = dist.Uniform(lower, upper)

    def normal(self, name, loc=0, scale=1):
        self.priors[name] = dist.Normal(loc, scale)

    def get_prior(self, name):
        return self.priors.get(name)

    def add_from_dict(self, priors_dict):
        for name, prior in priors_dict.items():
            self.add_prior(name, prior)
        # return priors

    @staticmethod
    def from_dict(priors_dict):
        priors = Priors()
        for name, prior in priors_dict.items():
            priors.add_prior(name, prior)
        return priors

    def __str__(self):
        for param_name, distribution in self.priors.items():
            distribution_name = distribution.__class__.__name__
            if isinstance(distribution, (dist.Uniform, dist.LogUniform)):
                low = f"{distribution.low:.2e}"
                high = f"{distribution.high:.2e}"
                bounds_info = f"Bounds: ({low}, {high})"

            elif isinstance(distribution, (dist.Normal, dist.LogNormal)):
                mean = f"{distribution.loc:.2e}"
                std_dev = f"{distribution.scale:.2e}"
                bounds_info = f"Mean: {mean}, StdDev: {std_dev}"
            else:
                bounds_info = "No Bounds"

            print(
                f"Parameter: {param_name}, Distribution: {distribution_name}, {bounds_info}"
            )
        return ""

    def __repr__(self):
        self.__str__()
        return ""


class SphGalaxy:
    _xk, _wk = np.polynomial.legendre.leggauss(10)
    _xk, _wk = jnp.array(_xk), jnp.array(_wk)
    _G = const.G.to(u.kpc * u.km**2 / u.solMass / u.s**2).value

    def __init__(
        self,
        tracer_model=models.Plummer,
        dark_model=models.gNFW,
        anisotropy_model=models.BetaConstant,
        priors: Priors = None,
    ):
        self._light = tracer_model
        self._dark = dark_model
        self._anisotropy = anisotropy_model

        self._density = self._light._density_fit
        self._projection = self._light._projection_function
        self._mass = self._dark._mass_fit
        self._anisotropy = self._anisotropy._beta

        self.mass = jax.vmap(self._mass, in_axes=(0, None))
        self.vec_dispn = jax.vmap(self.dispersion, in_axes=(0, None, None, None))
        self.vec_dispb = jax.vmap(self.test_dispersion, in_axes=(0, None, None, None))

        y0 = 0.0  # lowerbound
        y1 = jnp.pi / 2.0  # upperbound

        self.xk = 0.5 * (y1 - y0) * self._xk + 0.5 * (y1 + y0)
        self.wk = 0.5 * (y1 - y0) * self._wk

        self.cosk = jnp.cos(self.xk)
        self.sink = jnp.sin(self.xk)

        # Checks to see if default values are being used
        warnings.formatwarning = self.custom_formatwarning
        if not hasattr(self, "tracer_model"):
            warnings.warn("\nNo tracer model defined.\nUsing default Plummer model.\n")
        if not hasattr(self, "dark_model"):
            warnings.warn("\nNo DM model defined.\nUsing default gNFW model.\n")
        if not hasattr(self, "anisotropy_model"):
            warnings.warn(
                "\nNo anisotropy model defined.\nusing default BetaConstant model\n"
            )

        self._priors = priors
        self._tracer_model = tracer_model
        self._dm_model = dark_model

        if (
            priors == None
        ):  # if priors aren't defined, use the defaults one --- this is ugly, should rewrite
            warnings.warn("\nNo priors defined.\nUsing default priors for each model")

            self._priors = Priors()
            self._priors.add_from_dict(tracer_model._tracer_priors)
            self._priors.add_from_dict(dark_model._dm_priors)
            self._priors.add_from_dict(anisotropy_model._priors)

        # TODO: add consitency check to make sure all necessary priors are defined
        # should just be a count check

    @partial(jax.jit, static_argnums=(0,))
    def dispersion(self, r, dm_param_dict, tr_param_dict, beta_param_dict):
        @jax.jit
        def func_(q, dm_param_dict, tr_param_dict, beta_param_dict):
            return (
                q ** (2.0 * self._anisotropy(q, beta_param_dict))
                * self.mass(q, dm_param_dict)
                * self._density(q, tr_param_dict)
                / q**2
            )

        # func = jax.vmap(func_,in_axes=(0,None,None,None))
        coeff = self._G * r

        return (
            coeff
            * jnp.sum(
                self.wk
                * self.cosk
                * func_(r / self.sink, dm_param_dict, tr_param_dict, beta_param_dict)
                / self.sink**2,
                axis=0,
            )
            / r ** (2 * self._anisotropy(r, beta_param_dict))
        )

    @partial(jax.jit, static_argnums=(0,))
    def test_dispersion(
        self, R_, dm_param_dict, tr_param_dict, beta_param_dict
    ) -> float:
        """
        first using the transform y = rcsc(x)
        then using Gauss-Legendre transformation.
        """

        @jax.jit
        def func_22(x, R_, dm_param_dict, tr_param_dict, beta_param_dict):
            sn = self.vec_dispn(x, dm_param_dict, tr_param_dict, beta_param_dict)
            # sn = self.dispersion(x,dm_param_dict,tr_param_dict,beta_param_dict)
            return (
                (1.0 - self._anisotropy(x, beta_param_dict) * (R_**2 / x**2))
                * sn
                * x
                / jnp.sqrt(x**2 - R_**2)
            )

        # func2 = jax.vmap(func_22,in_axes=(0,None,None,None,None))

        return (
            2
            * R_
            * jnp.sum(
                self.wk
                * self.cosk
                * func_22(
                    R_ / self.sink, R_, dm_param_dict, tr_param_dict, beta_param_dict
                )
                / self.sink**2,
                axis=0,
            )
            / self._projection(R_, tr_param_dict)
        )

    def fit_dSph(
        self,
        data,
        rng_key=random.PRNGKey(0),
        num_samples=1000,
        num_warmup=1000,
        num_chains=2,
        progress_bar=True,
        jfactor=False,
        jfactor_params=None,
        init_strategy=None,
    ):
        # First get data from data class
        R = data._cached_dispersion["los"][0]
        v = data._cached_dispersion["los"][1]
        v_err = data._cached_dispersion["los"][2]
        data_ = jnp.array([R, v, v_err])

        def model_flat(data, priors):
            # set up priors on parameters
            samples = {}
            sub_dictionaries = {"tracer": {}, "dm": {}, "beta": {}}
            for param_name, prior_dist in priors.priors.items():
                samples[param_name] = numpyro.sample(param_name, prior_dist)

                if param_name.startswith("tracer_"):
                    sub_param_name = param_name[len("tracer_") :]
                    sub_dictionaries["tracer"][sub_param_name] = samples[param_name]

                elif param_name.startswith("dm_"):
                    sub_param_name = param_name[len("dm_") :]

                    sub_dictionaries["dm"][sub_param_name] = samples[param_name]
                elif param_name.startswith("beta_"):
                    sub_param_name = param_name[len("beta_") :]
                    sub_dictionaries["beta"][sub_param_name] = samples[param_name]
                    # sub_dictionaries['beta']['0'] = 0.0

            # deterministic function for j_factor
            if jfactor:
                jf = numpyro.deterministic(
                    "jfactor",
                    self._dark._jFactor(
                        theta=jfactor_params["theta"],
                        param_dict=sub_dictionaries["dm"],
                        d=jfactor_params["D"],
                        rt=jfactor_params["rt"],
                    ),
                )

            with numpyro.plate("data", len(data[1])):
                sigma2 = jnp.sqrt(
                    self.vec_dispb(
                        data[0],
                        sub_dictionaries["dm"],
                        sub_dictionaries["tracer"],
                        sub_dictionaries["beta"],
                    )
                )

                numpyro.sample("y", dist.Normal(sigma2, data[2]), obs=data[1])

        rng_key, rng_key_ = random.split(rng_key)

        # Run NUTS.
        kernel = NUTS(model_flat)
        # num_samples = 5000
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )
        mcmc.run(rng_key_, data=data_, priors=self._priors)
        return mcmc
        # mcmc.print_summary()
        # samples_1 = mcmc.get_samples()

    def fit_unbinned(
        self,
        data,
        rng_key=random.PRNGKey(0),
        num_samples=1000,
        num_warmup=1000,
        num_chains=2,
        progress_bar=True,
    ):
        # First get data from data class
        R = data._R
        v = data._vlos
        v_err = data.d_vlos
        data = jnp.array([R, v, v_err])

        def model_flat(data, priors):
            # set up priors on parameters
            v_mean = numpyro.sample("v_mean", dist.Uniform(-500, 500))
            samples = {}
            sub_dictionaries = {"tracer": {}, "dm": {}, "beta": {}}
            for param_name, prior_dist in priors.priors.items():
                samples[param_name] = numpyro.sample(param_name, prior_dist)

                if param_name.startswith("tracer_"):
                    sub_param_name = param_name[len("tracer_") :]
                    sub_dictionaries["tracer"][sub_param_name] = samples[param_name]
                elif param_name.startswith("dm_"):
                    sub_param_name = param_name[len("dm_") :]
                    sub_dictionaries["dm"][sub_param_name] = samples[param_name]
                elif param_name.startswith("beta_"):
                    sub_param_name = param_name[len("beta_") :]
                    sub_dictionaries["beta"][sub_param_name] = samples[param_name]

            with numpyro.plate("data", len(data[1, :])):
                sigma2 = self.vec_dispb(
                    data[0, :],
                    sub_dictionaries["dm"],
                    sub_dictionaries["tracer"],
                    sub_dictionaries["beta"],
                )

                epsilon = jnp.sqrt(sigma2 + data[2, :] ** 2)

            numpyro.sample("y", dist.Normal(v, epsilon), obs=data[1:])

        rng_key, rng_key_ = random.split(rng_key)

        # Run NUTS.
        kernel = NUTS(model_flat)
        # num_samples = 5000
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        mcmc.run(rng_key_, data=data, priors=self._priors)
        return mcmc
        # mcmc.print_summary()
        # samples_1 = mcmc.get_samples()

    def custom_formatwarning(self, message, category, filename, lineno, line=None):
        return f"{category.__name__}: {message}\n"
