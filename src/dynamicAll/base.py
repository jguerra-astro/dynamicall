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
import warnings

config.update("jax_enable_x64", True)


class JaxDensity(ABC):
    """
    Base class for Potentials.

    Parameters
    ----------
    ABC : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    # Class variables

    # Priors for the tracer density profile
    _tracer_priors = {}
    _dm_priors = {}

    # Integration stuff -- 100 is more than enough for the accuracy we need.
    # Others are here for mostly for testing purposes
    _xk, _wk = np.polynomial.legendre.leggauss(10)
    _xk, _wk = jnp.array(_xk), jnp.array(_wk)
    _xm, _wm = np.polynomial.legendre.leggauss(128)
    _xm, _wm = jnp.array(_xm), jnp.array(_wm)
    _x256, _w256 = np.polynomial.legendre.leggauss(256)
    _x256, _w256 = jnp.array(_x256), jnp.array(_w256)

    # Need a different accuracy for the J-factor integral
    _xj, _wj = np.polynomial.legendre.leggauss(1000)
    _xj, _wj = jnp.array(_xj), jnp.array(_wj)

    # unit conversion used in the J-factor calculation
    _GeV2cm5 = (1 * u.solMass**2 * u.kpc**-5 * const.c**4).to(u.GeV**2 / u.cm**5).value
    _GeVcm2 = (1 * u.solMass * u.kpc**-2 * const.c**2).to(u.GeV / u.cm**2).value

    # Gravitational constant in units useful units.
    _G = const.G.to(u.kpc * u.km**2 / u.solMass / u.s**2).value
    # Gravitational constant in units useful units.
    G = const.G.to(u.kpc * u.km**2 / u.solMass / u.s**2).value

    def __init__(
        self,
        int_order: int = 100,
        JfactorN: int = 1000,
        Gunit=u.kpc * u.km**2 / u.solMass / u.s**2,
    ):
        self._G = const.G.to(u.kpc * u.km**2 / u.solMass / u.s**2).value

        # Intregation stuff. 256 is more than enough for the accuracy we need.
        xi, wi = np.polynomial.legendre.leggauss(int_order)
        self._x, self._w = jnp.array(xi), jnp.array(wi)

        # Need a different accuracy for the J-factor integral
        xj, wj = np.polynomial.legendre.leggauss(JfactorN)
        self._xj, self._wj = jnp.array(xj), jnp.array(wj)

    def density(self, r):
        return self._density(r, self._params)

    def mass(self, r):
        return self._mass(r, self._params)

    def potential(self, r):
        return self._potential(r, self._params)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def DistributionFunction(cls, eps: jnp.ndarray, params):
        """
        Eddington Formula for the distribution function of an isotropic system
        """
        r_e = cls.test_calculate_re(eps, params, 1e10)
        return jax.vmap(cls.dF_helper, in_axes=(0, 0, None))(eps, r_e, params)
        # return jax.vmap(cls.dF_helper, in_axes=(0, 0, None))(eps, r_e, params)
        # return jax.vmap(lambda e, r: cls.dF_helper(e, r, params))(eps, r_e)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def dF_helper(cls, eps, r_e, params):
        r"""
        Eddington formula for the distribution function of a spherical isotropic system.
        This is using the standard variable names and definitions from Binney & Tremaine (2008) where
        :math:`\psi(r) = -\Phi(r) + \Phi_{0}`  and :math:`\mathcal{E} = \psi - \frac{1}{2}v^{2}`.
        In most cases we'll take \Phi_{0} = 0 in which case :math:`\mathcal{E}` is the binding energy.

        .. math::
            f(\mathcal{E}) = \frac{1}{\sqrt{8}\pi^2}\int_{r_\mathcal{E}}^{\infty} \frac{k(r)dr}{\sqrt{\mathcal{E} - \psi(r)}}

        where k(r)

        .. math::
            k(r) = \frac{r^{2}}{GM(r)}\left[\rho''(r)\ + \rho'(r)\left(\frac{2}{r} - \frac{4\pi r^{2}\rho(r)}{M(r)}\right)\right]

        See the documentation for more details on the derivation and tests.

        Parameters
        ----------
        eps : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        rho = jax.vmap(cls._density, in_axes=(0, None))
        drho = jax.vmap(jax.grad(cls._density), in_axes=(0, None))
        d2rho = jax.vmap(jax.grad(jax.grad(cls._density)), in_axes=(0, None))

        mass = jax.vmap(cls._mass, in_axes=(0, None))
        phi = jax.vmap(cls._potential, in_axes=(0, None))

        def k(x):
            term1 = x**2 / cls.G / mass(x, params)
            term2 = 2 / x - 4 * jnp.pi * rho(x, params) * x**2 / mass(x, params)
            term3 = d2rho(x, params) + drho(x, params) * term2
            return term1 * term3

        def integrand(x, epsilon):
            coeff = 1 / jnp.sqrt(8) / jnp.pi**2
            return coeff * k(x, params) / jnp.sqrt(epsilon + phi(x, params))

        coeff = 1 / jnp.sqrt(8) / jnp.pi**2

        def test_integrand(x, epsilon):
            def small():
                delta = jnp.abs(x - r_e)
                term1 = cls.G * mass(x, params) * delta / x**2
                term2 = delta**2 * (
                    2 * jnp.pi * rho(x, params) - cls.G * mass(x, params) / x**3
                )
                dphi = term1 + term2
                return coeff * k(x) / jnp.sqrt(dphi)

            def large():
                dphi = epsilon + phi(x, params)
                return coeff * k(x) / jnp.sqrt(dphi)

            return jnp.where(x - r_e < 1e-4, small(), large())

        # Integrate from r_epsilon to infinity

        xi, wi = cls._xm, cls._wm
        x0 = 0
        x1 = jnp.pi / 2

        xk = 0.5 * (x1 - x0) * xi + 0.5 * (x1 + x0)
        wk = 0.5 * (x1 - x0) * wi

        _integrand = test_integrand

        return r_e * jnp.sum(
            wk * _integrand(r_e / jnp.sin(xk), eps) * jnp.cos(xk) / jnp.sin(xk) ** 2
        )

    @classmethod
    def calculate_re(cls, epsilon, rlim, params):
        """
        Helper function for calculating the distribution function.
        Calculates the radius at which:math:`\Psi(r) = \epsilon`
        See the documentation for more details on the derivation and tests.

        Parameters
        ----------
        epsilon : _type_
            :math:`\epsilon = \psi(r) - \frac{1}{2}v^{2}`  | [kpc^{2}~s^{-2}]
        rlim : _type_
            Limit for the radius at which to calculate the distribution function, For small energies, the radius at which the potential is equal to epsilon can get very large.
        params : _type_
            parameters for the specific model being used
        """

        def psi_minus_epsilon(r):
            potential = cls._potential(r, params)
            result = -potential - epsilon
            print(f"r: {r}, potential: {potential}, psi_minus_epsilon: {result}")
            return result

        # Check function values at bounds
        lower = 1e-12
        upper = rlim
        f_lower = psi_minus_epsilon(lower)
        f_upper = psi_minus_epsilon(upper)
        print(f"Function value at lower bound: {f_lower}")
        print(f"Function value at upper bound: {f_upper}")

        if f_lower * f_upper >= 0:
            print("Warning: Function does not change sign between bounds!")

        solver = Bisection(
            psi_minus_epsilon,
            lower=lower,
            upper=upper,
            maxiter=100,
            tol=1e-20,
            check_bracket=False,
        )

        result = solver.run()
        print(f"Bisection result: {result}")
        return result.params

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def test_calculate_re(cls, epsilon, params, rlim):
        """
        Helper function for calculating the distribution function.
        Calculates the radius at which:math:`\Psi(r) = \epsilon`
        See the documentation for more details on the derivation and tests.

        Parameters
        ----------
        epsilon : _type_
            :math:`\epsilon = \psi(r) - \frac{1}{2}v^{2}`  | [kpc^{2}~s^{-2}]
        rlim : _type_
            Limit for the radius at which to calculate the distribution function, For small energies, the radius at which the potential is equal to epsilon can get very large.
        params : _type_
            parameters for the specific model being used
        """
        phi = cls._potential

        @jax.jit
        def psi_minus_epsilon(x, eps):
            return jnp.abs(phi(x, params)) - eps

        def bisection_solver(eps):
            solver = Bisection(
                lambda x: psi_minus_epsilon(x, eps),
                lower=1e-12,
                upper=rlim,
                maxiter=100,
                tol=1e-20,
                check_bracket=False,
            )
            return solver.run().params

        vectorized_bisect = jax.vmap(bisection_solver)
        r_epsilon = vectorized_bisect(epsilon)
        return r_epsilon

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def DF(cls, eps, params, **kwargs):
        """
        Eddington Formula for the distribution function of an isotropic system
        """

        def calculate_re(epsilon, rlim):
            phi = cls._potential

            @jax.jit
            def psi_minus_epsilon(x, eps):
                return jnp.abs(phi(x)) - eps

            def bisection_solver(eps):
                solver = Bisection(
                    lambda x: psi_minus_epsilon(x, eps),
                    lower=1e-12,
                    upper=rlim,
                    maxiter=100,
                    tol=1e-20,
                    check_bracket=False,
                )
                return solver.run().params

            vectorized_bisect = jax.vmap(bisection_solver)
            r_epsilon = vectorized_bisect(epsilon)
            return r_epsilon

        # get rlim from keyword arguments
        rlim = kwargs.get("rlim", 1e10)
        r_e = calculate_re(eps, rlim)
        return r_e
        # return cls.dF_helper(eps, params, r_e)

    def dJdOmega(self, theta, D, rt):
        r"""
        .. math:
            \frac{dJ}{d\omega} = \int{\rho^{2}(r) d\ell}

        Parameters
        ----------
        theta : _type_
            _description_
        """
        l0 = D * jnp.cos(theta) - jnp.sqrt(rt**2 - (D * jnp.sin(theta)) ** 2)
        l1 = D * jnp.cos(theta) + jnp.sqrt(rt**2 - (D * jnp.sin(theta)) ** 2)
        xi = 0.5 * (l1 - l0) * self._xj + 0.5 * (l1 + l0)
        wi = 0.5 * (l1 - l0) * self._wj
        return jnp.sum(
            wi * self.density(jnp.sqrt(xi**2 + D**2 - 2 * xi * D * jnp.cos(theta))) ** 2
        )

    @partial(jax.jit, static_argnums=(0,))
    def jFactor(self, theta, d, rt):
        r"""
        J-factor

        .. math:
            J= \int\int_{\rm los} \rho^{2}(r) d\Omega dl

        where we write r as :math:`r^2=\ell^2+d^2-2 \ell d \cos \theta` where :math:`\ell` is the distance along the line of sight and d is the distance to the center of the galaxy.
        Spherical symmetry lets us write :math:`d\Omega=2\pi\sin\theta d\theta`.

        The bounds on the line of sight are $\ell_{ \pm}=d \cos \theta \pm \sqrt{r_{200}^2-d^2 \sin ^2 \theta}$.
        where $r_{200}$ is the 'size' of the system.

        substituting in the integral we get
        $$
        J(\theta_{\rm max}) = 2 \pi \int_{0}^{\theta_{\rm max}} \int_{\ell_{-}}^{\ell_{+}} \left[\rho\left(\sqrt{\ell^2+d^2-2 \ell d \cos \theta}\right)\right]^{2}\sin(\theta) d\ell d\theta
        $$

        Parameters
        ----------
        theta : float
            _description_
        d : float
            distance to system | [kpc]
        rt : float
            tidal radius of system | [kpc]

        Returns
        -------
        float
            J-factor | [:math:`M_{\odot}^{2}~kpc^{-5}]`

        Notes
        -----
        Should rt just be r_200? should i just change the bounds on the line of sight part to be -inf to inf?
        Note the Units.
        Maybe I should just return log10(J/[GeV^{2} cm^{-5}]) instead of J?
        """

        x0 = 0
        x1 = theta
        xi = 0.5 * (x1 - x0) * self._xj + 0.5 * (x1 + x0)
        wi = 0.5 * (x1 - x0) * self._wj
        vectorized_func = jax.vmap(self.dJdOmega, in_axes=(0, None, None))
        return jnp.log10(
            2
            * jnp.pi
            * jnp.sum(wi * vectorized_func(xi, d, rt) * jnp.sin(xi))
            * self._GeV2cm5
        )

    @partial(jax.jit, static_argnums=(0,))
    def dDdOmega(self, theta, d, rt):
        """
        _summary_

        Parameters
        ----------
        theta : _type_
            _description_
        d : _type_
            _description_
        rt : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        l0 = d * jnp.cos(theta) - jnp.sqrt(rt**2 - (d * jnp.sin(theta)) ** 2)
        l1 = d * jnp.cos(theta) + jnp.sqrt(rt**2 - (d * jnp.sin(theta)) ** 2)
        xi = 0.5 * (l1 - l0) * self._xj + 0.5 * (l1 + l0)
        wi = 0.5 * (l1 - l0) * self._wj
        return jnp.sum(
            wi * self.density(jnp.sqrt(xi**2 + d**2 - 2 * xi * d * jnp.cos(theta)))
        )

    @partial(jax.jit, static_argnums=(0,))
    def dFactor(self, theta, d, rt):
        """
        Decay Factor

        Parameters
        ----------
        theta : _type_
            _description_
        d : _type_
            _description_
        rt : _type_
            _description_

        Returns
        -------
        float
            Decay Factor | [:math:`M_{\odot}~kpc^{-2}]`

        Notes
        -----
        TODO: Decide on the units and log10 or not
        """
        x0 = 0
        x1 = theta
        xi = 0.5 * (x1 - x0) * self._xj + 0.5 * (x1 + x0)
        wi = 0.5 * (x1 - x0) * self._wj
        vectorized_func = jax.vmap(self.dDdOmega, in_axes=(0, None, None))
        return jnp.log10(
            2
            * jnp.pi
            * jnp.sum(wi * vectorized_func(xi, d, rt) * jnp.sin(xi))
            * self._GeVcm2
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _density(cls, r: float, params):
        pass

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _mass(cls, r: float, params):
        q = r
        rs = params["rs"]

        xi = JaxPotential._x256
        wi = JaxPotential._w256

        def r_le_rs():
            xk = 0.5 * q * xi + 0.5 * q
            wk = 0.5 * q * wi
            return 4 * jnp.pi * jnp.sum(wk * xk**2 * cls._density(xk, params), axis=0)

        def r_gt_rs():
            # first ingrate from 0 to rs
            # then integrate from rs to r

            x1 = 0.5 * rs * xi + 0.5 * rs
            w1 = 0.5 * rs * wi
            m_rs = 4 * jnp.pi * jnp.sum(w1 * x1**2 * cls._density(x1, params), axis=0)

            x2 = 0.5 * (r - rs) * xi + 0.5 * (r + rs)
            w2 = 0.5 * (r - rs) * wi
            m_r = 4 * jnp.pi * jnp.sum(w2 * x2**2 * cls._density(x2, params), axis=0)

            return m_rs + m_r

        return jax.lax.cond(
            r <= rs,
            r_le_rs,
            r_gt_rs,
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def test_mass(cls, r, params):
        """
        Integration using Gauss-Legendre quadrature + trigonometric substitution, didn't make much of a difference,
        but i'll leave it around for more testing

        Parameters
        ----------
        r : _type_
            _description_
        params : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        rs = params["rs"]

        xi = JaxPotential._xk
        wi = JaxPotential._wk

        x0 = 0.0
        x1 = jnp.pi / 2
        xk = 0.5 * x1 * xi + 0.5 * x1
        wk = 0.5 * x1 * wi

        def r_le_rs():
            return (
                4
                * jnp.pi
                * r
                * jnp.sum(
                    wk
                    * (r * jnp.sin(xk)) ** 2
                    * cls._density(r * jnp.sin(xk), params)
                    * jnp.cos(xk),
                    axis=0,
                )
            )

        def r_gt_rs():
            # first ingrate from 0 to rs
            # then integrate from rs to r
            m_rs = (
                4
                * jnp.pi
                * rs
                * jnp.sum(
                    wk
                    * (rs * jnp.sin(xk)) ** 2
                    * cls._density(rs * jnp.sin(xk), params)
                    * jnp.cos(xk),
                    axis=0,
                )
            )

            x2 = 0.5 * (x1 - jnp.arcsin(rs / r)) * xi + 0.5 * (x1 + jnp.arcsin(rs / r))
            w2 = 0.5 * (x1 - jnp.arcsin(rs / r)) * wi
            m_r = (
                4
                * jnp.pi
                * r
                * jnp.sum(
                    w2
                    * (r * jnp.sin(x2)) ** 2
                    * cls._density(r * jnp.sin(x2), params)
                    * jnp.cos(x2),
                    axis=0,
                )
            )

            return m_rs + m_r

        return jax.lax.cond(
            r <= rs,
            r_le_rs,
            r_gt_rs,
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def test_potential(cls, r, params):
        r"""
        Calculates the potential at a given radius r numerically

        .. math::
            \phi(r) = -4\pi G\left[\frac{M}{r}+ \int_{r}^{\infty} \rho(r) r dr \right]

        Parameters
        ----------
        r : _type_
            float

        Returns
        -------
        float
            potential at a given radius r | [kpc^{2}~s^{-2}]
        """
        G = JaxPotential.G
        q = r
        phi_in = cls._mass(r, params) / r

        # we'll use Jax's flow control to split up integration into two regimes r << rs and r >> rs
        # this will avoid numerical issues with the trig substitution for r << rs
        # for r >> rs, the trig substitution is needed in order to turn the integral into a finite integral
        # TODO: for when there is a limit on the upper bound of the integral we again swith integration schemes - yet to be implemented
        x, w = cls._xk, cls._wk
        x0 = 0.0
        x1 = jnp.pi / 2
        xk = 0.5 * (x1 - x0) * x + 0.5 * (x1 + x0)
        wk = 0.5 * (x1 - x0) * w

        def less_than_rs():
            """
            for r << rs, (\frac{r}{r_{s}} < 1e-5, depending on the order of the integration) the trig substitution usually used leads to numerical issues.
            Instead just use regular Gauss-Legendre quadrature from r to rs. then use the trig substitution to
            """
            # x0_ = r
            # x1_ = params["rs"]
            # xi = 0.5 * (x1_ - x0_) * x + 0.5 * (x1_ + x0_)
            # wi = 0.5 * (x1_ - x0_) * w

            # # integrate from r to rs
            # t1 = jnp.sum(wi * cls._density(xi, params) * xi, axis=0)
            # x0 = jnp.arcsin(r / params["rs"])
            # x1 = jnp.pi / 2

            xi = 0.5 * (x1 - x0) * x + 0.5 * (x1 + x0)
            wi = 0.5 * (x1 - x0) * w
            t1 = r * jnp.sum(
                wi
                * cls._density(r / jnp.sin(xi), params)
                * r
                * jnp.cos(xi)
                / jnp.sin(xi) ** 3,
                axis=0,
            )
            # integrate from rs to infinity
            t2 = params["rs"] * jnp.sum(
                wk
                * cls._density(params["rs"] / jnp.sin(xk), params)
                * params["rs"]
                * jnp.cos(xk)
                / jnp.sin(xk) ** 3,
                axis=0,
            )  # integrate from rs to infinity

            # sum the two integrals
            return t1 + t2

        def greater_than_rs():
            """
            fro r > rs, the trig substitution behaves well and transforms the infinity integral into a finite one
            """
            phi_out = q * jnp.sum(
                wk
                * cls._density(q / jnp.sin(xk), params)
                * q
                * jnp.cos(xk)
                / jnp.sin(xk) ** 3,
                axis=0,
            )
            return phi_out

        phi_out = jax.lax.cond(r <= params["rs"], less_than_rs, greater_than_rs)

        return -G * (phi_in + 4 * jnp.pi * phi_out)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _potential(cls, r, params):
        G = JaxPotential.G
        q = r
        phi_in = cls._mass(r, params) / r
        x, w = cls._xm, cls._wm
        x0, x1 = 0.0, jnp.pi / 2
        xk = 0.5 * (x1 - x0) * x + 0.5 * (x1 + x0)
        wk = 0.5 * (x1 - x0) * w

        def very_small_r(r, params):
            """
            For r < 1e-4, we use a specialized integration technique or approximation
            """
            # Implement your integration technique for very small r
            # This is a placeholder implementation; adjust as needed
            lambda_ = jnp.log(params["rs"] / r)

            def transformed_integrand(u):
                y = r * jnp.exp(lambda_ * u)
                return cls._density(y, params) * y**2 * lambda_

            u_min, u_max = 0, 1
            u_vals = 0.5 * (u_max - u_min) * (x + 1) + u_min
            weights = 0.5 * (u_max - u_min) * w
            t1 = jnp.sum(weights * transformed_integrand(u_vals), axis=0)

            t2 = params["rs"] * jnp.sum(
                wk
                * cls._density(params["rs"] / jnp.sin(xk), params)
                * params["rs"]
                * jnp.cos(xk)
                / jnp.sin(xk) ** 3,
                axis=0,
            )
            return t1 + t2

        def less_than_rs(r, params):
            x0_ = r
            x1_ = params["rs"]
            xi = 0.5 * (x1_ - x0_) * x + 0.5 * (x1_ + x0_)
            wi = 0.5 * (x1_ - x0_) * w
            t1 = jnp.sum(wi * cls._density(xi, params) * xi, axis=0)

            t2 = params["rs"] * jnp.sum(
                wk
                * cls._density(params["rs"] / jnp.sin(xk), params)
                * params["rs"]
                * jnp.cos(xk)
                / jnp.sin(xk) ** 3,
                axis=0,
            )
            return t1 + t2

        def greater_than_rs(r, params):
            return q * jnp.sum(
                wk
                * cls._density(q / jnp.sin(xk), params)
                * q
                * jnp.cos(xk)
                / jnp.sin(xk) ** 3,
                axis=0,
            )

        # Nested conditional to handle three cases
        def inner_condition(r, params):
            return jax.lax.cond(
                r <= params["rs"],
                lambda r, p: less_than_rs(r, p),
                lambda r, p: greater_than_rs(r, p),
                r,
                params,
            )

        phi_out = jax.lax.cond(
            r <= 1e-1,
            lambda r, p: very_small_r(r, p),
            lambda r, p: inner_condition(r, p),
            r,
            params,
        )

        return -G * (phi_in + 4 * jnp.pi * phi_out)

    def v_circ(self, r):
        r"""
        circular velocity at a given radius r

        .. math::
            v_{circ} = \sqrt{r\frac{d\phi}{dr}} = \sqrt{\frac{GM(r)}{r}}

        Parameters
        ----------
        r : _type_
            _description_
        """
        return jnp.sqrt(self._G * self.mass(r) / r)

    def v_esc(self, r):
        """
        Calculates the escape velocity at a given radius r

        .. math::
            v_{esc} = \sqrt{-2\phi(r)}

        Parameters
        ----------
        r : _type_
            _description_
        """
        return jnp.sqrt(-2 * self.potential(r))

    # @partial(jax.jit, static_argnums=(0,))
    def total_energy(self, x, v):
        """
        Calculates the total energy per unit mass

        Parameters
        ----------
        x : _type_
            _description_
        v : _type_
            _description_

        Returns
        -------
        float
            total energy per unit mass | [km^{2}~s^{-2}]
        """
        T = 0.5 * jnp.dot(v, v)  # kinetic energy
        r = jnp.linalg.norm(x, axis=0)

        return T + self.potential(r)

    @partial(jax.jit, static_argnums=(0,))
    def gamma(self, r):
        r"""
        log-slope of density profile.

        .. math:
            \gamma = -\frac{d\log(\rho)}{d\log(r)}`

        Parameters
        ----------
        r : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        grad = jax.grad(self.density)
        return -r * grad(r) / self.density(r)

    def r200(self):
        """

        Calculate r200 between 1e-2 kpc and 300 kpc
        I'm assuming thats a safe range in which r200 for a dwarf galaxy would be, but I COULD be wrong

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        check_bracket = False otherwise the function is not jittable
        This might actually be more useful
        BUG: lower=1e-20 does not work with the NFW class for some reason even though its analytical... must look into this at some point
        """

        def func(x: float):
            r"""
            helper function for calculating r_{200} for a given density/mass profile.

            Parameters
            ----------
            x : float
                Has to be in kpc for the units to work out
                but it cant have units otherwise the function won't work..

            Returns
            -------
            float
                average density - 200\rho_{crit}

            Notes
            -----
            This doesnt really belong here -- at least not in this structure
            """
            # x = jnp.atleast_1d(x)
            rho_crit = 133.3636319527206  # critical density from astropy's WMAP9 with units [solMass/kpc**3]
            Delta = 200.0
            return self.mass(x) / (4 * jnp.pi * (x**3) / 3) - Delta * rho_crit

        bisec = Bisection(
            optimality_fun=func, lower=1e-10, upper=300, check_bracket=False
        )

        return bisec.run().params

    def rtidal(self, r_dsph):
        """
        _summary_

        Parameters
        ----------
        r_dsph : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        from . import models

        def func(r, r_dsph):
            # e.g. gala's default parameters for the Milky Way Halo
            mw_ms = 5.4e11  # Msun - scale mass
            mw_rs = 15.62  # kpc  - scale radius

            dSph_mass = self.mass(r)[-1]
            mw_mass = models.NFW.from_ms_rs(mw_ms, mw_rs).mass(r_dsph - r)
            return dSph_mass * (r_dsph - r) ** 3 / r**3 - mw_mass

        bisec = Bisection(
            optimality_fun=func,
            lower=1e-10,
            upper=r_dsph,
            check_bracket=False,
            maxiter=50,
            tol=1e-10,
        )
        return bisec.run(r_dsph=r_dsph).params

    def r200_notjax(self):
        """
        Radius at which the average density of the halo is equal to 200 times the critical density

        Notes
        -----
        This one works on models that aren't jax friendly
        """
        rho_crit = cosmo.critical_density(0).to(u.solMass / u.kpc**3).value
        Delta = 200

        def func(x):
            return (
                self.get_mass(x) / (4 * np.pi * x**3 / 3) - Delta * rho_crit
            )  # set average density - 200* rho_crit = 0

        self._r200 = scipy.optimize.bisect(func, 1e-2, 300)
        self._M200 = self.get_mass(self._r200)
        return self._r200

    def sample_w(
        self,
        N: int,
        r_range=[0, np.inf],
        v_range=[0, None],
        nwalkers: int = 32,
        N_burn: int = 10_000,
    ) -> np.ndarray:
        r"""
        Use emcee to generate samples of :math:`vec{w} = (\vec{x},\vec{v}) from the distribution function f(vec{w}) = f(E)`

        Parameters
        ----------
        N : int
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        G = 4.300917270036279e-06  # gravitational constant in units of kpc km^2 s^-2 Msol^-1

        def df(w):
            x = w[:3]
            v = w[3:]
            r = np.linalg.norm(x)
            E = self.potential(r) + 0.5 * np.dot(v, v)
            return self.logDF(E)

        def log_prior(w):
            x = w[:3]
            v = w[3:]
            r = np.linalg.norm(x)

            Energy = self.potential(r) + 0.5 * np.dot(v, v)

            # if (Energy < 0) and (Energy > -G*self._M/self._a): and (r < 50):
            if (Energy < 0) and (r < 70) and (r > 1e-5):
                # TODO: must change r < 70 to something like r < r_200, also include v_esc?s
                return 0
            return -np.inf

        def log_probability(w):
            lp = log_prior(w)
            if not np.isfinite(lp):
                return -np.inf
            return lp + df(w)

        ndim = 6
        r_temp = np.logspace(-3, 1, nwalkers)
        # print(r_temp.shape)
        v_temp = self.v_circ(r_temp)

        x, y, z = self.spherical_to_cartesian(r_temp)
        vx, vy, vz = self.spherical_to_cartesian(v_temp)

        p0 = np.array([x, y, z, vx, vy, vz]).reshape(nwalkers, ndim)

        # p0 = np.random.rand(nwalkers, ndim) # need to make p0 better
        # print(p0.shape)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        state = sampler.run_mcmc(p0, N_burn, progress=True)
        sampler.reset()
        sampler.run_mcmc(state, N, progress=True)
        samples = sampler.get_chain(flat=True)
        # print(
        #     "Mean acceptance fraction: {0:.3f}".format(
        #         np.mean(sampler.acceptance_fraction)
        #     )
        # )
        # print(
        #     "Mean autocorrelation time: {0:.3f} steps".format(
        #         np.mean(sampler.get_autocorr_time())
        #     )
        # )
        return samples

    def spherical_to_cartesian(self, quantity):
        N = len(quantity)
        xyz = np.zeros((3, N))
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        temp = np.random.uniform(size=N)
        theta = np.arccos(1 - 2 * temp)

        xyz[0] = quantity * np.cos(phi) * np.sin(theta)
        xyz[1] = quantity * np.sin(phi) * np.sin(theta)
        xyz[2] = quantity * np.cos(theta)
        return xyz

    @partial(jax.jit, static_argnums=(0,))
    def project(self, R):
        r"""
        Forward Abel transform : Project an optically thin, spherically symmetric function onto a plane.
        :math:`f(r(x,y,z))\rightarrow F(R(x,y))`

        .. math::
            F(R)=2 \int_R^{\infty} \frac{f(r) r}{\sqrt{r^2-R^2}} d r

        Parameters
        ----------
        func : Callable
            function of projected (on-sky) radii, R
        R : jax.Array
            Projected distance :math:`(R = \sqrt{x^2 +y^2})`

        params :
            parameters for func

        Returns
        -------
        jax.Array
            projected positions
        """

        f_abel = lambda x, q: 2 * self.density(x) * x / jnp.sqrt(x**2 - R**2)
        # integral goes from R to infinity
        # use u-sub wit u = R csc(theta) -> du = -R cos(theta)/sin^2(theta) dtheta
        x0 = 0
        x1 = jnp.pi / 2
        # x,w = np.polynomial.legendre.leggauss(100) # are these tabulated -- does it take a long time to calculate?

        xk = 0.5 * (x1 - x0) * self._xj + 0.5 * (x1 + x0)
        wk = 0.5 * (x1 - x0) * self._wj
        return np.sum(
            wk * f_abel(R / jnp.sin(xk), R) * R * jnp.cos(xk) / jnp.sin(xk) ** 2, axis=0
        )

    @partial(jax.jit, static_argnums=(0,))
    def deproject(self, r):
        r"""
        Inverse Abel transform is used to calculate the emission function given a projection, i.e.
        :math:`F(R(x,y)) \rightarrow f( r(x,y,z) )`

        .. math::
            f(r)=-\frac{1}{\pi} \int_r^{\infty} \frac{d F}{d y} \frac{d y}{\sqrt{y^2-r^2}}

        Parameters
        ----------
        func : Callable
            function of :math:`r = \sqrt{x^2+y^2+z^2}.` This must be written in a jax friendly way so that it can be differentiated
        R : np.ndarray
            _description_
        params : _type_
            _description_

        Returns
        -------
        np.ndarray
            _description_
        """
        # write everything out for clarity
        x0 = 0
        x1 = jnp.pi / 2

        xk = 0.5 * (x1 - x0) * self._xk + 0.5 * (x1 + x0)
        wk = 0.5 * (x1 - x0) * self._wk

        dfunc = jax.vmap(jax.grad(self.project, argnums=0))

        # dfunc = jax.vmap(jax.grad(self.projected_density,argnums=0))
        # return dfunc(r)
        def f_abel(y, q):
            return -1 * dfunc(y) / jnp.sqrt(y**2 - q**2) / jnp.pi

        return np.sum(
            wk * f_abel(r / jnp.sin(xk), r) * r * jnp.cos(xk) / jnp.sin(xk) ** 2, axis=0
        )

    @classmethod
    def set_Norder(cls, N):
        new_xk, new_wk = np.polynomial.legendre.leggauss(N)
        cls._xk = jnp.array(new_xk)
        cls._wk = jnp.array(new_wk)

    @classmethod
    def get_tracer_priors(cls):
        """
        TODO: Add instance for Normal distribution
        """
        for param_name, distribution in cls._tracer_priors.items():
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

    @classmethod
    def set_tracer_priors(cls, new_priors):
        cls._tracer_priors = new_priors

    @classmethod
    def get_dm_priors(cls):
        for param_name, distribution in cls._dm_priors.items():
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

    @classmethod
    def set_dm_priors(cls, new_priors):
        cls._dm_priors = new_priors

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _peri(cls, pars, x0, v0):
        r0 = jnp.linalg.norm(x0, axis=0)
        bisec = Bisection(
            optimality_fun=cls._centre, lower=1e-12, upper=r0, check_bracket=False
        )
        r_peri = bisec.run(x0=x0, v0=v0, pars=pars).params
        return r_peri

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _apo(cls, pars, x0, v0):
        r0 = jnp.linalg.norm(x0, axis=0)
        bisec2 = Bisection(
            optimality_fun=cls._centre, lower=r0, upper=500, check_bracket=False
        )
        r_apo = bisec2.run(x0=x0, v0=v0, pars=pars).params
        return r_apo

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _centre(cls, r, x0, v0, pars) -> float:
        """
        paper eq. 4 -- subject to change

        """
        L = jnp.cross(x0, v0)  # angular momentum
        T = 0.5 * jnp.dot(v0, v0)  # kinetic energy
        r0 = jnp.linalg.norm(x0, axis=0)
        # print(r0)s
        energy = T + cls._potential(*pars, r=r0)  # total energy

        return 2 * r**2 * (energy - cls._potential(*pars, r=r)) - jnp.dot(L, L)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _action_r(
        cls,
        params,
        x: jax.Array,
        v: jax.Array,
    ) -> float:
        """
        Integrand for radial action (B-T 2nd Ed. - eq. 3.224)

        .. math::
            J_{r}(\boldsymbol{x}, \boldsymbol{v} \mid \Phi)=\frac{1}{2 \pi} \oint v_{r} \mathrm{~d} r=\frac{1}{\pi} \int_{r_{-}}^{r_{+}}\left[2(E-\Phi)-\frac{L^{2}}{r^{2}}\right]^{1 / 2} \mathrm{~d} r

        Notes
        -----

        Parameters
        ----------
        x : jax.Array
            position vector | [kpc]
        v : jax.Array
            velocity vector | [km s^{-1}]

        Returns
        -------
        float
            radial action | [kpc km s^{-1}]
        """
        x0 = cls._peri(params, x, v)
        x1 = cls._apo(params, x, v)

        # Gauss-Legendre integration
        xi = 0.5 * (x1 - x0) * JaxPotential._xm + 0.5 * (x1 + x0)
        wi = 0.5 * (x1 - x0) * JaxPotential._wm

        L = jnp.linalg.norm(jnp.cross(x, v), axis=0)  # abs(Angular momentum)
        T = 0.5 * jnp.dot(v, v)
        r0 = jnp.linalg.norm(x, axis=0)

        energy = T + cls._potential(*params, r=r0)  # total energy
        term2 = cls._potential(*params, r=xi)

        term1 = 2 * (energy - term2)
        term3 = L**2 / xi**2  # takes out the kpc units leaves it as km**2/s**2
        out = jnp.sqrt(term1 - term3) / jnp.pi
        return jnp.sum(wi * out)

    @staticmethod
    @jax.jit
    def action_theta(x: jax.Array, v: jax.Array) -> jax.Array:
        r"""
        Polar action
        :math:`x` and :math:`v` are the position and velocity vectors in cartesian coordinates

        .. math::
            J_\theta(\boldsymbol{x}, \boldsymbol{v} \mid \Phi)=L-\left|L_\phi\right|,

        where :math:`L_\phi` is the azimuthal component of the angular momentum and :math:`L = |\boldsymbol{x}\times\boldsymbol{v}|` is the magnitude of the angular momentum
        """
        L_vector = jnp.cross(x, v)
        L_phi = L_vector[2]
        L = jnp.sqrt(jnp.dot(L_vector, L_vector))
        return L - L_phi

    @staticmethod
    @jax.jit
    def action_phi(x, v):
        r"""
        Azimuthal action

        .. math::
            J_{\phi}(\boldsymbol{x}, \boldsymbol{v} \mid \Phi)=\frac{1}{2 \pi} \oint p_{\phi} \mathrm{d} \phi=L_{\phi}
        :math:`L_{\phi}= xv_{y} - yv_{x}` is the z-component of the angular momentum

        """
        L_phi = x[0] * v[1] - x[1] * v[0]
        return L_phi

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _dJdOmega(cls, theta, param_dict, d, rt):
        r"""
        .. math:
            \frac{dJ}{d\omega} = \int{\rho^{2}(r) d\ell}


        Parameters
        ----------
        theta : _type_
            _description_
        """
        # First integrate from r_min to l_

        # l_min = d*jnp.cos(theta) - jnp.sqrt(rmin**2 - (d*jnp.sin(theta))**2)

        l0 = d * jnp.cos(theta) - jnp.sqrt(rt**2 - (d * jnp.sin(theta)) ** 2)
        l_mid = d * jnp.cos(theta)
        xi = 0.5 * (l_mid - l0) * cls._xj + 0.5 * (l_mid + l0)
        wi = 0.5 * (l_mid - l0) * cls._wj
        sum1 = jnp.sum(
            wi
            * cls._density_fit(
                jnp.sqrt(xi**2 + d**2 - 2 * xi * d * jnp.cos(theta)), param_dict
            )
            ** 2
        )

        # Now integrate from rm to l+
        l_max = d * jnp.cos(theta) + jnp.sqrt(rt**2 - (d * jnp.sin(theta)) ** 2)
        xv = 0.5 * (l_max - l_mid) * cls._xj + 0.5 * (l_max + l_mid)
        wv = 0.5 * (l_max - l_mid) * cls._wj
        sum2 = jnp.sum(
            wv
            * cls._density_fit(
                jnp.sqrt(xv**2 + d**2 - 2 * xv * d * jnp.cos(theta)), param_dict
            )
            ** 2
        )
        return sum1 + sum2

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _jFactor(cls, theta, param_dict, d, rt):
        r"""
        J-factor
        .. math:
            J= \int\int_{\rm los} \rho^{2}(r) d\Omega dl

        where we write r as :math:`r^2=\ell^2+d^2-2 \ell d \cos \theta` where :math:`\ell` is the distance along the line of sight and d is the distance to the center of the galaxy.
        Spherical symmetry lets us write :math:`d\Omega=2\pi\sin\theta d\theta`.

        The bounds on the line of sight are $\ell_{ \pm}=d \cos \theta \pm \sqrt{r_{200}^2-d^2 \sin ^2 \theta}$.
        where $r_{200}$ is the 'size' of the system.

        substituting in the integral we get
        $$
        J(\theta_{\rm max}) = 2 \pi \int_{0}^{\theta_{\rm max}} \int_{\ell_{-}}^{\ell_{+}} \left[\rho\left(\sqrt{\ell^2+d^2-2 \ell d \cos \theta}\right)\right]^{2}\sin(\theta) d\ell d\theta
        $$

        Parameters
        ----------
        theta : float
            _description_
        d : float
            distance to system | [kpc]
        rt : float
            tidal radius of system | [kpc]

        Returns
        -------
        float
            J-factor | [:math:`M_{\odot}^{2}~kpc^{-5}]`

        Notes
        -----
        Should rt just be r_200? should i just change the bounds on the line of sight part to be -inf to inf?
        Note the Units.
        Maybe I should just return log10(J/[GeV^{2} cm^{-5}]) instead of J?
        """
        # theta = jnp.atleast_1d(theta)
        x0 = 0
        x1 = theta
        xi = 0.5 * (x1 - x0) * cls._xj + 0.5 * (x1 + x0)
        wi = 0.5 * (x1 - x0) * cls._wj
        vectorized_func = jax.vmap(cls._dJdOmega, in_axes=(0, None, None, None))
        return jnp.log10(
            2
            * jnp.pi
            * jnp.sum(wi * vectorized_func(xi, param_dict, d, rt) * jnp.sin(xi))
            * cls._GeV2cm5
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _dDdOmega(self, theta, d, rt):
        """
        _summary_

        Parameters
        ----------
        theta : _type_
            _description_
        d : _type_
            _description_
        rt : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        l0 = d * jnp.cos(theta) - jnp.sqrt(rt**2 - (d * jnp.sin(theta)) ** 2)
        l1 = d * jnp.cos(theta) + jnp.sqrt(rt**2 - (d * jnp.sin(theta)) ** 2)
        xi = 0.5 * (l1 - l0) * self._xj + 0.5 * (l1 + l0)
        wi = 0.5 * (l1 - l0) * self._wj
        return jnp.sum(
            wi * self.density(jnp.sqrt(xi**2 + d**2 - 2 * xi * d * jnp.cos(theta)))
        )

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _dFactor(cls, param_dict, theta, d, rt):
        """
        Decay Factor

        Parameters
        ----------
        theta : _type_
            _description_
        d : _type_
            _description_
        rt : _type_
            _description_

        Returns
        -------
        float
            Decay Factor | [:math:`M_{\odot}~kpc^{-2}]`

        Notes
        -----
        TODO: Decide on the units and log10 or not
        """
        x0 = 0
        x1 = theta
        xi = 0.5 * (x1 - x0) * cls._xj + 0.5 * (x1 + x0)
        wi = 0.5 * (x1 - x0) * cls._wj
        vectorized_func = jax.vmap(cls.dDdOmega, in_axes=(0, None, None, None))
        return jnp.log10(
            2
            * jnp.pi
            * jnp.sum(wi * vectorized_func(xi, d, rt) * jnp.sin(xi), param_dict)
            * cls._GeVcm2
        )

    def __add__(self, other):
        if not isinstance(other, JaxDensity):
            return NotImplemented
        return CompositePotential(self, other)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)


class JaxPotential(JaxDensity):
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _potential(cls, r, params):
        raise NotImplementedError

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _mass(cls, r, params):
        """
        Define from potential function

        .. math::
            M(r) = r^{2} \frac{d\phi}{dr}

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        dphi_dr = jax.grad(cls._potential)
        return r**2 * dphi_dr(r, params)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _density(cls, r, params):
        """
        ..math::
            \rho(r) = -\frac{1}{4\pi G} \nabla^{2} \phi(r)

        Parameters
        ----------
        r : _type_
            _description_
        params : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        d2phi_dr2 = jax.grad(jax.grad(cls._potential))
        return -1 / (4 * jnp.pi * cls._G) * d2phi_dr2(r, params)


class JaxMass(JaxDensity):
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _mass(cls, r, params):
        raise NotImplementedError

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _density(cls, r, params):
        """
        _summary_

        Parameters
        ----------
        r : _type_
            _description_
        params : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        dM_dr = jax.grad(cls._mass)
        return dM_dr(r, params) / (4 * jnp.pi * r**2)


class JaxDistribuitionFunction(JaxDensity):
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def df(cls, E, params):
        """
        _summary_

        Parameters
        ----------
        E : _type_
            _description_
        params : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        raise NotImplementedError

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _potential(cls, r, params):
        """

        Parameters
        ----------
        r : _type_
            _description_
        params : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        raise NotImplementedError
        pass


class CompositePotential(JaxPotential):
    """
    Composite potential class that combines multiple potentials into a single potential.

    There are two ways to combine potentials:

    1. Add potentials using the '+' operator.
    2. Use the CompositePotential class to combine multiple potentials.

    Functions Here just sum the density, mass, and potential of each potential in the composite potential.
    """

    def __init__(self, *potentials):
        self.potentials = potentials
        super().__init__()

        # Merge _dm_priors from individual potentials with unique prefixes
        self._dm_priors = {}
        for i, p in enumerate(potentials):
            for key, value in p._dm_priors.items():
                unique_key = f"{p.__class__.__name__}_{i}_{key}"
                self._dm_priors[unique_key] = value

    def density(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Sum the density of each potential in the composite potential.

        Parameters
        ----------
        r : jnp.ndarray
            radii at which to calculate the density units of [kpc]

        Returns
        -------
        jnp.ndarray
            density units of :math:`[ \\rm M_{\odot}~kpc^{-3}]`
        """
        return jnp.sum(jnp.array([p.density(r) for p in self.potentials]))

    def mass(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        sum of the mass of each potential in the composite potential.

        Parameters
        ----------
        r : jnp.ndarray
            radii, units of [kpc]

        Returns
        -------
        jnp.ndarray
            mass units of :math:`[ \\rm M_{\odot}]`
        """
        return jnp.sum(jnp.array([p.mass(r) for p in self.potentials]))

    def potential(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        sum of the potential of each potential in the composite potential.

        Parameters
        ----------
        r : jnp.ndarray
            radii, units of [kpc]

        Returns
        -------
        jnp.ndarray
            potential units of :math:`[ \\rm km^{2}~s^{-2}]`
        """
        return jnp.sum(jnp.array([p.potential(r) for p in self.potentials]))

    def __str__(self):
        description = "CompositePotential combining the following potentials:\n"
        for i, p in enumerate(self.potentials):
            description += f"  {i+1}. {p.__class__.__name__} with parameters:\n"
            for key, value in p._dm_priors.items():
                prior_str = self._format_prior(value)
                description += f"      {key}: {prior_str}\n"
            # Assuming each potential class has a method or attribute to get its parameters
            if hasattr(p, "_params"):
                description += f"      parameters: {p._params}\n"
            else:
                description += "      parameters: Not available\n"
        return description

    def _format_prior(self, prior):
        if isinstance(prior, dist.Uniform):
            return f"Uniform(low={prior.low}, high={prior.high})"
        # Add other distributions as needed
        # elif isinstance(prior, dist.Normal):
        #     return f"Normal(mean={prior.mean}, std={prior.std})"
        else:
            return str(prior)

    def __repr__(self):
        return (
            f"<CompositePotential with {len(self.potentials)} potentials: "
            + ", ".join([p.__class__.__name__ for p in self.potentials])
            + ">"
        )


class Data(ABC):
    def __init__(self):
        if not hasattr(self, "_r"):
            warnings.warn(
                r"The '_r' component is not defined.\nOnly projected moments may be available."
            )

        if not hasattr(self, "_pmr") or not hasattr(self, "_pmt"):
            warnings.warn(
                r"'pmr' and 'pmt' components are not defined.\nVelocity dispersion for these components will not be available.\n only 'los' moment will be available"
            )

        self._cached_dispersion = {}  # Initialize cache dictionary if it doesn't exist # could put this in an __init__function

        # required_components = ('_r', '_vr', '_vtheta', '_vphi', '_R', '_vlos', '_pmr', '_pmt')
        # if any(getattr(self, comp) is None for comp in required_components):
        #     warnings.warn("Some required instance variables are not defined.\nThis will limit the functionality of this function.")

        self.N_star = len(self._R)
        self._component_map = self.construct_component_map()

    def construct_component_map(self):
        component_map = {}

        if hasattr(self, "_r") and hasattr(self, "_vr"):
            component_map["radial"] = (self._r, self._vr, getattr(self, "d_vr", None))

        if hasattr(self, "_r") and hasattr(self, "_vtheta"):
            component_map["theta"] = (
                self._r,
                self._vtheta,
                getattr(self, "d_vtheta", None),
            )

        if hasattr(self, "_r") and hasattr(self, "_vphi"):
            component_map["phi"] = (self._r, self._vphi, getattr(self, "d_vphi", None))

        if hasattr(self, "_R") and hasattr(self, "_vlos"):
            component_map["los"] = (self._R, self._vlos, getattr(self, "d_vlos", None))

        if hasattr(self, "_R") and hasattr(self, "_pmr"):
            component_map["pmr"] = (self._R, self._pmr, getattr(self, "d_pmt", None))

        if hasattr(self, "_R") and hasattr(self, "_pmt"):
            component_map["pmt"] = (self._R, self._pmt, getattr(self, "d_pmr", None))

        return component_map

    def dispersion_i(
        self,
        component: str = "los",
        binfunc: Callable = histogram,
        bins="blocks",
        clear_cache: bool = False,
        plot=False,
    ):
        r"""
        Calculate the dispersion of a given component e.g

        r_center,dispersion_i, dispersion_i_err = dispersion_i('los')

        Parameters
        ----------
        component : str, optional
            The component for which to calculate the dispersion.
            Available components: 'radial', 'theta', 'phi', 'los', 'pmr', 'pmt'.
            Defaults to 'los'.
        binfunc : Callable, optional
            function used to bin data, by default histogram.
            As in astropy.stats.histogram
        bins : int or sequence of scalars or str, optional
            bin edges, by default 'blocks'

        Returns
        -------
        result : tuple
        A tuple containing three values:
            - r_center : array
              The center values for the dispersion bins.
            - dispersion_i : array
              The calculated dispersion for the specified component.
            - dispersion_error : array
              The uncertainty associated with the dispersion calculation.

        Raises
        ------
        Warning
            If the component type is unknown or missing.

        Notes
        -----
        - This function utilizes caching to store and reuse previously calculated dispersion results.

        Examples
        --------
        >>> # Use mock data set
        >>> dataSet = pd.read_csv('data/gs010_bs050_rcrs025_rarcinf_cusp_0064mpc3_df_1000_3_err.dat')
        >>> error = np.full_like(dataSet['x'],2.0)
        >>> # append error to dataSet
        >>> dataSet['error'] = error
        >>> obj = data.Mockdata(dataset)
        >>> r_center, dispersion_i, dispersion_error = obj.dispersion_i('radial')
        >>> r_center, dispersion_default, dispersion_error = obj.dispersion_i()  # Calculate dispersion for the default component ('los')
        """

        component = (
            component.casefold()
        )  # Just in case there's random capitalizations by the user

        if component in self._cached_dispersion and not clear_cache:
            print("here actually")
            return self._cached_dispersion[component]

        if component in self._component_map:
            data = self._component_map[component]
            out = self.dispersion(
                *data, component, binfunc=binfunc, bins=bins, plot=False
            )
            print(out)
            self._cached_dispersion[component] = out
            print("here")
            return out

        else:
            warnings.warn(
                "\nUnknown or missing component type.\n make sure you have the right data in order to calculate this component of the dispersion"
            )

        return None  # Optionally return a value if needed

    def global_dispersion(self, ri, vi, d_vi, component=None, plot=False):
        def global_velocity_model(vi, error):
            # Priors for mean velocity and global dispersion
            mean_velocity = numpyro.sample("mean_velocity", dist.Uniform(-100, 100))
            global_sigmav = numpyro.sample("global_sigmav", dist.Uniform(0, 100))

            # Likelihood for each observed velocity
            with numpyro.plate("data", len(vi)):
                total_variance = jnp.sqrt(global_sigmav**2 + error**2)
                numpyro.sample(
                    "obs", dist.Normal(mean_velocity, total_variance), obs=vi
                )

        # MCMC
        rng_key = random.PRNGKey(0)
        kernel = NUTS(global_velocity_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
        mcmc.run(rng_key, vi, d_vi)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        self.mean_velocity = samples["mean_velocity"]
        self.global_sigmav = samples["global_sigmav"]

        meanv, sigma = jnp.mean(self.mean_velocity), jnp.mean(self.global_sigmav)

        # Plotting
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            # Histogram of velocities
            N, bin_edges = histogram(vi, bins="blocks")
            import numpy as np

            # import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import gaussian_kde

            sns.histplot(vi, bins=30, kde=False, color="skyblue", stat="density")

            # Density Plot with KDE
            kde = gaussian_kde(vi, weights=1 / (d_vi**2))
            x = np.linspace(min(vi), max(vi), 1000)
            kde_values = kde(x)
            # Confidence interval (e.g., 95%)
            ci = np.percentile(kde_values, [2.5, 97.5])
            plt.fill_between(
                x,
                kde_values,
                where=(kde_values >= ci[0]) & (kde_values <= ci[1]),
                color="gray",
                alpha=0.5,
            )

            ax.hist(vi, bins=bin_edges, density=True, alpha=0.6, color="g")

            # Best-fit Gaussian curve
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = scipy.stats.norm.pdf(x, meanv, sigma)
            ax.plot(x, p, "k", linewidth=2)

            title = "Fit results: mean = {:.2f},  std = {:.2f}".format(meanv, sigma)
            plt.title(title)
            plt.xlabel("Velocity")
            plt.ylabel("Density")

            plt.show()
        return jnp.mean(self.mean_velocity), jnp.mean(self.global_sigmav)

    def dispersion(
        self,
        ri,
        vi,
        d_vi,
        component=None,
        binfunc: Callable = histogram,
        bins="blocks",
        plot=False,
    ):
        """
        calculates the dispersion of velocity component, v_i at radii r
        -- or at project radius R. Also calcualtes the
        error (see dispersion_error() for details)

        Parameters
        ----------
        v_i : _type_
            component of velocity you want to calculate the dispersion of
            e.g vx,vy,vz,vr,vtheta,vphi etc...
        binfunc : Callable, optional
        binning function -- must return two values like np.histogram, by default histogram
        bins : str, optional
            binning scheme I guess , by default 'blocks'

        Returns
        -------
        _type_
            _description_

        """

        # first bin positions -- Note: this will only take anytime the first time this function is run -- results are then cached
        N, bin_edges = binfunc(ri, bins=bins)

        r_center = (
            0.5 * (bin_edges[:-1] + bin_edges[1:])
        )  # Using the center of the bins seems like an ok method -- a mean or median might be better though

        # Next get global dispersion and mean velocity
        meanv, sigma = self.global_dispersion(ri, vi, d_vi, component, plot=plot)

        N, _, binnumber = binned_statistic(ri, vi, statistic="count", bins=bin_edges)

        num_bins = len(bin_edges) - 1

        def velocity_dispersion_model(ri, vi, error, global_dispersion, bin_edges):
            # Number of bins
            num_bins = len(bin_edges) - 1

            # Dispersion hyperprior based on global dispersion
            with numpyro.plate("dispersion_hyperprior", num_bins):
                dispersion = numpyro.sample(
                    "dispersion", dist.HalfNormal(global_dispersion)
                )

            # Use binnumber to assign data to bins
            for i in range(1, num_bins + 1):
                bin_mask = binnumber == i
                bin_vi = vi[bin_mask]
                bin_error = error[bin_mask]

                with numpyro.plate(f"bin_{i}", len(bin_vi)):
                    obs_name = f"obs_{i}"
                    numpyro.sample(
                        obs_name,
                        dist.Normal(
                            meanv, jnp.sqrt(dispersion[i - 1] ** 2 + bin_error**2)
                        ),
                        obs=bin_vi,
                    )

        # MCMC
        rng_key = random.PRNGKey(0)
        kernel = NUTS(velocity_dispersion_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
        mcmc.run(rng_key, ri, vi, d_vi, sigma, bin_edges)
        mcmc.print_summary()
        samples = mcmc.get_samples()
        self.dispersion = samples["dispersion"]
        # Assuming 'mcmc' is your MCMC object after running the sampling
        samples = mcmc.get_samples()
        dispersion_samples = samples["dispersion"]
        # mean_velocity_samples = samples['mean_velocity']
        dispersion_means = jnp.mean(dispersion_samples, axis=0)

        # Calculate the standard deviation
        dispersion_std = jnp.std(dispersion_samples, axis=0)

        # Plotting
        if plot:
            # Create a new figure and set of subplots
            fig, ax = plt.subplots(figsize=(10, 6))

            # Histogram of velocities for each bin
            for i in range(1, num_bins + 1):
                bin_mask = binnumber == i
                bin_vi = vi[bin_mask]

                # Plot histogram for each bin
                ax.hist(bin_vi, bins=30, density=True, alpha=0.6, label=f"Bin {i}")

                # Best-fit Gaussian curve for each bin
                bin_meanv = meanv  # If mean is constant across bins
                bin_dispersion = jnp.mean(self.dispersion[i - 1])
                xmin, xmax = min(bin_vi), max(bin_vi)
                x = np.linspace(xmin, xmax, 100)
                p = scipy.stats.norm.pdf(x, bin_meanv, bin_dispersion)
                ax.plot(x, p, linewidth=2)

            ax.set_title("Velocity Histogram and Gaussian Fits for Each Bin")
            ax.set_xlabel("Velocity")
            ax.set_ylabel("Density")
            ax.legend()
            plt.show()

        return (
            jnp.array(r_center),
            dispersion_means,
            jnp.array(dispersion_std),
            bin_edges,
        )

    def dispersion_errors(
        self, ri, vi, d_vi, component, bin_edges, Nmonte: int, Nbins, error=None
    ):
        """
        Very Basic Error calculation for dispersion

        Parameters
        ----------
        Nmonte : int
            Number of times to sample errors  <= Number of observations

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        Maybe I should do this using EMCEE instead or Numpyro I guess
        TODO: add observational error as input? atm its just set to 2
        """
        # if (error.any() == None): error= 2 # gotta change that 2 but need real errors first
        N = self.N_star
        stemp = np.zeros((Nmonte, len(Nbins)))
        meanv = jnp.mean(vi)  # Probably shouldn't assume that v_{com} = 0
        for i in range(Nmonte):
            # Draw new velocities by sampling a gaussian distribution
            # centered at the measured velocity with a sigma = observational error
            vlos_new = vi + error * np.random.normal(size=N)
            # next calculate the dispersion of the new velocities
            mu, _, _ = binned_statistic(
                ri, (meanv - vlos_new) ** 2, "mean", bins=bin_edges
            )
            stemp[i] = mu

        meanv = jnp.mean(vi)  # Probably shouldn't assume that v_{com} = 0
        mu, _, bin_numbers = binned_statistic(
            ri, (meanv - vi) ** 2, "mean", bins=bin_edges
        )
        N, _, bin_numbers = binned_statistic(ri, ri, "count", bins=bin_edges)

        # Get standard deviation of the Nmonte iterations
        mError = np.std(stemp, axis=0)
        # Add Poisson error
        self.s_error = np.sqrt(mError**2 + mu**2 / N)
        return self.s_error

    def plot_dispersion_i(
        self,
        component=None,
        binfunc: Callable = histogram,
        bins="blocks",
        ax=None,
        fig=None,
    ):
        if component in self._cached_dispersion:
            r_center, dispersion, dispersion_error = self._cached_dispersion[component]

        else:
            r_center, dispersion, dispersion_error = self.dispersion_i(component)

        # fig,ax = plt.subplots()
        ax.step(r_center, dispersion, where="mid")
        ax.errorbar(
            r_center,
            dispersion,
            yerr=dispersion_error,
            fmt="none",
            ecolor="k",
            elinewidth=1.5,
        )
        ax.set(
            xscale="log",
            xlabel="R [kpc]",
            ylabel=r"$\sigma_{los}$ [km/s]",
            ylim=(0, 20),
        )

    def plot_spatial(self, component):
        fig, ax = plt.subplots(ncols=2, nrows=2)
        ax = ax.flatten()
        component = component.lower()

    def fourth_moment(self, ri, vi, d_vi, binfunc: Callable = histogram, bins="blocks"):
        N, bin_edges = binfunc(ri, bins=bins)
        r_center = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        meanv = jnp.mean(vi)  # Probably shouldn't assume that v_{com} = 0
        mu, _, bin_numbers = binned_statistic(
            ri, (meanv - vi.value) ** 4, "mean", bins=bin_edges
        )
        s4_error = self.dispersion_errors(1000)
        # s_error = s4_error/2/np.sqrt(mu)

        return r_center, mu, s4_error

    def spatial_density(
        self, bin_method: Callable = histogram, bins="blocks", dim: int = 2
    ):
        """
        Given a set of N stars with either 2D (default) or 3D(set dim=3),
        compute a spatial density

        Parameters
        ----------
        bin_method : Callable, optional
            binning function (e.g. np.histogram)
            must return just two things so dont use plt.hist
            by default histogram (astropy's histogram)
        bins : str or ndarray, optional, default is 'blocks'
            bin edges -- 'blocks' only works with astropy's histogram method
        dim : int, optional
            Either Calculate, by default 2

        Returns
        -------
        _type_
            _description_
        """
        if dim == 3:
            try:
                N, bin_edges = bin_method(self._r, bins=self.bin_edges)
            except:
                N, bin_edges = bin_method(self._r, bins=bins)

            v_shell = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
            r_center = (np.roll(bin_edges, -1) + bin_edges)[:-1] / 2
            nu = N / v_shell
        else:
            try:
                N, bin_edges = bin_method(self._R, bins=self.bin_edges)
                v_shell = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
                r_center = (np.roll(bin_edges, -1) + bin_edges)[:-1] / 2
                nu = N / v_shell
            except:
                N, bin_edges = bin_method(self._R, bins=bins)
                v_shell = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)
                r_center = (np.roll(bin_edges, -1) + bin_edges)[:-1] / 2
                nu = N / v_shell

        return r_center, nu

    def spatial_nonpar(self, dims=2):
        """
        _summary_
        TODO: write this lol -- see APWs jax implementation maybe?
        Parameters
        ----------
        dims : int, optional
            _description_, by default 2

        Returns
        -------
        _type_
            _description_
        """
        return -1

    def spatial_errors(self, Nmonte: int, errors=0.5):
        stemp = np.zeros((Nmonte, len(self._N)))

        for i in range(Nmonte):
            vlos_new = self.R.value + errors * np.random.normal(size=len(self.R.value))
            mu, _, _ = binned_statistic(
                self.R.value, vlos_new**2, "mean", bins=self.bin_edges
            )
            stemp[i] = mu
        mError = np.std(stemp, axis=0)
        error = np.sqrt(mError**2 + mu**2 / self._N)
        return error

    def bspline_projected(self, R: np.ndarray) -> np.ndarray:
        """
        Calculate the projected density profile of a set of stars
        given their projected radii R

        Parameters
        ----------
        R : np.ndarray
            Projected Radii of stars

        Returns
        -------
        np.ndarray
            projected density profile at radius R
        NOTES
        -----
        """
        chi = np.log(R)
        spl = Data.bspline(self._R)
        return np.exp(spl(chi) - 2 * chi) / (2 * np.pi)
        # return  0.5*np.exp(spl(chi))/(R**2*np.pi)

    def bspline_3d(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate the density profile of a set of stars
        given their radius r

        Parameters
        ----------
        r : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            density profile at radius r
        TODO: clean up notation on this and projected version
        """
        chi = np.log(r)
        spl = Data.bspline(self._r)
        return 0.25 * np.exp(spl(chi) - 3 * chi) / np.pi

    def spherical(self):
        r"""
        Convert cartesian coordinates :math:`(x,y,z,vx,vy,vz)` to spherical coordinates :math:`(r,\theta,\phi,v_{r},v_{\theta},v_{\phi})`

        This is done by the following transformations:
        - :math:`r = \sqrt{x^2+y^2+z^2}`
        - :math:`\theta = arccos(z/r)`
        - :math: `\phi  = arctan2(y,x)  = sign(y)arcos(R/z)`
        - :math:`v_{r} = \frac{xv_{x}+yv_{y}+zv_{z}}{r}`
        - :math:`v_{\phi}= \frac{yv_{x}-xv_{y}}{R}`
        - :math:`v_{\theta] =\frac{\frac{zxv_{x}}[R] + \frac{zyv_{y}}{R} - Rv_{z}}{r}`

        Returns
        -------
        _type_
            _description_
        """
        # (x,y,z) -> r
        self._R = np.sqrt(self._x**2 + self._y**2)  # for conveniences
        self._r = np.sqrt(self._x**2 + self._y**2 + self._z**2)

        # (vx,vy,vz) -> (vr,v_{\theta},v_{\phi})
        self._vr = (
            self._x * self._vx + self._y * self._vy + self._z * self._vz
        ) / self._r
        self._vphi = (self._y * self._vx - self._x * self._vy) / self._R
        self._vtheta = (
            self._z * self._x * self._vx / self._R
            + self._z * self._y * self._vy / self._R
            - self._R * self._vz
        ) / self._r
        # return self._r, self._vr, self._vtheta,self._vphi

    def cylindrical(self):
        r"""
        convert cartesian velocities to cylindrical units

        .. math::
            (v_x,v_y,v_z) \rightarrow (v_{\rho},v_{\phi},v_{z})

        Notes
        -----

        """
        # proper motions
        # using theta
        # theta = jnp.arccos(self._z/self._r) # or jnp.arctan2(self._R,self._z)
        # self._pmr     = self._vr*jnp.sin(theta) + self._vtheta*jnp.cos(theta)
        # self._vz_test = self._vr*jnp.cos(theta) - self._vtheta*jnp.sin(theta)

        self._pmr = self._vr * self._R / self._r + self._vtheta * self._z / self._r
        self._vz_test = self._vr * self._R / self._r + self._vtheta * self._z / self._r
        self._pmt = self._vphi
        # return self._pmr,self._pmt,self._vz_test

    # @classmethod
    def fit_projection(self, profile: Callable, p0=[], bounds=(0, np.inf)):
        R_center, nu = self.spatial_density()
        # fit to projected density profile
        popt, pcov = curve_fit(profile, R_center, nu, bounds=bounds, p0=p0)

        fig, ax = plt.subplots()
        ax.step(R_center, nu, where="mid", label="data")
        ax.plot(R_center, profile(R_center, *popt), label="fit")
        ax.set(
            xlabel="R [kpc]",
            ylabel=r"$\rm\Sigma(R)~N~kpc^{-2}$",
            xscale="log",
            yscale="log",
        )
        return fig, ax, popt, pcov

    def fit_gaussian(self, v_i, error_i, rhalf=None):
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)
        # Run NUTS.
        kernel = NUTS(Data.gaussian)
        num_samples = 3000
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
        mcmc.run(rng_key_, data=v_i, error=error_i, rhalf=rhalf)
        mcmc.print_summary()
        samples_1 = mcmc.get_samples()
        # corner.corner(samples_1,var_names=['mu','sigma','mwolf'],quantiles=[0.16, 0.5, 0.84],show_titles=True,labels=[r'$\mu$',r'$\sigma$',r'$M_{\rm wolf}$'],title_fmt='2.f');
        return samples_1

    def mass_estimator(self, estimator: str = "Wolf", rhalf=None) -> float:
        r"""Use the average velocity dispersion of your data to estimate the mass at the half-light radius
        using the estimator of your choice.

        For radial velocities, you have the option of Wolf, or Walker.

        If your data contains proper motions, you can use the estimator proposed in Errani et al. 2018.

        Parameters
        ----------
        estimator : str, optional
            The estimator to use for the mass calculation
            Options: 'Wolf', 'Walker', 'Errani'
            Defaults to 'Wolf'
        rhalf : float, optional
            The half-light radius of the system
            Defaults to None

        Returns
        -------
        float
            The estimated mass at the half-light radius
        """
        # If rhalf is non then check if it is in the data, if not raise an error
        if rhalf is None:
            if self._rhalf is None:
                raise ValueError("rhalf must be specified")
            else:
                rhalf = self._rhalf

        # fit the velocity data to a gaussian using match case to select the estimator
        match estimator:
            case "Wolf":
                samples = self.fit_gaussian(self._vlos, self._vlos_error, rhalf=rhalf)
                disp = samples["sigma"].mean()

            case "Walker":
                samples = self.fit_gaussian(self._vlos, self._vlos_error, rhalf=rhalf)
                disp = samples["sigma"].mean()

            case "Errani":
                return -1

            case _:
                return -1

    @staticmethod
    def bspline(x: np.ndarray):
        """
        See e.g. Rehemtulla 2022

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        TODO: must make tuning paramaters be an input to this function

        """
        q = np.log(x)  # Definining this for convenience
        r = np.logspace(start=q.min(), stop=q.max(), num=6, base=np.exp(1))  # This make
        chi = np.log(r)
        # chi = np.linspace(np.log(x.min()), np.log(x.max()), 100)
        spl = agama.splineLogDensity(chi, q, w=np.full_like(q, 1))
        return spl

    @staticmethod
    def random_rotation(pos: np.ndarray):
        angles = 360 * np.random.rand(3)
        rot = Rotation.from_euler("xyz", angles, degrees=True)
        return np.dot(rot.as_matrix(), pos)

    @staticmethod
    def to_spherical(pos: np.ndarray, vel: np.ndarray):
        # (x,y,z) -> r
        Rp = np.sqrt(pos["x"] ** 2 + pos["y"] ** 2)  # for conveniences
        r = np.sqrt(pos["x"] ** 2 + pos["y"] ** 2 + pos["z"] ** 2)

        # (vx,vy,vz) -> (vr,v_{\theta},v_{\phi})
        v_r = (pos["x"] * vel["vx"] + pos["y"] * vel["vy"] + pos["z"] * vel["vz"]) / r
        v_phi = (pos["y"] * vel["vx"] - pos["x"] * vel["vy"]) / Rp
        v_theta = (
            pos["z"] * pos["x"] * vel["vx"] / Rp
            + pos["z"] * pos["y"] * vel["vy"] / Rp
            - Rp * vel["vz"]
        ) / r
        return r, v_r, v_phi, v_theta

    @staticmethod
    def gaussian(data, error, rhalf=None):
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
        mu = numpyro.sample("mu", dist.Uniform(jnp.min(data), jnp.max(data)))
        sigma = numpyro.sample("sigma", dist.Uniform(0.01, 100))

        if rhalf != None:
            mwolf = numpyro.deterministic("mwolf", (4 * rhalf * sigma**2) / (3 * G))

        s = jnp.sqrt(sigma**2 + error**2)
        numpyro.sample("obs", dist.Normal(mu, s), obs=data)

    @staticmethod
    def dynamical_mass(system, q_eval, l_grav=87 * u.kpc):
        r"""
        Calculate the mass profile of a system using the spherical Jeans equation:

        .. math::
            M(<r)=-\frac{r \overline{v_r^2}}{G}\left(\frac{\mathrm{d} \ln \rho}{\mathrm{d} \ln r} + \frac{\mathrm{d} \ln \overline{v_r^2}}{\mathrm{~d} \ln r}+2 \beta\right)

        All terms in the Equation above are calculated using B-splines

        Parameters
        ----------
        system : Data
            Should be either a KeckData, DCJLData or MockData object (subclasses of Data clas)
        q_eval : ndarray
            radii at which to evaluate the mass | units: [kpc]

        Returns
        -------
        ndarray
            Mass evaluated at q_eval | units: :math:`[\rm M_{\odot}]`

        Notes
        -----
        TODO: Currently only works for the specific DCJL simulated Dwarf's
        TODO: Need a better way to do this so that I dont have to calculate every term everytime i call this
        TODO: Get rid of Agama and instead use a Jax implementation of of these functions.

        References
        ----------
        `Rehemtulla et al. 2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.511.5536R/abstract>`_

        """

        # get all stellar positions and velocities
        radii = system.r  # [kpc]
        vr_sq = system._vr**2  # [km^{2}~s^{-2}]
        vtheta_sq = system._vtheta**2  # [km^{2}~s^{-2}]
        vphi_sq = system._vphi**2  # [km^{2}~s^{-2}]

        # calculate knots (i.e Where to evaluate B-splines)
        # NOTE: NEED to change this, but its ok for know since i'm only using it on the simulated dwarfs
        q = np.log(radii)
        lBound = np.log(
            4 * l_grav.to(u.kpc).value
        )  # only trust the velocity values after 4 l_grav's
        uBound = q.max()

        rknot = np.logspace(
            start=lBound,  # only trust the velocity values after 4 l_grav's
            stop=uBound,
            num=6,  # TODO: this should be a parameter -- but i'm still unsure of how many I should use
            base=np.exp(1),
        )  # There's definitely a better way of doing this

        knots = np.log(rknot)  # mostly use log(rknots so)

        vr_bspline = agama.splineApprox(knots, q, vr_sq)
        vtheta_bspline = agama.splineApprox(knots, q, vtheta_sq)
        vphi_bspline = agama.splineApprox(knots, q, vphi_sq)

        logq_eval = np.log(q_eval)
        vr_fit = vr_bspline(logq_eval)
        vtheta_fit = vtheta_bspline(logq_eval)
        vphi_fit = vphi_bspline(logq_eval)

        beta = 1 - ((vtheta_fit + vphi_fit) / (2 * vr_fit))
        G = 4.3e-6  # kpc km2 Msun-1 s-2
        dvr = vr_bspline(logq_eval, der=1)
        S = agama.splineLogDensity(knots, x=np.log(radii), w=np.ones(len(radii)))
        rho = rho_eval(S, q_eval)  # This has units of N_{\star}/kpc^3
        dlnrho_dlnr = dlnrho_dlnr_eval(S, q_eval)  # unitless

        # Split Spherical Jeans Eq (e.g. Eq. 1 in Rehemtulla et al. 2022) into three terms
        a = -dlnrho_dlnr
        b = -dvr / vr_fit
        c = -2 * beta
        M_jeans = (np.array(vr_fit) * q_eval / G) * (
            np.array(a) + np.array(b) + np.array(c)
        )
        return q_eval, np.array(M_jeans)


def rho_eval(S, r):
    r"""
    Calculates rho(r)
    S(log r) is log of tracer count (intended to be used with agama.splineLogDensity and weights=1)
    r is point or array of points where rho should be evaluated
    """
    return np.exp(S(np.log(r))) / (4.0 * np.pi * (r**3))


def dlnrho_dlnr_eval(S, r):
    r"""
    Calcualtes dlnrho / dlnr
    S(log r) is log of tracer count (intended to be used with agama.splineLogDensity and weights=1)
    r is point or array of points where rho should be evaluated
    """
    return S(np.log(r), der=1) - 3
