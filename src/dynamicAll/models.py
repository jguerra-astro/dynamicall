#third-party imports
from typing import Callable
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special as sc
from jax._src.config import config
from jaxopt import Bisection
config.update("jax_enable_x64", True)
from typing import Union
import arviz as az
import corner
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS
# project 
from . import abel
from . import distributions as jdists
from .base import JaxPotential
from functools import partial


def centre(r:float,x0:jax.Array,v0:jax.Array,M:float,b:float) -> float:
    
    L      = jnp.cross(x0,v0)                   # Angular momentum
    T      = 0.5*jnp.dot(v0,v0)                 # Kinetic energy 
    r0     = jnp.linalg.norm(x0,axis=0)
    energy =  T + Isochrone._potential(M,b,r0)  # total energy
    return 2*r**2 *(energy - Isochrone._potential(M,b,r)) - jnp.dot(L,L) 

class Hernquist(JaxPotential):
    
    def __init__(self, M:float, a:float):
        self._M = M
        self._a = a

    def density(self, r:Union[float,jax.Array]) -> Union[float,jax.Array]:
        _units = self._M/2/jnp.pi/self._a**3
        q = r/self._a         
        return _units/q/(1+q)**3

    def mass(self, r:Union[float,jax.Array]) -> Union[float,jax.Array]: 
        _units = self._M
        q = r/self._a
        return _units* q**2/(1+q)**2

    def potential(self, r:Union[float,jax.Array]) -> Union[float,jax.Array]:
        q = r/self._a
        G =  4.300917270036279e-06 
        _units = -G*self._M/self._a
        return _units/(1+q)

    def v_circ(self, r:Union[float,jax.Array]) -> Union[float,jax.Array]:
        G =  4.300917270036279e-06 
        return jnp.sqrt(G*self._M*r)/(r+self._a)

    def dispersion(self, r:Union[float,jax.Array]) -> Union[float,jax.Array]:
        '''
        Analytic expression for the velocity dispersion of the Hernquist profile.
        Useful for testing DF sampling.
        Useful 

        Parameters
        ----------
        r : Union[float,jax.Array]
            

        Returns
        -------
        Union[float,jax.Array]
            _description_
        '''
        G =  4.300917270036279e-06 
        term1 =  G*self._M/(12*self._a)
        term2 = 12*r*(r+self._a)**3*np.log((r+self._a)/r)/self._a**4
        term3 = -r/(r+self._a) * (25 + 52*r/self._a + 42*(r/self._a)**2 + 12*(r/self._a)**3) 
        return  term1 *(term2 + term3)

    def logDF(self,E):
        '''
        log of the DF for the Hernquist profile -- Note this is not normalized.
        This is just meant to be used with the MCMC sampler.

        Parameters
        ----------
        E : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        G =  4.300917270036279e-06 
        q = jnp.sqrt(-self._a * E/(G*self._M))

        term1 = (1-q**2)**(-5/2)
        term2 = 3*jnp.arcsin(q) + q*(1-q**2)**(1/2)*(1-2*q**2)*(8*q**4-8*q**2-3)
        return np.log(term1*term2)

    def pdf_r(self,r):
        
        _units = 2/self._a**3
        q = r/self._a         
        return _units*r**2/q/(1+q)**3

class Jaffe(JaxPotential):

    def __init__(self, M:float, a:float):
        self._M = M
        self._a = a

    def mass(self,r):

        q= r/self._a
        return self._M*q/(1+q) 

    def potential(self,r):
        
        G = 4.300917270036279e-06 # G in kpc km^2 s^-2 Msun^-1
        return - G*self._M* jnp.log(1+self._a/r)


    def logDF(self,r):
        '''
        TODO: finish this one

        Parameters
        ----------
        r : _type_
            _description_
        '''
        fm = lambda x: .5*jnp.sqrt(jnp.pi)*jnp.exp(x**2)*jax.scipy.special.erf(x)
        # fp 
    
class HernquistZhao(JaxPotential):
    r'''
    Hernquist-Zhao density profile
    
    .. math::
        \rho(r) =\frac{\rho_s}{\left(\frac{r}{r_s}\right)^a \left[1+\left(\frac{r}{r_s}\right)^{b}\right]^{\frac{c-a}{b}}}

    Parameters
    ----------
    rhos : float
        scale density -- units: [M L^-3] if mass density,  [L^-3] for number density
    rs : float
        scale radius -- units: [L]
    a : float
        inner slope -- for  :math:`\frac{r}{r_s} \ll 1 ~ \rightarrow \rho \sim r^{-a}`
    b : float
        characterizes width of transition between inner slope and outer slope
    c : float
        outer slope -- for :math:`r/r_s \gg 1~ \rho \sim r^{-c}`
    
    Notes
    -----
    for :math:`c \ge 3` mass diverges as :math:`r\rightarrow \infty` therefore default priors on c go from 1 to 3.
    You can get around this, by limiting relevant integrations to a finite radius e.g. :math:`r_{200}`.

    Note however 
    
    References
    ----------
    Baes 2021      : http://dx.doi.org/10.1093/mnras/stab634
    
    An & Zhao 2012 : http://dx.doi.org/10.1093/mnras/sts175
    
    Zhao 1996      : http://dx.doi.org/10.1093/mnras/278.2.488
    '''
    
    _tracer_priors = {
        'tracer_rhos' : dist.LogNormal(0.0, 1.0),
        'tracer_rs'   : dist.LogNormal(0.0, 1.0),
        'tracer_a'    : dist.Uniform(0.0, 1.0),
        'tracer_b'    : dist.Uniform(0.0, 1.0),
        'tracer_c'    : dist.Uniform(0.0, 1.0),
    }#TODO: make sure these priors are within physical limits

    _dm_priors = {
        'dm_rhos' : dist.LogUniform(jnp.exp(5),jnp.exp(30)),
        'dm_rs'   : dist.LogUniform(jnp.exp(-10),jnp.exp(10)),
        'dm_a'    : dist.Uniform(-1.0, 3.0),
        'dm_b'    : dist.Uniform(0.1, 5.0),
        'dm_c'    : dist.Uniform(1.0, 3.0),
    }

    def __init__(self,rhos: float,rs: float,a: float,b: float,c: float):
        
        self._rhos = rhos
        self._rs   = rs
        self._a    = a
        self._b    = b
        self._c    = c

        super().__init__()

    def density(self,r:Union[float,jax.Array]) -> Union[float,jax.Array]:
        r'''
        Hernquist-Zhao density profile

        .. math::
            \rho(r) =\frac{\rho_s}{\left(\frac{r}{r_s}\right)^a \left[1+\left(\frac{r}{r_s}\right)^{b}\right]^{\frac{c-a}{b}}}

        Parameters
        ----------
        r : float | jax.Array
            :math:`r = \sqrt{x^2+y^2+z^2}` | units: [L]

        Returns
        -------
        float | jax.Array

        '''
        return HernquistZhao._density(r,self._rhos,self._rs, self._a, self._b, self._c)

    def mass(self,r):
        r = jnp.atleast_1d(r)
        vec_mass  = jax.vmap(HernquistZhao._mass,in_axes=(0, None,None,None, None,None))
        # return HernquistZhao._mass(r,self._rhos,self._rs, self._a, self._b, self._c)  
        return vec_mass(r,self._rhos,self._rs, self._a, self._b, self._c)  
        
    def potential_scipy(self,r):
        r'''
        Potential for Hernquist density

        Must be calculated numerically
        
        .. math::
            \phi(r) = -4\pi G\left[\frac{M}{r}+ \int_{r}^{\infty} \rho(r) r dr \right]
        
        calculated using a slightly different hypergeometric function -- evaluated at r and a very large number
        shouldn't this just be r to r200?? 
        
        TODO: need a jax implementation of this - DONE
        TODO: write a test for this

        Parameters
        ----------
        r : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        G =  4.517103049894965e-39 # G in kpc km^2/Msun/s^2
        # for readability
        a     = self._a
        b     = self._b
        c     = self._c
        rs   = self._rs
        rhos = self._rhos

        q = r/rs

        func = lambda r:    -r**2 *(q)**(-a) *sc.hyp2f1((2 - a)/b, -(a - c)/b, 1 + (2 - a)/b, -q**b)/(-2 + a)
        
        phi_in  = -G*self.mass_scipy(r, 1.0, rs, a, b, c)/r 
        
        phi_out =  -(4*np.pi*G*rhos*(func(1e20)-func(r)))
        return phi_in+phi_out

    def potential(self,r):
        r'''
        Potential for Hernquist-Zhao density profile
        
        .. math::
            \phi(r) = -4\pi G\left[\frac{M}{r}+ \int_{r}^{\infty} \rho(r) r dr \right]

        Must be calculated numerically.

        Parameters
        ----------
        r : float | array
            array of radii | units: [L]

        Returns
        -------
        float| array
            potential | units: :math:`km^{2} s^{-2}`
            
        '''
        # for readability
        G       = self._G 
        a       = self._a
        b       = self._b
        c       = self._c
        rs      = self._rs   # kpc 
        rhos    = self._rhos # Msun/kpc^3
        
        # vec_mass  = jax.vmap(HernquistZhao._mass,in_axes=(0, None,None,None, None,None))
        q       = r/rs # always convenient to make all integrals dimensionless
        # phi_in  = rs*vec_mass(q,1.0,1.0,a,b,c)/r  # M/r
        phi_in  = rs*self._mass(q,1.0,1.0,a,b,c)/r  # M/r

        # integration points and weights for Gauss-Legendre quadrature + trig substitution
        x,w = self._xj,self._wj
        x0 = 0.0
        x1 = jnp.pi/2
        xk = 0.5*(x1-x0)*x + 0.5*(x1+x0) 
        wk = 0.5*(x1-x0)*w

        # we'll use Jax's flow control to split up integration into two regimes r << rs and r >> rs
        # this will avoid numerical issues with the trig substitution for r << rs
        # for r >> rs, the trig substitution is needed in order to turn the integral into a finite integral
        # TODO: for when there is a limit on the upper bound of the integral we again swith integration schemes - yet to be implemented

        
        def less_than_rs():
            '''
            for r << rs, (\frac{r}{r_{s}} < 1e-5, depending on the order of the integration) the trig substitution usually used leads to numerical issues.
            Instead just use regular Gauss-Legendre quadrature from r to rs. then use the trig substitution to
            '''
            x0 = q
            x1 = 1          
            xi = 0.5*(x1-x0)*x + 0.5*(x1+x0) 
            wi = 0.5*(x1-x0)*w

            t1 = jnp.sum(wi*HernquistZhao._density(xi,1.0,1.0,a,b,c)*xi,axis=0) # integrate from r to rs
            
            t2 = 1*jnp.sum(wk*HernquistZhao._density(1/jnp.sin(xk),1.0,1.0,a,b,c)*1*jnp.cos(xk)/jnp.sin(xk)**3,axis=0) # integrate from rs to infinity
            
            return t1+t2

        def greater_than_rs():
            '''
            fro r > rs, the trig substitution behaves well and transforms the infinity integral into a finite one
            '''
            phi_out = q*jnp.sum(wk*HernquistZhao._density(q/jnp.sin(xk),1.0,1.0,a,b,c)*q*jnp.cos(xk)/jnp.sin(xk)**3,axis=0)
            return phi_out

        phi_out = jax.lax.cond(r <= rs, less_than_rs, greater_than_rs)
        
        return -G*rhos*rs**2 *(phi_in+ 4*jnp.pi*phi_out)

    @staticmethod
    @jax.jit
    def _density(r,rhos : float,rs : float, a : float, b : float, c : float):
        '''
        Hernquist-zhao density profile.. at least thats what i call it
        Useful for both dark matter and stellar distributions

        Parameters
        ----------
        r : _type_
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        Im writing these as static functions for a couple of reasons, but mainly because i want to be able to use them
        as fitting functions
        
        '''
        q = r/rs
        nu = (a-c)/b
        return rhos * q**-a * (1+q**b)**(nu)

    @staticmethod    
    def mass_scipy(r,rhos: float,rs: float,a: float,b: float,c: float):
        '''
        Mass profile using Gauss's Hypergeometric function
        
        Notes
        -----
        See mass_error- too see comparision with calculating the mass by integrating the desity profile -- no errors until r/rs > 1e8
        #? Coeff calculation throws a warning at some point -- but doesnt when I try individual values?
        #! treatment of units is very clunky
    
        '''        
        q      = r/rs
        coeff  = 4*np.pi*q**(-a)/(3-a) 
        units = rhos*(r**3) # units are here: [Mass] = [Mass * L**-3] * [L**3]   
        
        gauss  = sc.hyp2f1((3-a)/b,(c-a)/b,(-a+b+3)/b,-q**b) # unitless function
        return coeff*gauss *units

    @staticmethod
    @jax.jit
    def _mass(r: float,rhos: float,rs: float,a: float,b: float,c: float) -> float:
        r'''
        
        Parameters
        ----------
        r : float
            radius | units [L]
        rhos : float
            scale densisty | units [Mass * L**-3]
        rs : float
            scale radius | units [L]
        a : float
            inner slope | unitless
        b : float
            smoothness of transition from inner to outer slope | unitless
        c : float
            outer slope | unitless

        Returns
        -------
        float
            Mass enclosed within radius r | units [Mass]
        '''
        q     = r/rs
        xk    = 0.5*q*HernquistZhao._xk + 0.5*q
        wk    = 0.5*q*HernquistZhao._wk
        units = 4*jnp.pi*rhos*rs**3
        return units* jnp.sum(wk*xk**2 *HernquistZhao._density(xk,1.0,1.0,a,b,c),axis=0)

    @staticmethod
    @jax.jit
    def _mass_fit(r: float,param_dict) -> float:
        r'''
        
        Parameters
        ----------
        r : float
            radius | units [L]
        rhos : float
            scale densisty | units [Mass * L**-3]
        rs : float
            scale radius | units [L]
        a : float
            inner slope | unitless
        b : float
            smoothness of transition from inner to outer slope | unitless
        c : float
            outer slope | unitless

        Returns
        -------
        float
            Mass enclosed within radius r | units [Mass]
        '''
        q     = r/param_dict['rs']
        xk    = 0.5*q*HernquistZhao._xk + 0.5*q
        wk    = 0.5*q*HernquistZhao._wk
        units = 4*jnp.pi*param_dict['rhos']*param_dict['rs']**3
        return units* jnp.sum(wk*xk**2 *HernquistZhao._density(xk,1.0,1.0,param_dict['a'],param_dict['b'],param_dict['c']),axis=0)
    
    @staticmethod
    def _mass_test(r: float,rhos: float,rs: float,a: float,b: float,c: float) -> float:
        r'''
        update: use branches to calculate mass for r < rs and r > rs.
        This SHOULD improve accuracy of overall calculation, and be more stable for extreme values of b

        Parameters
        ----------
        r : float
            radius | units [L]
        rhos : float
            scale densisty | units [Mass * L**-3]
        rs : float
            scale radius | units [L]
        a : float
            inner slope | unitless
        b : float
            smoothness of transition from inner to outer slope | unitless
        c : float
            outer slope | unitless

        Returns
        -------
        float
            Mass enclosed within radius r | units [Mass]
        '''
        def less_than_rs():
            # When r > rs
            x0 = 0.0                          # lower bound of integral
            x1 = jnp.pi/2.0                   # upper bound of integral
            xi = 0.5*(x1-x0)*x + 0.5*(x1+x0) 
            wi = 0.5*(x1-x0)*w
            coeff = 4*jnp.pi*r
            mrltrs = coeff*jnp.sum(wi*jnp.cos(xi)* HernquistZhao._density(r*jnp.sin(xi),rhos,rs,a,b,c)*(r*jnp.sin(xi))**2)
            return mrltrs

        def greater_than_rs():
            # When r = rs, first do integral from 0 to rs
            x0 = 0.0         # lower bound of integral
            x1 = jnp.pi/2    # upper bound of integral
            xi = 0.5*(x1-x0)*x + 0.5*(x1+x0) 
            wi = 0.5*(x1-x0)*w
            coeff = 4*jnp.pi*rs**3
            mrltrs = coeff*jnp.sum(wi*jnp.cos(xi)* HernquistZhao._density(rs*jnp.sin(xi),rhos,rs,a,b,c)*(jnp.sin(xi))**2)

            # Next When r > rs, first do integral from 0 to rs
            x2 = jnp.arcsin(rs/r)          # lower bound of integral
            x3 = jnp.pi/2.0                # upper bound of integral
            xk = 0.5*(x3-x2)*x + .5*(x3+x2) 
            wk = 0.5*(x3-x2)*w
            coeff2 = 4*jnp.pi* rhos*r**3 
            mrgtrs = coeff2*jnp.sum(wk*jnp.cos(xk)* HernquistZhao._density(r*jnp.sin(xk),1.0,rs,a,b,c)*(jnp.sin(xk))**2)

            return mrltrs + mrgtrs

        mass_calc = jax.lax.cond(r <= rs, less_than_rs, greater_than_rs)
        return mass_calc

class NFW(JaxPotential):
    r'''
    NFW(rhos: float,rs: float)


    Spherical Navarro-Frenk-White model
    This will be useful for testing purposes.

    Parameters
    ----------
    rhos : float
        scale density :math:`\rho_{s}`
    rs   : float
        scale radius :math:`r_{s}`
    
    '''

    _dm_priors = {
        'dm_rhos': dist.LogUniform(1e-3,1e3),
        'dm_rs': dist.LogUniform(1e-3,1e3)
    }

    def __init__(self,rhos: float,rs: float):
        self._rhos   = rhos
        self._rs     = rs
        self._params = [self._rhos,self._rs]

        super().__init__()
    
    def density(self,r):
        r'''
        .. math::
            \rho(r) = \frac{\rho_s}{\frac{r}{r_s}(1+\frac{r}{r_s})^2}

        Parameters
        ----------
        r : _type_
            radius [Length]

        Returns
        -------
        _type_
            _description_
        '''
        return NFW._density(r,*self._params)

    def mass(self,r): 
        r'''
        Mass Profile for Spherical NFW profile

        .. math::
            M(r) = 4\pi \rho_s r_s^3 \left[ \ln\left(\frac{r+r_s}{r_s}\right) - \frac{r}{r+r_s} \right]

        Parameters
        ----------
        r : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        return NFW._mass(r,self._rhos,self._rs)

    def cdf(self,r):
        r'''
        Cumulative distribution function for NFW profile
        We cut off the integral at :math:`r_{200}` 

        .. math::
            cdf(r) = \frac{\left[ \ln\left(\frac{r+r_s}{r_s}\right) - \frac{r}{r+r_s} \right]}{\left[ \ln\left(\frac{r_{200}+r_s}{r_s}\right) - \frac{r_{200}}{r_{200}+r_s} \right]}

        Parameters
        ----------
        r : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        r_200 = self.r200()

        return self._mass(r,self._rhos,self._rs)/self._mass(r_200,self._rhos,self._rs)

    def pdf(self,r):
        r'''
        Probability density function for NFW profile

        .. math::
            pdf(r) = \frac{d}{dr}cdf(r)

        Parameters
        ----------
        r : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        r_200 = self.r200()

        return 4*np.pi*r**2*NFW._density(self._rhos,self._rs,r)/NFW._mass(r_200,self._rhos,self._rs)

    def potential(self,r):
        r'''
        Potential for NFW profile

        .. math::
            \Phi(r) = -4\pi G \rho_s r_s^3 \ln\left(1+\frac{r}{r_s}\right)/r

        Parameters
        ----------
        r : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        G  = self._G
        return -4*np.pi*G*self._rhos*self._rs**3 *jnp.log(1.0+r/self._rs)/r
    
    @partial(jax.jit, static_argnums=(0,))
    def J_analytic(self,theta,D):
        r'''
        .. math::
            \mathrm{J(\theta | D,\rho_s,r_s)}
            =\frac{\pi \rho_0^2 r_{\mathrm{s}}^3}{3 D^2 \Delta^4}\left[2 y\left(7 y-4 y^3+3 \pi \Delta^4\right)\right.\left.+6\left(2 \Delta^6-2 \Delta^2-y^4\right) X(y)\right]
        
        where :math:`y=D\theta/r_s`, :math:`\Delta=\sqrt{1-y^2}` and
        .. math:: 
            X(s)= \begin{cases}\frac{1}{\sqrt{1-s^2}} \operatorname{Arcsech} s, & 0 \leq s \leq 1 \\ \frac{1}{\sqrt{s^2-1}} \operatorname{Arcsec} s, & s \geq 1.\end{cases}
    
        This is using the approximation that :math:`d\Omega d\mathcal{l} =\frac{2πRdRdz}{D^{2}}`.

        Parameters
        ----------
        theta : _type_
            _description_
        D : float
            _description_

        References
        ----------
        Evans et. al. 2016:  https://ui.adsabs.harvard.edu/abs/2016PhRvD..93j3512E/abstract

        '''
        theta = jnp.atleast_1d(theta)
        def chi(s):
            '''
            Overkill at the moment,but in case I want to make it more complicated later.
            Also note the use of jax.lax.cond.
            To be Tested: is this still differentiable?

            Parameters
            ----------
            s : _type_
                _description_
            '''
            def s_lt_one():
                return jnp.arccosh(1/s)/jnp.sqrt(1-s**2)

            def s_gt_one():
                return jnp.arccos(1/s)/jnp.sqrt(s**2-1)

            def s_eq_one():
                return 1.0
            return jax.lax.cond(s< 1, s_lt_one, lambda: jax.lax.cond(s>1, s_gt_one, s_eq_one))
        
        chiv = jax.vmap(chi)
        y = D*theta/rs
        Delta2 = 1.0 - y**2

        coeff = jnp.pi*rhos**2*rs**3/(3*D**2*Delta2**2)
        term1 = 2*y*(7*y -4*y**3 + 3*jnp.pi*Delta2**2)
        term2 = 6*(2*Delta2**3 - 2*Delta2 -y**4)*chiv(y)

        return coeff*(term1+term2)
        
    @staticmethod 
    @jax.jit
    def _density(r,rhos: float ,rs: float):
        q= r/rs
        return rhos * q**(-1) * (1+q)**(-2)

    @staticmethod
    @jax.jit
    def _mass(r,rhos: float,rs: float):
        '''
        _summary_

        Parameters
        ----------
        r : _type_
            _description_
        rhos : float
            scale radius
        rs : float
            scale density

        Returns
        -------
        _type_
            _description_
        
        NOTES
        -----
        
        '''
        q    = (rs+r)/rs
        mNFW = 4*jnp.pi*rhos*rs**3 * (jnp.log(q) - (r/(rs+r))) # Analytical NFW mass profile
        return mNFW
    @staticmethod
    @jax.jit
    def _potential(r,rhos: float,rs: float):
                
        G  = NFW._G
        return -4*np.pi*G*rhos*rs**3 *jnp.log(1.0+r/rs)/r

    @staticmethod
    def from_ms_rs(M: float,rs: float):
        '''
        alternate constructor for NFW profile

        Parameters
        ----------
        M : _type_
            _description_
        rs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        rhos = M/(4*np.pi*rs**3)
        
        return NFW(rhos,rs)

class Isochrone(JaxPotential):
    
    def __init__(self,M:float,b:float)-> None:
        '''
        _summary_ 
        G = 4.30091731e-6 # Gravitational constant units :math:`[$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]`

        Parameters
        ----------
        M : float
            total Mass in solar Masses
        b : float
            scale radius
        
        '''
        self.G  =  4.300917270036279e-06 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        # G =  4.517103049894965e-39 # G in kpc3/Msun/s^2        
        self._M = M 
        self._b = b

    def density(self,r):
        
        return Isochrone._density(self._M,self._b,r)

    def potential(self,r):
        
        return Isochrone._potential(self._M,self._b,r)

    def v_circ(self,r:float) -> float:
        '''
        circular speed at radius r B&T Eq 2.49

        Parameters
        ----------
        r : float
            _description_

        Returns
        -------
        float
            v_{circular}| [km/s]
        
        '''        
        # unnecessary redefining of variables
        G = self.G
        M = self._M
        b = self._b
        a = jnp.sqrt(b**2 + r**2)
        
        return jnp.sqrt(G*M*r**2/a/(b+a)**2)

    def action_r(self,x,v):
        '''
        Analytical radial action for Ischrone Potential
        Useful for testing numerical schemes

        Parameters
        ----------
        x : _type_
            _description_
        v : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        G   =  4.30091731e-6                  # Gravitational constant | [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        L   = jnp.linalg.norm(jnp.cross(x,v)) # magnitude of angular momentum |
        T   = 0.5* jnp.dot(v,v)               # kinetic energy
        r   = jnp.linalg.norm(x,axis=0)       # radius from position vector 
        energy = T + self.potential(r)        # total energy
        term1 = G*self._M/jnp.sqrt(-2*energy)    
        term2 = .5*(L + jnp.sqrt(L**2 + 4*G*self._M*self._b))
        return term1 - term2

    def actions(self,x,v):
        return self.action_r(x,v),self.action_theta(x,v),self.action_phi(x,v)

    def peri(self,x0,v0):
        r0     = jnp.linalg.norm(x0,axis=0)
        bisec  = Bisection(optimality_fun= centre, lower=1e-12, upper=r0,check_bracket=True)

        r_peri = bisec.run(x0=x0,v0=v0,M=self._M,b=self._b).params
        return r_peri

    def apo(self,x0,v0):
        r0    = jnp.linalg.norm(x0,axis=0)
        bisec2 = Bisection(optimality_fun= centre, lower= r0, upper = 500,check_bracket=True)
        r_apo = bisec2.run(x0=x0,v0=v0,M=self._M,b=self._b).params
        return r_apo

    def dispersion(self,r):
        pass

    @staticmethod
    @jax.jit
    def _potential(M: float,b: float,r: float) -> float:
        '''
        B&T2 Eq 2.49

        Parameters
        ----------
        r : float
            [kpc]
        M : float
            [M_{\odot}] (solar mass )
        b : float
            [kpc]

        Returns
        -------
        float
            potential energy [km^2 s^{-2}]
        '''        
        G     =  4.300917270036279e-06 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]s
        return -G*M/(b+ jnp.sqrt(b**2+r**2))

    @staticmethod
    def _density(M: float,b: float,r: float) -> float:
        '''
        B&T2 Eq 2.49

        Parameters
        ----------
        r : float
            _description_
        M : float
            _description_
        b : float
            _description_

        Returns
        -------
        float
            _description_
        '''        
        a = jnp.sqrt(b**2 + r**2)
        
        num = 3*(b+a)*a**2 - r**2*(b+3*a)
        den = 4*jnp.pi*(b+a)**3*a**3 
        return M*num/den

    @staticmethod
    def _DF(w,M,b):
        '''
        Distribution function for the Isochrone potential

        Parameters
        ----------
        x : jnp.array
            position vector
        v : jnp.array
            velocity vector
        M : float
            total mass
        b : float
            scale radius

        Returns
        -------
        float
            distribution function
        '''
        x = w[:3]
        v = w[3:]
        G = 4.300917270036279e-06 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        Energy = 0.5 * jnp.dot(v,v) + Isochrone._potential(M,b,jnp.linalg.norm(x))
        E = -Energy

        def E_gt_zero():
            return -jnp.inf

        def E_lt_zero():
            tE = -Energy * b / (G * M)
            denominator = jnp.sqrt(2) * (2 * jnp.pi) ** 3 * (G * M * b) ** (3 / 2)
            numerator = jnp.sqrt(E) / ((2 * (1 - tE)) ** 4)
        
            term1 = 27 - 66 * tE + 320 * tE ** 2 - 240 * tE ** 3 + 64 * tE ** 4
            term2 = 3 * (16 * tE ** 2 + 28 * tE - 9)
        
            f_I = numerator * term1 + numerator * term2 * jnp.arcsin(jnp.sqrt(E)) / jnp.sqrt(tE * (1 - tE))
            
            return jnp.log(f_I)
        
        f_I = jax.lax.cond(Energy > 0, E_gt_zero, E_lt_zero)

        return f_I

    # @staticmethod
    def logDF(self,E):
        '''
        Distribution function for the Isochrone potential

        Parameters
        ----------
        x : jnp.array
            position vector
        v : jnp.array
            velocity vector
        M : float
            total mass
        b : float
            scale radius

        Returns
        -------
        float
            distribution function
        '''
        # x = w[:3]
        # v = w[3:]
        # G = 4.300917270036279e-06 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        # Energy = 0.5 * jnp.dot(v,v) + Isochrone._potential(M,b,jnp.linalg.norm(x))
        Epsilon = -E
        
        M = self._M
        b = self._b
        G = 4.300917270036279e-06 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        
        tE = Epsilon * b / (G * M)
        
        denominator = jnp.sqrt(2) * (2 * jnp.pi) ** 3 * (G * M * b) ** (3 / 2)
        numerator = jnp.sqrt(tE) / ((2 * (1 - tE)) ** 4)
    
        term1 = 27 - 66 * tE + 320 * tE ** 2 - 240 * tE ** 3 + 64 * tE ** 4
        term2 = 3 * (16 * tE ** 2 + 28 * tE - 9)
    
        f_I = numerator * term1 + numerator * term2 * jnp.arcsin(jnp.sqrt(tE)) / jnp.sqrt(tE * (1 - tE))
        return jnp.log(f_I)
        
class Gaussians(JaxPotential):
    r'''
    Class for a desribing the density profile as a sum of Gaussians
    
    .. math::
        \rho(r) = \sum_{i=1}^{N} \frac{M_i}{(2\pi\sigma_i^2)^{3/2}} \exp\left(-\frac{r^2}{2\sigma_i^2}\right)

    Parameters
    ----------
    params : jax.Array
        array of shape (2,N) where the first row is the mass and the second row is the sigma
    N : int
        Number of Gaussians    
    '''
    def __init__(self,params, N:int):

        self._N = N
        self._params = params.reshape(2,N) #M_i, sigma_i pairs
    
    def density(self,r):

        density= jnp.zeros(len(r))
        for i in range(self._N):
            density +=Gaussians.density_i(r,self._params[0,i],self._params[1,i])

        return density
    
    def mass(self,r):
        # BUG: only calling one component
        return Gaussians.mass_j(r)

    @staticmethod
    @jax.jit
    def density_i(r,M: float,sigma: float):

        rho = M/(jnp.sqrt(2*jnp.pi) * sigma)**3 # part with the units  [Mass * Length**-3] 

        return rho * jnp.exp(- r**2/(2*sigma**2))

    @staticmethod
    @jax.jit
    def _density(r,M: jax.Array,sigma: jax.Array,N=4):
        rho_vec = jax.vmap(Gaussians.density_i, in_axes=(0,None,None))
        return jnp.sum(rho_vec(r,M,sigma),axis=1)

    @staticmethod
    @jax.jit
    def mass_j(r,M:float,sigma:float):
        return M*(jax.scipy.special.erf(r/(jnp.sqrt(2)*sigma)) - jnp.sqrt(2/jnp.pi) * r * jnp.exp(-r**2/ (2*sigma**2))/sigma)

    @staticmethod
    @jax.jit
    def _mass(r: float,M:jax.Array,sigma: jax.Array):
        m_vec = jax.vmap(Gaussians.mass_i, in_axes=(0,None,None))
        return jnp.sum(m_vec(r,M,sigma),axis=1)

class Gaussian(JaxPotential):
    def __init__(self,M,sigma):
        self._M = M
        self._sigma = sigma

    def density(self,r):
        density = Gaussian.density_i(r,self._M,self._sigma)
        return density

    def mass(self,r):
        mass = Gaussian.density_i(r,self._M,self._sigma)

    @staticmethod
    @jax.jit
    def density_i(r,M: float,sigma: float):

        rho = M/(jnp.sqrt(2*jnp.pi) * sigma)**3 # part with the units  [Mass * Length**-3] 

        return rho * jnp.exp(- r**2/(2*sigma**2))

    @staticmethod
    @jax.jit
    def mass_j(r,M:float,sigma:float):
        
        return M*(jax.scipy.special.erf(r/(jnp.sqrt(2)*sigma)) - jnp.sqrt(2/jnp.pi) * r * jnp.exp(-r**2/ (2*sigma**2))/sigma)

class Plummer(JaxPotential):
    r'''
    Class for a spherical Plummer model
    .. math::
        \rho(r) = \frac{3M}{4\pi a^3} \left(1+\frac{r^2}{a^2}\right)^{-5/2}

    Parameters
    ----------
    M : float
        'Mass' of system | :math:`[Msun]`
    a : float
        scale length | [kpc]
    '''
    _tracer_priors = {
        'tracer_M':dist.LogUniform(1e-2,1e8),
        'tracer_a':dist.LogUniform(1e-3,1e3),
    }

    _dm_priors = {
        'dm_M':dist.LogUniform(1e3,1e12),
        'dm_a':dist.LogUniform(1e-10,1e10),
    }
    def __init__(self,M,a):

        self._M = M 
        self._a = a
        #calculate density
        self._rho = 3*self._M/(4*np.pi*self._a**3)
        self.G    = 4.300917270036279e-06 #kpc km^2 s^-2 Msun^-1
        

    def density(self,r:np.ndarray)->np.ndarray:
        r'''
        .. math::
            \rho(r) = \frac{3M}{4\pi a^3} \left(1+\frac{r^2}{a^2}\right)^{-5/2}

        Parameters
        ----------
        r : np.ndarray
            radii | [kpc]

        Returns
        -------
        np.ndarray
            density | :math:`[M_{\odot} kpc^{-3}]`
        '''
        return Plummer._density(r,self._M,self._a)

    def dispersion(self,r:np.ndarray)->np.ndarray:
        '''
        .. math::
            \sigma_{r}^{2}(r) = \frac{GM}{6\sqrt{r^2+a^2}}

        Parameters
        ----------
        r : np.ndarray
            radii | [kpc]

        Returns
        -------
        np.ndarray
            dispersion :math:`sigma_{r}^{2}` | :math:`[km^{2} s^{-2}]`
        '''
        return Plummer._dispersion(r,self._M,self._a)

    def mass(self,r:np.ndarray)-> np.ndarray:
        '''
        Mass enclosed within radius r

        .. math::
            M(r) = \frac{M r^3}{(r^2+a^2)^{3/2}}

        Parameters
        ----------
        r : np.ndarray
            radii | [kpc]

        Returns
        -------
        np.ndarray
            _description_
        '''
        return Plummer._mass(self._M,self._a,r)
    
    def potential(self,r:jax.Array)->jax.Array:
        '''
        .. math::
            \Phi(r) = -\frac{GM}{\sqrt{r^2+a^2}}

        Parameters
        ----------
        r : jax.Array
            _description_

        Returns
        -------
        jax.Array
            _description_
        '''
        return Plummer._potential(r,self._M,self._a)

    def projected_density(self,R: np.ndarray) -> np.ndarray:
        '''
        2D projection of plummer model

        Parameters
        ----------
        R : np.ndarrray
            Projected radii
        M : float
            'Mass' of system -- could just be total number of objects though
        a : float
            scale radius

        Returns
        -------
        np.ndarray
            [M.unit kpc^{-2}]
        '''
        return Plummer.projection(R,self._M,self._a)

    def sample_r(self,N: int) -> np.ndarray:
        '''
        sample from plummer sphere -- inverse transform-sampling
        '''
        y = np.random.uniform(size=N) # Draw random numbers from 0 to 1
        r = self._a/ np.sqrt(y**(-2 / 3) - 1)
        return r

    def sample_xyz(self,N: int) -> np.ndarray:
        '''
            draw x,y,z by sampling uniformly from a sphere
        '''
        r = self.sample_r(N)    
        xyz = np.zeros((3,N))
        phi   = np.random.uniform(0, 2 * np.pi, size=N)
        temp  = np.random.uniform(size=N)
        theta = np.arccos( 1 - 2 *temp)

        xyz[0] = r * np.cos(phi) * np.sin(theta)
        xyz[1] = r * np.sin(phi) * np.sin(theta)
        xyz[2] = r * np.cos(theta)

        return xyz

    def sample_R(self,N: int)-> np.ndarray:
        '''
        Assuming z is the line-of-sight, returs random samples of projection positions

        Parameters
        ----------
        N : _type_
            Number of samples

        Returns
        -------
        np.ndarray
            Projected sample of objects whose distribution follows as plummer sphere
        '''
        r      = self.sample_r(N)    
        xyz    = np.zeros((2,N))
        phi    = np.random.uniform(0, 2 * np.pi, size=N)
        temp   = np.random.uniform(size=N)
        theta  = np.arccos( 1 - 2 *temp)
        xyz[0] = r * np.cos(phi) * np.cos(theta)
        xyz[1] = r * np.sin(phi) * np.cos(theta)
        xyz[2] = r * np.sin(theta) #* Check the angle conventions again! 
        return np.sqrt(xyz[0]**2+xyz[1]**2)

    def sample_w_conditional(self,N:int, evolve=False,save=False,fileName='./out.txt') -> np.ndarray:
        # first sample from r
        x,y,z= self.sample_xyz(N)

        r     = jnp.sqrt(x**2+y**2+z**2)
        vesc = self.v_esc(r)
    
        # then sample from v
        def g_q(q):
            return q**2*(1-q**2)**(7/2)* (512/(7*np.pi))

        out = []
        while len(out) < N:
            x1 = np.random.uniform(0,1)
            x2 = np.random.uniform(0,50176*np.sqrt(7)/19683 *np.pi) 
            if x2 < g_q(x1):
                out.append(x1)
        
        out = jnp.array(out)*vesc
        phi   = np.random.uniform(0, 2 * np.pi, size=N)
        temp  = np.random.uniform(size=N)
        theta = np.arccos( 1 - 2 *temp)
        
        vx = out * np.cos(phi) * np.sin(theta)
        vy = out * np.sin(phi) * np.sin(theta)
        vz = out * np.cos(theta)

        v = np.sqrt(vx**2+vy**2+vz**2)
        vr = (x*vx+y*vy+z*vz)/r
        if evolve:
            import astropy.units as u
            import gala.dynamics as gd
            import gala.potential as gp
            import gala.integrate as gi
            from gala.units import galactic
            pos = [x, y, z] * u.kpc
            vel = [vx, vy, vz] * u.km / u.s
            w0 = gd.PhaseSpacePosition(pos=pos, vel=vel)
            M = self._M * u.Msun
            b = self._a * u.kpc

            potential = gp.PlummerPotential(m=M, b=b, units=galactic)
            Hamiltonian = gp.Hamiltonian(potential)

            orbits = Hamiltonian.integrate_orbit(
                w0, t=np.linspace(0, 1, 100) * u.Gyr, Integrator=gi.DOPRI853Integrator
            )
            r = orbits.spherical.distance
            r, r_unit = r.value, r.unit
            v_xyz= orbits.v_xyz
            vx,vy,vx = v_xyz.to(u.km/u.s).value
            v_new = jnp.sqrt(vx[-1]**2+vy[-1]**2+vz[-1]**2)
        if save:
            try:
                np.savetxt(fileName, np.c_[x,y,z,vx,vy,vz], delimiter=',')
              
            except:
                print('Error saving file, use abosulte path in fileName')
        return np.c_[x,y,z,vx,vy,vz]

    def sample_w(self,
                N:int, 
                evolve=False,
                save=False,
                fileName='./out.txt'
                ) -> np.ndarray:
        '''
        Use emcee to generate samples from a plummer sphere.
        Since you can easily generate samples from a plummer sphere using different methods, 
        i'll use this as a test and compare the results between both methods

        Parameters
        ----------
        N : int
            _description_
        evolve : bool, optional
            _description_, by default False
        save : bool, optional
            _description_, by default False
        fileName : str, optional
            _description_, by default './out.txt'

        Returns
        -------
        np.ndarray
            _description_
        '''

        def E_plummer(w,M,b):
            x = w[:3]
            v = w[3:]
            r = np.linalg.norm(x)
            E = Plummer._potential(r,M,b) + 0.5*np.dot(v,v)
            return 7*np.log(-E)/2

        def log_prior(w,M,b):
            x = w[:3]
            v = w[3:]
            r = np.linalg.norm(x)

            Energy = Plummer._potential(r,M,b) + 0.5*np.dot(v,v)

            if (Energy < 0) and (r < 70):
                # TODO: must change r < 70 to something like r < r_200, also include v_esc?s
                return 0
            return -np.inf


        def log_probability(theta,M,b):
            lp = log_prior(theta,M,b)
            if not np.isfinite(lp):
                return -np.inf
            return lp + E_plummer(theta,M,b)
        
        import emcee
        ndim = 6
        nwalkers = 32
        p0 = np.random.rand(nwalkers, ndim) # need to make p0 better
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=[self._M,self._a])
        state = sampler.run_mcmc(p0, 10000)
        sampler.reset()
        sampler.run_mcmc(state,N)
        samples = sampler.get_chain(flat=True)
        print(
            "Mean acceptance fraction: {0:.3f}".format(
                np.mean(sampler.acceptance_fraction)
            )
        )
        print(
            "Mean autocorrelation time: {0:.3f} steps".format(
                np.mean(sampler.get_autocorr_time())
            )
        )
        return samples

    def probability(self, x: np.ndarray) -> np.ndarray:
        '''
        '''
        return 4*np.pi*x**2 * self._density(x,1,self._a)
    
    def pdf_proj(self, x: np.ndarray) -> np.ndarray:
        '''
        '''
        return 2*np.pi*x * self.projection(x,1,self._a)

    def logDF(self,E):
        '''
        Distribution function for a Plummer shere

        .. math::
            f(E) =
            \frac{24\sqrt{2}}{7\pi^3}
            \frac{a^2}{G^5 M^4}(-E)^{7/2}
        '''
        return 7*np.log(-E)/2

    # @property
    # def tracer_priors(self):
    #     return self._tracer_priors.copy()

    # @tracer_priors.setter
    # def tracer_priors(self, priors):
    #     self._tracer_priors = priors


    # @property
    # def default_priors(self):
    #     return self._default_priors.copy()

    # @default_priors.setter
    # def default_priors(self, priors):
    #     self._default_priors = priors


    @staticmethod
    def _density(r: np.ndarray,M: float,a: float) -> np.ndarray:
        '''
        Making these static methods for fitting purposes

        Parameters
        ----------
        r : np.ndarray
            radius : np.sqrt(x**2+y**2 + z**2)
        M : float
            _description_
        a : float
            _description_

        Returns
        -------
        np.ndarray
            _description_
        '''
        q     = r/a
        coeff = 3*M/(4*np.pi*a**3)
        
        return coeff * (1+q**2)**(-5/2)

    @staticmethod
    def _density_fit(r: np.ndarray,param_dict) -> np.ndarray:
        '''
        Making these static methods for fitting purposes

        Parameters
        ----------
        r : np.ndarray
            radius : np.sqrt(x**2+y**2 + z**2)
        M : float
            _description_
        a : float
            _description_

        Returns
        -------
        np.ndarray
            _description_
        '''
        q     = r/param_dict['a']
        coeff = 3*param_dict['M']/(4*np.pi*param_dict['a']**3)
        
        return coeff * (1+q**2)**(-5/2)

    @staticmethod
    def _dispersion(r: np.ndarray,M: float,a: float) -> np.ndarray:
        '''
        velocity dispersion

        Parameters
        ----------
        r : np.ndarray
            radii | :math:`[kpc]`
        M : float
            mass | :math:`[M_{\odot}]`
        a : float
            scale radius :math:`[kpc]`

        Returns
        -------
        np.ndarray
            velocity dispersion at r | :math:`[km^{2}s^{-2}]`
        '''
        G    = 4.300917270036279e-06 #kpc km^2 s^-2 Msun^-1
        return G*M/jnp.sqrt(r**2+a**2)/6

    @staticmethod
    def from_scale_density_radius(rho: float, a:float):
        '''
        Alternate way to create instance of Plummer model from
        scale density and radius
        '''
        M = (rho*4*np.pi*a**3)/3
        return Plummer(M,a) 

    @staticmethod
    def _mass(M:float,a:float,x:np.ndarray) -> np.ndarray:
        '''
        Making these static methods for fitting purposes
        
        Parameters
        ----------
        x : np.ndarray
            _description_
        M : float
            _description_
        a : float
            _description_

        Returns
        -------
        np.ndarray
            _description_
        '''
        term1 = M *x**3
        term2 = (x**2 + a**2)**(3/2)
        return term1/term2

    @staticmethod
    def _potential(r: jax.Array,M: float, a: float) -> jax.Array:
        '''
        Analitycal potential for plummer sphere

        Parameters
        ----------
        r : jax.Array
            _description_

        Returns
        -------
        jax.Array
            Potential at r | units [G] = #kpc km^2 s^-2 Msun^-1
        '''
        G = 4.300917270036279e-06 #kpc km^2 s^-2 Msun^-1
        q= r/a
        return -G*M/a/jnp.sqrt(1+q**2)

    @staticmethod
    def projection(R: np.ndarray, M: float, a: float) -> np.ndarray:
        '''
        2D projection of plummer model

        Parameters
        ----------
        R : np.ndarray
            Projected radii
        M : float
            'Mass' of system -- could just be total number of objects though
        a : float
            scale radius

        Returns
        -------
        np.ndarray
            [M.unit kpc^{-2}]
        '''
        q     = R/a
        coeff = M/(np.pi*a**2) #units [M.unit/a.unit**2]
        return coeff * (1 + q**2)**(-2)

    @staticmethod
    def _projection_function(R: np.ndarray,param_dict) -> np.ndarray:
        '''
        2D projection of plummer model

        Parameters
        ----------
        R : np.ndarray
            Projected radii
        M : float
            'Mass' of system -- could just be total number of objects though
        a : float
            scale radius

        Returns
        -------
        np.ndarray
            [M.unit kpc^{-2}]
        '''
        q     = R/param_dict['a']
        coeff = param_dict['M']/(np.pi*param_dict['a']**2) #units [M.unit/a.unit**2]
        return coeff * (1 + q**2)**(-2)

    @staticmethod
    def prob_func(x: np.ndarray,a: float) -> np.ndarray:
        return 4*np.pi*x**2 * Plummer._density(x,1,a)

    @staticmethod
    def bin_centers(bin_edges):
        '''
        this is lazy. I know.
        '''
        r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
        return r_center

    @staticmethod
    def radial_distribution(r,bin_method ='doane'):
        N,bin_edges = np.histogram(r,bins =bin_method)
        v_shell = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
        nu = N/v_shell
        return r_center,nu,bin_edges,v_shell

    @staticmethod
    def fit(data):
        def model(data=None):
            # Define the prior distribution for a
            a = numpyro.sample('a', numpyro.distributions.Uniform(0, 10)) # should maybe leave this as a parameter
            # Define the likelihood function using the Normal3D distribution qq
            # with numpyro.plate('data', len(data)):
            numpyro.sample('obs', jdists.Plummer(a), obs=data)

        # Run the MCMC sampler
        num_samples = 1000
        num_warmup = 1000
        kernel = NUTS(model)
        mcmc = MCMC(kernel, 
                    num_warmup=2000,
                    num_samples=4000,
                    num_chains=2,)
        mcmc.run(random.PRNGKey(0), data=data)
        return mcmc

class King(JaxPotential):

    _dm_priors = {}
    _tracer_priors = {}

    def __init__(self,rc,rt):
        self._rc = rc
        self._rt = rt
    
    def density(self, r):
        
        return King._density(r,self._rc,self._rt)

    @staticmethod    
    def _projection(R,rc: float,rt: float):
        '''                                                                                                                                                                     
        King Profile -surface brightness King 1962)                                                                                                                             
        '''
        w = 1.0 +R**2/rc**2
        v = 1.0 +rt**2/rc**2
        term1 = jnp.sqrt(w)**-1
        term2 = jnp.sqrt(v)**-1
        return (term1-term2)**2
    
    @staticmethod        
    def _density(r,rc: float,rt: float):
        '''                                                                                                                          
        King Profile - stellar density King 1962                                                                                     
        '''
        w = 1 +r**2/rc**2
        v = 1 +rt**2/rc**2
        # z     = jnp.sqrt((1+(r/rc)**2)/(1+(rt/rc)**2))
        z     = jnp.sqrt(w/v)
        
        term1 = (jnp.pi*rc* v**(3/2)*z**2)**(-1)
        
        term2 = (jnp.arccos(z)/z) - jnp.sqrt(1 - z**2)
        return term1*term2

class gEinasto(JaxPotential):
    r'''
    Generalized Einasto profile

    Defined by the following density profile:
    
    .. math::
        \rho(r)=\rho_{\mathrm{s}}\left(\frac{r}{r_{\mathrm{s}}}\right)^{-\gamma} \exp \left(-\frac{2}{\alpha}\left[\left(\frac{r}{r_{\mathrm{s}}}\right)^\alpha-1\right]\right),
    
    Parameters
    ----------
    rhos : float
        central density
    rs : float
        scale radius
    alpha : float
        shape parameter
    gamma : float
        inner slope
    
    '''
    def __init__(self,rhos: float,rs: float,alpha: float,gamma: float) -> None:
        self._rhos  = rhos
        self._rs    = rs
        self._alpha = alpha
        self._gamma = gamma

    def density(self,r: jax.Array) -> jax.Array:
        '''
        Parameters
        ----------
        r : np.ndarray
            radius

        Returns
        -------
        np.ndarray
            density
        '''
        return gEinasto._density(r,self._rhos,self._rs,self._alpha,self._gamma)

    @staticmethod
    def _density(r: jax.Array,rhos: float,rs: float,alpha: float,gamma: float) -> jax.Array:
        '''
        Parameters
        ----------
        r : np.ndarray
            radius
        rhos : float
            central density
        rs : float
            scale radius
        alpha : float
            shape parameter
        gamma : float
            shape parameter

        Returns
        -------
        np.ndarray
            density
        '''
        q = r/rs
        return rhos*(r/rs)**(-gamma) * np.exp(-(2/alpha) * (q**alpha - 1))

class NFW_truncated(NFW):

    def __init__(self,rs,rhos,rt=None):
        super().__init__(rs,rhos)
        if rt is None:
            self._rt = self.r200()

    def density(self,r):
        '''
        This is the basic idea, but it's not quite right... i think

        Parameters
        ----------
        r : _type_
            _description_
        '''
        def r_lteq_rt():
            return super().density(r)

        def r_gt_rt():
            return 0.0
        
        return jax.lax.cond(r<=self._rt,r_lteq_rt,r_gt_rt)
        
class PowerLaws(JaxPotential):
    '''
    See e.g GravSphere

    Parameters
    ----------
    JaxPotential : _type_
        _description_
    '''
    _dm_priors = {
        'dm_rhos':dist.LogUniform(1e-2,1e8),
        'dm_R_half':dist.LogUniform(1e-3,1e3),
        'dm_gamma':dist.Uniform(-3,3),
        'dm_beta':dist.Uniform(-3,3),
        'dm_delta':dist.Uniform(-3,3),
        'dm_sigma':dist.Uniform(-3,3),
        'dm_alpha':dist.Uniform(-3,3),
    }
    def __init__(self,
        rhos: float,
        R_half: float,
        gamma: float,
        beta: float,
        delta: float,
        sigma: float,
        alpha: float,
        radial_bins: jax.Array = jnp.array([0.25,0.5,1.0,2.0,4.0]),
        ):
        self._rhos = rhos
        self._R_half = R_half
        self._gamma = gamma
        self._beta = beta
        self._delta = delta
        self._sigma = sigma
        self._alpha = alpha
        self._radial_bins = radial_bins

    def density(self,r):
        return PowerLaws._density(r,
                self._R_half,
                self._rhos,
                self._gamma,
                self._beta,
                self._delta,
                self._sigma,
                self._alpha,
                self._radial_bins)
    

    @staticmethod
    @jax.jit
    def _density(r,
        R_half : float,
        rhos   : float,
        gamma  : float,
        beta   : float,
        delta  : float,
        sigma  : float,
        alpha  : float,
        radial_bins: jax.Array = jnp.array([0.25,0.5,1.0,2.0,4.0])):
        '''
        Describe the density profile as a sequence of power laws e.g Gravsphere

        ..math::
            \rho(r) = \rho_{s} 
            \begin{cases}
                \left(\frac{r}{0.25R_{1/2}}\right)^{\gamma} & r \lt 0.25 R_{1/2} \\
                \left(\frac{r}{0.5R_{1/2}}\right)^{\beta}
                    \left(\frac{0.5 R_{1/2}}{0.25 R_{half}}\right)^{\beta} & 0.25 R_{1/2} \le r \lt 0.5 R_{1/2} \\
                \left(\frac{r}{R_{1/2}}\right)^{\delta}
                    \left(\frac{R_{1/2}}{0.5 R_{half}}\right)^{\delta}
                    \left(\frac{0.5 R_{1/2}}{0.25 R_{half}}\right)^{\beta} & 0.5 R_{1/2} \le r \lt R_{1/2} \\
                \left(\frac{r}{2R_{1/2}}\right)^{\sigma}
                    \left(\frac{2 R_{1/2}}{R_{half}}\right)^{\sigma}
                    \left(\frac{R_{1/2}}{0.5 R_{half}}\right)^{\delta}
                    \left(\frac{0.5 R_{1/2}}{0.25 R_{half}}\right)^{\beta}  & R_{1/2} \le r \lt 2 R_{1/2} \\
                \left(\frac{r}{4R_{1/2}}\right)^{\alpha}
                    \left(\frac{4 R_{1/2}}{2 R_{half}}\right)^{\alpha}
                    \left(\frac{2 R_{1/2}}{R_{half}}\right)^{\sigma}
                    \left(\frac{R_{1/2}}{0.5 R_{half}}\right)^{\delta}
                    \left(\frac{0.5 R_{1/2}}{0.25 R_{half}}\right)^{\beta}  & r \ge 2 R_{1/2} \\
            \end{cases}

        Parameters
        ----------
        r : _type_
            _description_
        R_half : float
            _description_
        rhos : float
            _description_
        gamma : float
            _description_
        beta : float
            _description_
        delta : float
            _description_
        sigma : float
            _description_
        alpha : float
            _description_
        radial_bins : jax.Array, optional
            _description_, by default jnp.array([0.25,0.5,1.0,2.0,4.0])

        Returns
        -------
        _type_
            _description_
        '''
        r_c = radial_bins* R_half

        def r_gamma():
            return (r/r_c[0])**gamma
        def r_beta():
            # Note Additional terms are added to ensure continuity
            return (r/r_c[1])**beta * (r_c[1]/r_c[0])**(beta)
        def r_delta():
            return (r/r_c[2])**delta * (r_c[2]/r_c[1])**(delta) * (r_c[1]/r_c[0])**(beta)
        def r_sigma():
            return (r/r_c[3])**sigma * (r_c[3]/r_c[2])**(sigma) * (r_c[2]/r_c[1])**(delta) * (r_c[1]/r_c[0])**(beta)
        def r_alpha():
            return (r/r_c[4])**alpha * (r_c[4]/r_c[3])**(alpha) * (r_c[3]/r_c[2])**(sigma) * (r_c[2]/r_c[1])**(delta) * (r_c[1]/r_c[0])**(beta)


        density = rhos* jax.lax.cond(r < r_c[0],
                            r_gamma,
                            lambda: jax.lax.cond(r<r_c[1],
                                        r_beta,
                                        lambda: jax.lax.cond(r<r_c[2],r_delta,
                                                lambda: jax.lax.cond(r<r_c[3],r_sigma, r_alpha)
                                                    )
                                        )
                                    )

        return density
    
    @staticmethod
    def _mass(r,
        R_half: float,
        rhos: float,
        gamma: float,
        beta: float,
        delta: float,
        sigma: float,
        alpha: float,
        radial_bins: jax.Array = jnp.array([0.25,0.5,1.0,2.0,4.0])):
        '''
        _summary_

        Parameters
        ----------
        r : _type_
            _description_
        R_half : float
            _description_
        rhos : float
            _description_
        gamma : float
            _description_
        beta : float
            _description_
        delta : float
            _description_
        sigma : float
            _description_
        alpha : float
            _description_
        radial_bins : jax.Array, optional
            _description_, by default jnp.array([0.25,0.5,1.0,2.0,4.0])
        '''
        ...


class JeansOM:    

    def __init__(self,data: np.ndarray,model_dm,model_tr,model_beta):
        '''
        Parameters
        ----------
        data : np.ndarray
            (projected positions,line-of-sight velocities)

        Notes
        -----
        
        '''
        self._G    = 4.5171031e-39 # Gravitational constant in the right units
        self._data = data          # data
        self._dm   = model_dm      # dark matter
        self._tr   = model_tr      # tracers
        self._beta = model_beta    # stellar anisotropy model

    @staticmethod
    def likelihood(data: np.ndarray,model: Callable,theta: np.ndarray) -> float:
        '''
        I'll keep this around so that it's easy to evaluate the likelihood at a value with some theta

        Parameters
        ----------
        data : np.ndarray
            _description_
        model : Callable
            _description_
        theta : np.ndarray
            parameters we're fitting

        Returns
        -------
        float
            _description_
        '''
        R      = data[0]
        vi     = data[1]
        err    = np.full_like(R,0) # here in case i want to change this to include errors
        N      = len(vi)
        e2     = model(R,theta) + err**2
        

        term1 = - 0.5 * np.sum(np.log(e2))
        term2 = - 0.5 * np.sum(vi**2/e2)   # This assumes that the velocities are gaussian centered at 0 
        term3 = - 0.5 * N *np.log(2*np.pi) # doesnt really need to be here
        return term1+term2+term3

    @staticmethod
    def beta(r:float,a: float,alpha = 1):
        r'''
        Osipkov-Merrit stellar anisotropy
        \beta = 0 as r -> 0
        \beta = 1 as r -> infinity
        a determines how quickly the trasition from \beta= 0 to 1 happens 
        
        Parameters
        ----------
        r : float
            _description_
        a : float
            scale length
            determines how quickly the trasition from \beta= 0 to 1 happens 

        Returns
        -------
        _type_
            0 as r -> 0
            1 as r -> infinity
            unitless
        
        Notes
        -----
        TODO: Need to change to include alpha parameter as a paramater to fit so that the anisotropy can be negative
        TODO: change 2 to a parameter
        TODO: look into the symmetrized velocity anisotropy
        
        '''
        # alpha = 1
        return alpha*r**2/(r**2+a**2)

    @staticmethod
    def f_beta(r:float,a:float,alpha = 1):
        r'''
        f(r) = exp[2 \inta_{0}^{r}{\beta(r)/r} dr ]

        Parameters
        ----------
        r : float
            _description_
        a : float
            scale radius for stellar anisotropy

        Returns
        -------
        _type_
            _description_
        
        Notes
        -----
        '''
        return (a**2 + r**2)**alpha

    @staticmethod
    def mass(r,rhos,rs,a,b,c):
        
        return HernquistZhao.mass(r,rhos,rs,a,b,c)

    @staticmethod
    def nu(r,ap):
        return Plummer._density(r,1.0,ap)

    @staticmethod
    def projected_stars(R,ap):
        return Plummer.projection(R,1.0,ap)

    @staticmethod
    def beta_func(f,r,a):
        grad = jax.grad(f)
        return r*grad(r,a)/f(r,a)

    
    @staticmethod
    def dispersion(r: float,r200: float,abeta: float,rhos: float,rs: float,a: float,b: float,c: float) -> float:
        '''
        velocity dispersion form a system with:
        1. Hernquist-Zhao density profile
        2. Asipkov-Merrit stellar anisotropy profile

        see e.g: B&T

        Parameters
        ----------
        r : float
            _description_
        r200 : float
            _description_
        abeta : float
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        float
            radial velocity dispersion
            units: [kpc**2 s**-2]

        Notes
        -----
        Needs to be vectorize in order to accept an array of radii
        '''
        G  = 4.5171031e-39 # Gravitational constant
        x0 = r    # lowerbound 
        x1 = r200 # upperbound
        
        # Gauss-Legendre integration
        xi = (x1-x0)*0.5*x + .5*(x1+x0) # scale from a = 0, b= 1 to a= r, b = r_{200}
        wi = (x1-x0)*0.5*w              # modify weights


        # term1 = G* JeansOM.mass(xi,rhos,rs,a,b,c)* JeansOM.f_beta(xi,abeta)*JeansOM.nu(xi,.25)/xi**2 #GM\nu *f_beta
        # return JeansOM.mass(xi,rhos,rs,a,b,c)
        term1  = JeansOM.mass(xi,rhos,rs,a,b,c)*JeansOM.nu(xi,1.0)*JeansOM.f_beta(xi,abeta) /xi**2
        return G * jnp.sum(wi*term1,axis=0)/JeansOM.f_beta(r,abeta)/JeansOM.nu(r,1.0)

    @staticmethod
    def nusigma(
        r     : float,
        r200  : float,
        abeta : float,
        rhos  : float,
        rs    : float,
        a     : float,
        b     : float,
        c     : float) -> float:
        '''
        velocity dispersion form a system with:
            1. Hernquist-Zhao density profile
            2. Asipkov-Merrit stellar anisotropy profile

        see e.g: B&T

        Parameters
        ----------
        r : float
            _description_
        r200 : float
            _description_
        abeta : float
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        float
            radial velocity dispersion
            units: [kpc**2 s**-2]

        Notes
        -----
        Needs to be vectorize in order to accept an array of radii
        '''
        G  = 4.5171031e-39 # Gravitational constant [kpc**3 solMass**-1 s**-2] 
        x0 = r             # lower bound 
        x1 = r200          # upper bound
        
        # Gauss-Legendre integration
        xi = (x1-x0)*0.5*x + .5*(x1+x0) # xcale from (0,1) ->  (r,r_{200}) 
        wi = (x1-x0)*0.5*w              # modify weights


        # term1 = G* JeansOM.mass(xi,rhos,rs,a,b,c)* JeansOM.f_beta(xi,abeta)*JeansOM.nu(xi,.25)/xi**2 #GM\nu *f_beta

        term1  = JeansOM.mass(xi,rhos,rs,a,b,c)*JeansOM.nu(xi,1.0) * JeansOM.f_beta(xi,abeta)/xi**2
        return G* jnp.sum(wi*term1,axis=0)/JeansOM.f_beta(r,abeta)


    @staticmethod
    def los_dispersion(R: float,r200: float,abeta: float,rhos: float,rs: float,a: float,b: float,c: float) -> float:
        '''
        velocity dispersion form a system with:
            1. Hernquist-Zhao density profile
            2. Asipkov-Merrit stellar anisotropy profile

        see e.g: B&T

        Parameters
        ----------
        r : float
            _description_
        r200 : float
            _description_
        abeta : float
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        float
            radial velocity dispersion
            units: [kpc**2 s**-2]

        Notes
        -----
        Needs to be vectorize in order to accept an array of radii
        '''
        y0 = R    # lowerbound 
        y1 = r200 # upperbound
        
        # Gauss-Legendre integration
        xj = (y1-y0)*0.5*x + .5*(y1+y0) # scale from a = 0, b= 1 to a= r, b = r_{200}
        wj = (y1-y0)*0.5*w              # scale weights

        # unit_converstion = 9.5214061e32
        unit_converstion = 1
        term1 = 1-JeansOM.beta(xj,abeta)*(R/xj)**2                # 1 - beta(r)* R**2 / r**2
        
        term2 = JeansOM.nusigma(xj,r200,abeta,rhos,rs,a,b,c)*xj/jnp.sqrt(xj**2-R**2)     #\nu(r)\sigma_{r}(r)

        return 2 *unit_converstion* jnp.sum(wj*term1*term2,axis=0)/JeansOM.projected_stars(R,1.0)

    @staticmethod
    def dispersion0(r: float,r200: float,abeta: float,rhos: float,rs: float,a: float,b: float,c: float) -> float:
        '''
        velocity dispersion form a system with:
        1. Hernquist-Zhao density profile
        2. Asipkov-Merrit stellar anisotropy profile

        see e.g: B&T

        Parameters
        ----------
        r : float
            _description_
        r200 : float
            _description_
        abeta : float
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        float
            radial velocity dispersion
            units: [kpc**2 s**-2]

        Notes
        -----
        Needs to be vectorize in order to accept an array of radii
        '''
        G  = 4.5171031e-39 # Gravitational constant
        x0 = r    # lowerbound 
        x1 = r200 # upperbound
        
        # Gauss-Legendre integration
        xi = (x1-x0)*0.5*x + .5*(x1+x0) # scale from a = 0, b= 1 to a= r, b = r_{200}
        wi = (x1-x0)*0.5*w              # modify weights

        return G * jnp.sum(wi*JeansOM.mass(xi,rhos,rs,a,b,c)*JeansOM.nu(xi,1.0)/xi**2)/JeansOM.nu(r,1.0)

    @staticmethod
    def nusigma0(r: float, r200: float, abeta : float, rhos: float, rs: float, a: float, b: float, c: float) -> float:
        '''
        velocity dispersion form a system with:
        1. Hernquist-Zhao density profile
        2. Asipkov-Merrit stellar anisotropy profile

        see e.g: B&T

        Parameters
        ----------
        r : float
            _description_
        r200 : float
            _description_
        abeta : float
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        float
            radial velocity dispersion
            units: [kpc**2 s**-2]

        Notes
        -----
        Needs to be vectorize in order to accept an array of radii
        '''
        G  = 4.5171031e-39 # Gravitational constant [kpc**3 solMass**-1 s**-2] 
        x0 = r             # lower bound 
        x1 = r200          # upper bound
        
        # Gauss-Legendre integration
        xi = (x1-x0)*0.5*x + .5*(x1+x0) # xcale from (0,1) ->  (r,r_{200}) 
        wi = (x1-x0)*0.5*w              # modify weights

        return G* jnp.sum(wi*JeansOM.mass(xi,rhos,rs,a,b,c)*JeansOM.nu(xi,0.25)/xi**2,axis=0)

    @staticmethod
    def los_dispersion0(R: float,r200: float,abeta: float,rhos: float,rs: float,a: float,b: float,c: float) -> float:
        '''
        velocity dispersion form a system with:
            1. Hernquist-Zhao density profile
            2. Asipkov-Merrit stellar anisotropy profile

        see e.g: B&T

        Parameters
        ----------
        r : float
            _description_
        r200 : float
            _description_
        abeta : float
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        float
            radial velocity dispersion
            units: [kpc**2 s**-2]

        Notes
        -----
        Needs to be vectorize in order to accept an array of radii
        
        '''
        y0 = R    # lowerbound 
        y1 = r200 # upperbound
        
        # Gauss-Legendre integration
        xj = (y1-y0)*0.5*x + .5*(y1+y0) # scale from a = 0, b= 1 to a= r, b = r_{200}
        wj = (y1-y0)*0.5*w              # scale weights
        
        term2 = JeansOM.nusigma(xj,r200,abeta,rhos,rs,a,b,c)*xj/jnp.sqrt(xj**2-R**2)     #\nu(r)\sigma_{r}(r)

        return 2* jnp.sum(wj*term2,axis=0)/JeansOM.projected_stars(R,1.0)

class gOM:
    def __init__(self,a,alpha):

        self._a     = a
        self._alpha = a

    @staticmethod
    def beta(r,a,alpha):

        return alpha*r**2/(r**2+a**2)

    @staticmethod
    def g_beta(r,a,alpha):

        return (r**2+a**2)**alpha



class Anisotropy:
    _priors: dict = {}
    def beta(self):
        pass
    def f_beta(self):
        pass

    @classmethod
    def get_priors(cls):
        for param_name, distribution in cls._priors.items():
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

            print(f"Parameter: {param_name}, Distribution: {distribution_name}, {bounds_info}")

class BetaConstant(Anisotropy):
    r'''
    Constant anisotropy profile

    .. math::
        \beta(r) = \beta_0

    Parameters
    ----------
    beta0 : float
        _description_
    '''
    beta0: float

    _priors = {
        'beta0': dist.Uniform(-5,1)
        }

    def __init__(self,beta0):
        self._beta0 = beta0

    @staticmethod
    def _beta(r,param_dict):
        return param_dict['beta0']

    @staticmethod
    def _f_beta(r,param_dict):
        return 1

class BetaSymmetric(Anisotropy):
    '''
    This is the symmetric version of a constant anisotropy profile.
    Define by
    .. math::
        \tilde{\beta} = \frac{\beta}{2-\beta}

    or equivalently
    .. math::
        \beta = \frac{2\tilde{\beta}}{1+\tilde{\beta}}.

    This allows for a simple prior on :math:`\tilde{\beta}` (-1,1) that maps to :math:`\beta` (-infinity,1).
    and :math:`\tilde{\beta} = 0` corresponds to :math:`\beta = 0`.

    Parameters
    ----------

    Returns
    -------
    _type_
        _description_
    '''
    beta0: float

    _anisotropy_prior = {
        'beta_tilde': dist.Uniform(-1,1)
        }
    def __init__(self,beta_tilde):
        self._beta_tilde = beta_tilde

    @staticmethod
    def beta(beta_tilde):
        return 2*beta_tilde/(1+beta_tilde)

    @staticmethod
    def f_beta(beta0):
        return 1

class BetaOsipkovMerrit(Anisotropy):
    r'''
    Osipkov-Merrit anisotropy profile

    .. math::
        \beta(r) = \frac{r^2}{r^2+a^2}


    Parameters
    ----------
    a : float
        _description_
    beta0 : float
        _description_
    '''
    _anisotropy_prior = {
        'ra': dist.Uniform(0,10),
        'beta': dist.Uniform(-5,1),
        }
    def __init__(self,ra):
        self._ra    = ra

    @staticmethod
    def beta(r,a,beta):
        return beta/(1+(r/a)**2)

    @staticmethod
    def f_beta(r,a,beta):
        return (1+(r/a)**2)**beta
