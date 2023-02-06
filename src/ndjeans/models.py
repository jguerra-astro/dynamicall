#third-party
from typing import Callable

import jax
import jax.numpy as jnp
from jaxopt import Bisection
import numpy as np
import astropy.units as u
import scipy.special as sc
from scipy.integrate import quad
from jax._src.config import config
from functools import partial
config.update("jax_enable_x64", True)
# import astropy.units as u
# import scipy.special as sc
from scipy.integrate import quad
from scipy.integrate import fixed_quad

import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary

from numpyro.infer import MCMC, NUTS
from jax import random

# project 
from . import abel
from .base import Pot

#? What other classes should be here?
# TODO: Some of the static methods in plummer, don't need to be there so i should get rid of them?
# TODO: Gross gotta clean thise up...
try:
    x,w  = np.loadtxt('/home/jjg57/DCJL/newjeans/src/data/gausleg_100',delimiter=',')
except:
    x,w  = np.loadtxt('/Users/juan/phd/projects/weird-jeans/src/data/gausleg_100',delimiter=',')

x = jnp.array(x)
w = jnp.array(w)

def centre(r,x0,v0,M,b) -> float:
    '''
    paper eq. 4 -- subject to change
    TODO: delete
    '''
    L      = jnp.cross(x0,v0)                   # angular momentum
    T      = 0.5*jnp.dot(v0,v0)                 # kinetic energy 
    r0     = jnp.linalg.norm(x0,axis=0)
    energy =  T + Isochrone._potential(M,b,r0) # total energy
    
    return 2*r**2 *(energy - Isochrone._potential(M,b,r)) - jnp.dot(L,L) 

class HernquistZhao(Pot):
    param_names = {
        'rhos': 1,
        'rs': 1,
        'a': 1,
        'b': 1,
        'c': 1,
    }
    def __init__(self,rhos: float,rs: float,a: float,b: float,c: float):
        '''
        _summary_

        Parameters
        ----------
        rhos : float
            scale density -- units: [M L^-3] if mass density,  [L^-3] for number density
        rs : float
            scale radius -- units: [L]
        a : float
            inner slope -- for  r/r_s < 1  rho_star ~ r^{-a} 
        b : float
            characterizes width of transition between inner slope and outer slope
        c : float
            outer slope -- for r/r_s >> 1 rho_star ~ r^{-c}
        Notes
        -----
        #!for c > 3 mass diverges r -> infinity
        References:
            Baes 2021      : http://dx.doi.org/10.1093/mnras/stab634
            An & Zhao 2012 : http://dx.doi.org/10.1093/mnras/sts175
            Zhao 1996      : http://dx.doi.org/10.1093/mnras/278.2.488
        TODO: Can i get away with only using M(r:float)? 
        '''
        self._rhos = rhos
        self._rs   = rs
        self._a    = a
        self._b    = b
        self._c    = c

    def density(self,r):

        return HernquistZhao._density(r,self._rhos,self._rs, self._a, self._b, self._c)

    def mass(self,r):

        return HernquistZhao._mass(r,self._rhos,self._rs, self._a, self._b, self._c)  

    def potential(self,r):
        '''
        Potential for Hernquist density:
        phi_in = -GM/r  term -- see mass()
        phi_out = integral_{r}^{\infty} \rho(r) r dr -- calculated using a slightly different hypergeometric function -- evaluated at r and a very large number

        shouldn't this just be r to r200?? 
        TODO: need a jax implementation of this
        '''
        G =  4.5171031e-39
        # for readability
        a     = self._a
        b     = self._b
        c     = self._c
        rs   = self._rs
        rhos = self._rhos

        q = r/rs

        func = lambda r:    -r**2 *(q)**(-a) * \
                            sc.hyp2f1((2 - a)/b, -(a - c)/b, 1 + (2 - a)/b, -q**b)/(-2 + a)
        
        phi_in  = -G*self.mass(r)/r 
        
        phi_out =  -(4*np.pi*G*rhos*(func(1e20)-func(r)))
        
        return phi_in+phi_out

    def projection(self,R):
        '''
        TODO: What the hell is this doing here???

        Parameters
        ----------
        func : _type_
            _description_
        R : _type_
            _description_
        params : _type_
            _description_
        '''

        return abel.project(self.density,R,(self._rhos,self._rs,self._a,self._b,self._c))

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
    def _mass(r: float,rhos: float,rs: float,a: float,b: float,c: float):
        q  = r/rs
        xk = 0.5*q*x + 0.5*q
        wk = 0.5*q*w
        units = 4*jnp.pi*rhos*rs**3
        
        return units* jnp.sum(wk*xk**2 *HernquistZhao.density(xk,1.0,1.0,a,b,c),axis=0)

class NFW(Pot):
    param_names ={
        'rhos': 1,
        'rs:' : 1
    }
    def __init__(self,rhos: float,rs: float):
        '''
        Navarro-Frenk-White model
        This will be useful for testing purposes. I dont think i'll actually use this for any science

        Parameters
        ----------
        rhos : float
            scale density
        rs : float
            scale radius
        '''
        self._rhos   = rhos
        self._rs     = rs
        self._params = [self._rhos,self._rs]
    
    def density(self,r):
        
        return NFW.density(*self._params,r=r)

    def mass(self,r): 

        return NFW.mass(r,self._rhos,self._rs)

    @staticmethod 
    @jax.jit
    def _density(rhos: float ,rs: float,r):
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
        # q = rs/(rs+r)
        # mNFW = 4*jnp.pi*rhos*rs**3 * (q - jnp.log(q) -1) # Analytical NFW mass profile
        # tr
        q = (rs+r)/rs
        mNFW = 4*jnp.pi*rhos*rs**3 * (jnp.log(q) + 1/q - 1) # Analytical NFW mass profile
        return mNFW

class Isochrone(Pot):
    param_names = {
        "M": 1,  # the log number density
        'b':1
        }
    
    def __init__(self,M:float,b:float)-> None:
        '''
        _summary_

        Parameters
        ----------
        M : float
            total Mass in solar Masses
        b : float
            scale radius
        '''
        self.G  =  4.30091731e-6 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]        
        self._M = M 
        self._b = b

    def density(self,r):
        
        return Isochrone._density(self._M,self._b,r)

    def potential(self,r):
        
        return Isochrone._potential(self._M,self._b,r)

    def v_spherical(self,r:float) -> float:
        '''
        circular speed at radius r B&T Eq 2.49

        Parameters
        ----------
        r : float
            _description_

        Returns
        -------
        float
            _description_
        '''        
        
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
        G     =  4.30091731e-6 # u.kpc*u.km**2*u.solMass**-1 *u.s**-2# Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        # print(b.unit,M.unit,r.unit)
        # print(jnp.sqrt(b**2+r**2))
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
    
class Gaussians(Pot):

    def __init__(self,params, N:int):
        self._N = N
        self._params = params.reshape(2,N) #M_i, sigma_i pairs
    
    def density(self,r):
        '''
        BUG: not the right method
        '''
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
        '''
        _summary_

        Parameters
        ----------
        r : _type_
            _description_
        M : float
            units: Masss
        sigma : float
            units: Length

        Returns
        -------
        _type_
            _description_
        '''

        rho = M * (jnp.sqrt(2*jnp.pi) * sigma**3)**(-1) # part with the units  [Mass * Length**-3] 

        return rho * jnp.exp(- 1*r**2/(2*sigma**2))

    @staticmethod
    @jax.jit
    def _density(r,M: jnp.DeviceArray,sigma: jnp.DeviceArray,N=4):
        rho_vec = jax.vmap(Gaussians.density_i, in_axes=(0,None,None))
        return jnp.sum(rho_vec(r,M,sigma),axis=1)

    @staticmethod
    @jax.jit
    def mass_j(r,M:float,sigma:float):
        
        return M*(jax.scipy.special.erf(r/(jnp.sqrt(2)*sigma)) - jnp.sqrt(2/jnp.pi) * r * jnp.exp(-r**2/ (2*sigma**2))/sigma)

    @staticmethod
    @jax.jit
    def _mass(r: float,M:jnp.DeviceArray,sigma: jnp.DeviceArray):
        '''
        _summary_

        Parameters
        ----------
        r : float
            _description_
        M : jnp.DeviceArray
            _description_
        sigma : jnp.DeviceArray
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        m_vec = jax.vmap(Gaussians.mass_i, in_axes=(0,None,None))
        return jnp.sum(m_vec(r,M,sigma),axis=1)

class Plummer(Pot):
    
    def __init__(self,M,a):
        '''
        TODO: Implement sampling from a self consistent plummer
        TODO: Project into 2D
        TODO: implement multiple components
        '''
        self._M = M 
        self._a = a
        #calculate density
        self._rho = 3*self._M/(4*np.pi*self._a**3)
    
    def density(self,r:np.ndarray)->np.ndarray:
        
        return Plummer._density(self._M,self._a,r)

    def mass(self,r:np.ndarray)-> np.ndarray:
        return Plummer._mass(self._M,self._a,r)
    
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
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        temp = np.random.uniform(size=N)
        theta = np.arccos( 1 - 2 *temp)

        xyz[0] = r * np.cos(phi) * np.sin(theta)
        xyz[1] = r * np.sin(theta)*np.cos(theta)
        xyz[2] = r * np.cos(theta) #* Check the angle conventions again! 

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
        xyz[0] = r * np.cos(phi) * np.sin(theta)
        xyz[1] = r * np.sin(theta)*np.cos(theta)
        return np.sqrt(xyz[0]**2+xyz[1]**2)

    def probability(self, x: np.ndarray) -> np.ndarray:
        '''
        '''
        return 4*np.pi*x**2 * self.density_func(x,1,self._a)

    @staticmethod
    def from_scale_density_radius(rho: float, a:float):
        '''
        Alternate way to create instance of Plummer model from
        scale density and raidus
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
    def prob_func(x: np.ndarray,a: float) -> np.ndarray:
        return 4*np.pi*x**2 * Plummer.density_func(x,1,a)

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
    def params():
        print('number of parameters for a Plummer model are 2, a scale density, and a scale radii')
        print('two options for priors are currently available, flat and gaussian')
        print('If you are fitting multiple components, this is organized as a 4xN array')
        return  

class King(Pot):
    param_names ={
        'rc': 1,
        'rt': 1,
    }
    def __init__(self,rc,rt):
        self._rc = rc
        self._rt = rt
    def density(self, r):
        
        return King.density(r,self._rc,self._rt)

    @staticmethod    
    def projection(R,rc,rt):
        '''                                                                                                                                                                     
        King Profile -surface brightness King 1962)                                                                                                                             
        '''
        w = 1 +R**2/rc**2
        v = 1 +rt**2/rc**2
        term1 = np.sqrt(w)**-1
        term2 = np.sqrt(v)**-1
        return (term1-term2)**2
    @staticmethod        
    def _density(r,rc,rt):
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

# THESE SHOULD GO INTO A DIFFERENT FILE

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
        '''
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
        '''
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
        return Plummer.density_func(r,1.0,ap)

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


    
