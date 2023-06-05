import astropy.units as u
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

# xmass,wmass  = np.loadtxt('/Users/juan/phd/projects/dynamicAll/src/data/gausleg_100',delimiter=',')
xmass,wmass = np.polynomial.legendre.leggauss(100)
xmass = jnp.array(xmass)
wmass = jnp.array(wmass)

x,w = np.polynomial.legendre.leggauss(100) # are these tabulated -- does it take a long time to calculate?

x,w = jnp.array(x),jnp.array(w)



class JaxPotential(ABC):
    '''
    Base class for Potentials.
    All Subclasses are implemented using jax.numpy.
    This allows us to leverage the use of GPUs and automatic differentiation.

    Parameters
    ----------
    ABC : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    '''
    
    @abstractmethod
    def density(self,r):
        pass
    
    def mass(self,r):
        pass 

    def potential(self,r):
        r'''
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
            
        '''
        G =  4.517103049894965e-39 # G in kpc**3/Msun/s**2

        phi_in  = self.mass(r)/r  # M/r | [Msun kpc**-1]  
        #! HERE mass must be a vectorized. This works for the Hernquist_zhao formula, but may be an issue for other classes

        x0 = 0.0
        x1 = jnp.pi/2
        
        xk = 0.5*(x1-x0)*x + 0.5*(x1+x0) 
        wk = 0.5*(x1-x0)*w
        phi_out = jnp.sum(wk*self.density(r/jnp.sin(xk))*(r/jnp.sin(xk))*r*jnp.cos(xk)/jnp.sin(xk)**2,axis=0)
        return -G*(phi_in+ 4*jnp.pi*phi_out)

    def total_energy(self,x,v):
        '''
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
        '''
        T = 0.5* jnp.dot(v,v)         # kinetic energy 
        r = jnp.linalg.norm(x,axis=0)
    
        return T +self.potential(r)

    def gamma(self,r):
        r'''
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
        '''
        grad = jax.grad(self.density)
        return -r*grad(r)/self.density(r)

    def func(self,x: float):
        r'''
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
        '''
        rho_crit = 133.3636319527206 # critical density from astropy's WMAP9 with units [solMass/kpc**3]
        Delta    = 200.0
        return self.mass(x)/(4*jnp.pi*(x**3)/3) - Delta*rho_crit
    
    def r200(self):
        '''
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
        '''
        # rho_crit = cosmo.critical_density(0).to(u.solMass/u.kpc**3).value
        # rho_crit = 133.3636319527206 # critical density from astropy's WMAP9 with units [solMass/kpc**3]
        # Delta    = 200
        # func     = lambda x: self.get_mass(x)/(4*np.pi*(x**3)/3) - Delta*rho_crit
        
        bisec = Bisection(optimality_fun=self.func, lower= 1e-20, upper = 300,check_bracket=False)
        
        return bisec.run().params

    def r200_notjax(self):
        '''
        Radius at which the average density of the halo is equal to 200 times the critical density
        
        Notes
        -----
        This one works on models that aren't jax friendly
        '''
        rho_crit = cosmo.critical_density(0).to(u.solMass/u.kpc**3).value
        Delta    = 200
        
        func     = lambda x: self.get_mass(x)/(4*np.pi*(x**3)/3) - Delta*rho_crit # set average density - 200* rho_crit = 0
        
        self._r200 = scipy.optimize.bisect(func,1e-2,300)
        self._M200 = self.get_mass(self._r200)
        return self._r200

    @partial(jax.jit, static_argnums=(0,))
    def project(self,R):
        r'''
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
        '''
        
        f_abel = lambda x,q: 2*self.density(x)*x/jnp.sqrt(x**2-R**2)
        # integral goes from R to infinity
        # use u-sub wit u = R csc(theta) -> du = -R cos(theta)/sin^2(theta) dtheta
        x0 = 0
        x1 = jnp.pi/2
        # x,w = np.polynomial.legendre.leggauss(100) # are these tabulated -- does it take a long time to calculate?
        
        xk = 0.5*(x1-x0)*x + 0.5*(x1+x0) 
        wk = 0.5*(x1-x0)*w
        return np.sum(wk*f_abel(R/jnp.sin(xk),R)*R*jnp.cos(xk)/jnp.sin(xk)**2,axis=0)
    @partial(jax.jit, static_argnums=(0,))
    def deproject(self,r):
        r'''
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
        '''        
        # write everything out for clarity
        x0 = 0
        x1 = jnp.pi/2
        
        xk = 0.5*(x1-x0)*x + 0.5*(x1+x0) 
        wk = 0.5*(x1-x0)*w
        
        dfunc = jax.vmap(jax.grad(self.project,argnums=0))
        # dfunc = jax.vmap(jax.grad(self.projected_density,argnums=0))
        # return dfunc(r)
        f_abel = lambda y,q: -1*dfunc(y)/jnp.sqrt(y**2-q**2)/jnp.pi
        
        return np.sum(wk*f_abel(r/jnp.sin(xk),r)*r*jnp.cos(xk)/jnp.sin(xk)**2,axis=0)
    
    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _peri(cls,pars,x0,v0):
        r0     = jnp.linalg.norm(x0,axis=0)
        bisec  = Bisection(optimality_fun= cls._centre, lower=1e-12, upper=r0,check_bracket=False)
        r_peri = bisec.run(x0=x0,v0 = v0,pars = pars).params
        return r_peri

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _apo(cls,pars,x0,v0):
        r0     = jnp.linalg.norm(x0,axis=0)
        bisec2 = Bisection(optimality_fun= cls._centre, lower= r0, upper = 500,check_bracket=False)
        r_apo = bisec2.run(x0=x0,v0 = v0,pars = pars).params
        return r_apo

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _centre(cls,r,x0,v0,pars) -> float:
        '''
        paper eq. 4 -- subject to change

        '''
        L      = jnp.cross(x0,v0)                   # angular momentum
        T      = 0.5* jnp.dot(v0,v0)                   # kinetic energy 
        r0     = jnp.linalg.norm(x0,axis=0)
        # print(r0)s
        energy =  T + cls._potential(*pars,r=r0) # total energy
        
        return 2*r**2 *(energy - cls._potential(*pars,r=r)) - jnp.dot(L,L) 

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _action_r(cls,params,x:jnp.DeviceArray,v:jnp.DeviceArray,) -> float:
        '''
        Integrand for radial action (B-T 2nd Ed. - eq. 3.224) 

        Notes
        -----
        DONE: get rid of gala -- needed it for testing, but need to get rid of it for numpyro 
        DONE: get rid of quad -- same reason as above
        TODO: move this to BaseModel
        
        Parameters
        ----------
        x : jnp.DeviceArray
            position vector | [kpc]
        v : jnp.DeviceArray
            velocity vector | [km s^{-1}]

        Returns
        -------
        float
            radial action | [kpc km s^{-1}]
        '''    
        x0 = cls._peri(params,x,v)
        x1 = cls._apo(params,x,v)

        # Gauss-Legendre integration
        xi = 0.5*(x1-x0)*xmass + .5*(x1+x0) # scale from (r,r_{200}) -> (-1,1)
        wi = 0.5*(x1-x0)*wmass
        
        L     = jnp.linalg.norm(jnp.cross(x,v),axis=0) # abs(Angular momentum)
        T     = .5*jnp.dot(v,v)
        r0     = jnp.linalg.norm(x,axis=0)
        
        energy =  T + cls._potential(*params,r=r0) # total energy
        term2  =  cls._potential(*params,r=xi)
        
        term1  = 2*(energy - term2)
        term3  = L**2/xi**2      # takes out the kpc units leaves it as km**2/s**2 
        out    = jnp.sqrt(term1 - term3)/jnp.pi
        return jnp.sum(wi*out)

    @staticmethod
    @jax.jit
    def action_theta(x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        '''
        Equation 3.
        Doesn't depend on the potential
        '''
        L_vector = jnp.cross(x,v)
        L_phi    = L_vector[2]
        L        = jnp.sqrt(jnp.dot(L_vector,L_vector))
        return (L - L_phi)
    
    @staticmethod
    @jax.jit
    def action_phi(x,v):
        '''
        Equation 2.
        Doesn't depend on the potential
        '''
        L_phi = x[0]*v[1] - x[1]*v[0]
        return L_phi

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
        STOLEN FROM APW
        This function takes a parameter array and unpacks it into a dictionary
        with the parameter names as keys.
        """
        p_dict = {}
        j = 0
        for name, size in cls.param_names.items():
            # print(name,size)
            p_dict[name] = jnp.squeeze(p_arr[j : j + size])
            j += size
        return p_dict

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def pack_pars(cls, p_dict):
        """
        STOLEN FROM APW
        This function takes a parameter dictionary and packs it into a JAX array
        where the order is set by the parameter name list defined on the class.
        """
        p_arrs = []
        for name in cls.param_names.keys():
            p_arrs.append(jnp.atleast_1d(p_dict[name]))
        return jnp.concatenate(p_arrs)

    @classmethod
    def infer(cls,model,data,errors):
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        # Run NUTS.
        kernel = NUTS(model)
        num_samples = 5000
        mcmc = MCMC(kernel, num_warmup=1000,  num_samples=num_samples)
        mcmc.run(rng_key_,data= data,error=errors)
        mcmc.print_summary()
        samples_1 = mcmc.get_samples()

        return -1

class Data:

    def dispersion_i(
        self,
        component: str,
        binfunc  : Callable = histogram,
        bins = 'blocks'):
        
        component = component.casefold() # in case theres capitals or something
        
        # return self.dispersion(self.pos['component'],self.vel)
        
        match component:
            case 'radial':
                return self.dispersion(self._r,self._vr,self.d_vr,component,binfunc=binfunc,bins=bins)
            case 'theta':
                return self.dispersion(self._r,self._vtheta,self.d_vtheta,component,binfunc=binfunc,bins=bins)
            case 'phi':
                return self.dispersion(self._r,self._vphi,self.d_vphi,component,binfunc=binfunc,bins=bins)
            case 'los':
                return self.dispersion(self._R,self._vlos,self.d_vlos,component,binfunc=binfunc,bins=bins)
            case 'pmr':
                return self.dispersion(self._R,self._pmr,self.d_pmt,component,binfunc=binfunc,bins=bins)
            case 'pmt':
                return self.dispersion(self._R,self._pmt,self.d_pmr,component,binfunc=binfunc,bins=bins)
            case _:
                print('Uknown or missing component type')
    
    def dispersion(self,ri,vi,d_vi,component, binfunc: Callable = histogram,bins='blocks'):
        '''
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
        
        Notes
        -----
        TODO: Figure out how to make this cleaner

        '''
        N, bin_edges = binfunc(ri.value,bins = bins)
        r_center     = .5*(bin_edges[:-1]+ bin_edges[1:]) 
        self.bin_edges[component] = bin_edges
        meanv = jnp.mean(vi) # Probably shouldn't assume that v_{com} = 0
        mu,_,bin_numbers   = binned_statistic(ri.value,(meanv- vi)**2,'mean',bins=self.bin_edges[component])
        s2_error = self.dispersion_errors(ri,vi,component,1000,N,error=d_vi)
        s_error = s2_error/2/np.sqrt(mu)
        
        return r_center,np.sqrt(mu), s_error

    def dispersion_errors(self,ri,vi,component,Nmonte:int,Nbins,error=None):
        '''
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
        '''
        if error==None: error= 2 # gotta change that 2 but need real errors first
        N = self.N_star
        stemp = np.zeros((Nmonte,len(Nbins)))
        meanv = jnp.mean(vi) # Probably shouldn't assume that v_{com} = 0
        for i in range(Nmonte):
            #Draw new velocities by sampling a gaussian distribution 
            # centered at the measured velocity with a sigma = observational error
            vlos_new = vi + error*np.random.normal(size=N)
            #next calculate the dispersion of the new velocities
            mu,_,_   = binned_statistic(ri,(meanv-vlos_new)**2,'mean',bins=self.bin_edges[component])
            stemp[i] = mu
        
        meanv = jnp.mean(vi) # Probably shouldn't assume that v_{com} = 0
        mu,_,bin_numbers   = binned_statistic(ri.value,(meanv- vi)**2,'mean',bins=self.bin_edges[component])
        N,_,bin_numbers   = binned_statistic(ri.value,ri.value,'count',bins=self.bin_edges[component])

        # Get standard deviation of the Nmonte iterations
        mError = np.std(stemp,axis=0)
        # Add Poisson error
        self.s_error = np.sqrt(mError**2 + mu**2/N)
        return self.s_error
    
    def fourth_moment(self,ri,vi,d_vi,binfunc: Callable = histogram, bins ='blocks'):
        
        N, bin_edges = binfunc(ri,bins = bins)
        r_center     = .5*(bin_edges[:-1]+ bin_edges[1:]) 
        meanv = jnp.mean(vi) # Probably shouldn't assume that v_{com} = 0
        mu,_,bin_numbers   = binned_statistic(ri,(meanv- vi.value)**4,'mean',bins=bin_edges)
        s4_error = self.dispersion_errors(1000)
        # s_error = s4_error/2/np.sqrt(mu)
        
        return r_center,mu, s4_error

    def spatial_density(self,
                bin_method: Callable = histogram,
                bins                 = 'blocks',
                dim:int              = 2
                ):
        '''
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
        '''
        if dim==3:
            try: 
                N,bin_edges = bin_method(self.r,bins =self.bin_edges)
                v_shell = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
                r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
                nu = N/v_shell
            except:
                N,bin_edges = bin_method(self.r,bins =bins)
                v_shell = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
                r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
                nu = N/v_shell
        else:
            try:
                N,bin_edges= bin_method(self.R.value,bins =self.bin_edges)
                v_shell = np.pi*(bin_edges[1:]**2 - bin_edges[:-1]**2)
                r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
                nu = N/v_shell
            except:
                N,bin_edges = bin_method(self.R.value,bins =self.bin_edges)
                v_shell = np.pi*(bin_edges[1:]**2 - bin_edges[:-1]**2)
                r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
                nu = N/v_shell  
        
        return r_center,nu

    def spatial_nonpar(self,dims=2):
        '''
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
        '''
        return -1

    def spatial_errors(self,Nmonte:int, errors = .5):
        stemp = np.zeros((Nmonte,len(self._N)))
        
        for i in range(Nmonte):
            vlos_new = self.R.value + errors*np.random.normal(size=len(self.R.value))
            mu,_,_   = binned_statistic(self.R.value,vlos_new**2,'mean',bins=self.bin_edges)
            stemp[i] = mu
        mError = np.std(stemp,axis=0)
        error = np.sqrt(mError**2 + mu**2/self._N)
        return error

    def bspline_projected(self,R: np.ndarray) -> np.ndarray:
        '''
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
        '''
        chi = np.log(R)        
        spl = Data.bspline(R)
        return  0.5*np.exp(spl(chi) - 2*chi)/np.pi

    def bspline_3d(self,r: np.ndarray) -> np.ndarray:
        '''
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
        '''        
        chi = np.log(r)
        spl = Data.bspline(r)
        return  0.25*np.exp(spl(chi) - 3*chi)/np.pi
    
    def spherical(self):
        '''
        _summary_
        (x,y,z) -> r
        (v_{x},v_{y},v_{z}) -> (v_{r},v_{\theta},v_{\phi}) 
        
        theta = arccos(z/r)
        phi   = arctan2(y,x)  = sign(y)arcos(R/z)
        Returns
        -------
        _type_
            _description_
        '''
        #(x,y,z) -> r
        self._R  = np.sqrt(self._x**2+self._y**2) #for conveniences
        self._r   = np.sqrt(self._x**2+self._y**2+self._z**2)

        # (vx,vy,vz) -> (vr,v_{\theta},v_{\phi})
        self._vr     =  ( self._x * self._vx  + self._y * self._vy + self._z * self._vz )/ self._r    
        self._vphi   = (self._y * self._vx - self._x * self._vy)/ self._R
        self._vtheta = (self._z * self._x * self._vx/self._R  + self._z * self._y * self._vy/self._R - self._R * self._vz)/self._r
        # return self._r, self._vr, self._vtheta,self._vphi

    def cylindrical(self):
        r'''
        convert cartesian velocities to cylindrical units
        
        .. math::
            (v_x,v_y,v_z) \rightarrow (v_{\rho},v_{\phi},v_{z})     
        
        Notes
        -----

        '''
        # proper motions
        # using theta 
        # theta = jnp.arccos(self._z/self._r) # or jnp.arctan2(self._R,self._z) 
        # self._pmr     = self._vr*jnp.sin(theta) + self._vtheta*jnp.cos(theta)
        # self._vz_test = self._vr*jnp.cos(theta) - self._vtheta*jnp.sin(theta)

        self._pmr     = self._vr*self._R/self._r + self._vtheta*self._z/self._r        
        self._vz_test = self._vr*self._R/self._r + self._vtheta*self._z/self._r
        self._pmt     = self._vphi
        # return self._pmr,self._pmt,self._vz_test

    def fit_gaussian(self,v_i,error_i,rhalf=None):
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)
        # Run NUTS.
        kernel = NUTS(Data.gaussian)
        num_samples = 3000
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
        mcmc.run(rng_key_,data=v_i,error=error_i,rhalf=rhalf)
        mcmc.print_summary()
        samples_1 = mcmc.get_samples()
        # corner.corner(samples_1,var_names=['mu','sigma','mwolf'],quantiles=[0.16, 0.5, 0.84],show_titles=True,labels=[r'$\mu$',r'$\sigma$',r'$M_{\rm wolf}$'],title_fmt='2.f');
        return samples_1

    @staticmethod
    def bspline(x:np.ndarray):
        '''
        See e.g. Rehemtulla 2022 

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        TODO: must make tuning paramaters be an input to this function

        '''
        q= np.log(x) # Definining this for convenience
        r = np.logspace(
                start = q.min(),
                stop  = q.max(),
                num   = 6,
                base  = np.exp(1)
                ) # This make
        chi = np.log(r)
        spl  = agama.splineLogDensity(chi,q,w=np.full_like(q,1))
        return spl
    
    @staticmethod
    def random_rotation(pos: np.ndarray):

        angles = 360*np.random.rand(3)
        rot    = Rotation.from_euler('xyz', angles, degrees=True)
        return np.dot(rot.as_matrix(),pos)
    
    @staticmethod
    def to_spherical(pos: np.ndarray,vel: np.ndarray):
        #(x,y,z) -> r
        Rp  = np.sqrt(pos['x']**2+pos['y']**2) #for conveniences
        r   = np.sqrt(pos['x']**2+pos['y']**2+pos['z']**2)

        # (vx,vy,vz) -> (vr,v_{\theta},v_{\phi})
        v_r     = ( pos['x'] * vel['vx']  + pos['y'] * vel['vy'] + pos['z'] * vel['vz'] )/ r    
        v_phi   = (pos['y'] * vel['vx'] - pos['x'] * vel['vy'])/ Rp
        v_theta = (pos['z'] * pos['x'] * vel['vx']/Rp  + pos['z'] * pos['y'] * vel['vy']/Rp - Rp * vel['vz'])/r
        return r, v_r,v_phi,v_theta

    @staticmethod
    def gaussian(data,error,rhalf=None):
        r'''
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
        
        '''
        G     =  4.30091731e-6 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
        mu    = numpyro.sample('mu'   ,dist.Uniform(jnp.min(data),jnp.max(data)))
        sigma = numpyro.sample('sigma',dist.Uniform(.01,100))
        
        if rhalf != None:
            mwolf    = numpyro.deterministic('mwolf',(4*rhalf*sigma**2)/(3*G))

        s = jnp.sqrt(sigma**2+error**2)
        numpyro.sample("obs", dist.Normal(mu,s), obs=data)

    @staticmethod
    def dynamical_mass(system,q_eval,l_grav = 87*u.kpc):
        r'''
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

        '''
        
        # get all stellar positions and velocities
        radii      = system.r            # [kpc]
        vr_sq      = system._vr**2       # [km^{2}~s^{-2}]
        vtheta_sq  = system._vtheta**2   # [km^{2}~s^{-2}]
        vphi_sq    = system._vphi**2     # [km^{2}~s^{-2}]
        
        #calculate knots (i.e Where to evaluate B-splines)
        #NOTE: NEED to change this, but its ok for know since i'm only using it on the simulated dwarfs
        q = np.log(radii)
        lBound = np.log(4*l_grav.to(u.kpc).value) # only trust the velocity values after 4 l_grav's
        uBound = q.max()
        
        rknot = np.logspace(
            start = lBound,    # only trust the velocity values after 4 l_grav's
            stop  = uBound,
            num   = 6,         # TODO: this should be a parameter -- but i'm still unsure of how many I should use
            base  = np.exp(1)
            ) # There's definitely a better way of doing this
        
        knots= np.log(rknot) # mostly use log(rknots so)

        vr_bspline     = agama.splineApprox(knots,q, vr_sq)
        vtheta_bspline = agama.splineApprox(knots,q, vtheta_sq)
        vphi_bspline   = agama.splineApprox(knots,q, vphi_sq)
        
        
        logq_eval = np.log(q_eval)
        vr_fit     = vr_bspline(logq_eval)
        vtheta_fit = vtheta_bspline(logq_eval)
        vphi_fit   = vphi_bspline(logq_eval)

        beta = 1 - ( (vtheta_fit + vphi_fit) / (2*vr_fit) )
        G = 4.3e-6 # kpc km2 Msun-1 s-2
        dvr = vr_bspline(logq_eval, der=1)
        S = agama.splineLogDensity(knots, x=np.log(radii), w=np.ones(len(radii)))
        rho = rho_eval(S, q_eval) # This has units of N_{\star}/kpc^3
        dlnrho_dlnr = dlnrho_dlnr_eval(S, q_eval) #unitless
        
        # Split Spherical Jeans Eq (e.g. Eq. 1 in Rehemtulla et al. 2022) into three terms
        a = -dlnrho_dlnr
        b = -dvr/vr_fit
        c = -2*beta
        M_jeans = (np.array(vr_fit) * q_eval / G) * (np.array(a) + np.array(b) + np.array(c))
        return q_eval,np.array(M_jeans)

def rho_eval(S, r):
    r'''
    Calculates rho(r)
    S(log r) is log of tracer count (intended to be used with agama.splineLogDensity and weights=1)
    r is point or array of points where rho should be evaluated
    '''
    return np.exp(S(np.log(r))) / (4.0 * np.pi * (r**3))

def dlnrho_dlnr_eval(S, r):
    r'''
    Calcualtes dlnrho / dlnr
    S(log r) is log of tracer count (intended to be used with agama.splineLogDensity and weights=1)
    r is point or array of points where rho should be evaluated
    '''
    return (S(np.log(r), der=1) - 3)


# class Fit:

#     def __init__(self,data,priors:dict):
#         '''
#         _summary_

#         Parameters
#         ----------
#         data : _type_
#             data['pos']: positions
#             data['vel']: velocities
#             data['dv]  : error on velocities
#         priors : dict
#             dictionary of priors for parameters
#             Depends on which model we're trying to use to fit the data
#         '''
#         self.data   = data
#         self.priors = priors



#     @jax.jit
#     @staticmethod
#     def gaussian(data,error,rhalf=None):
#         '''
#         simple model function for finding the mean velocity of stars
#         $P(v|\mu,\sigma) = (\sqrt{2\pi}\sigma)^{-1}exp{\frac{1/2}{}}$

#         Parameters
#         ----------
#         data : _type_
#             presumably velocities
#         error : _type_
#             error on velocities 
#         rhalf : _type_, optional
#             If you want to calculate the Wolf mass, you can supply an r_{1/2}, by default None
#         NOTES
#         -----
#         Where do  the r_{1/2}'s come from
#         It'd be nice if i could use r_{1/2}'s with an error
#         e.g. 
#             rh       = numpyro.sample('rhalf',dist.Normal(rhalf,drhalf)) 
#             mwolf    = numpyro.deterministic('mwolf',(4*rh*sigma**2)/(3*G))
#         should add a with numpyro plate
#         '''
        
#         G     =  4.30091731e-6 # Gravitational constant units [$kpc~km^{2}~M_{\odot}^{-1}~s^{-2}$]
#         mu    = numpyro.sample('mu'   ,dist.Uniform(jnp.min(data),jnp.max(data)))
#         sigma = numpyro.sample('sigma',dist.Uniform(.01,100))
        
#         if rhalf != None:
#             mwolf    = numpyro.deterministic('mwolf',(4*rhalf*sigma**2)/(3*G))

#         s = jnp.sqrt(sigma**2+error**2)
#         numpyro.sample("obs", dist.Normal(mu,s), obs=data)

#         def model(self,priors,data):
#             # set up priors on parameters
#             # m_beta = numpyro.sample("beta"   , priors['beta']) # stellar anisotropy
#             m_a    = numpyro.sample("gamma"  , priors['gamma'])  # inner-slope
#             m_rs   = numpyro.sample("lnrs"   , priors['lnrs'])   # ln{scale density} -- dont want negatives 
#             m_rhos = numpyro.sample("lnrhos" , priors['lnrhos']) # ln{scale density} -- dont want negatives 
#             m_v    = numpyro.sample("v_mean" , priors['v_mean'])

#             q     = data['pos']
#             obs   = data['vel']
#             error = data['dv']

#             # with numpyro.plate("data", len(data[1,:])):
#                 # sigma2 = jnp.sqrt(Jeans.vec_dispb(data[0,:],m,b,m_a,1,3))
#                 # numpyro.sample("y", dist.Normal(sigma,error), obs=obs)


def run_inference(self, model, num_warmup=1000, num_samples=1000):

    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup = num_warmup,
        num_samples= num_samples,
        num_chains = 1,
        progress_bar =True
    )
    mcmc.run(random.PRNGKey(0))
    summary_dict = summary(mcmc.get_samples(), group_by_chain=False)