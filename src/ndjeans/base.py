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

def gaussian(data,error,rhalf=None):
    '''
    simple model function for finding the mean velocity of stars
    $P(v|\mu,\sigma) = (\sqrt{2\pi}\sigma)^{-1}exp{\frac{1/2}{}}$

    Parameters
    ----------
    data : _type_
        presumably velocities
    error : _type_
        error on velocities 
    rhalf : _type_, optional
        If you want to calculate the Wolf mass, you can supply an r_{1/2}, by default None
    NOTES
    -----
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


class BaseModel:
    param_names = {}
    def density(self,r):
        pass

    def mass(self,r):
        pass 
    
    def func(self,x: float):
        '''
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
        '''
        rho_crit = 133.3636319527206 # critical density from astropy's WMAP9 with units [solMass/kpc**3]
        Delta    = 200
        return self.get_mass(x)/(4*jnp.pi*(x**3)/3) - Delta*rho_crit
    
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

    def potential(self,r):
        pass

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


    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def unpack_pars(cls, p_arr):
        """
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
        This function takes a parameter dictionary and packs it into a JAX array
        where the order is set by the parameter name list defined on the class.
        """
        p_arrs = []
        for name in cls.param_names.keys():
            p_arrs.append(jnp.atleast_1d(p_dict[name]))
        return jnp.concatenate(p_arrs)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _peri(cls,pars,*args):
        r0     = jnp.linalg.norm(args[0],axis=0)
        bisec  = Bisection(optimality_fun= cls._centre, lower=1e-12, upper=r0,check_bracket=False)
        r_peri = bisec.run(x0=args[0],v0 = args[1],pars = pars).params
        return r_peri

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _apo(cls,pars,*args):
        r0     = jnp.linalg.norm(args[0],axis=0)
        bisec2 = Bisection(optimality_fun= cls._centre, lower= r0, upper = 500,check_bracket=False)
        r_apo = bisec2.run(x0=args[0],v0 = args[1],pars = pars).params
        return r_apo

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def _centre(cls,r,x0,v0,pars) -> float:
        '''
        paper eq. 4 -- subject to change

        '''
        # print(cls)
        # print(x0,v0)
        # print(pars)
        L      = jnp.cross(x0,v0)                   # angular momentum
        T      = 0.5* jnp.dot(v0,v0)                   # kinetic energy 
        r0     = jnp.linalg.norm(x0,axis=0)
        # print(r0)s
        energy =  T + cls._potential(*pars,r=r0) # total energy
        
        return 2*r**2 *(energy - cls._potential(*pars,r=r)) - jnp.dot(L,L) 



    # @partial(jax.jit, static_argnums=(0,))
    # def centre(self,r, x0, v0) -> float:
        '''
        paper eq. 4 -- subject to change

        '''
        L      = jnp.cross(x0,v0)                   # angular momentum
        T      = jnp.dot(v0,v0)/2                   # kinetic energy 
        r0     = jnp.linalg.norm(x0,axis=0)
        energy =  T + self._potential(self._M,self._b,r0) # total energy
        
        return 2*r**2 *(energy - self._potential(self._M,self._b,r)) - jnp.dot(L,L) 

    @staticmethod
    def action_theta(x,v):
        '''
        Equation 3.
        Doesn't depend on the potential
        '''
        L_vector = np.cross(x,v)
        L_phi    = L_vector[2]
        L        = np.sqrt(np.dot(L_vector,L_vector))
        return (L - L_phi)
    
    @staticmethod
    def action_phi(x,v):
        '''
        Equation 2.
        Doesn't depend on the potential
        '''
        L_phi = x[0]*v[1] - x[1]*v[0]
        return L_phi

class Data:
    '''
    Class for dealing with kinematic data from `spherical systems'
    '''
    def dispersion_i(
        self,
        component: str,
        binfunc  : Callable = histogram,
        bins = 'blocks'
    ):
        
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

    def spatial_par(self, bin_method= histogram, bins = 'blocks',dim =2):
        '''
        Given a set of N stars with either 2D (default) or 3D(set dim=3),
        compute a spatial density

        Parameters
        ----------
        bin_method : _type_, optional
            binning function (e.g. np.histogram)
            must return just two things so dont use plt.hist
            by default histogram (astropy's histogram)
        bins : str, optional
            bin edges -- 'blocks' only works with astropy's histogram method
            by default 'blocks'
        dim : int, optional
            _description_, by default 2

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
        '''
        convert velocities to cylindrical units
        Already calculated \rho/R and z so no need to do anything for those
        (vr,vtheta,vphi) -> (v_{\rho},v_{phi},v_{z})
        
        Notes
        -----
        I guess i should really do (vx,vy,vz) -> (v_{\rho},v_{phi},v_{z})
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
        kernel = NUTS(gaussian)
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


#     def run_inference(self, model, num_warmup=1000, num_samples=1000):

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