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

from numpyro.infer import MCMC, NUTS
from jax import random

import matplotlib.pyplot as plt
class BaseModel:

    def get_density(self,r):
        pass

    def get_mass(self,r):
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

class Data:
    '''
    Class for dealing with kinematic data from `spherical systems'
    '''
    def dispersion(self,binfunc=histogram,bins='blocks'):
        
        self._N, self.bin_edges = binfunc(self.R.value,bins = bins)
        self.r_center = .5*(self.bin_edges[:-1]+ self.bin_edges[1:])
        self.meanv = jnp.mean(self.v) # Probably shouldn't assume that v_{com} = 0
        self.mu,_,self.bin_numbers   = binned_statistic(self.R,(self.meanv- self.v.value)**2,'mean',bins=self.bin_edges)
        # Keep bin numbers to track stars going into each bin.
        self.s_error = self.dispersion_errors(1000)
        return self.r_center,np.sqrt(self.mu), self.s_error/2/np.sqrt(self.mu)

    def dispersion_errors(self,Nmonte:int,error=None):
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
        stemp = np.zeros((Nmonte,len(self._N)))
        
        for i in range(Nmonte):
            #Draw new velocities by sampling a gaussian distribution 
            # centered at the measured velocity with a sigma = observational error
            vlos_new = self.v.value + error*np.random.normal(size=N)
            
            #next calculate the dispersion of the new velocities
            mu,_,_   = binned_statistic(self.R.value,vlos_new**2,'mean',bins=self.bin_edges)
            stemp[i] = mu
        # Get standard deviation of the Nmonte iterations
        self.mError = np.std(stemp,axis=0)
        # Add Poisson error
        self.s_error = np.sqrt(self.mError**2 + self.mu**2/self._N)
        return self.s_error
    
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
        #(x,y,z) -> r
        Rp  = np.sqrt(self.x**2+self.y**2) #for conveniences
        r   = np.sqrt(self.x**2+self.y**2+self.z**2)

        # (vx,vy,vz) -> (vr,v_{\theta},v_{\phi})
        self.v_r     =  ( self.x * self.vx  + self.y * self.vy + self.z * self.vz )/ r    
        self.v_phi   = (self.y * self.vx - self.x * self.vy)/ Rp
        self.v_theta = (self.z * self.x * self.vx/Rp  + self.z * self.y * self.vy/Rp - Rp * self.vz)/r
        return r, self.v_r, self.v_theta,self.v_phi


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
    def random_rotation(pos):

        angles = 360*np.random.rand(3)
        rot = Rotation.from_euler('xyz', angles, degrees=True)
        return np.dot(rot.as_matrix(),pos)
    
    @staticmethod
    def to_spherical(pos,vel):
        #(x,y,z) -> r
        Rp  = np.sqrt(pos['x']**2+pos['y']**2) #for conveniences
        r   = np.sqrt(pos['x']**2+pos['y']**2+pos['z']**2)

        # (vx,vy,vz) -> (vr,v_{\theta},v_{\phi})
        v_r     = ( pos['x'] * vel['vx']  + pos['y'] * vel['vy'] + pos['z'] * vel['vz'] )/ r    
        v_phi   = (pos['y'] * vel['vx'] - pos['x'] * vel['vy'])/ Rp
        v_theta = (pos['z'] * pos['x'] * vel['vx']/Rp  + pos['z'] * pos['y'] * vel['vy']/Rp - Rp * vel['vz'])/r
        return r, v_r,v_phi,v_theta

class Fit:

    def __init__(self,data,priors:dict):
        '''
        _summary_

        Parameters
        ----------
        data : _type_
            data['pos']: positions
            data['vel']: velocities
            data['dv]  : error on velocities
        priors : dict
            dictionary of priors for parameters
            Depends on which model we're trying to use to fit the data
        '''
        self.data   = data
        self.priors = priors

    def model(self,priors,data):
        # set up priors on parameters
        # m_beta = numpyro.sample("beta"   , priors['beta']) # stellar anisotropy
        m_a    = numpyro.sample("gamma"  , priors['gamma'])  # inner-slope
        m_rs   = numpyro.sample("lnrs"   , priors['lnrs'])   # ln{scale density} -- dont want negatives 
        m_rhos = numpyro.sample("lnrhos" , priors['lnrhos']) # ln{scale density} -- dont want negatives 
        m_v    = numpyro.sample("v_mean" , priors['v_mean'])

        q     = data['pos']
        obs   = data['vel']
        error = data['dv']

        # with numpyro.plate("data", len(data[1,:])):
            # sigma2 = jnp.sqrt(Jeans.vec_dispb(data[0,:],m,b,m_a,1,3))
            # numpyro.sample("y", dist.Normal(sigma,error), obs=obs)


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