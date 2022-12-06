import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from jax._src.config import config
from jaxopt import Bisection
from scipy.optimize import root
import scipy

config.update("jax_enable_x64", True)

class BaseModel:

    def get_density(self,r):
        pass

    def get_mass(self,r):
        pass 
    
    def func(self,x):
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