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

from . import models, base 

xmass,wmass = np.polynomial.legendre.leggauss(100)

xmass = jnp.array(xmass)
wmass = jnp.array(wmass)
xdisp = jnp.array(xmass)
wdisp = jnp.array(wmass)

class SphGalaxy:

    def __init__(self,light,dark,anisotropy):

        self._light = light
        self._dark  = dark 
        self._anisotropy = anisotropy


        self._vec_mass  = jax.vmap(self._dark.mass              , in_axes=(0, None, None, None, None, None))
        self.vec_dispn  = jax.vmap(self.dark.dispersion_integral, in_axes=(0, None, None, None, None, None))
        self.vec_dispb  = jax.vmap(self.dark.test_dispersion    , in_axes=(0, None, None, None, None, None))


    def func2(self,x,R_,rhos,rs,a,b,c):
        sn = self.vec_dispn(x,rhos,rs,a,b,c)
        return sn*x/jnp.sqrt(x**2-R_**2)

    def test_dispersion(R_: float,rhos: float,rs: float,a: float,b: float,c: float) -> float:
        '''
        first using the transform y = rcsc(x)
        then using Gauss-Legendre transformation to do the integral from equation...
        
        Notes
        ----- 
        y=0 -- corresponds to
        '''
        y0 = 0        # lowerbound 
        y1 = jnp.pi/2 # upperbound
        # # Modified weights and points for 
        xk = 0.5*(y1-y0)* xdisp + 0.5*(y1+y0) 
        wk = 0.5*(y1-y0)* wdisp 
        cosk= jnp.cos(xk)
        sink = jnp.sin(xk)
        # return 2* R_* jnp.sum(wk*jnp.cos(xk)*func2(R_/jnp.sin(xk),R_,rhos,rs,a,b,c)/jnp.sin(xk)**2,axis=0)/JeansOM.projected_stars(R_,0.25)
        # return 2* R_* jnp.sum(wk*cosk*self.func2(R_/sink,R_,rhos,rs,a,b,c)/sink**2,axis=0)/JeansOM.projected_stars(R_,0.25)


    

    def model(self,data,errors):
            m_rhos = numpyro.sample("m_rhos", dist.Uniform(5,30))   # log_{10}{scale density} -- dont want negatives 
            m_rs   = numpyro.sample("m_rs", dist.Uniform(-10, 10))  # log_{10}{scale density} -- dont want negatives 
            m_a    = numpyro.sample("m_a", dist.Uniform(-1,5))
            m_b    = numpyro.sample("m_b", dist.Normal(1,1))
            m_c    = numpyro.sample("m_c", dist.Normal(3,1))
            # m_d    = numpyro.sample("m_d",dist.Normal(np.ones(5),np.ones(5))) # agh ok so this works, yay!
            m = 10**m_rhos
            b = 10**m_rs

            sigma2 = self.vec_dispb(data[0,:],m,b,m_a,m_b,m_c) + errors**2
            with numpyro.plate("data", len(data[1])):
                numpyro.sample("y", dist.Normal(data[1,:],sigma2), obs=data[1:])

            # return -1




import numpyro.distributions as dist

class Priors:
    '''
    Class for defining priors for the model
    '''
    def __init__(self):
        self.priors = {}

    def add_prior(self, name, prior_dist,base='one'):
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


def custom_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


class Fit:
    '''
    TODO: Add useful docstring
    ToDO: This could be a class defined withhin a bigger class
    '''
    def __init__(self,tracer_model= models.Plummer,dm_model= models.HernquistZhao,anisotropy_model = models.BetaConstant,priors  : Priors = None):
        
        
        # Checks to see if default values are being used
        warnings.formatwarning = custom_formatwarning
        if not hasattr(self, 'tracer_model'):
            warnings.warn("\nNo tracer model defined.\nUsing default Plummer model.\n")
        if not hasattr(self, 'dm_model'):
            warnings.warn("\nNo DM model defined.\nUsing default HernquistZhao model.\n")
        if not hasattr(self, 'anisotropy_model'):
            warnings.warn("\nNo anisotropy model defined.\nusing default BetaConstant model\n")  

        self._priors = priors
        self._tracer_model = tracer_model
        self._dm_model = dm_model
        
        if priors == None: # if priors aren't defined, use the defaults one --- this is ugly, should rewrite
            warnings.warn("\nNo priors defined.\nUsing default priors for each model")

            self._priors = Priors()
            self._priors.add_from_dict(tracer_model.get_tracer_priors())
            self._priors.add_from_dict(dm_model.get_dm_priors())
            self._priors.add_from_dict(anisotropy_model.get_priors())


        #TODO: add consitency check to make sure all necessary priors are defined
        # should just be a count check

        
     
    # @staticmethod
    def fit(self,data,priors=None):
        
        # first fit the tracer model
        
        R = self._data._R
        v = self._data._v
        v_err = self._data._v_err
        




        return -1

    def fit_dSph(self,priors=None):
        # First get data from data class
        R = self._data._R
        v = self._data._v
        v_err = self._data._v_err

        # define model

        def model_flat(data,error,priors):
            # set up priors on parameters
            samples = {}
            for param_name, prior_dist in priors.priors.items():
                samples[param_name] = numpyro.sample(param_name, prior_dist)




            m_rhos = numpyro.sample("m_rhos", dist.Uniform(5,30)   )  # \ln{scale density} -- dont want negatives 
            m_rs   = numpyro.sample("m_rs"  , dist.Uniform(-10,10) )  # \ln{scale density} -- dont want negatives 
            m_a    = numpyro.sample("m_a"   , dist.Uniform(-1,5)   )  #inner-slope

            # m = numpyro.deterministic('m',jnp.exp(m_rhos))
            # b = numpyro.deterministic('b',jnp.exp(m_rs))

            # sigma2 = vec_dispb(data[0,:],m,b,m_a,1,3)
            with numpyro.plate("data", len(data[1,:])):
                # sigma2 = vec_dispb(data[0,:],m,b,m_a,1,3)\   
                m = numpyro.deterministic('m',jnp.exp(m_rhos))
                b = numpyro.deterministic('b',jnp.exp(m_rs))
                sigma2 = jnp.sqrt(vec_dispb(data[0,:],m,b,m_a,1,3))
                numpyro.sample("y", dist.Normal(sigma2,error), obs=data[1:])


        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        # Run NUTS.
        kernel = NUTS(model_flat)
        num_samples = 5000
        mcmc = MCMC(kernel, num_warmup=1000,  num_samples=num_samples)
        mcmc.run(rng_key_,data= data_,error=errors)
        mcmc.print_summary()
        samples_1 = mcmc.get_samples()

    def dispersion_integral(r):
        x0 = 0.0
        x1 = jnp.pi/2.0
        # Gauss-Legendre integration
        xi = 0.5*(x1-x0)*x + .5*(x1+x0)
        wi = 0.5*(x1-x0)*w
        coeff = G*r

        return coeff*jnp.sum(wi*jnp.cos(xi)*func(r/jnp.sin(xi),rhos,rs,a,b,c)/jnp.sin(xi)**2,axis=0)
