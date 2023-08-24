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

class SphGalaxy:
    
    _xk,_wk = np.polynomial.legendre.leggauss(100)
    _xk,_wk = jnp.array(_xk),jnp.array(_wk)
    _G = const.G.to(u.kpc*u.km**2/u.solMass/u.s**2).value
    
    def __init__(self,
                light,
                dark,
                anisotropy,
                priors  : Priors = None):

        self._light      = light
        self._dark       = dark 
        self._anisotropy = anisotropy

        self._density    = self._light._density_fit          #  density from the tracer model
        self._projection = self._light._projection_function
        self._mass       = self._dark._mass_fit           #  assume this is already vectorized,
        self._anisotropy = self._anisotropy._beta

        self.mass       = jax.jit(jax.vmap(self._mass,in_axes=(0,None)))
        self.vec_dispn  = jax.jit(jax.vmap(self.dispersion, in_axes=(0, None, None, None)))        
        self.vec_dispb  = jax.jit(jax.vmap(self.test_dispersion, in_axes=(0, None, None, None)))

        y0 = 0        # lowerbound 
        y1 = jnp.pi/2 # upperbound
        # Modified weights and points for 
        self.xk = 0.5*(y1-y0)* self._xk + 0.5*(y1+y0) 
        self.wk = 0.5*(y1-y0)* self._wk 

        self.cosk= jnp.cos(self.xk)
        self.sink = jnp.sin(self.xk)
    
    
    @partial(jax.jit, static_argnums=(0,))
    def dispersion(self,r,dm_param_dict,tr_param_dict,beta_param_dict):


        @jax.jit
        def func_(q,dm_param_dict,tr_param_dict,beta_param_dict):
            # I see, the difference, i dont have to vectorize both func and mass... i think
            return self._mass(q,dm_param_dict)*self._density(q,tr_param_dict)/q**2

        func = jax.vmap(func_,in_axes=(0,None,None,None))
        coeff = self._G*r

        return coeff*jnp.sum(self.wk*jnp.cos(self.xk)*func(r/jnp.sin(self.xk),dm_param_dict,tr_param_dict,beta_param_dict)/jnp.sin(self.xk)**2,axis=0)   
    

    @partial(jax.jit, static_argnums=(0,))
    def test_dispersion(self,R_,dm_param_dict,tr_param_dict,beta_param_dict) -> float:
        '''
        first using the transform y = rcsc(x)
        then using Gauss-Legendre transformation.
        '''

        @jax.jit
        def func_2(x,R_,dm_param_dict,tr_param_dict,beta_param_dict):
            sn = self.vec_dispn(x,dm_param_dict,tr_param_dict,beta_param_dict)
            return sn*x/jnp.sqrt(x**2-R_**2)

        func2 = jax.vmap(func_2,in_axes=(0,None,None,None,None))

        # y0 = 0        # lowerbound 
        # y1 = jnp.pi/2 # upperbound
        # # Modified weights and points for 
        # xk = 0.5*(y1-y0)* self._xk + 0.5*(y1+y0) 
        # wk = 0.5*(y1-y0)* self._wk 

        # cosk= jnp.cos(xk)
        # sink = jnp.sin(xk)

        return 2* R_* jnp.sum(self.wk*self.cosk*func_2(R_/self.sink,R_,dm_param_dict,tr_param_dict,beta_param_dict)/self.sink**2,axis=0)/self._projection(R_,tr_param_dict)



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


            with numpyro.plate("data", len(data[1,:])):

                sigma2 = jnp.sqrt(self.vec_dispb(data[0,:],dm_param_dict,tr_param_dict,beta_param_dict))

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

            print(f"Parameter: {param_name}, Distribution: {distribution_name}, {bounds_info}")
        return ""
    def __repr__(self):
        self.__str__()
        return ""

class Fit:
    '''

    '''
    def __init__(self,
                tracer_model     = models.Plummer,
                dm_model         = models.HernquistZhao,
                anisotropy_model = models.BetaConstant,
                priors  : Priors = None):
        
        
        # Checks to see if default values are being used
        warnings.formatwarning = self.custom_formatwarning
        if not hasattr(self, 'tracer_model'):
            warnings.warn("\nNo tracer model defined.\nUsing default Plummer model.\n")
        if not hasattr(self, 'dm_model'):
            warnings.warn("\nNo DM model defined.\nUsing default HernquistZhao model.\n")
        if not hasattr(self, 'anisotropy_model'):
            warnings.warn("\nNo anisotropy model defined.\nusing default BetaConstant model\n")  

        self._priors       = priors
        self._tracer_model = tracer_model
        self._dm_model     = dm_model
        
        if priors == None: # if priors aren't defined, use the defaults one --- this is ugly, should rewrite
            warnings.warn("\nNo priors defined.\nUsing default priors for each model")

            self._priors = Priors()
            self._priors.add_from_dict(tracer_model._tracer_priors)
            self._priors.add_from_dict(dm_model._dm_priors)
            self._priors.add_from_dict(anisotropy_model._priors)


        #TODO: add consitency check to make sure all necessary priors are defined
        # should just be a count check

        
    def test(self,
            data: data.Data,
            rng_key     = random.PRNGKey(0),
            num_samples  : int  = 5000,
            num_warmup   : int  = 1000,
            num_chains   : int  = 2,
            progress_bar : bool = True):
        '''
        _summary_

        Parameters
        ----------
        rng_key : _type_, optional
            _description_, by default random.PRNGKey(0)
        num_samples : int, optional
            _description_, by default 5000
        num_warmup : int, optional
            _description_, by default 1000
        '''
        # First set up data for model
        R = data['R']
        v = data['v']
        v_err = data['v_err']
        
        vec_mass  = jax.vmap(self._dm_model._mass,in_axes=(0, None,None,None, None,None))


        def model_normal(data=None,error=None,priors=None):
            # set up priors on parameters
            samples = {}
            for param_name, prior_dist in priors.priors.items():
                samples[param_name] = numpyro.sample(param_name, prior_dist)
            
            with numpyro.plate("data", len(data[1,:])):

                sigma2 = jnp.sqrt(vec_dispb(data[0,:],
                                    samples['dm_rhos'],
                                    samples['dm_rs'],
                                    samples['dm_a'],
                                    sampes['dm_b'],
                                    samples['dm_c']))
                
                numpyro.sample("y", dist.Normal(sigma2,error), obs=data[1:])
        
        # Standard way of runnig MCMC using numpyro
        rng_key,rng_key_ = random.split(rng_key)
        kernel = NUTS(model_normal)
        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=True,
                    )
        
        mcmc.run(rng_key_,
                data=None,
                error=None,
                priors=self._priors)
        # mcmc.print_summary()
        return mcmc
     
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


    def fit_dSph2(self,priors=None):
        # First get data from data class
        R = self._data._R
        v = self._data._v
        v_err = self._data._v_err

    def dispersion_integral(r):
        x0 = 0.0
        x1 = jnp.pi/2.0
        # Gauss-Legendre integration
        xi = 0.5*(x1-x0)*x + .5*(x1+x0)
        wi = 0.5*(x1-x0)*w
        coeff = G*r

        return coeff*jnp.sum(wi*jnp.cos(xi)*func(r/jnp.sin(xi),rhos,rs,a,b,c)/jnp.sin(xi)**2,axis=0)

    def custom_formatwarning(self,message, category, filename, lineno, line=None):
        return f"{category.__name__}: {message}\n"