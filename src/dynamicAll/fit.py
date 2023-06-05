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



#temporary
G  = 4.5171031e-39 # Gravitational constant [kpc**3 solMass**-1 s**-2] 
#need different accuracy for each integral
xmass,wmass  = np.loadtxt('/Users/juan/phd/projects/dynamicAll/src/data/gausleg_10',delimiter=',')

xmass = jnp.array(xmass)
wmass = jnp.array(wmass)

xdisp,wdisp  = np.loadtxt('/Users/juan/phd/projects/dynamicAll/src/data/gausleg_100',delimiter=',')

xdisp = jnp.array(xdisp)
wdisp = jnp.array(wdisp)

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


