from dynamicAll import models
import sampler
import jax.numpy as jnp
import numpy as np
import pytest
from tests import abel


def test_vplummer():
    '''
    Hernquist - Zhao density
    We can test it against the analytical profiles for a plummer and a NFW profile

    Notes
    -----
    Dont use 1 cause errors might go unnoticed
    '''
    # Plummer
    M  = 3.1e7
    rs = 1.2
    # Equivalent  Hernquist-zhao density profile
    rhos= 3*(M)/(4*np.pi *rs**3)
    a  = 0
    b  = 2
    c  = 5
    r = np.logspace(-2,2,50)
    rho_plummer = models.Plummer._density(r,M,rs)
    rho_HZ = models.HernquistZhao._density(r,rhos,rs,a,b,c)
    assert pytest.approx(rho_HZ) == rho_plummer

def test_vNFW():
    # Plummer
    M =1
    rs=1
    # Equivalent  Hernquist-zhao density profile
    rhos= 3*(M)/(4*np.pi *rs**3)
    # now NFW: 
    a  = 1
    b  = 1
    c  = 3
    r = np.logspace(-2,2,50)
    rho_nfw = models.NFW._density(rhos,rs,r)
    rho_HZ = models.HernquistZhao._density(r,rhos,rs,a,b,c)
    assert pytest.approx(rho_HZ) == rho_nfw