from ndjeans import abel,models
import sampler
import numpy as np
import pytest


def test_vplummer():
    '''
    Hernquist - Zhao density
    We can test it against the analytical profiles for a plummer and a NFW profile
    '''
    # Plummer
    M =1
    rs=1
    # Equivalent  Hernquist-zhao density profile
    rhos= 3*(M)/(4*np.pi *rs**3)
    a  = 0
    b  = 2
    c  = 5
    r = np.logspace(-2,2,50)
    rho_plummer = models.Plummer.density_func(r,M,rs)
    rho_HZ = models.HernquistZhao.density(r,rhos,rs,a,b,c)
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
    rho_nfw = models.NFW.density(r,rhos,rs)
    rho_HZ = models.HernquistZhao.density(r,rhos,rs,a,b,c)
    assert pytest.approx(rho_HZ) == rho_nfw


    
