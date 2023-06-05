from dynamicAll import abel,models
import sampler
import numpy as np
import pytest
import jax
import jax.numpy as jnp
def test_projection():
    '''
    Notes
    ------
    Old
    '''
    M,a = 1, 0.25
    R = np.logspace(-2,2,50) 
    
    # Analytical functions 
    density_func = sampler.Plummer.density_func
    projection = sampler.Plummer.projection
    # --------------------------------------------
    
    # Abel transform calculation
    abel_projection  = abel.project(density_func,R,params=(M,a))
    # --------------------------------------------
    
    assert projection(R,M,a) == pytest.approx(abel_projection)


def test_deprojection():
    '''
    Notes
    ------
    Old
    '''
    M,a = 1, 0.25
    R = np.logspace(-2,2,50) 

    # Analytical functions
    density_func = sampler.Plummer.density_func
    projection = sampler.Plummer.projection
    # --------------------------------------------
    
    # Abel transform calculation
    abel_deprojection = abel.deproject(projection,R,params=(M,a))

    assert density_func(R,M,a) == pytest.approx(abel_deprojection)

def test_transform():
    '''
    Notes
    ------
    Old
    '''
    M,a = 2, 0.25
    R = np.logspace(-2,2,50) 

    # Analytical functions
    density_func = sampler.Plummer.density_func
    projection   = sampler.Plummer.projection
    # --------------------------------------------

    # define model
    model = models.Plummer(M,a)


    # Abel transform calculation
    abel_deprojection = jax.vmap(model.deproject)(R)

    # assert density_func(R,M,a) == pytest.approx(abel_deprojection)
    np.testing.assert_allclose(abel_deprojection,density_func(R,M,a))


def test_transform():
    '''
    Notes
    ------
    Old
    '''
    M,a = 2, 0.25
    R = np.logspace(-2,2,50) 

    # Analytical functions
    density_func = sampler.Plummer.density_func
    projection   = sampler.Plummer.projection
    # --------------------------------------------

    # define model
    model = models.Plummer(M,a)


    # Abel transform calculation
    abel_deprojection = jax.vmap(model.deproject)(R)

    # assert density_func(R,M,a) == pytest.approx(abel_deprojection)
    np.testing.assert_allclose(abel_deprojection,density_func(R,M,a))