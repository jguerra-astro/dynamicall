from typing import Callable
import numpy as np

def lnLikelihood(data: np.ndarray,obs_error: np.ndarray,model: Callable, theta: np.ndarray) -> float:
    '''
    log likelihood function for 2+1 dimensional data e.g (x,y,v_{los}) equivalenetly (R,v_{los})

    Parameters
    ----------
    data : np.ndarray
        Projected positions and line of sight velocities
        R    = data[0]
        vlos = data[1]
    error : np.ndarray
        assuming that measurements are indepentt, error.shape = vlos.shape
    theta : np.ndarray
        Parameters we're trying to fit
        e.g. for basic models: (inner-slope, rho_d,r_d,stellar-anisotropy)


    Returns
    -------
    float
        _description_


    Notes
    -------
    #?: Maybe it would be better to accept x,y,v intead of R,v?

    TODO: currently we fit the stellar component seperately, but we should probably do everything at once.
    '''

    q  = data[0]            # projected radii
    v  = data[1]            # line-of-sight/"radial" velocitiess
    N  = v.shape[0]         # number of observed stars
    sigmasq  = model(q,theta) + obs_error**2
        
    # log of likelihood
    term1 = - 0.5 * np.sum(np.log(sigmasq))
    term2 = - 0.5 * np.sum(v**2/sigmasq)   # This assumes that the velocities are gaussian centered at 0 -- could also fit for this
    term3 = - 0.5 * N *np.log(2*np.pi) # doesnt really need to be here

    return term1+term2+term3


def lnLikelihood2(data: np.ndarray,obs_error: np.ndarray,model: Callable, theta: np.ndarray) -> float:
    '''
    Log likelihood assuming 3+1 dimenstional data (x,y,z,v_{los})


    Parameters
    ----------
    data : np.ndarray
        _description_
    obs_error : np.ndarray
        _description_
    model : Callable
        _description_
    theta : np.ndarray
        _description_

    Returns
    -------
    float
        _description_

    Notes
    -------
    Ok, so whats up? What does adding positions do

    '''
    q  = data[0]            # projected radii
    v  = data[1]            # line-of-sight/"radial" velocitiess
    N  = v.shape[0]         # number of observed stars
    sigmasq  = model(q,theta) + obs_error**2

    return -1