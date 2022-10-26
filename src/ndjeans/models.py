#third-party
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
from jax._src.config import config
config.update("jax_enable_x64", True)
import scipy.special as sc
from scipy.integrate import quad
import astropy.units as u

# project 
from . import abel
from .base import BaseModel

x,w  = np.loadtxt('/Users/juan/phd/projects/weird-jeans/src/data/gausleg_200',delimiter=',')
x = jnp.array(x)
w = jnp.array(w)
class HernquistZhao(BaseModel):
    
    def __init__(self,rhos: float,rs: float,a: float,b: float,c: float):
        '''
        _summary_

        Parameters
        ----------
        rhos : float
            scale density -- units: [M L^-3] if mass density,  [L^-3] for number density
        rs : float
            scale radius -- units: [L]
        a : float
            inner slope -- for  r/r_s < 1  rho_star ~ r^{-a} 
        b : float
            characterizes width of transition between inner slope and outer slope
        c : float
            outer slope -- for r/r_s >> 1 rho_star ~ r^{-c}
        Notes
        -----
        #!for c > 3 mass diverges r -> infinity
        References:
            Baes 2021      : http://dx.doi.org/10.1093/mnras/stab634
            An & Zhao 2012 : http://dx.doi.org/10.1093/mnras/sts175
            Zhao 1996      : http://dx.doi.org/10.1093/mnras/278.2.488
        TODO: Can i get away with only using M(r:float)? 
        '''
        self._rhos = rhos
        self._rs   = rs
        self._a    = a
        self._b    = b
        self._c    = c

    def get_density(self,r):

        return HernquistZhao.density(r,self._rhos,self._rs, self._a, self._b, self._c)

    def get_mass(self,r):

        return HernquistZhao.mass(r,self._rhos,self._rs, self._a, self._b, self._c)  

    @staticmethod
    @jax.jit
    def density(r,rhos : float,rs : float, a : float, b : float, c : float):
        '''
        Hernquist-zhao density profile.. at least thats what i call it
        Useful for both dark matter and stellar distributions

        Parameters
        ----------
        r : _type_
            _description_
        rhos : float
            _description_
        rs : float
            _description_
        a : float
            _description_
        b : float
            _description_
        c : float
            _description_

        Returns
        -------
        _type_
            _description_

        Notes
        -----
        Im writing these as static functions for a couple of reasons, but mainly because i want to be able to use them
        as fitting functions
        '''
        q = r/rs
        nu = (a-c)/b
        return rhos * q**-a * (1+q**b)**(nu)

    def projection(self,R):
        '''
        _summary_

        Parameters
        ----------
        func : _type_
            _description_
        R : _type_
            _description_
        params : _type_
            _description_
        '''

        return abel.project(self.density,R,(self._rhos,self._rs,self._a,self._b,self._c))

    @staticmethod    
    def mass_scipy(r,rhos: float,rs: float,a: float,b: float,c: float):
        '''
        Mass profile using Gauss's Hypergeometric function
        Notes
            See mass_error- too see comparision with calculating the mass by integrating the desity profile -- no errors until r/rs > 1e8
            #? Coeff calculation throws a warning at some point -- but doesnt when I try individual values?
            #! treatment of units is very clunky
        '''        
        q      = r/rs
        coeff  = 4*np.pi*q**(-a)/(3-a) 
        units = rhos*(r**3) # units are here: [Mass] = [Mass * L**-3] * [L**3]   
        
        gauss  = sc.hyp2f1((3-a)/b,(c-a)/b,(-a+b+3)/b,-q**b) # unitless function
        return coeff*gauss *units

    @staticmethod
    @jax.jit
    def mass(r: float,rhos: float,rs: float,a: float,b: float,c: float):
        a=0 
        xi=(r-a)*0.5*x[:,None] + .5*(r+a)
        wi=(r-a)*0.5*w[:,None]
        return jnp.sum(wi*4*jnp.pi*xi**2 *HernquistZhao.density(xi,rhos,rs,a,b,c),axis=0)

    @staticmethod
    @jax.jit
    def mass_fixed2(r: float,rhos: float,rs: float,a: float,b: float,c: float):
        xi=r*(0.5*x + .5)
        wi=r*0.5*w
        return jnp.sum(wi*4*jnp.pi*xi**2 *HernquistZhao.density(xi,rhos,rs,a,b,c),axis=0)

    def potential(self,r):
        '''
        Potential for Hernquist density:
        phi_in = -GM/r  term -- see mass()
        phi_out = integral_{r}^{\infty} \rho(r) r dr -- calculated using a slightly different hypergeometric function -- evaluated at r and a very large number

        shouldn't this just be r to r200?? 
        '''
        # for readability
        a = self._a
        b = self._b
        c = self._c
        r_s = self._r_s
        rho_s = self._rhos
        G =  4.5171031e-39
        q = r/r_s

        func = lambda r: -r**2 *(q)**(-a) * sc.hyp2f1((2 - a)/b, -(a - c)/b, 1 + (2 - a)/b, -q**b)/(-2 + a)
        
        phi_in  = (-G*self.mass(r)/r)
        phi_out =  -(4*np.pi*G*rho_s*(func(1e20)-func(r)))
        
        return phi_in+phi_out

class NFW(BaseModel):

    def __init__(self,rhos,rs):

        self._rhos   = rhos
        self._rs     = rs
        self._params = [self._rhos,self._rs]
    
    def get_density(self,r):
        
        return NFW.density(r,*self._params)

    def get_mass(self,r): 

        return NFW.mass(r,self._rhos,self._rs)

    @staticmethod 
    @jax.jit
    def density(r,rhos: float ,rs: float):
        q= r/rs
        return rhos * q**(-1) * (1+q)**(-2)

    @staticmethod
    @jax.jit
    def mass(r,rhos,rs):
    
        # r = np.logspace(-2,1,20)
        q = rs/(rs+r)
        mNFW = 4*jnp.pi*rhos*rs**3 * (q -jnp.log(q) -1) # Analytical NFW mass profile
        return mNFW

class Gaussians(BaseModel):

    def __init__(self,params, N:int):
        self._N = N
        self._params = params.reshape(2,N) #M_i, sigma_i pairs
    
    def get_density(self,r):
        return Gaussians.density_i(r)
    
    def get_mass(self,r):
        return Gaussians.mass_j(r)

    @staticmethod
    @jax.jit
    def density_i(r,M: float,sigma: float):

        rho = M * (jnp.sqrt(2*jnp.pi) * sigma**3)**-1 # part with the units  [Mass * Length**-3] 

        return rho * jnp.exp(-r**2/(2*sigma**2))


    @staticmethod
    @jax.jit
    def mass_j(r,M:float,sigma:float):
        
        return M*(jax.scipy.special.erf(r/(jnp.sqrt(2)*sigma)) - jnp.sqrt(2/jnp.pi) * r * jnp.exp(-r**2/ (2*sigma**2))/sigma)

    @staticmethod
    @jax.jit
    def mass(r: float,M:jnp.DeviceArray,sigma: jnp.DeviceArray):
        '''
        _summary_

        Parameters
        ----------
        r : float
            _description_
        M : jnp.DeviceArray
            _description_
        sigma : jnp.DeviceArray
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        _mass = jnp.sum(Gaussians.mass_j(r,M,sigma))
        return _mass

class Plummer:
    
    def __init__(self,M,a):
        '''
        TODO: 
            - implement sampling from a self consistent plummer
            - project into 2D
        '''
        self._M = M 
        self._a = a
        #calculate density
        self._rho = 3*self._M/(4*np.pi*self._a**3)
    
    def density(self,r:np.ndarray)->np.ndarray:
        
        return Plummer.density_func(r,self._M,self._a)

    def mass(self,r:np.ndarray)-> np.ndarray:
        '''
        three-dimensional mass profile for a Plummer sphere

        Parameters
        ----------
        r : np.ndarray
            radii [Length]

        Returns
        -------
        np.ndarray
            mass enclosed withing r [Mass]
        '''
        a   = self._a
        M = self._M
        term1 = M *r**3
        term2 = (r**2 + a**2)**(3/2)
        
        return term1/term2
    
    def projected_density(self,R: np.ndarray) -> np.ndarray:
        '''
        2D projection of plummer model

        Parameters
        ----------
        R : np.ndarrray
            Projected radii
        M : float
            'Mass' of system -- could just be total number of objects though
        a : float
            scale radius

        Returns
        -------
        np.ndarray
            [M.unit kpc^{-2}]
        '''
        return Plummer.projection(R,self._M,self._a)

    def sample_r(self,N: int) -> np.ndarray:
        '''
        sample from plummer sphere -- inverse transform-sampling
        '''
        y = np.random.uniform(size=N) # Draw random numbers from 0 to 1
        r = self._a/ np.sqrt(y**(-2 / 3) - 1)
        return r

    def sample_xyz(self,N: int) -> np.ndarray:
        '''
            draw x,y,z by sampling uniformly from a sphere
        '''
        r = self.sample_r(N)    
        xyz = np.zeros((3,N))
        phi = np.random.uniform(0, 2 * np.pi, size=N)
        temp = np.random.uniform(size=N)
        theta = np.arccos( 1 - 2 *temp)

        xyz[0] = r * np.cos(phi) * np.sin(theta)
        xyz[1] = r * np.sin(theta)*np.cos(theta)
        xyz[2] = r * np.cos(theta) #* Check the angle conventions again! 

        return xyz

    def sample_R(self,N: int)-> np.ndarray:
        '''
        Assuming z is the line-of-sight, returs random samples of projection positions

        Parameters
        ----------
        N : _type_
            Number of samples

        Returns
        -------
        np.ndarray
            Projected sample of objects whose distribution follows as plummer sphere
        '''
        r      = self.sample_r(N)    
        xyz    = np.zeros((2,N))
        phi    = np.random.uniform(0, 2 * np.pi, size=N)
        temp   = np.random.uniform(size=N)
        theta  = np.arccos( 1 - 2 *temp)
        xyz[0] = r * np.cos(phi) * np.sin(theta)
        xyz[1] = r * np.sin(theta)*np.cos(theta)
        return np.sqrt(xyz[0]**2+xyz[1]**2)

    def probability(self, x: np.ndarray) -> np.ndarray:
        '''
        '''
        return 4*np.pi*x**2 * self.density_func(x,1,self._a)

    @staticmethod
    def from_scale_density_radius(rho: float, a:float):
        '''
        Alternate way to create instance of Plummer model from
        scale density and raidus
        '''
        M = (rho*4*np.pi*a**3)/3
        return Plummer(M,a) 

    @staticmethod
    def mass_func(x:np.ndarray,M:float,a:float) -> np.ndarray:
        '''
        Making these static methods for fitting purposes
        
        Parameters
        ----------
        x : np.ndarray
            _description_
        M : float
            _description_
        a : float
            _description_

        Returns
        -------
        np.ndarray
            _description_
        '''
        term1 = M *x**3
        term2 = (x**2 + a**2)**(3/2)
        return term1/term2

    @staticmethod
    def density_func(x: np.ndarray,M: float,a: float) -> np.ndarray:
        '''
        Making these static methods for fitting purposes

        Parameters
        ----------
        x : np.ndarray
            _description_
        M : float
            _description_
        a : float
            _description_

        Returns
        -------
        np.ndarray
            _description_
        '''
        q  = x/a
        return 3*M/(4*np.pi*a**3)*(1+q**2)**(-5/2)

    @staticmethod
    def projection(R: np.ndarray, M: float, a: float) -> np.ndarray:
        '''
        2D projection of plummer model

        Parameters
        ----------
        R : np.ndarray
            Projected radii
        M : float
            'Mass' of system -- could just be total number of objects though
        a : float
            scale radius

        Returns
        -------
        np.ndarray
            [M.unit kpc^{-2}]
        '''
        coeff = M/(np.pi*a**2) #units [M.unit/a.unit**2]
        x = R/a
        return coeff * (1 + x**2)**(-2)

    @staticmethod
    def prob_func(x: np.ndarray,a: float) -> np.ndarray:
        return 4*np.pi*x**2 * Plummer.density_func(x,1,a)

    @staticmethod
    def bin_centers(bin_edges):
        '''
        this is lazy. I know.
        '''
        r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
        return r_center

    @staticmethod
    def radial_distribution(r,bin_method ='doane'):

        N,bin_edges = np.histogram(r,bins =bin_method)
        v_shell = (4/3)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        r_center = (np.roll(bin_edges,-1)+bin_edges)[:-1]/2
        nu = N/v_shell
        return r_center,nu,bin_edges,v_shell



#? What other classes should be here?
# TODO: Some of the static methods in plummer, don't need to be there so i should get rid of them


class JeansOM:    
    def __init__(self,data: np.ndarray,model_dm,model_tr):
        '''
        Parameters
        ----------
        data : np.ndarray
            (projected positions,line-of-sight velocities)

        Notes
        -----
        
        '''
        self._G    = 4.5171031e-39 # Gravitational constant in the right units
        
        self._data = data
        self._dm   = model_dm # dark matter
        self._tr   = model_tr # tracers 

    def disp_integrand(self,x: float,R: float,kernel: Callable):
        '''
        integrand for line-of-sight velocity dipersion at position R

        Parameters
        ----------
        x : float
            dummy variable in integration
        R : float
            lower bound of integral
        kernel : Callable
            kernel for stellar anisotropy model

        Returns
        -------
        _type_
            _description_

        Notes
        -------
        TODO: ADD upperbound to integral. Either cut off stellar distribution or set upper bound of integral to be r200
        '''
        kern    = kernel(x,R,self._beta)
        nu_star = self._tracer.profile3d(x).value
        mass_dm = self._dm.mass(x).value
        
        return kern * nu_star*mass_dm/x**(2-2*self._beta)

    def dispersion(self,R):
        '''
        Project Velocity Dispersion
        
        Notes
        -------
        TODO: ADD upperbound to integral. Either cut off stellar distribution or set upper bound of integral to be r200
 
        '''
        coeff = 2*self._G/self._tracer.profile2d(R).value
        
        R,units = R.value,R.unit
        
        output = np.array([quad(self.disp_integrand,i,np.inf,args=(i,JeansOM.kernel))[0] for i in R])
        return ((coeff * output)*u.kpc**2/u.s**2).to(u.km**2/u.s**2).value

    @staticmethod
    def likelihood(data: np.ndarray,model: Callable,theta: np.ndarray) -> float:
        '''
        _summary_

        Parameters
        ----------
        data : np.ndarray
            _description_
        model : Callable
            _description_
        theta : np.ndarray
            parameters we're fitting

        Returns
        -------
        float
            _description_
        '''
        R      = data[0]
        vi     = data[1]
        err    = np.full_like(R,0) # here in case i want to change this to include errors
        N      = len(vi)
        e2     = model(R,theta) + err**2
        

        term1 = - 0.5 * np.sum(np.log(e2))
        term2 = - 0.5 * np.sum(vi**2/e2)   # This assumes that the velocities are gaussian centered at 0 
        term3 = - 0.5 * N *np.log(2*np.pi) # doesnt really need to be here
        return term1+term2+term3

    @staticmethod
    def kernel(r:float,R:float,beta:float)-> float:
        '''
        Integration kernel for constant anisotropy models
        See e.g. Capellari 2008 eq.43

        Notes
        -------
        The special functions needed to make this integral simple make auto-grad capabilities undoable.
        It's probably possible if we do out the full integral under no assumptions about the form of the stellar anisotropy.
        ...might be worth looking into

        '''
        t1 = .5* R**(1-2*beta)
        w  = (R/r)**2
        
        if beta >1/2:
            t2  = beta * sc.betainc(beta+0.5,0.5,w) - sc.betainc(beta-.5,.5,w)

            t3  = np.sqrt(np.pi)*(3/2 -beta)*sc.gamma(beta-1/2)/sc.gamma(beta)
        else: 
            a = beta+0.5
            b = 0.5
            betainc = ((w**(a))/a) *  sc.hyp2f1(a,1-b,a+1,w)
            a2=beta-0.5
            betainc2 = ((w**(a2))/a2) *  sc.hyp2f1(a2,1-b,a2+1,w)
            t2  = beta * betainc - betainc2
            t3  = np.sqrt(np.pi)*(3/2 -beta)*sc.gamma(beta-1/2)/sc.gamma(beta)
        
        out = t1*(t2+t3)
        return out 
    
    @staticmethod
    def disp_integrand(x: float,R: float,nu3: Callable,theta: np.ndarray) -> float:
        '''
        integrand for line-of-sight velocity dipersion at position R

        Parameters
        ----------
        x : float
            dummy variable in integration
        R : float
            lower bound of integral
        kernel : Callable
            kernel for stellar anisotropy model
        Returns
        -------
        _type_
            _description_

        Notes
        -------
        TODO: ADD upperbound to integral. Either cut off stellar distribution or set upper bound of integral to be r200
        '''

        kern    = JeansOM.kernel(x,R,theta[-1])
        nu_star = nu3(x)
        mass_dm = JeansOM.mass(x,theta[:-1])
        
        return kern * nu_star*mass_dm/x**(2-2*theta[-1])

    @staticmethod
    def projected_dispersion(R:np.ndarray,nu2: Callable,nu3: Callable,theta: np.ndarray) -> np.ndarray:
        '''
        Project dispersion at radii R

        Parameters
        ----------
        R : np.ndarray
            _description_
        nu2 : Callable
            Projected stellar distribution
        nu3 : Callable
            3D stellar distribution (one of these is calculated using the abel transform)
        theta : np.ndarray
            [inner-slope,scale density, scale radius, stellar anisotropy]

        Returns
        -------
        np.ndarray
            _description_
        
        Notes
        _______

        TODO: Change upperbound of integral to be r_{200} for those model parameters -- Some
        
        '''
        G = 4.5171031e-39 # Gravitational constant [kpc^{3} solMass^{-1}  s^{-2}]
        coeff = 2*G/nu2(R)
        output = np.array([quad(JeansOM.disp_integrand,i,np.inf,args=(i,nu3,theta))[0] for i in R])
        return coeff* output


# WORKING ON TODAY October 18: 
    @staticmethod
    def beta(r:float,a: float):
        '''
        Osipkov-Merrit stellar anisotropy

        Parameters
        ----------
        r : float
            _description_
        a : float
            _description_

        Returns
        -------
        _type_
            0 as r -> 0
            1 as r -> infinity
        '''
        return r**2/(r**2+a**2)

    @staticmethod
    def f_beta(r:float,a:float):
        return r**2+a**2

    @staticmethod
    def beta_func(f,r,a):
        grad = jax.grad(f)
        return r*grad(r,a)/f(r,a)

    @staticmethod
    def nu_sigma(r,r200,abeta,rhos,rs,a,b,c):
        G    = 4.5171031e-39
        x0=r
        x1=r200
        xi=(x1-x0)*0.5*x[:,None] + .5*(x1+x0)
        wi=(x1-x0)*0.5*w[:,None]
        term1 = JeansOM.f_beta(r,abeta)
        term2 = jnp.sum(wi*JeansOM.f_beta(xi,abeta)*Plummer.density_func(xi,1,.25)*HernquistZhao.mass(xi,rhos,rs,a,b,c)/xi**2,axis=0)
        return G*term2/term1

    @staticmethod
    def sigma(r,r200,abeta,rhos,rs,a,b,c):
        G    = 4.5171031e-39
        x0=r
        x1=r200
        xi=(x1-x0)*0.5*x[:,None] + .5*(x1+x0)
        wi=(x1-x0)*0.5*w[:,None]
        term1 = JeansOM.f_beta(r,abeta)
        term2 = jnp.sum(wi*JeansOM.f_beta(xi,abeta)*Plummer.density_func(xi,1,.25)*HernquistZhao.mass(xi,rhos,rs,a,b,c)/xi**2,axis=0)
        return G*term2/term1/Plummer.density_func(r,1,.25)
    
    @staticmethod
    def dispersion(R,r200,abeta,rhos,rs,a,b,c):
        x0=R
        x1=r200
        xi=(x1-x0)*0.5*x[:,None] + .5*(x1+x0)
        wi=(x1-x0)*0.5*w[:,None]
        
        term1 = xi* (1 - JeansOM.beta(xi,abeta))* R**2/xi**2
        term2 = jnp.sqrt(xi**2 - R**2)
        return jnp.sum(wi*2*JeansOM.nu_sigma(xi,r200,abeta,rhos,rs,a,b,c)*term1/term2,axis=0)



