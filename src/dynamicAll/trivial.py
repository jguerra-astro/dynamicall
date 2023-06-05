
from functools import partial

import jax
from jax._src.config import config
config.update("jax_enable_x64", True)
class HarmonicOscillator:

    def __init__(self,w: float) -> None:

        self.w = w
    
    def potential(self,x):

        return 0.5*self.w**2*x**2
    
    def force(self,x: float) -> float:

        return -self.w**2*x
    
    def energy(self, x: float, v: float) -> float:

        return self.potential(x)+ 0.5*v**2
    
    def action(self,x,v):
        
        return self.energy(x,v)/self.w
    
    @staticmethod
    @jax.jit
    def _action(w,x,v):
        
        return (0.5*v**2 + 0.5*w**2*x**2)/w
