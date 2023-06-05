import numpy as np

from ._base_funcs import V_d, avg_ln_f, entropy

_G = 4.300831457814024e-06 # kpc Msun^-1 (km/s)2

#-----------------------
def S_E(E, g):
    """
    Estimate the entropy for a system whose DF is a function of energy only
    This applies for a phase-mixed spherical and isotropic system

    Parameters
    ----------

    E: array [N]
       Energy of particles
    g: array [N]
       density of states g(E) = (4pi)^2 int_0^r_m(E) dr r^2 sqrt(2(E - phi)),
       where phi(r) is the gravitational potential
    """
    return entropy(E, J=g)

#-----------------------------


