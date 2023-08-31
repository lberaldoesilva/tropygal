import numpy as np

from ._base_funcs import V_d, entropy, C_k, renyi_entropy, tsallis_entropy

_G = 4.300831457814024e-06 # kpc Msun^-1 (km/s)2

#-----------------------
def S_E(E, g, k=1):
    """
    Estimate the differential entropy for a system whose DF is a function of energy only
    This applies for a phase-mixed spherical and isotropic system

    Parameters
    ----------

    E: array [N]
       Energy of particles
    g: array [N]
       density of states g(E) = (4pi)^2 int_0^r_m(E) dr r^2 sqrt(2(E - phi)),
       where phi(r) is the gravitational potential
    """
    return entropy(E, mu=g, k=k)
#-----------------------
def S_EL(E, L, g, k=1):
    """
    Estimate the differential entropy for a system whose DF is f = f(E,L). 
    This applies for a phase-mixed spherical and anisotropic system

    Parameters
    ----------

    E: array [N]
       Energy of particles
    L: array [N]
       Angular momentum amplitude of particles
    g: array [N]
       density of states g(E,L) = 8pi^2 L T_r(E,L),
       where T_r = 2 * int_rper^rapo dr 1/sqrt(2(E - phi) - L^2/r^2)
       is the radial period - Binney & Tremaine Eq. (3.17)
       and phi(r) is the gravitational potential
    """

    data = np.column_stack((E, L))
    return entropy(data, mu=g, k=k)
#-----------------------
def S_renyi_E(E, g, q=2, k=1):
    """
    Estimate the Rényi entropy for a system whose DF is a function of energy only
    This applies for a phase-mixed spherical and isotropic system

    Parameters
    ----------

    E: array [N]
       Energy of particles
    g: array [N]
       density of states g(E) = (4pi)^2 int_0^r_m(E) dr r^2 sqrt(2(E - phi)),
       where phi(r) is the gravitational potential
    """
    return renyi_entropy(E, mu=g, q=q, k=k)
#-----------------------
def S_renyi_EL(E, L, g, q=2, k=1):
    """
    Estimate the Rényi entropy for a system whose DF is f = f(E,L). 
    This applies for a phase-mixed spherical and anisotropic system

    Parameters
    ----------

    E: array [N]
       Energy of particles
    L: array [N]
       Angular momentum amplitude of particles
    g: array [N]
       density of states g(E,L) = 8pi^2 L T_r(E,L),
       where T_r = 2 * int_rper^rapo dr 1/sqrt(2(E - phi) - L^2/r^2)
       is the radial period - Binney & Tremaine Eq. (3.17)
       and phi(r) is the gravitational potential
    """

    data = np.column_stack((E, L))
    return renyi_entropy(data, mu=g, q=q, k=k)
#-----------------------
def S_tsallis_E(E, g, q=2, k=1):
    """
    Estimate the Tsallis entropy for a system whose DF is a function of energy only
    This applies for a phase-mixed spherical and isotropic system

    Parameters
    ----------

    E: array [N]
       Energy of particles
    g: array [N]
       density of states g(E) = (4pi)^2 int_0^r_m(E) dr r^2 sqrt(2(E - phi)),
       where phi(r) is the gravitational potential
    """
    return tsallis_entropy(E, mu=g, q=q, k=k)
#-----------------------
def S_tsallis_EL(E, L, g, q=2, k=1):
    """
    Estimate the Tsallis entropy for a system whose DF is f = f(E,L). 
    This applies for a phase-mixed spherical and anisotropic system

    Parameters
    ----------

    E: array [N]
       Energy of particles
    L: array [N]
       Angular momentum amplitude of particles
    g: array [N]
       density of states g(E,L) = 8pi^2 L T_r(E,L),
       where T_r = 2 * int_rper^rapo dr 1/sqrt(2(E - phi) - L^2/r^2)
       is the radial period - Binney & Tremaine Eq. (3.17)
       and phi(r) is the gravitational potential
    """

    data = np.column_stack((E, L))
    return tsallis_entropy(data, mu=g, q=q, k=k)

#-----------------------------


