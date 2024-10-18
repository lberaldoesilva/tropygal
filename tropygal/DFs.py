import numpy as np
import agama

_G = 4.300831457814024e-06 # kpc Msun^-1 (km/s)2

#-----------------------------
def DF_Isochrone(E, M, b, G=_G):
    """ Distribution Function (DF) of isochrone model - see Binney & Tremaine (2008), Eq. (4.54)
    
    Parameters
    ----------
    E: float
         energy
    M: float
         total mass
    b: float
         scale lenght
    G: float
         gravitational constant (standard value in kpc Msun^-1 (km/s)2)

    Returns
    -------
    float
      the DF

    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    e = -b*E/(G*M)
    A = 1./(np.sqrt(2)*(2.*np.pi)**3*(G*M*b)**1.5)
    return (A*(np.sqrt(e)/(2*(1.-e))**4)*
            (27 - 66*e + 320*e**2 - 240*e**3 + 64*e**4 + 3*(16*e**2 + 28*e -9)*np.arcsin(np.sqrt(e))/np.sqrt(e*(1-e))))
#------------------------
def g_Isochrone(E, M, b, G=_G):
    """ The density of states for the isochrone model
    
    Parameters
    ----------
    E: float
         energy
    M: float
         total mass
    b: float
         scale lenght
    G: float
         gravitational constant (standard value in kpc Msun^-1 (km/s)2)

    Returns
    -------
    float
      the density of states

    References
    ----------
    .. [1] Binney, J., & Petrou, M. 1985, MNRAS, 214, 449
    """
    e = -b*E/(G*M)
    return (2.*np.pi)**3*np.sqrt(G*M)*b**2.5*(1-2*e)**2/(2*e)**2.5

#------------------------
def Tr_Isochrone(E, M, G=_G):
    """ The period of radial oscillation for the isochrone model

    Parameters
    ----------
    E: float
         energy
    M: float
         total mass
    G: float
         gravitational constant (standard value in kpc Msun^-1 (km/s)2)

    Returns
    -------
    float
      the period of radial motion

    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    return (2.*np.pi)*M*G/(-2*E)**1.5

def gEL_Isochrone(E, L, M, G=_G):
    """ The density of states for the isochrone model, but assuming (generically) a DF that depends on energy and angular momentum

    Parameters
    ----------
    E: float
         energy
    L: float
         angular momentum
    M: float
         total mass
    G: float
         gravitational constant (standard value in kpc Msun^-1 (km/s)2)

    Returns
    -------
    float
      the density of states

    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    return 8.*np.pi**2*L*Tr_Isochrone(E, M, G=G)

#------------------------
def Tr_Spherical(coords, pot):
    """ The period of radial oscillation for the a generic spherical potential. Current version requires agama
    
    Parameters
    ----------
    coords: 6D array
         coordinates of particle
    pot: potential object
         total potential

    Returns
    -------
    float
      the period of radial motion

    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    actF = agama.ActionFinder(pot)
    Omega = actF(coords, actions=False, frequencies=True, angles=False)

    Om_r = Omega[:,0]
    return (2.*np.pi)/Om_r

def gEL_Spherical(coords, pot):
    """ The density of states assuming a DF that depends on energy and angular momentum, in a generic spherical potential. Current version requires agama
    
    Parameters
    ----------
    coords: 6D array
         coordinates of particle
    pot: potential object
         total potential

    Returns
    -------
    float
      the density of states

    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    x = coords[:,0]
    y = coords[:,1]
    z = coords[:,2]
    vx = coords[:,3]
    vy = coords[:,4]
    vz = coords[:,5]
    Lx = Lx = (y*vz - z*vy)
    Ly = (z*vx - x*vz)
    Lz = (x*vy - y*vx)
    L = np.sqrt(Lx**2 + Ly**2 + Lz**2)
    
    return 8.*np.pi**2*L*Tr_Spherical(coords, pot)
