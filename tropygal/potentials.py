import numpy as np
import scipy

from .constants import _G_gal

#----------------
def find_Phi_eff_min(L, dPhi_dr, params, range_0=(1e-3, 1e+3), max_range=1e+8):
    """ Find the location of the minimum of the effective potential.
    The effective potential is defined as Phi_eff = Phi + L**2/(2.*r**2).
    This function is primarily to calculate the peri- and apo-center radii rper and rapo, but can also be used with any other purposes.
    rper and rapo, on the other hand, are primarily calculated to calculate the period of radial motion T_r.
    T_r on its turn can be used to calculate the density of states for DFs assumed to depend on energy and angular momentum as g(E,L) = 8 pi^2 L T_r(E,L).

    Parameters
    ----------
    L: array (or scalar)
        angular momentum of particle
    dPhi_dr: function
        function with signature (r, params), evaluating the derivative of the potential at each radius
    params: array
        the parameters to be used in Phi and dPhi_dr
    range_0: tuple
        the initial range within which we search for the root of dPhi_eff_dr

    Returns
    -------
    array, array
        Arrays containing rper and rapo for all particles used at input
    """
    
    L = np.atleast_1d(L)
    rmin = np.atleast_1d(np.full(np.size(L), np.nan, dtype=float))
    
    for i in range(np.size(L)):
        # derivative of effective potential:
        df = lambda r: dPhi_dr(r, params) - L[i]**2/r**3
    
        this_range = range_0
        # Try progressively larger intervals
        while np.isnan(rmin[i]):
            try:
                rmin[i] = scipy.optimize.brentq(df, this_range[0], this_range[1])
            except ValueError:
                this_range = (this_range[0]/2., this_range[1]*2) 
    
            if this_range[1] > max_range:
                raise ValueError("tropygal: Could not find minimum of Phi_eff - Try increasing max_range (standard is 1e+8).")

    return rmin
#----------------
def find_rper_rapo(E, L, Phi, dPhi_dr, params, range_0=(1e-2, 1e+2), max_range=1e+8):
    """ Find peri- and apo-center distances with interval adjustments.
    It first identifies rmin, the location of the minimum of the effective potential.
    The effective potential is defined as Phi_eff = Phi + L**2/(2.*r**2).
    Then it defines the range where brentq should look for roots, such that the pericenter radius rper is searched from rmin inwards, and rapo is searched from rmin outwards.
    rper and rapo are primarily calculated to calculate the period of radial motion T_r.
    T_r on its turn can be used to calculate the density of states for DFs assumed to depend on energy and angular momentum as g(E,L) = 8 pi^2 L T_r(E,L).

    Parameters
    ----------
    E: array (or scalar)
        energy of particle
    L: array (or scalar)
        angular momentum of particle
    Phi: function
        function with signature (r, params), evaluating the potential at each radius
    dPhi_dr: function
        function with signature (r, params), evaluating the derivative of the potential at each radius
    params: array
        the parameters to be used in Phi and dPhi_dr
    range_0: tuple
        the initial range within which we search for the roots defining rper and rapo

    Returns
    -------
    array, array
        Arrays containing rper and rapo for all particles used at input
    """
    rper = np.full(np.size(E), np.nan, dtype=float)
    rapo = np.full(np.size(E), np.nan, dtype=float)

    # Before calculating rper and rapo, find the minimum of Phi_eff:
    rmin = find_Phi_eff_min(L, dPhi_dr, params)

    for i in range(np.size(E)):
        # define the function for which we seek roots (in the denominator of the integrand of Tr, the radial period)
        # f = lambda r: E[i] - Phi_eff(r, L[i], Phi, params)
        f = lambda r: E[i] - Phi(r, params) - L[i]**2/(2.*r**2)
        
        this_range = range_0
        # Make sure rmin in within the initial range where brentq will look for rper and rapo:
        if (this_range[0] > rmin[i]):
            this_range = (rmin[i]/2, this_range[1])
        if (this_range[1] < rmin[i]):
            this_range =  (this_range[0], rmin[i]*2)

        new_range_0 = this_range
        # Try progressively larger intervals
        #-------------
        # for rper:
        while np.isnan(rper[i]):
            try:
                rper[i] = scipy.optimize.brentq(f, this_range[0], rmin[i])
            except ValueError:
                this_range = (this_range[0]/2., rmin[i])  # Expand lower bound 
        #-------------
        # for rapo:
        this_range = new_range_0
        while np.isnan(rapo[i]):
            try:
                rapo[i] = scipy.optimize.brentq(f, rmin[i], this_range[1])
            except ValueError:
                this_range = (rmin[i], this_range[1]*2)  # Expand upper bound
            
        if this_range[1] > max_range:
            raise ValueError("tropygal: Could not find peri- and apo-center distances in specified range.")

    return rper, rapo
#------------------------
def Tr_Spherical(E, L, Phi, dPhi_dr, params):
    """ The period of radial oscillation, T_r, for the a generic spherical potential. See Binney & Tremaine (2008) - eq. (3.17).
    T_r can be used, among other things, to calculate the density of states for DFs assumed to depend on energy and angular momentum as g(E,L) = 8 pi^2 L T_r(E,L).
    
    Parameters
    ----------
    E: array (or scalar)
        energy of particle
    L: array (or scalar)
        angular momentum of particle
    Phi: function
        function with signature (r, params), evaluating the potential at each radius
    dPhi_dr: function
        function with signature (r, params), evaluating the derivative of the potential at each radius
    params: array
        the parameters to be used in Phi and dPhi_dr
     
    Returns
    -------
    float array
      The period of radial motion
    
    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """

    rper, rapo = find_rper_rapo(E, L, Phi, dPhi_dr, params)
    #-----------------
    def integrand(r, E, L, Phi, params):
        # return 2./np.sqrt(2.*( E - Phi_eff(r, L, Phi, params)))
        return 2./np.sqrt(2.*( E - Phi(r, params)) - L**2/r**2)
    #-----------------
    T_r = np.full(np.size(E), np.nan, dtype=float)
    for i in range(np.size(E)):
        if (rper[i] != rapo[i]):
            T_r[i] = scipy.integrate.quad(integrand, a=rper[i], b=rapo[i], args=(E[i], L[i], Phi, params), full_output=1)[0]
        else:
            # For orbits very close to circular:
            # we estimete the second derivative of the effective potential numerically and approximate
            # Tr ~ 2pi/sqrt(d^2Phi_eff/dr^2 (rc)), where rc is the radius of circular motion
            h = 1e-4*rper[i]
            # d2Phi_eff_dr2 = (Phi_eff(rper[i] + h, L[i], Phi, params) - 2 * Phi_eff(rper[i], L[i], Phi, params) + Phi_eff(rper[i] - h, L[i], Phi, params)) / h**2
            d2Phi_eff_dr2 = (Phi(rper[i] + h, params) - 2 * Phi(rper[i], params) + Phi(rper[i] - h, params)) / h**2 + 3 * L[i]**2/(rper[i]**4)
            T_r[i] = 2.*np.pi/np.sqrt(d2Phi_eff_dr2)
            print ('tropygal: Aproximating Tr for very circular orbit.')
    return T_r
#----------------------
def Phi_Isochrone(r, params):
    """ The gravitational potential for the Isochrone model.
    
    Parameters
    ----------
    r: array (or scalar)
         radial coordinate of particle
    params: array
        the parameters to be used in the potential.
     
    Returns
    -------
    float
      The potential evaluated at radius r.
    
    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    M = params[0] # total mass
    b = params[1] # scale length
    G = params[2] # Gravitational constant
    
    return -(G*M/b)*(1./(1 + np.sqrt(1 + (r/b)**2)))
#----------------------
def dPhi_dr_Isochrone(r, params):
    """ The derivative of the gravitational potential for the Isochrone model.
    
    Parameters
    ----------
    r: array (or scalar)
         radial coordinate of particle
    params: array
        the parameters to be used in the potential.
     
    Returns
    -------
    float
        The derivative of the potential, evaluated at a radius r.
    
    References
    ----------
    .. [1] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    M = params[0] # total mass
    b = params[1] # scale length
    G = params[2] # Gravitational constant

    return (G*M/b**2)*(r/b)/( np.sqrt(1 + (r/b)**2)*(1 + np.sqrt(1 + (r/b)**2))**2 )
#----------------------
def Phi_Hernquist(r, params):
    """ The gravitational potential for the Hernquist model.
    
    Parameters
    ----------
    r: array (or scalar)
         radial coordinate of particle
    params: array
        the parameters to be used in the potential.
     
    Returns
    -------
    float
      The potential evaluated at radius r.
    
    References
    ----------
    .. [1] Hernquist, L (1990).
    .. [2] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    M = params[0] # total mass
    b = params[1] # scale length
    G = params[2] # Gravitational constant
    
    return -(G*M/b)*( 1./(1 + r/b) )
#----------------------
def dPhi_dr_Hernquist(r, params):
    """ The derivative of the gravitational potential for the Hernquist model.
    
    Parameters
    ----------
    r: array (or scalar)
         radial coordinate of particle
    params: array
        the parameters to be used in the potential.
     
    Returns
    -------
    float
        The derivative of the potential, evaluated at a radius r.
    
    References
    ----------
    .. [1] Hernquist, L (1990).
    .. [2] Binney, J., & Tremaine, S. (2008). Galactic Dynamics (2nd ed.). Princeton University Press
    """
    M = params[0] # total mass
    b = params[1] # scale length
    G = params[2] # Gravitational constant

    return (G*M/b**2)*( 1./(1 + r/b)**2 )
