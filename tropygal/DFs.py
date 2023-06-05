import numpy as np

_G = 4.300831457814024e-06 # kpc Msun^-1 (km/s)2


#-----------------------------
def DF_Isochrone(E, M, b, G=_G):
    e = -b*E/(G*M)
    A = 1./(np.sqrt(2)*(2.*np.pi)**3*(G*M*b)**1.5)
    return (A*(np.sqrt(e)/(2*(1.-e))**4)*
            (27 - 66*e + 320*e**2 - 240*e**3 + 64*e**4 + 3*(16*e**2 + 28*e -9)*np.arcsin(np.sqrt(e))/np.sqrt(e*(1-e))))
#------------------------
def g_Isochrone(E, M, b, G=_G):
    e = -b*E/(G*M)
    return (2.*np.pi)**3*np.sqrt(G*M)*b**2.5*(1-2*e)**2/(2*e)**2.5
