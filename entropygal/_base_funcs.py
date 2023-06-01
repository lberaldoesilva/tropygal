import numpy as np
from scipy.special import gamma as Gamma
from scipy.spatial import KDTree


pisq = np.pi**2.

#-----------------
def V_d(dim):
    """ Volume of (unit-radius) hyper-sphere of dimension d

    Parameters
    ----------
    dim: int
         dimension

    Returns
    -------
    float number
      the volume [pi^(dim/2.)]/Gamma(dim/2. + 1)
    """
    # Particular cases (for performance):
    if (dim==1):
        return 2.
    if (dim==2):
        return np.pi
    if (dim==3):
        return 4.*np.pi/3.
    if (dim==4):
        return pisq/2.
    if (dim==5):
        return 8.*pisq/15.
    if (dim==6):
        return pisq*np.pi/6.
    
    return np.pi**(dim/2.)/Gamma(dim/2. + 1)
# -----------------------

def avg_ln_f(dim, N, data):
    """ average of ln(f) over the sample

    This is defined as (1/N) * sum_i=1^N ln(f_i),
    where f_i is the estimate of the DF f around point/particle/star i
    From e.g. Eq. (10) in Leonenko, Pronzato, Savani (2008):
    f_i = 1/[ (N-1) * exp(gamma) * V_d * D**dim ]

    Parameters
    ----------
    dim: int
       Dimension of phase-space
    N: int
       Number of data points
    data: array [dim, N]
       Data points

    Returns
    -------
    float
       <ln(f)> = (1/N) * sum_i=1^N ln(f_i)
    """
    tree = KDTree(data, leafsize=10) # default leafsize=10
    dist, ind = tree.query(data, k=2, workers=-1) # workers is number of threads. -1 means all threads
    idx = np.where(dist[:,1]>0)[0]
    N_zero_dist = N - len(idx) # Number of points with zero distance (typically, if not zero,  very small compared to N)
    
    if (N_zero_dist > 0):
        print (N_zero_dist,' points with zero D_NN neglected')
        N = N - N_zero_dist

    ln_D = np.log(dist[idx,1])
    ln_f = -np.log(N-1.) - np.euler_gamma - np.log(V_d(dim)) - dim*ln_D
    return np.mean(ln_f)
#-----------------------------
def entropy(dim, N, data, J=1):
    """
    Estimate of Boltzmann/Shannon entropy
    S = - int f' ln (f'/J) d^dim x'
    Here f' and x' are assumed to be dimensionless
    Also, x' is typically of order unit, i.e. x' ~ x/sigma_x,
    where sigma_x is a measure of the dispersion of the physical quantity x

    Parameters
    ----------
    dim: int
       Dimension of phase-space
    N: int
       Number of data points
    data: array [dim, N]
       Data points
    J: float number
       J = |del x/del x'| is the jacobian of transf. from (x, y,...) -> (x', y', ...)
       If x' = x/sigma_x, y' = y/sigma_y... -> J = |sigma_x*sigma_y...|

    Returns
    -------
    float
       entropy estimate -<ln(f/J)> = - (1/N)*sum_i=1^N ln(f_i/J_i)
    """
    
    avg_ln_J = np.mean(np.log(J))
    return -avg_ln_f(dim, N, data) + avg_ln_J
