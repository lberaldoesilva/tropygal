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
def avg_ln_f(data):
    """ average of ln(f) over the sample

    This is defined as (1/N) * sum_i=1^N ln(f_i),
    where f_i is the estimate of the DF f around point/particle/star i
    From e.g. Eq. (10) in Leonenko, Pronzato, Savani (2008):
    f_i = 1/[ (N-1) * exp(gamma) * V_d * D**dim ]

    Parameters
    ----------
    data: array [N, dim]
       Data points

    Returns
    -------
    float
       <ln(f)> = (1/N) * sum_i=1^N ln(f_i)
    """

    if (len(np.shape(data)) == 1):
        dim = 1
        data = np.reshape(data, (len(data), 1))
    elif (len(np.shape(data)) == 2):
        dim = np.shape(data)[1]
    else:
        print ('Array with data should be of form [N, dim]')
        return np.nan
    
    N = np.shape(data)[0]
    
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
def entropy(data, J=1):
    """
    Estimate of Boltzmann/Shannon entropy
    S = - int f ln (f/J) d^dim x
    The factor J ensures that f/J is dimensionless and 
    that S is invariant for changes of variable x -> x', in which case J = |del x' / del x|
    For precise estimates, we also want all (x_1, x_2..., x_dim) to be order unit.
    So, if x and f are dimensionless and x is order unit, we can define J = 1.

    Parameters
    ----------
    data: array [N, dim]
       Data points
    J: float number or array of size N
       J = |del x'/del x| is the jacobian of transf. from (x, y,...) -> (x', y', ...)
       If x' = x/sigma_x, y' = y/sigma_y... -> J = 1/|sigma_x*sigma_y...|

    Returns
    -------
    float
       entropy estimate -<ln(f/J)> = - (1/N)*sum_i=1^N ln(f_i/J_i)
    """
    
    avg_ln_J = np.mean(np.log(J))
    
    return -avg_ln_f(data) + avg_ln_J

# -----------------------
def cross_avg_ln_f(data1, data2):
    """ average of ln(f) over the sample sampled by f0

    This is defined as (1/N) * sum_i=1^N ln(f_i),
    where f_i based on the distance D between points in the first data set (with N points)
    to points in the second data set (with M points)

    From e.g. Eq. (11) in Leonenko, Pronzato, Savani (2008):
    f_i = 1/[ M * exp(gamma) * V_d * D**dim ]

    Parameters
    ----------
    data1: array [N, dim]
       Data points
    data2: array [N, dim]
       Data points

    Returns
    -------
    float
       <ln(f)> = (1/N) * sum_i=1^N ln(f_i)
    """
    if (np.shape(data1) != np.shape(data2)):
        print ('Both data arrays should be of form [N, dim]')
        return np.nan
    if (len(np.shape(data1)) == 1):
        dim = 1
        data1 = np.reshape(data1, (len(data), 1))
        data2 = np.reshape(data2, (len(data2), 1))
    elif (len(np.shape(data1)) == 2):
        dim = np.shape(data1)[1]
    else:
        print ('Data arrays should be of form [N, dim]')
        return np.nan
    
    N = np.shape(data1)[0]
    M = np.shape(data2)[0]
    
    tree = KDTree(data2, leafsize=10) # default leafsize=10
    dist, ind = tree.query(data1, k=2, workers=-1) # workers is number of threads. -1 means all threads
    idx = np.where(dist[:,1]>0)[0]
    N_zero_dist = N - len(idx) # Number of points with zero distance (typically, if not zero,  very small compared to N)
    
    if (N_zero_dist > 0):
        print (N_zero_dist,' points with zero D_NN neglected')
        N = N - N_zero_dist

    ln_D = np.log(dist[idx,1])
    ln_f = -np.log(M) - np.euler_gamma - np.log(V_d(dim)) - dim*ln_D
    return np.mean(ln_f)
#-----------------------------
def cross_entropy(data1, data2, J=1):
    """
    Estimate of the cross entropy
    H = - int f0 ln (f/J) d^dim x
    The factor J ensures that f/J is dimensionless and 
    that S is invariant for changes of variable x -> x', in which case J = |del x' / del x|
    For precise estimates, we also want all (x_1, x_2..., x_dim) to be order unit.
    So, if x and f are dimensionless and x is order unit, we can define J = 1.

    Parameters
    ----------
    data: array [N, 2*dim]
       Data points
    J: float number or array of size N
       J = |del x'/del x| is the jacobian of transf. from (x, y,...) -> (x', y', ...)
       If x' = x/sigma_x, y' = y/sigma_y... -> J = 1/|sigma_x*sigma_y...|

    Returns
    -------
    float
       entropy estimate -<ln(f/J)> = - (1/N)*sum_i=1^N ln(f_i/J_i)
    """
    
    avg_ln_J = np.mean(np.log(J))
    
    return -cross_avg_ln_f(data1, data2) + avg_ln_J
