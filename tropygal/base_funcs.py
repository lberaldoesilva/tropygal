import numpy as np
from scipy.special import gamma as Gamma
from scipy.special import psi # digamma function
#from scipy.special import betainc
from scipy.spatial import KDTree
from scipy import integrate

#pi2 = np.pi**2.
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
        # return 4.*np.pi/3.
        return 4.18879020478639
    if (dim==4):
        # return pi2/2.
        return 4.93480220054468
    if (dim==5):
        # return 8.*pi2/15.
        return 5.26378901391432
    if (dim==6):
        # return pi2*np.pi/6.
        return 5.16771278004997 # From here, V_d starts decreasing with d (the curse of dimensionality)
    
    return np.pi**(dim/2.)/Gamma(dim/2. + 1)
#-----------------------------
def entropy(data, mu=1, k=1, correct_bias=False, vol_correction='cube', l_cube_over_d=None, workers=-1):
    """
    Estimate of (Shannon/Jaynes) differential entropy
    S = - int f ln (f/mu) d^dim x
    The factor mu ensures that f/mu is dimensionless and 
    that S is invariant for changes of variable x -> x', in which case mu = |del x' / del x|
    For precise estimates, we also want all (x_1, x_2..., x_dim) to be order unit.
    So, if x and f are dimensionless and x is order unit, we can set mu = 1.

    S is estimated as (1/N) * sum_i=1^N ln(f_i),
    where f_i is the estimate of the DF f around point/particle/star i
    For NN (Nerest Neighbor) method:
    From e.g. Eq. (10) in Leonenko, Pronzato, Savani (2008):
    f_i = 1/[ (N-1) * exp(-psi(k)) * V_d * D^d ], where:
    N is sample size
    psi is the digamma function
    V_d is the volume of unitary hypersphere in d-dimensions
    D is the Euclidean distance to kth neighbor

    Parameters
    ----------
    data: array [N, dim]
       Data points
    mu: float number or array of size N
       mu = |del x'/del x| is the jacobian of transf. from (x, y,...) -> (x', y', ...)
       If x' = x/sigma_x, y' = y/sigma_y... -> mu = 1/|sigma_x*sigma_y...|
       It is also the density of states, e.g. mu = g(E), mu = g(E,L) or mu = (2pi)^3, in cases where the DF
       depends only on integrals, e.g. energy, or energy and angular momentum, or actions, respectively
    k: int value
       kth nearest neighbor
    correct_bias: Boolean
       If correct for bias due to boundary effects as proposed by Charzynska & Gambin 2015
    vol_correction: string
       strategy for correction:
       - 'cube': the support is assumed to be a paralelepiped, and the volume around each point is a cube
    l_cube_over_d: float
      side of cube around each point divided by D, the distance to the k-neighbor.
      A typically good choice is a cube inscribed in the sphere, i.e. side/D = 2/sqrt(dim). If None, this is set by default
    workers: int (default: -1, meaning using all CPUs available)
       Number of CPUs to be used to parallelize the seacrh for NNs.

    Returns
    -------
    float
       entropy estimate -<ln(f/mu)> = - (1/N)*sum_i=1^N ln(f_i/mu_i)
    """

    if (len(np.shape(data)) == 1):
        dim = 1
        data = np.reshape(data, (len(data), 1))
    elif (len(np.shape(data)) == 2):
        dim = np.shape(data)[1]
    else:
        raise ValueError('tropygal: Array with data should be of form [N, dim]')
    
    N = np.shape(data)[0]
    
    tree = KDTree(data, leafsize=10) # default leafsize=10
    
    dist, ind = tree.query(data, k=k+1, workers=workers) # workers is number of threads. -1 means all threads; k+1 because the first is the particle itself
    dist_kNN = dist[:,k]
    
    idx = np.where(dist_kNN > 0)[0]
    N_zero_dist = N - len(idx) # Number of points with zero distance (typically, if not zero, very small compared to N)
    
    if (N_zero_dist > 0):
        print ('tropygal:', N_zero_dist,' points with zero D_NN neglected')
        N = N - N_zero_dist

    ln_D = np.log(dist_kNN[idx])
#    ln_f = -np.log(N-1.) - np.euler_gamma - np.log(V_d(dim)) - dim*ln_D
#    ln_f = -psi(N) + psi(k) - np.log(V_d(dim)) - dim*ln_D
    avg_ln_f = np.mean(-np.log(N-1.) + psi(k) - np.log(V_d(dim)) - dim*ln_D)
    avg_ln_mu = np.mean(np.log(mu))

    if (correct_bias==True):
        data = data[idx] # consider only data points with positive distance to NN
        dist_kNN = dist_kNN[idx]
        # The fraction of the volume around particle i inside the domain:
        log_frac_vol = np.zeros(len(idx))
        
        if (vol_correction=='cube'):
            if l_cube_over_d is None:
                l_cube_over_d = 2/np.sqrt(dim)
            l_cube = l_cube_over_d * dist_kNN
            for j in range(dim):
                xmax = max(data[:, j])
                xmin = min(data[:, j])

                dx_max_over_l_cube = (xmax - data[:, j]) / l_cube # we only need to correct if this is < 1/2, i.e. if the volume of the cube goes beyond support
                dx_min_over_l_cube = (data[:, j] - xmin) / l_cube # we only need to correct if this is < 1/2

                needs_correc = dx_max_over_l_cube < 0.5
                log_frac_vol[needs_correc] = log_frac_vol[needs_correc] + np.log(0.5 + dx_max_over_l_cube[needs_correc])

                needs_correc = dx_min_over_l_cube < 0.5
                log_frac_vol[needs_correc] = log_frac_vol[needs_correc] + np.log(0.5 + dx_min_over_l_cube[needs_correc])
            correction = np.mean(log_frac_vol)
        else:
            correction = 0
            # raise ValueError("tropygal: vol_correction is the volume around each point assumed for the bias correction. Current possible values: 'cube', 'actions' or 'sph'")
            raise ValueError("tropygal: vol_correction is the volume around each point assumed for the bias correction. Current possible values: 'cube'.")
        return -avg_ln_f + avg_ln_mu + correction
    else:
        return -avg_ln_f + avg_ln_mu
#-----------------------------
def cross_entropy(data1, data2, mu=1, k=1, correct_bias=False, vol_correction='cube', l_cube_over_d=None, workers=-1):
    """
    Estimate of the cross entropy
    H = - int f0 ln (f/mu) d^dim x
    The factor mu ensures that f/mu is dimensionless and 
    that H is invariant for changes of variable x -> x', in which case mu = |del x' / del x|
    For precise estimates, we also want all (x_1, x_2..., x_dim) to be order unit.
    So, if x and f are dimensionless and x is order unit, we can define mu = 1.

    H is estimated as (1/N) * sum_i=1^N ln(f_i),
    where f_i is the estimate of the DF f based on the dist. of point i in sample 1 to its kth neighbor in sample 2
    For NN (Nerest Neighbor) method:
    From e.g. Eq. (11) in Leonenko, Pronzato, Savani (2008):
    f_i = 1/[ M * exp(-psi(k)) * V_d * D^d ], where:
    N is size of sample 1
    M is size of sample 2
    psi is the digamma function
    V_d is the volume of unitary hypersphere in d-dimensions
    D is the Euclidean distance of particle i in sample 1 to its kth neighbor in sample 2

    Parameters
    ----------
    data1: array [N, dim]
       Data points of sample 1
    data2: array [M, dim]
       Data points of sample 2
    mu: float number or array of size N
       mu = |del x'/del x| is the jacobian of transf. from (x, y,...) -> (x', y', ...)
       If x' = x/sigma_x, y' = y/sigma_y... -> mu = 1/|sigma_x*sigma_y...|
    k: int value
       kth nearest neighbor
    correct_bias: Boolean
       If correct for bias due to boundary effects as proposed by Charzynska & Gambin 2015
    vol_correction: string
       strategy for correction:
       - 'cube': the support is assumed to be a paralelepiped, and the volume around each point is a cube
    l_cube_over_d: float
      Side of cube around each point divided by D, the distance to the k-neighbor.
      A typically good choice is a cube inscribed in the sphere, i.e. side/D = 2/sqrt(dim). If None, this is set by default
    workers: int (default: -1, meaning using all CPUs available)
       Number of CPUs to be used to parallelize the seacrh for NNs.

    Returns
    -------
    float
       cross entropy estimate -<ln(f/mu)> = - (1/N)*sum_i=1^N ln(f_i/mu_i)
    """

    if (len(np.shape(data1)) != len(np.shape(data2))):
        raise ValueError("tropygal: Data arrays should be of form [N, dim] and [M, dim], or 1D arrays")
        return np.nan
    if (len(np.shape(data1)) == 1):
        dim = 1
        data1 = np.reshape(data1, (len(data1), 1))
        data2 = np.reshape(data2, (len(data2), 1))
    elif (len(np.shape(data1)) == 2):
        if (np.shape(data1)[1] != np.shape(data2)[1]):
            raise ValueError("tropygal: Data arrays should be of form [N, dim] and [M, dim], or 1D arrays")
        dim = np.shape(data1)[1]
    else:
        raise ValueError("tropygal: Data arrays should be of form [N, dim] and [M, dim], or 1D arrays")
        return np.nan
    
    N = np.shape(data1)[0]
    M = np.shape(data2)[0]
    
    tree = KDTree(data2, leafsize=10) # default leafsize=10
    dist, ind = tree.query(data1, k=k+1, workers=workers) # workers is number of threads. -1 means all threads
    #dist, ind = tree.query(data1, k=k, workers=-1) # workers is number of threads. -1 means all threads
    # if (k==1):
    #     dist_kNN = dist
    # else:
    #     dist_kNN = dist[:,k-1] # changed here too in respect to entropy
    dist_kNN = dist[:,k]
    
    idx = np.where(dist_kNN > 0)[0]
    N_zero_dist = N - len(idx) # Number of points with zero distance (typically, if not zero,  very small compared to N)
    
    if (N_zero_dist > 0):
        print ("tropygal:", N_zero_dist,' points with zero D_NN neglected')
        N = N - N_zero_dist

    ln_D = np.log(dist_kNN[idx])
    avg_ln_f = np.mean(-np.log(M) + psi(k) - np.log(V_d(dim)) - dim*ln_D)
    avg_ln_mu = np.mean(np.log(mu))

    if (correct_bias==True):
        data1 = data1[idx] # consider only data points with positive distance to NN
        dist_kNN = dist_kNN[idx]
        # The fraction of the volume around particle i inside the domain:
        log_frac_vol = np.zeros(len(idx))

        if (vol_correction=='cube'):
            if l_cube_over_d is None:
                l_cube_over_d = 2/np.sqrt(dim)
            l_cube = l_cube_over_d * dist_kNN
            for j in range(dim):
                xmax = max(data2[:, j])
                xmin = min(data2[:, j])

                y = np.minimum(np.maximum(data1[:, j], xmin), xmax)

                dx_max_over_l_cube = (xmax - y) / l_cube # we only need to correct if this is < 1, i.e. if the volume of the ball goes beyond support
                dx_min_over_l_cube = (y - xmin) / l_cube # we only need to correct if this is < 1

                needs_correc = dx_max_over_l_cube < 0.5
                log_frac_vol[needs_correc] = log_frac_vol[needs_correc] + np.log(0.5 + dx_max_over_l_cube[needs_correc])

                needs_correc = dx_min_over_l_cube < 0.5
                log_frac_vol[needs_correc] = log_frac_vol[needs_correc] + np.log(0.5 + dx_min_over_l_cube[needs_correc])
            correction = np.mean(log_frac_vol)
        else:
            correction = 0
            # raise ValueError("tropygal: vol_correction is the volume around each point assumed for the bias correction. Current possible values: 'cube', 'actions' or 'sph'")
            raise ValueError("tropygal: vol_correction is the volume around each point assumed for the bias correction. Current possible values: 'cube'.")
        return -avg_ln_f + avg_ln_mu + correction
    else:
        return -avg_ln_f + avg_ln_mu
#-----------------------------
def C_k(q, k=1):
    """
    For Renyi entropy.
    Check Eq. (7) in Leonenko, Pronzato, Savani (2008)

    Parameters
    ----------
    q: float number
       q index in Rényi entropy (must be != 1)
    k: int value
       kth nearest neighbor

    Returns
    -------
    float
       C_k
    """
    return (Gamma(k)/Gamma(k+1.-q))**(1./(1.-q))
#-----------------------------
def renyi_entropy(data, mu=1, q=2, k=1):
    """
    Estimate of Rényi entropy
    S_q = [1/(1 - q)] ln int f (f/mu)^(q-1) d^dim x, for q != 1
    The factor mu ensures that f/mu is dimensionless and 
    that S_q is invariant for changes of variable x -> x', in which case mu = |del x' / del x|
    For precise estimates, we also want all (x_1, x_2..., x_dim) to be order unit.
    So, if x and f are dimensionless and x is order unit, we can set mu = 1.

    S_q is estimated as [1/(1-q)] ln (1/N) * sum_i=1^N (f_i/mu_i)^(q-1),
    where f_i is the estimate of the DF f around point/particle/star i
    For NN (Nerest Neighbor) method:
    From e.g. Eq. (7) in Leonenko, Pronzato, Savani (2008):
    f_i = 1/[ (N-1) * C_k * V_d * D^dim ], where
    C_k = [Gamma(k)/Gamma(k+1-q)]^[1/(1-q)]

    Parameters
    ----------
    data: array [N, dim]
       Data points
    mu: float number or array of size N
       mu = |del x'/del x| is the jacobian of transf. from (x, y,...) -> (x', y', ...)
       If x' = x/sigma_x, y' = y/sigma_y... -> mu = 1/|sigma_x*sigma_y...|
    q: float value
       q-parameter of the entropy; needs to be q != 1
    k: int value
       kth nearest neighbor

    Returns
    -------
    float
       entropy estimate [1/(1 - q)] ln <(f/mu)^(q-1)>
    """

    if (len(np.shape(data)) == 1):
        dim = 1
        data = np.reshape(data, (len(data), 1))
    elif (len(np.shape(data)) == 2):
        dim = np.shape(data)[1]
    else:
        raise ValueError("tropygal: Array with data should be of form [N, dim].")
        return np.nan

    if (q==1):
        raise ValueError ('tropygal: For the Rényi entropy, q needs to be != 1')
        return np.nan
    
    N = np.shape(data)[0]
    
    tree = KDTree(data, leafsize=10) # default leafsize=10
    dist, ind = tree.query(data, k=k+1, workers=-1) # workers is number of threads. -1 means all threads
    dist_kNN = dist[:,k]
    
    idx = np.where(dist_kNN > 0)[0]
    N_zero_dist = N - len(idx) # Number of points with zero distance (typically, if not zero,  very small compared to N)
    
    if (N_zero_dist > 0):
        print ('tropygal:', N_zero_dist,' points with zero D_NN neglected')
        N = N - N_zero_dist
        
    D = dist_kNN[idx]
    f = 1./( (N-1) * C_k(q, k) * V_d(dim) * D**dim )

    return (1./(1-q))*np.log(np.mean((f/mu)**(q-1.)))
#-----------------------------
def tsallis_entropy(data, mu=1, q=2, k=1):
    """
    Estimate of Tsallis entropy
    S_q = [1/(q - 1)][ 1 - int f (f/mu)^(q-1) d^dim x ], for q != 1
    The factor mu ensures that f/mu is dimensionless and 
    that S is invariant for changes of variable x -> x', in which case mu = |del x' / del x|
    For precise estimates, we also want all (x_1, x_2..., x_dim) to be order unit.
    So, if x and f are dimensionless and x is order unit, we can set mu = 1.

    S_q is estimated as [1/(q - 1)] [ 1 - (1/N) * sum_i=1^N (f_i/mu_i)^(q-1) ],
    where f_i is the estimate of the DF f around point/particle/star i
    For NN (Nerest Neighbor) method:
    From e.g. Eq. (7) in Leonenko, Pronzato, Savani (2008):
    f_i = 1/[ (N-1) * C_k * V_d * D^dim ],
    where C_k = [Gamma(k)/Gamma(k+1-q)]^[1/(1-q)]

    Parameters
    ----------
    data: array [N, dim]
       Data points
    mu: float number or array of size N
       mu = |del x'/del x| is the jacobian of transf. from (x, y,...) -> (x', y', ...)
       If x' = x/sigma_x, y' = y/sigma_y... -> mu = 1/|sigma_x*sigma_y...|
    q: int value
       q-parameter of the entropy; needs to be q != 1
    k: int value
       kth nearest neighbor

    Returns
    -------
    float
       entropy estimate [1/(q - 1)] [ 1 -  <f^(q-1)> ]
    """

    if (len(np.shape(data)) == 1):
        dim = 1
        data = np.reshape(data, (len(data), 1))
    elif (len(np.shape(data)) == 2):
        dim = np.shape(data)[1]
    else:
        raise ValueError("tropygal: Array with data should be of form [N, dim].")
        return np.nan

    if (q==1):
        raise ValueError ('tropygal: For the Tsallis entropy, q needs to be != 1.')
        return np.nan
    
    N = np.shape(data)[0]
    
    tree = KDTree(data, leafsize=10) # default leafsize=10
    dist, ind = tree.query(data, k=k+1, workers=-1) # workers is number of threads. -1 means all threads
    dist_kNN = dist[:,k]
    
    idx = np.where(dist_kNN > 0)[0]
    N_zero_dist = N - len(idx) # Number of points with zero distance (typically, if not zero,  very small compared to N)
    
    if (N_zero_dist > 0):
        print ('tropygal:', N_zero_dist,' points with zero D_NN neglected')
        N = N - N_zero_dist
        
    D = dist_kNN[idx]
    f = 1./( (N-1) * C_k(q, k) * V_d(dim) * D**dim )

    return (1./(q - 1))*( 1. - np.mean((f/mu)**(q-1.)))
