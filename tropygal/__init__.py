""" tropygal: entropy estimators for galactic dynamics"""
from ._base_funcs import V_d, l_cube_sph, entropy, cross_entropy, C_k, renyi_entropy, tsallis_entropy
from .entropies import S_E, S_EL, S_renyi_E, S_renyi_EL, S_tsallis_E, S_tsallis_EL
from .DFs import DF_Isochrone, g_Isochrone, gEL_Isochrone, gEL_Spherical
# from . import _base_funcs
