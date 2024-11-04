""" tropygal: entropy estimators for galactic dynamics"""
from .base_funcs import V_d, entropy, cross_entropy, C_k, renyi_entropy, tsallis_entropy
from .DFs import (
    DF_Isochrone, g_Isochrone, Tr_Isochrone, gEL_Isochrone,
    DF_Hernquist, g_Hernquist,
    gEL_Spherical
    )
from .potentials import(
    find_Phi_eff_min, find_rper_rapo, Tr_Spherical,
    Phi_Isochrone, dPhi_dr_Isochrone,
    Phi_Hernquist, dPhi_dr_Hernquist
    )
# from . import _base_funcs
