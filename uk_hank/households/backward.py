"""
Household backward step for UK-HANK model.

Phase 1: One-asset consumption-savings with EGM, following SSJ's @het pattern.
Uses sequence_jacobian.interpolate for grid interpolation.
"""

import numpy as np
from numba import njit
from sequence_jacobian.blocks.het_block import het
from sequence_jacobian import interpolate, grids


def hh_init(a_grid, e_grid, z, r, sigma):
    """Initial guess for marginal value of assets."""
    coh = (1 + r) * a_grid[np.newaxis, :] + z * e_grid[:, np.newaxis]
    Va = (1 + r) * np.maximum(coh, 1e-8) ** (-sigma)
    return Va


@het(exogenous='Pi', policy='a', backward='Va', backward_init=hh_init)
def hh(Va_p, a_grid, e_grid, z, r, beta, sigma):
    """
    Single backward step of the household problem via EGM.

    Households solve:
        max E sum_t beta^t u(c_t)
        s.t. a' + c = (1+r)*a + z*e
             a' >= a_min

    where u(c) = c^{1-sigma}/(1-sigma) and z is after-tax wage per efficiency unit.

    Parameters
    ----------
    Va_p : array (n_e, n_a)
        Expected marginal value of assets next period (post-multiplication by Pi)
    a_grid : array (n_a,)
        Asset grid
    e_grid : array (n_e,)
        Productivity grid (levels, mean 1)
    z : float
        Aggregate after-tax income per efficiency unit
    r : float
        Real interest rate
    beta : float
        Discount factor
    sigma : float
        CRRA coefficient

    Returns
    -------
    Va, a, c : arrays (n_e, n_a)
    """
    # Step 1: Euler equation on the endogenous grid
    # u'(c) = beta*(1+r)*E[Va'] => c = (beta*(1+r)*Va_p)^(-1/sigma)
    uc_nextgrid = beta * Va_p
    c_nextgrid = uc_nextgrid ** (-1 / sigma)

    # Step 2: Endogenous cash-on-hand grid
    # coh_endo = c + a'  (where a' is a_grid)
    coh_endo = c_nextgrid + a_grid[np.newaxis, :]

    # Step 3: Actual cash-on-hand on exogenous grid
    # coh = (1+r)*a + z*e
    coh = (1 + r) * a_grid[np.newaxis, :] + z * e_grid[:, np.newaxis]

    # Step 4: Interpolate consumption onto exogenous grid
    c = interpolate.interpolate_y(coh_endo, coh, c_nextgrid)

    # Step 5: Savings policy and borrowing constraint
    a = coh - c
    iconst = np.nonzero(a < a_grid[0])
    a[iconst] = a_grid[0]
    c[iconst] = coh[iconst] - a_grid[0]

    # Step 6: Envelope condition
    Va = (1 + r) * c ** (-sigma)

    return Va, a, c
