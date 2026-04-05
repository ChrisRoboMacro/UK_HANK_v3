"""
General equilibrium assembly for UK-HANK model.

Phase 1: Minimal one-asset closed-economy HANK.
Households save in liquid deposits, firms produce with capital and labour,
government taxes proportionally and sets rates via Taylor rule.
"""

import numpy as np
from sequence_jacobian import het, simple, create_model, steady_state
from sequence_jacobian.grids import asset_grid

from uk_hank.households.income import rouwenhorst
from uk_hank.households.backward import household_backward_step


# ============================================================================
# Household block using sequence-jacobian's @het decorator
# ============================================================================

def make_household_block(n_e=11, n_a=200, a_min=-2.0, a_max=200.0,
                         rho_e=0.966, sigma_e=0.5):
    """
    Create the household heterogeneous agent block.

    This wraps the EGM backward step into SSJ's @het framework.
    """
    # Create grids
    e_grid, Pi, pi_e = rouwenhorst(n_e, rho_e, sigma_e)
    a_grid = asset_grid(a_min, a_max, n_a)

    @het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init)
    def household(Va_p, Pi_p, a_grid, e_grid, z, r, beta, sigma):
        Va, a, c = household_backward_step(Va_p, Pi_p, a_grid, e_grid, z, r, beta, sigma)
        return Va, a, c

    # Attach grids as defaults
    household_block = household.add_hetinputs([
        make_hetinput(e_grid, a_grid, Pi)
    ])

    return household_block, e_grid, a_grid, Pi


def household_init(a_grid, e_grid, z, r, sigma):
    """Initial guess for Va (marginal value of assets)."""
    n_e = len(e_grid)
    n_a = len(a_grid)
    # Assume consumption = income + r*a as initial guess
    coh = np.maximum((1 + r) * a_grid[np.newaxis, :] + z * e_grid[:, np.newaxis], 1e-8)
    Va = (1 + r) * coh ** (-sigma)
    return Va


def make_hetinput(e_grid, a_grid, Pi):
    """Create hetinput function that provides grids to the household block."""
    @simple
    def hetinput(z_agg, tau, w, N, div):
        """
        After-tax income per efficiency unit.
        Phase 1: z = (1-tau) * w * N  (simple proportional tax on wage income)
        """
        z = (1 - tau) * w
        return z

    return hetinput


# ============================================================================
# Phase 1 model assembly
# ============================================================================

def build_phase1_model():
    """
    Build the Phase 1 minimal HANK model.

    Blocks:
    1. Household (HetBlock): consumption-savings with EGM
    2. Firms: Rotemberg pricing, marginal cost
    3. Capital: Tobin's Q, investment
    4. Government: Taylor rule, simple fiscal
    5. Market clearing: goods, assets, Fisher equation

    This is a proof-of-concept to verify the SSJ toolchain works.
    """
    print("Phase 1 model assembly — coming in next iteration")
    print("Components built:")
    print("  - households/income.py: Rouwenhorst discretization")
    print("  - households/backward.py: EGM backward step")
    print("  - firms/intermediate.py: Rotemberg NKPC")
    print("  - firms/capital.py: Tobin's Q")
    print("  - government/central_bank.py: Taylor rule")
    print("  - government/fiscal.py: Simple proportional tax")
    print("  - model/market_clearing.py: Goods + asset clearing")
    print()
    print("Next step: wire these into create_model() DAG")


if __name__ == '__main__':
    build_phase1_model()
