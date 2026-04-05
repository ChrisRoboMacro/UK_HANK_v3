"""
DC-EGM Household Block for UK-HANK Model
==========================================

Two-stage discrete-continuous choice via SSJ's StageBlock:
  Stage 1 (Exogenous): Income Markov process e → e'
  Stage 2 (LogitChoice): Discrete tenure choice h' ∈ {rent, own}
  Stage 3 (Continuous1D): EGM consumption-savings with Cobb-Douglas utility

State space: (n_e, n_h, n_a) where:
  n_e = 21 (7 persistent × 3 transitory income states)
  n_h = 2  (rent=0, own=1)
  n_a = 100 (asset grid)

Key equations from the paper:
  u(c,h) = (c^{1-φ_H} · x(h)^{φ_H})^{1-σ} / (1-σ)
  x(h) = H(h) · (1 + ω_oo · 1{owner})
  Budget: a' + c + c_H(h) = (1+r)·a + z(e)
  Borrowing: a' ≥ a_lower(h)
"""

import numpy as np
from sequence_jacobian import grids, interpolate
from sequence_jacobian.blocks.stage_block import StageBlock
from sequence_jacobian.blocks.support.stages import Continuous1D, LogitChoice, ExogenousMaker
from sequence_jacobian.blocks.support.law_of_motion import ShockedPolicyLottery1D


# ============================================================================
# Behavioral Continuous1D: cognitive discounting in the backward step
# ============================================================================

class BehavioralContinuous1D(Continuous1D):
    """Continuous1D stage with cognitive discounting (Gabaix 2020, Section 2.2).

    Cognitive discounting scales perturbations to AGGREGATE variables (r, w, etc.)
    by M_CD < 1, so households perceive future aggregate deviations as smaller.
    Backward variables (Va, V) from other stages are NOT discounted — these
    represent the household's own value function, not aggregate perceptions.
    """
    def __init__(self, M_CD=0.85, **kwargs):
        super().__init__(**kwargs)
        self.M_CD = M_CD

    def backward_step_shock(self, ss, shocks, precomputed):
        """Apply cognitive discounting to aggregate variable perturbations only."""
        backward_vars = set(self.backward_outputs)
        discounted_shocks = {}
        for k, v in shocks.items():
            if k in backward_vars:
                discounted_shocks[k] = v  # don't discount backward variables
            else:
                discounted_shocks[k] = v * self.M_CD  # discount aggregate shocks
        return super().backward_step_shock(ss, discounted_shocks, precomputed)


# ============================================================================
# EGM backward step (Continuous1D's f function)
# ============================================================================

def hh_egm(Va, V, a_grid, e_grid, h_grid, z_grid, r, beta, sigma, phi_H, x_h, c_H, a_lower_h):
    """
    EGM with Cobb-Douglas utility and tenure-specific budget constraints.

    State space: (n_e, n_h, n_a).
    Va, V are continuation values (post-expectations from Exogenous + LogitChoice).

    Returns Va, V, a, c — all shape (n_e, n_h, n_a).
    """
    n_e = len(e_grid)
    n_h = len(h_grid)
    n_a = len(a_grid)

    # Effective CRRA on consumption with Cobb-Douglas housing
    # u(c,h) = (c^{1-φ} · x^φ)^{1-σ} / (1-σ)
    # u_c = (1-φ) · c^{-σ_c} · x^{φ(1-σ)}
    # where σ_c = φ + σ(1-φ) = 1 - (1-σ)(1-φ)
    sigma_c = phi_H + sigma * (1 - phi_H)

    # Housing service term: x_h shape (n_h,) → broadcast to (1, n_h, 1)
    xh = x_h[np.newaxis, :, np.newaxis]
    xh_factor = xh ** (phi_H * (1 - sigma))  # enters marginal utility

    # Step 1: Euler equation on endogenous grid
    # u_c(c) = beta · Va  →  c = ((beta·Va) / ((1-φ)·xh_factor))^{-1/σ_c}
    uc_next = beta * Va
    c_next = (uc_next / ((1 - phi_H) * xh_factor)) ** (-1.0 / sigma_c)

    # Step 2: Endogenous cash-on-hand
    coh_endo = c_next + a_grid[np.newaxis, np.newaxis, :]  # (n_e, n_h, n_a)

    # Step 3: Actual cash-on-hand
    # coh = (1+r)·a + z(e) - c_H(h)
    coh = ((1 + r) * a_grid[np.newaxis, np.newaxis, :]
           + z_grid[:, np.newaxis, np.newaxis]
           - c_H[np.newaxis, :, np.newaxis])  # (n_e, n_h, n_a)

    # Step 4: Interpolate consumption onto exogenous grid
    # Reshape to 2D for vectorized interpolation, then reshape back
    c = interpolate.interpolate_y(
        coh_endo.reshape(-1, n_a), coh.reshape(-1, n_a), c_next.reshape(-1, n_a)
    ).reshape(n_e, n_h, n_a)

    # Step 5: Savings policy and borrowing constraint
    a = coh - c
    # a_lower_h: scalar per tenure state → (1, n_h, 1)
    a_min = np.maximum(a_lower_h[np.newaxis, :, np.newaxis], a_grid[0])
    iconst = a < a_min
    a = np.where(iconst, a_min, a)
    c = np.where(iconst, coh - a_min, c)
    c = np.maximum(c, 1e-10)

    # Step 6: Envelope condition — marginal value of assets
    Va = (1 + r) * (1 - phi_H) * c ** (-sigma_c) * xh_factor

    # Step 7: Value function level (needed for LogitChoice)
    # V(e,h,a) = u(c,h) + beta · V_continuation
    if abs(sigma - 1.0) < 1e-10:
        u_flow = np.log(np.maximum(c, 1e-10) ** (1 - phi_H) * xh)
    else:
        u_flow = (np.maximum(c, 1e-10) ** (1 - phi_H) * xh) ** (1 - sigma) / (1 - sigma)

    # Continuation value at optimal savings a'
    # V (input) is on a_grid. Interpolate to evaluate at a' (savings policy).
    V_cont = np.empty((n_e, n_h, n_a))
    for ih in range(n_h):
        for ie in range(n_e):
            V_cont[ie, ih, :] = np.interp(
                np.clip(a[ie, ih, :], a_grid[0], a_grid[-1]),
                a_grid, V[ie, ih, :])

    V = u_flow + beta * V_cont

    return Va, V, a, c


# ============================================================================
# Housing flow utility (LogitChoice's f function)
# ============================================================================

def housing_flow(h_grid, eta_move, e_grid, a_grid, F_cost):
    """
    Flow utility from tenure choice: moving costs + transaction costs.

    5-state: rent(0), mortgagor-flat(1), outright-flat(2),
             mortgagor-house(3), outright-house(4).
    Moving cost: eta_move for any tenure change.
    Transaction costs: F_cost for buying/selling property.
    Mortgage transitions: mortgagor <-> outright (refinance/payoff) are cheap.

    Returns shape (n_h_choice, 1, n_h_current, 1) for broadcasting with V.
    """
    n_h = len(h_grid)
    flow = np.zeros((n_h, 1, n_h, 1))
    # Groups: rent={0}, flat={1,2}, house={3,4}
    def is_owner(h): return h >= 1
    def is_flat(h): return h in (1, 2)
    def is_house(h): return h in (3, 4)
    def is_mortgagor(h): return h in (1, 3)
    def same_property_type(h1, h2):
        return (is_flat(h1) and is_flat(h2)) or (is_house(h1) and is_house(h2))

    for h_new in range(n_h):
        for h_old in range(n_h):
            if h_new != h_old:
                if same_property_type(h_new, h_old):
                    # Same property, different mortgage status (refinance/payoff)
                    cost = -eta_move * 0.1  # very low cost
                elif not is_owner(h_old) and is_owner(h_new):
                    # Rent -> buy: moving cost + buying transaction
                    cost = -eta_move - F_cost * 0.5
                elif is_owner(h_old) and not is_owner(h_new):
                    # Own -> rent: moving cost + selling transaction
                    cost = -eta_move - F_cost * 0.5
                elif is_owner(h_old) and is_owner(h_new):
                    # Own flat -> own house or vice versa: move + buy + sell
                    cost = -eta_move - F_cost
                else:
                    cost = -eta_move
                flow[h_new, 0, h_old, 0] = cost
    return flow


# ============================================================================
# Backward initialization
# ============================================================================

def hh_dc_init(a_grid, e_grid, h_grid, z_grid, r, sigma, phi_H, x_h):
    """
    Initialize Va and V for the 3D state space (n_e, n_h, n_a).

    SSJ's ExtendedFunction infers output names from the return statement:
    the variables MUST be named Va and V.
    """
    sigma_c = phi_H + sigma * (1 - phi_H)
    xh = x_h[np.newaxis, :, np.newaxis]
    xh_factor = xh ** (phi_H * (1 - sigma))

    coh = ((1 + r) * a_grid[np.newaxis, np.newaxis, :]
           + z_grid[:, np.newaxis, np.newaxis])
    coh = np.maximum(coh, 1e-8)

    Va = (1 + r) * (1 - phi_H) * coh ** (-sigma_c) * xh_factor

    if abs(sigma - 1.0) < 1e-10:
        V = np.log(coh ** (1 - phi_H) * xh) * 50  # scale up for level
    else:
        V = (coh ** (1 - phi_H) * xh) ** (1 - sigma) / (1 - sigma) * 50

    return Va, V


# ============================================================================
# Hetinput functions
# ============================================================================

def make_grids_dc(nE, nA, amax, rho_e, sigma_e, phi_H, omega_oo,
                  P_R, P_H, delta_H, omega_bor, kappa_H, H_F, H_H, r,
                  nT=3, rho_T=0.495, sigma_T=0.464):
    """Create grids for DC-EGM household (v2).

    5-state tenure: 0=renter, 1=mortgagor-flat, 2=outright-flat,
                    3=mortgagor-house, 4=outright-house
    21-state income: 7 persistent x 3 transitory (paper Table 3, ASHE)
    Variable-rate mortgage for mortgagors (r in c_H)
    """
    # --- Income grid: persistent x transitory ---
    e_P, pi_P, Pi_P = grids.markov_rouwenhorst(rho=rho_e, sigma=sigma_e, N=nE)
    e_T, pi_T, Pi_T = grids.markov_rouwenhorst(rho=rho_T, sigma=sigma_T, N=nT)
    # Combined: e = e_P * e_T (multiplicative)
    e_grid = np.outer(e_P, e_T).ravel()
    pi_e = np.outer(pi_P, pi_T).ravel()
    pi_e = pi_e / pi_e.sum()
    Pi = np.kron(Pi_P, Pi_T)

    # --- Asset grid ---
    a_min = -kappa_H * P_H * H_H
    a_grid = grids.agrid(amax=amax, n=nA, amin=max(a_min, -10.0))

    # --- 5-state tenure ---
    h_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Housing services
    x_h = np.array([
        H_F,                        # 0: renter
        H_F * (1 + omega_oo),       # 1: mortgagor-flat
        H_F * (1 + omega_oo),       # 2: outright-flat
        H_H * (1 + omega_oo),       # 3: mortgagor-house
        H_H * (1 + omega_oo),       # 4: outright-house
    ])

    # Housing costs: physical depreciation + mortgage interest for mortgagors
    # Mortgagor: delta_H*H (physical maintenance) + (r+omega_bor)*P_H*H (interest on house value)
    # Outright: delta_H*H only (no mortgage)
    # Renter: P_R*H_F (rental payment)
    c_H = np.array([
        P_R * H_F,                                          # 0: renter (0.15)
        delta_H * H_F + (r + omega_bor) * P_H * H_F,        # 1: mortgagor-flat (0.066)
        delta_H * H_F,                                       # 2: outright-flat (0.037)
        delta_H * H_H + (r + omega_bor) * P_H * H_H,        # 3: mortgagor-house (0.100)
        delta_H * H_H,                                       # 4: outright-house (0.056)
    ])

    # Borrowing limits
    a_lower_h = np.array([
        0.0,                                        # renter
        max(-kappa_H * P_H * H_F, a_grid[0]),       # mortgagor-flat
        0.0,                                        # outright-flat
        max(-kappa_H * P_H * H_H, a_grid[0]),       # mortgagor-house
        0.0,                                        # outright-house
    ])

    return e_grid, pi_e, Pi, a_grid, h_grid, x_h, c_H, a_lower_h


def income_dc(w, e_grid):
    """After-tax wage income by productivity."""
    z_grid = w * e_grid
    return z_grid


def transfers_dc(pi_e, div, Tax, e_grid):
    """Net transfers proportional to productivity."""
    n_e = len(e_grid)
    div_rule = e_grid
    tax_rule = e_grid
    d = div / np.sum(pi_e * div_rule) * div_rule
    t = Tax / np.sum(pi_e * tax_rule) * tax_rule
    z_transfer = d - t
    # Add transfers to z_grid (will be combined in income_dc or directly)
    return z_transfer


# ============================================================================
# Hetoutput: marginal utility of consumption (needed by wage Phillips Curve)
# ============================================================================

def marginal_utility_dc(c, sigma):
    """Marginal utility of consumption u'(c) = c^{-sigma}.

    This is a hetoutput on the Continuous1D stage. SSJ will aggregate
    it against the distribution: UCE = integral(c^{-sigma} dD).
    The output name 'uce' gets auto-capitalized to 'UCE' by StageBlock.
    """
    uce = c ** (-sigma)
    return uce


# ============================================================================
# StageBlock assembly
# ============================================================================

def build_dc_household(M_CD=0.85):
    """
    Build the DC-EGM household StageBlock.

    Three stages in forward/chronological order:
    1. Exogenous: income Markov process (e → e')
    2. LogitChoice: discrete tenure choice (h → h')
    3. BehavioralContinuous1D: EGM with cognitive discounting (M_CD=0.85)

    Backward iteration runs in reverse: Continuous1D → LogitChoice → Exogenous.
    Cognitive discounting applied inside the consumption stage's backward_step_shock.
    """
    stages = [
        ExogenousMaker('Pi', index=0, name='income'),
        LogitChoice(value='V', backward='Va', index=1,
                    taste_shock_scale='alpha_H', f=housing_flow, name='tenure'),
        BehavioralContinuous1D(M_CD=M_CD,
                               backward=['Va', 'V'], policy='a', f=hh_egm,
                               name='consumption',
                               hetoutputs=[marginal_utility_dc]),
    ]

    household = StageBlock(
        stages=stages,
        backward_init=hh_dc_init,
        hetinputs=[make_grids_dc, income_dc, transfers_dc],
        name='hh_dc'
    )

    return household


# ============================================================================
# Testing
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("DC-EGM Household Block Test")
    print("=" * 60)

    # Test 0: Build and solve steady state
    household = build_dc_household()
    print(f"\nStageBlock created: {household}")
    print(f"  inputs: {household.inputs}")
    print(f"  outputs: {household.outputs}")

    # Calibration
    cal = {
        'r': 0.005, 'beta': 0.99, 'sigma': 1.0,
        'w': 0.8, 'div': 0.1, 'Tax': 0.1,
        'phi_H': 0.24, 'omega_oo': 1.06,
        'eta_move': 0.32, 'alpha_H': 1.3, 'F_cost': 0.1,
        'P_R': 0.15, 'P_H': 3.6, 'delta_H': 0.037,
        'omega_bor': 0.00375, 'kappa_H': 0.95,
        'H_F': 1.0, 'H_H': 1.5,
        'nE': 7, 'nA': 50, 'amax': 50,
        'rho_e': 0.966, 'sigma_e': 0.5,
    }

    print("\nSolving steady state...")
    try:
        ss = household.steady_state(cal)
        print(f"  A (mean assets): {ss['A']:.4f}")
        print(f"  C (mean consumption): {ss['C']:.4f}")

        # Check tenure distribution from internals
        try:
            internals = ss.internals['hh_dc']
            # Look for distribution in stage internals
            for key in internals:
                if isinstance(internals[key], dict):
                    for k2 in internals[key]:
                        if k2 == 'D':
                            D = internals[key][k2]
                            renter_mass = D[:, 0, :].sum()
                            owner_mass = D[:, 1, :].sum()
                            total = renter_mass + owner_mass
                            print(f"\n  Renter share: {renter_mass/total*100:.1f}% (target: 33%)")
                            print(f"  Owner share: {owner_mass/total*100:.1f}% (target: 67%)")
                            break
                elif hasattr(internals[key], 'shape') and len(internals[key].shape) == 3:
                    print(f"\n  Internal '{key}': shape={internals[key].shape}")
        except Exception as e:
            print(f"\n  (Tenure diagnostics: {e})")
            # Try listing what's available
            if 'hh_dc' in ss.internals:
                print(f"  Available internals: {list(ss.internals['hh_dc'].keys())}")

        print("\n  DC-EGM StageBlock: WORKING!")

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
