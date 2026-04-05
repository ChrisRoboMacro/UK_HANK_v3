"""
Behavioral expectations for UK-HANK model.
==========================================

Implements Section 2.2 of the paper:

1. Cognitive discounting (Gabaix 2020):
   E_t^CD[X_{t+1}] = X_bar + M^CD * (E_t[X_{t+1}] - X_bar)

   Effect on Jacobians: multiply the (s,t) element by M^CD^(t-s)
   i.e., agents discount future deviations from SS geometrically.

2. Sticky expectations (Carroll et al. 2020):
   With prob (1-gamma), agent updates to current info.
   With prob gamma, agent keeps stale info from last period.

   Effect on Jacobians: J_sticky[s] = sum_{k=0}^{inf} (1-gamma)*gamma^k * J[s-k]
   i.e., geometric moving average of the rational Jacobian.

Key parameters (Table 5):
  M^CD = 0.85          (cognitive discounting factor)
  1-gamma_hh = 0.13    (household info update probability)
  1-gamma_f = 0.16     (firm info update probability)

Reference: Auclert et al. (2020) show these can be applied as
post-processing transforms on the sequence-space Jacobians.
"""

import numpy as np
from copy import deepcopy


def apply_cognitive_discounting(J, M_CD=0.85):
    """
    Apply cognitive discounting to a Jacobian matrix.

    J[s, t] represents the response at time s to a shock at time t.
    Cognitive discounting scales the response to future shocks:
    J_CD[s, t] = M_CD^(t-s) * J[s, t]  for t >= s

    This means agents partially ignore deviations far in the future,
    dampening the forward guidance puzzle.

    Parameters
    ----------
    J : ndarray (T, T)
        Rational expectations Jacobian
    M_CD : float
        Cognitive discounting factor (0 < M_CD <= 1)
        M_CD = 1 is rational expectations
        M_CD = 0.85 is the paper's baseline

    Returns
    -------
    J_CD : ndarray (T, T)
        Cognitively-discounted Jacobian
    """
    T = J.shape[0]
    J_CD = J.copy()

    for s in range(T):
        for t in range(T):
            horizon = abs(t - s)
            J_CD[s, t] *= M_CD ** horizon

    return J_CD


def apply_cognitive_discounting_fast(J, M_CD=0.85):
    """Vectorized version of cognitive discounting."""
    T = J.shape[0]
    s_grid = np.arange(T)[:, None]  # (T, 1)
    t_grid = np.arange(T)[None, :]  # (1, T)
    horizon = np.abs(t_grid - s_grid)  # (T, T)
    discount = M_CD ** horizon
    return J * discount


def apply_sticky_expectations(J, gamma=0.87, max_lag=80):
    """
    Apply sticky information updating to a Jacobian matrix.

    With probability (1-gamma), an agent updates to the latest info.
    The effective Jacobian is a geometric MA of the rational Jacobian:

    J_sticky[s, :] = sum_{k=0}^{inf} (1-gamma) * gamma^k * J[s-k, :]

    where J[s-k, :] = 0 for s-k < 0.

    This creates gradual absorption of news, producing hump-shaped responses.

    Parameters
    ----------
    J : ndarray (T, T)
        Input Jacobian (possibly already cognitively discounted)
    gamma : float
        Probability of NOT updating (stickiness)
        gamma = 0 is full information rational expectations
        gamma = 0.87 means 13% update per quarter (household)
        gamma = 0.84 means 16% update per quarter (firm)
    max_lag : int
        Truncation of geometric sum

    Returns
    -------
    J_sticky : ndarray (T, T)
        Sticky-information Jacobian
    """
    if gamma < 1e-10:
        return J.copy()

    T = J.shape[0]
    J_sticky = np.zeros_like(J)

    update_prob = 1 - gamma

    for s in range(T):
        for k in range(min(s + 1, max_lag)):
            weight = update_prob * gamma ** k
            J_sticky[s, :] += weight * J[s - k, :]
        # NO normalization — the magnitude reduction IS the behavioral effect
        # Early rows naturally have lower weight sums (information hasn't diffused yet)
        # This creates the hump shape: peak shifts from Q0 to ~Q6

    return J_sticky


def apply_sticky_expectations_fast(J, gamma=0.87, max_lag=80):
    """Vectorized sticky expectations with per-row normalisation."""
    if gamma < 1e-10:
        return J.copy()

    T = J.shape[0]
    update_prob = 1 - gamma
    weights = update_prob * gamma ** np.arange(max_lag)

    J_sticky = np.zeros_like(J)
    for s in range(T):
        for k in range(min(s + 1, max_lag)):
            J_sticky[s, :] += weights[k] * J[s - k, :]
        # NO normalization — same as apply_sticky_expectations

    return J_sticky


def transform_jacobian_dict(G, M_CD=0.85, gamma_hh=0.87, gamma_f=0.84,
                            household_outputs=None, firm_outputs=None):
    """
    Apply behavioral expectations to a full Jacobian dictionary.

    The SSJ model.solve_jacobian returns G as a nested dict:
    G[output_var][input_var] = ndarray (T, T)

    Household Jacobians get gamma_hh stickiness.
    Firm Jacobians get gamma_f stickiness.
    Both get cognitive discounting M_CD.

    Parameters
    ----------
    G : dict of dict of ndarray
        Jacobian dictionary from model.solve_jacobian
    M_CD : float
        Cognitive discounting factor
    gamma_hh : float
        Household stickiness (1 - update probability)
    gamma_f : float
        Firm stickiness
    household_outputs : set or None
        Output variables from household block (use gamma_hh)
        Default: {'C', 'A', 'NE'} (consumption, assets, labor supply)
    firm_outputs : set or None
        Output variables from firm block (use gamma_f)
        Default: all others

    Returns
    -------
    G_behav : same structure as G
    """
    if household_outputs is None:
        household_outputs = {'C', 'A', 'NE', 'c', 'a', 'ne'}

    G_behav = {}
    for out_var in G:
        G_behav[out_var] = {}
        for in_var in G[out_var]:
            J = G[out_var][in_var]

            # Determine stickiness based on which block produces this output
            if out_var in household_outputs:
                gamma = gamma_hh
            else:
                gamma = gamma_f

            # Apply both transforms
            J_cd = apply_cognitive_discounting_fast(J, M_CD)
            J_behav = apply_sticky_expectations_fast(J_cd, gamma)

            G_behav[out_var][in_var] = J_behav

    return G_behav


if __name__ == '__main__':
    # Quick test: verify transforms produce expected effects
    T = 40

    # Create a mock Jacobian: impulse response that peaks on impact
    J_rational = np.zeros((T, T))
    for s in range(T):
        for t in range(T):
            if t <= s:
                J_rational[s, t] = -0.5 * np.exp(-(s - t) / 5)

    # The "IRF" is the first column (response to shock at t=0)
    irf_rational = J_rational[:, 0]

    # Apply cognitive discounting
    J_cd = apply_cognitive_discounting_fast(J_rational, M_CD=0.85)
    irf_cd = J_cd[:, 0]

    # Apply sticky expectations
    J_sticky = apply_sticky_expectations_fast(J_rational, gamma=0.87)
    irf_sticky = J_sticky[:, 0]

    # Apply both
    J_both = apply_sticky_expectations_fast(J_cd, gamma=0.87)
    irf_both = J_both[:, 0]

    print("Behavioral expectations test:")
    print(f"  Rational: peak = {irf_rational.min():.3f} at Q{np.argmin(irf_rational)}")
    print(f"  + Cognitive disc (M=0.85): peak = {irf_cd.min():.3f} at Q{np.argmin(irf_cd)}")
    print(f"  + Sticky info (γ=0.87): peak = {irf_sticky.min():.3f} at Q{np.argmin(irf_sticky)}")
    print(f"  + Both: peak = {irf_both.min():.3f} at Q{np.argmin(irf_both)}")
    print()
    print(f"  Effect: peak moves from Q{np.argmin(irf_rational)} to Q{np.argmin(irf_both)}")
    print(f"  Dampening: {irf_both.min()/irf_rational.min():.1%} of rational peak")
