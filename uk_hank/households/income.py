"""
Household income process for UK-HANK model.

Implements the Kaplan et al. (2018) jump-shock income process from Table 3:
  - Persistent component: log(e^P) = (1-J^P)*rho_P*log(e^P_lag) + J^P*eps^P
  - Transitory component: log(e^T) = (1-J^T)*rho_T*log(e^T_lag) + J^T*eps^T
  - Total: e = e^P * e^T (log additive)
  - J^s ~ Bernoulli(xi_s), eps^s ~ N(0, sigma_s^2)

This is discretized into a finite Markov chain over (e^P, e^T) states.
"""

import numpy as np
from sequence_jacobian import grids


def rouwenhorst(n, rho, sigma):
    """
    Rouwenhorst method for discretizing AR(1) process.
    Returns grid in levels (mean 1) and transition matrix.
    """
    p = (1 + rho) / 2

    psi = sigma * np.sqrt(n - 1)
    e_log = np.linspace(-psi, psi, n)

    if n == 1:
        Pi = np.array([[1.0]])
    elif n == 2:
        Pi = np.array([[p, 1 - p], [1 - p, p]])
    else:
        Pi_old = np.array([[p, 1 - p], [1 - p, p]])
        for i in range(3, n + 1):
            Pi_new = np.zeros((i, i))
            Pi_new[:i-1, :i-1] += p * Pi_old
            Pi_new[:i-1, 1:i] += (1 - p) * Pi_old
            Pi_new[1:i, :i-1] += (1 - p) * Pi_old
            Pi_new[1:i, 1:i] += p * Pi_old
            Pi_new[1:-1, :] /= 2
            Pi_old = Pi_new
        Pi = Pi_new

    e_grid = np.exp(e_log)
    pi_ss = stationary_distribution(Pi)
    e_grid = e_grid / np.dot(pi_ss, e_grid)  # normalise mean to 1

    return e_grid, Pi, pi_ss


def jump_shock_markov(n, rho, xi, sigma_eps):
    """
    Discretize a jump-shock AR(1) process:
      log(e_t) = (1 - J_t) * rho * log(e_{t-1}) + J_t * eps_t
    where J ~ Bernoulli(xi), eps ~ N(0, sigma_eps^2).

    With probability (1-xi): the process follows AR(1) with persistence rho
    With probability xi: the process jumps to a new draw from N(0, sigma_eps^2)

    We approximate this with a Rouwenhorst grid for the AR(1) component,
    then modify the transition matrix to include jumps.
    """
    # Unconditional variance: solve sigma_e^2 = (1-xi)*rho^2*sigma_e^2 + xi*sigma_eps^2
    # => sigma_e = sigma_eps * sqrt(xi / (1 - (1-xi)*rho^2))
    var_e = xi * sigma_eps**2 / (1 - (1 - xi) * rho**2)
    sigma_e = np.sqrt(var_e)

    # Rouwenhorst grid for the unconditional distribution
    e_grid, Pi_rouw, pi_ss = rouwenhorst(n, rho, sigma_e)
    e_log = np.log(e_grid * np.dot(pi_ss, e_grid))  # log grid centered at 0

    # Modify transition matrix:
    # Pi[i,j] = (1-xi) * Pi_rouw[i,j]  (no jump: follow AR(1))
    #         + xi * Pi_jump[j]          (jump: draw from discretized N(0, sigma_eps^2))
    #
    # Pi_jump[j] = prob of landing in bin j when drawing eps ~ N(0, sigma_eps^2)
    from scipy.stats import norm

    # Jump distribution: discretize N(0, sigma_eps^2) onto the grid
    midpoints = np.zeros(n + 1)
    midpoints[0] = -np.inf
    midpoints[-1] = np.inf
    for i in range(1, n):
        midpoints[i] = (e_log[i-1] + e_log[i]) / 2

    Pi_jump = np.zeros(n)
    for j in range(n):
        Pi_jump[j] = norm.cdf(midpoints[j+1], 0, sigma_eps) - norm.cdf(midpoints[j], 0, sigma_eps)

    # Combined transition matrix
    Pi = (1 - xi) * Pi_rouw + xi * Pi_jump[np.newaxis, :]

    # Recompute stationary distribution
    pi_ss = stationary_distribution(Pi)

    # Renormalise grid to have mean 1
    e_grid = np.exp(e_log)
    e_grid = e_grid / np.dot(pi_ss, e_grid)

    return e_grid, Pi, pi_ss


def make_combined_income_process(n_P=7, n_T=3):
    """
    Create the full Kaplan et al. (2018) income process with persistent
    and transitory components.

    Parameters from Table 3:
      Transitory: xi_T=0.14, rho_T=0.495, sigma_T=0.464
      Persistent: xi_P=0.011, rho_P=0.995, sigma_P=0.825

    Total productivity: e = e^P * e^T (multiplicative in levels)

    Returns combined grid of shape (n_P * n_T,) and transition matrix.
    """
    # Persistent component
    e_P, Pi_P, pi_P = jump_shock_markov(n_P, rho=0.995, xi=0.011, sigma_eps=0.825)

    # Transitory component
    e_T, Pi_T, pi_T = jump_shock_markov(n_T, rho=0.495, xi=0.14, sigma_eps=0.464)

    # Combined grid: e = e^P * e^T
    n_combined = n_P * n_T
    e_grid = np.zeros(n_combined)
    for ip in range(n_P):
        for it in range(n_T):
            idx = ip * n_T + it
            e_grid[idx] = e_P[ip] * e_T[it]

    # Combined transition matrix: Pi = Pi_P ⊗ Pi_T
    Pi = np.kron(Pi_P, Pi_T)

    # Stationary distribution
    pi_ss = stationary_distribution(Pi)

    # Normalise mean to 1
    e_grid = e_grid / np.dot(pi_ss, e_grid)

    return e_grid, Pi, pi_ss


def stationary_distribution(Pi):
    """Find stationary distribution of Markov chain."""
    eigenvalues, eigenvectors = np.linalg.eig(Pi.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi_ss = np.real(eigenvectors[:, idx])
    pi_ss = pi_ss / pi_ss.sum()
    return pi_ss


def check_income_moments(e_grid, Pi, pi_ss, n_periods=10000):
    """
    Check income process moments against Table B.1 targets.

    Target moments (UK data from ASHE):
      Std log earnings: 0.70
      Std 1-yr change: 0.28
      Kurtosis 1-yr change: 24.8
    """
    log_e = np.log(e_grid)

    # Cross-sectional std of log earnings
    mean_log = np.dot(pi_ss, log_e)
    var_log = np.dot(pi_ss, (log_e - mean_log)**2)
    std_log = np.sqrt(var_log)

    # 1-year change moments (4 quarters)
    Pi_4 = np.linalg.matrix_power(Pi, 4)  # annual transition

    # Var of 1-year change: E[(log_e' - log_e)^2]
    # = E[log_e'^2] + E[log_e^2] - 2*E[log_e * log_e']
    # where expectations are over the joint (e, e') distribution
    E_log2 = np.dot(pi_ss, log_e**2)
    E_cross = 0
    for i in range(len(e_grid)):
        for j in range(len(e_grid)):
            E_cross += pi_ss[i] * Pi_4[i, j] * log_e[i] * log_e[j]
    var_1yr = 2 * E_log2 - 2 * E_cross
    std_1yr = np.sqrt(max(var_1yr, 0))

    print(f"  Income process moments:")
    print(f"    Std log earnings: {std_log:.2f} (target: 0.70, model: 0.65)")
    print(f"    Std 1-yr change:  {std_1yr:.2f} (target: 0.28, model: 0.26)")
    print(f"    Mean productivity: {np.dot(pi_ss, e_grid):.4f} (target: 1.0)")

    return {'std_log': std_log, 'std_1yr': std_1yr}


if __name__ == '__main__':
    print("Testing income process...")

    # Simple Rouwenhorst
    print("\n1. Simple Rouwenhorst (Phase 1):")
    e, Pi, pi = rouwenhorst(7, 0.966, 0.5)
    print(f"   Grid: {e.round(3)}")
    check_income_moments(e, Pi, pi)

    # Jump-shock (Phase 3)
    print("\n2. Jump-shock combined (Phase 3):")
    e, Pi, pi = make_combined_income_process(n_P=7, n_T=3)
    print(f"   Grid: {len(e)} points, range [{e.min():.3f}, {e.max():.3f}]")
    check_income_moments(e, Pi, pi)
