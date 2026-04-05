"""
Central bank for UK-HANK model.

Taylor rule for short-term nominal interest rate.
Phase 5 will add reserves and QE mechanism.
"""

from sequence_jacobian import simple
import numpy as np


@simple
def taylor_rule(pi_cpi, Y, rho_i, phi_pi, phi_y, i_lag, r_star, pi_star, Y_ss, epsilon_r=0.0):
    """
    Taylor rule (p.13 of the paper).

    log(1+i) = rho_i * log(1+i_lag) + (1-rho_i) * {
        log(1+r_bar) + log(1+pi_bar) + phi_pi * [log(1+pi_cpi) - log(1+pi_bar)]
        + phi_y * [log(Y) - log(Y_ss)]
    } + log(1 + epsilon_r)

    Simplified (first-order):
    i = rho_i * i_lag + (1-rho_i) * (r_star + pi_star + phi_pi*(pi_cpi - pi_star) + phi_y*(Y - Y_ss)/Y_ss)
        + epsilon_r
    """
    i_target = r_star + pi_star + phi_pi * (pi_cpi - pi_star) + phi_y * np.log(Y / Y_ss)
    i = rho_i * i_lag + (1 - rho_i) * i_target + epsilon_r
    return i
