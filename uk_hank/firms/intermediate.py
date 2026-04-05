"""
Intermediate goods firms for UK-HANK model.

Monopolistic competition with Rotemberg price adjustment costs.
Cobb-Douglas production: y = Omega * k^alpha_k * n^(1-alpha_k)

Delivers:
  - Price Phillips Curve (Eq 7)
  - Real marginal cost (Eq 8)
  - Factor demands
"""

from sequence_jacobian import simple


@simple
def firms_intermediate(Y, w, r_K, alpha_k, eta_x, varphi_x, pi, pi_star, r_ante, Omega=1.0):
    """
    Intermediate goods sector.

    Inputs:
        Y : output
        w : real wage (W_tilde / P)
        r_K : rental rate of capital
        alpha_k : capital share
        eta_x : demand elasticity (markup = eta_x / (eta_x - 1))
        varphi_x : price adjustment cost scale
        pi : current inflation (P_D,t^X / P_D,t-1^X - 1)
        pi_star : steady-state inflation
        r_ante : ex-ante real rate
        Omega : TFP (normalised to 1)

    Outputs:
        mc : real marginal cost (Eq 8)
        N : labour demand
        K : capital demand
        profit_x : intermediate goods sector profits
    """
    # Real marginal cost (Eq 8)
    mc = (1 / Omega) * (r_K / alpha_k) ** alpha_k * (w / (1 - alpha_k)) ** (1 - alpha_k)

    # Factor demands from cost minimisation (Eq 38)
    # w * n / (r_K * k) = (1-alpha_k) / alpha_k
    # y = Omega * k^alpha_k * n^(1-alpha_k)
    K = alpha_k * mc * Y / r_K
    N = (1 - alpha_k) * mc * Y / w

    # Profits (Eq 42): p_x * X - MC*X - adjustment costs
    # In symmetric equilibrium, profits per unit of output:
    profit_x = Y * (1 - mc - (varphi_x / 2) * (pi - pi_star) ** 2)

    return mc, N, K, profit_x


@simple
def price_phillips_curve(pi, pi_next, mc, Y, Y_next, eta_x, varphi_x, pi_star, r_ante):
    """
    New Keynesian Price Phillips Curve (linearised Eq 7).

    (pi - pi_star)(1 + pi) = (1-eta_x)/varphi_x * P_D^X + eta_x/varphi_x * MC
                              + beta * E[(pi_next - pi_star)(1+pi_next) * Y_next/Y]

    In steady state: mc_ss = (eta_x - 1) / eta_x  (inverse markup)
    """
    nkpc_res = ((pi - pi_star) * (1 + pi)
                - (1 - eta_x) / varphi_x
                - eta_x / varphi_x * mc
                - 1 / (1 + r_ante) * (pi_next - pi_star) * (1 + pi_next) * Y_next / Y)
    return nkpc_res
