"""
Capital firms for UK-HANK model.

Capital firms own physical capital K, invest I, and rent capital to
intermediate goods firms at rate r_K.

Eqs 9-11: Investment with adjustment costs, Tobin's Q.
"""

from sequence_jacobian import solved


@solved
def capital_investment(r_K, r_K_next, q_K, q_K_next, r_ante, delta_K=0.025, varphi_i=20.1):
    """
    Capital firm optimality conditions.

    Eq 10: q_K = E[r_K_next + q_K_next*(1-delta_K)] / (1 + r_ante)
    Eq 11: 1 = q_K * (1 - S(I/I_lag) - I/I_lag * S'(I/I_lag))
                + E[q_K_next / (1+r_ante) * (I_next/I)^2 * S'(I_next/I)]

    where S(x) = varphi_i/2 * (x - 1)^2

    For the solved block, we provide the residual equations that
    implicitly define the unknowns (q_K, I).
    """
    # Tobin's Q (Eq 10): asset pricing of capital
    q_K_residual = q_K - (r_K_next + q_K_next * (1 - delta_K)) / (1 + r_ante)

    return q_K_residual


@solved
def investment_euler(I, I_lag, I_next, q_K, q_K_next, r_ante, varphi_i=20.1):
    """
    Investment Euler equation (Eq 11).

    1 = q_K * [1 - S(I/I_lag) - (I/I_lag)*S'(I/I_lag)]
        + E[ q_K_next/(1+r_ante) * (I_next/I)^2 * S'(I_next/I) ]

    where S(x) = varphi_i/2 * (x-1)^2, S'(x) = varphi_i*(x-1)
    """
    x = I / I_lag
    x_next = I_next / I

    S = varphi_i / 2 * (x - 1) ** 2
    Sp = varphi_i * (x - 1)
    Sp_next = varphi_i * (x_next - 1)

    invest_residual = (1 - q_K * (1 - S - x * Sp)
                       - q_K_next / (1 + r_ante) * x_next ** 2 * Sp_next)

    return invest_residual
