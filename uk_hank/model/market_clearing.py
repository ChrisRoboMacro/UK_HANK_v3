"""
Market clearing conditions for UK-HANK model.

Phase 1: Goods market and asset market clearing for closed economy.
"""

from sequence_jacobian import simple


@simple
def goods_market_clearing(Y, C, I, G, delta_K, K):
    """
    Goods market clearing (simplified Eq 26 for closed economy).

    Y = C + I + G

    In Phase 6, this expands to include exports - imports.
    """
    goods_residual = Y - C - I - G
    return goods_residual


@simple
def asset_market_clearing(A, B, K):
    """
    Asset market clearing (simplified Eq 22 for Phase 1).

    Household assets = Government bonds + Physical capital
    A = B + K

    Phase 5 will add long-term bonds, reserves, foreign bonds.
    """
    asset_residual = A - B - K
    return asset_residual


@simple
def capital_accumulation(K, K_lag, I, delta_K):
    """
    Capital accumulation (Eq 9, without adjustment costs in levels).

    K = (1 - delta_K) * K_lag + I
    """
    K_residual = K - (1 - delta_K) * K_lag - I
    return K_residual


@simple
def fisher_equation(i, pi_next, r_ante):
    """
    Fisher equation (Eq 19, 1st equation).

    1 + i = E[(1 + pi_next)(1 + r_ante)]
    => r_ante = (1 + i) / (1 + pi_next) - 1

    Simplified: r_ante ≈ i - pi_next
    """
    fisher_res = r_ante - (i - pi_next)
    return fisher_res
