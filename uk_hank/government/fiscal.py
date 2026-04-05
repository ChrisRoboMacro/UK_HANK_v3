"""
Fiscal policy for UK-HANK model.

Phase 1: Simple proportional tax with government spending.
Phase 3 will add progressive tax retention function (Eq 13) and benefits.
Phase 5 will add government debt dynamics and fiscal feedback rules.
"""

from sequence_jacobian import simple


@simple
def fiscal_simple(Y, G_ss, tau_ss):
    """
    Phase 1 simplified fiscal block.

    Government levies proportional tax tau on output and spends G.
    Budget: T = G (balanced budget for now; debt comes in Phase 5).

    Returns after-tax income scaling factor and government demand.
    """
    T = tau_ss * Y    # Tax revenue
    G = G_ss          # Government spending (constant for now)
    tau = tau_ss      # Tax rate
    return T, G, tau
