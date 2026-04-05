"""
Block-level behavioral expectations for UK-HANK model (Step 1).
===============================================================

Correct implementation:
1. Build model via incremental_test.test_v3_plus_investment (proven stable)
2. Get GE Jacobian G[var][exog] = (T,T) matrix
3. Apply cognitive discounting + sticky expectations (NO normalization)
4. Compute IRF as G_transformed @ shock_path
"""

import numpy as np
from uk_hank.expectations.behavioral import (
    apply_cognitive_discounting_fast,
    apply_sticky_expectations_fast,
)


def run(verbose=True):
    """Run model with proper behavioral expectations."""
    if verbose:
        print("=" * 60)
        print("UK-HANK: BEHAVIORAL EXPECTATIONS (Step 1)")
        print("=" * 60)

    # Build proven-stable model from incremental test
    from uk_hank.model.incremental_test import test_v3_plus_investment
    result = test_v3_plus_investment()

    if not result or not result['stable']:
        print("  Model build FAILED")
        return None

    model = result['model']
    ss = result['ss']
    G = result['G']
    T = 300

    if verbose:
        print(f"\n  SS: beta={ss['beta']:.6f}")
        print(f"  Rational GDP: {result['gdp_peak']:.2f}% at Q0")

    # Shock path
    dr = 0.0025 * 0.5 ** np.arange(T)

    # Rational IRF (baseline)
    irf_rat = {}
    for var in G:
        if 'rstar' in G[var]:
            irf_rat[var] = np.array(G[var]['rstar']) @ dr

    # Behavioral: transform each G[var][rstar] matrix
    M_CD = 0.85
    gamma_hh = 0.87
    gamma_f = 0.84
    hh_vars = {'C', 'A', 'Ns', 'NE'}

    irf_beh = {}
    for var in G:
        if 'rstar' not in G[var]:
            continue
        J = np.array(G[var]['rstar'])

        # 1. Cognitive discounting
        J = apply_cognitive_discounting_fast(J, M_CD)

        # 2. Sticky expectations (NO normalization — this IS the behavioral effect)
        gamma = gamma_hh if var in hh_vars else gamma_f
        J = apply_sticky_expectations_fast(J, gamma, max_lag=80)

        # 3. IRF
        irf_beh[var] = J @ dr

    # Results
    Y_rat = irf_rat.get('Y', np.zeros(T)) / ss['Y'] * 100
    Y_beh = irf_beh.get('Y', np.zeros(T)) / ss['Y'] * 100

    if verbose:
        print(f"\n  RATIONAL:   GDP = {np.min(Y_rat):.2f}% at Q{np.argmin(Y_rat)}")
        print(f"  BEHAVIORAL: GDP = {np.min(Y_beh):.2f}% at Q{np.argmin(Y_beh)}")
        print(f"  Peak shift: Q{np.argmin(Y_rat)} -> Q{np.argmin(Y_beh)}")

        if 'C' in irf_beh:
            C_beh = irf_beh['C'] / ss['C'] * 100
            print(f"  C: {np.min(C_beh):.2f}% at Q{np.argmin(C_beh)}")
        if 'pi' in irf_beh:
            print(f"  CPI 3yr: {np.cumsum(irf_beh['pi'][:12])[-1]*100:.2f}pp")
        if 'RER' in irf_beh:
            print(f"  RER: {np.max(np.abs(irf_beh['RER']/ss['RER']*100)):.2f}%")

        print(f"\n  Paper target: GDP -0.71% at Q6, CPI -1.59pp")

    return {
        'model': model, 'ss': ss,
        'irf_rational': irf_rat, 'irf_behavioral': irf_beh,
        'G': G,
    }


if __name__ == '__main__':
    result = run()
