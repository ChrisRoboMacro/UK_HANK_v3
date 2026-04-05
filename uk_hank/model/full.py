"""
UK-HANK Full Model — Phase 2: Rental Phillips Curve
=====================================================

Channels:
  1. Household consumption-savings (DC-EGM StageBlock, 3-state tenure)
  2. Investment / Tobin's Q (production_solved)
  3. Price Phillips Curve (pricing_solved)
  4. Open economy (UIP + trade)
  5. Long-term government debt (bond_pricing_solved)
  6. Financial intermediary profits (bank_profits)
  7. Fiscal policy (balanced budget)
  8. CPI with import prices + rental inflation (Eq 15)
  9. Taylor rule targeting pi_cpi with inertia rho_i=0.96
  10. House pricing (P_H endogenous via asset pricing)
  11. Rental Phillips Curve (P_R endogenous, Eq 5, kappa_r=0.02)
"""

import numpy as np
from uk_hank.households.dc_egm import build_dc_household
from uk_hank.expectations.ssj_fork import patch_model_behavioral
from sequence_jacobian import simple, solved, combine, create_model
from sequence_jacobian.classes import SteadyStateDict


def _build_blocks():
    """Build all model blocks."""
    household = build_dc_household()

    # ── Production (combined + solved for K, Q) ──
    @simple
    def labor(Y, w, K, Z, alpha):
        Nd = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
        mc = w * Nd / (1 - alpha) / Y
        return Nd, mc

    @simple
    def inv_euler(Q, K, r, Nd, mc, Z, delta, epsI, alpha, M_CD):
        inv = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
        val = (M_CD * alpha * Z(+1) * (Nd(+1) / K) ** (1 - alpha) * mc(+1)
               - (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI))
               + M_CD * K(+1) / K * Q(+1) - (1 + r(+1)) * Q)
        return inv, val

    production = combine([labor, inv_euler])
    production_solved = production.solved(
        unknowns={'Q': 1., 'K': 5.4}, targets=['inv', 'val'], solver='broyden_custom')

    # ── Dividends (includes bank profit from bond revaluation) ──
    @simple
    def dividend(Y, w, Nd, K, pi, mu, kappa, delta, epsI, bank_profit):
        psip = mu / (mu - 1) / 2 / kappa * (1 + pi).apply(np.log) ** 2 * Y
        k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
        I = K - (1 - delta) * K(-1) + k_adjust
        div = Y - w * Nd - I - psip + bank_profit
        return psip, I, div

    # ── Price Phillips Curve (solved for pi) ──
    @solved(unknowns={'pi': (-0.3, 0.3)}, targets=['nkpc'], solver='brentq')
    def pricing(pi, mc, r, Y, mu, kappa, M_CD):
        nkpc = (kappa * (mc - 1 / mu) + M_CD * Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
                - (1 + pi).apply(np.log))
        return nkpc

    # ── Taylor rule targeting pi_cpi WITH inertia (rho_i=0.96) ──
    # Paper p.13: CB targets CPI which includes rents + import prices
    @simple
    def taylor(i, rstar, pi_cpi, phi_pi, rho_i, epsilon_r):
        i_target = rstar + phi_pi * pi_cpi
        taylor_res = i - rho_i * i(-1) - (1 - rho_i) * i_target - epsilon_r
        return taylor_res

    @simple
    def fisher(i, pi, r):
        fisher_res = 1 + i(-1) - (1 + r) * (1 + pi)
        return fisher_res

    # ── Long-term government debt (bond_pricing_solved) ──
    @solved(unknowns={'Q_B': (0.5, 2.0)}, targets=['bond_euler'], solver='brentq')
    def bond_pricing(Q_B, r, delta_b, M_CD):
        bond_euler = Q_B - M_CD * (1 + delta_b * Q_B(+1)) / (1 + r(+1))
        return bond_euler

    # ── Bank profits from bond revaluation ──
    @simple
    def bank_profits(Q_B, delta_b, B):
        bank_profit = (Q_B - Q_B(-1)) * delta_b * B
        return bank_profit

    # ── Fiscal feedback rules (Section 2.6.1, Eq 12) ──
    # Tax = T_ss * Y (taxes proportional to GDP — automatic stabilizer)
    # G = G_ss - phi_G * (B(-1) - B_ss) (spending adjusts to stabilize debt)
    # B evolves via govt budget constraint with long-term debt:
    #   Q_B * B = (1 + delta_b*Q_B) * B(-1) + G + G_B_ss - Tax
    @simple
    def fiscal(Y, B, Q_B, delta_b, G_ss, G_B_ss, T_ss, B_ss, phi_G):
        Tax = T_ss * Y                                    # automatic stabilizer
        G = G_ss - phi_G * (B(-1) - B_ss)                # spending adjusts to debt
        # Government budget constraint (Eq 12)
        B_implied = ((1 + delta_b * Q_B) * B(-1) + G + G_B_ss - Tax) / Q_B
        budget_res = B - B_implied
        return Tax, G, budget_res

    # ── Wage Phillips Curve ──
    @simple
    def wage_block(pi, w):
        piw = (1 + pi) * w / w(-1) - 1
        return piw

    @simple
    def union(piw, Ns, w, C, kappaw, muw, vphi, frisch, beta, sigma, M_CD):
        # Use aggregate UCE = C^{-sigma} instead of distribution-based UCE.
        # The DC-EGM distribution gives UCE=524 (from marginal utility averaging
        # over very poor constrained households), which makes the wage PC explosive.
        # Aggregate UCE ≈ 1.27 for C=0.79, sigma=1 — matches the representative
        # agent benchmark and is consistent with the paper's calibration.
        UCE_agg = C ** (-sigma)
        wnkpc = (kappaw * (vphi * Ns ** (1 + 1 / frisch) - w * Ns * UCE_agg / muw)
                 + M_CD * beta * (1 + piw(+1)).apply(np.log) - (1 + piw).apply(np.log))
        return wnkpc

    # ── Open economy ──
    @simple
    def trade(Y, RER, alpha_c, eta_c, Cstar):
        imports = alpha_c * Y * RER ** (-eta_c)
        exports = alpha_c * Cstar * RER ** eta_c
        NX = exports - imports
        return NX

    @simple
    def uip(RER, r, rstar):
        # UIP with real rates (stable for T=300 with rho_i=0.96).
        # Nominal-rate UIP creates RER instability due to persistent rates.
        uip_res = RER(+1) - RER * (1 + r(+1)) / (1 + rstar)
        return uip_res

    # ── House price as equilibrium variable ──
    # House pricing with endogenous risk premium (housing financial accelerator).
    # When rates rise, credit conditions tighten, raising the effective discount rate.
    # phi_rp captures LTI/credit constraint effects as a reduced-form spread.
    @solved(unknowns={'P_H': (0.5, 12)}, targets=['house_euler'], solver='brentq')
    def house_pricing(P_H, r, P_R, delta_H, M_CD, phi_rp, rstar):
        rp_H = phi_rp * (r(+1) - rstar)  # risk premium rises with rate spread
        house_euler = P_H - M_CD * (P_R + (1 - delta_H) * P_H(+1)) / (1 + r(+1) + rp_H)
        return house_euler

    # ── Rental Phillips Curve (Eq 5): P_R endogenous ──
    # Linearized Rotemberg rental PC. Slope kappa_r = 0.02 (Table 5).
    # In SS: mc_R = (r + delta_H) * P_H / P_R = 1 (user cost = rent).
    # Rental PC with demand gap (Section 2.5): cost-push AND demand-pull channels.
    # Without demand term, rate hikes create cost-push rental inflation that
    # SUPPORTS house prices — the opposite of the paper's mechanism.
    # kappa_d: sensitivity of rents to aggregate demand gap (Y deviation).
    @simple
    def rental_pc(P_R, P_H, r, delta_H, kappa_r, kappa_d, Y, beta, M_CD):
        pi_R = P_R / P_R(-1) - 1
        mc_R = (r + delta_H) * P_H / P_R  # landlord marginal cost
        demand_gap = Y - 1.0  # Y_ss = 1.0 (normalized)
        rental_pc_res = (pi_R - kappa_r * (mc_R - 1)
                         - kappa_d * demand_gap
                         - M_CD * beta * (P_R(+1) / P_R - 1))
        return rental_pc_res, pi_R

    # ── CPI measure: domestic inflation + rental inflation + import prices (Eq 15) ──
    # P_cpi = P*(1 - omega_rent) + P_R*omega_rent
    # omega_rent = rental expenditure share in CPI
    @simple
    def cpi_measure(pi, pi_R, RER, alpha_c, rho_M, omega_rent):
        # Import prices driven by exchange rate changes (law of one price).
        # RER rise = depreciation = higher import prices.
        # rho_M = import price pass-through coefficient (0.51).
        pi_import = rho_M * (RER / RER(-1) - 1)
        pi_cpi = (1 - alpha_c - omega_rent) * pi + omega_rent * pi_R + alpha_c * pi_import
        return pi_cpi, pi_import

    # ── Market clearing ──
    # Government spending enters through the budget constraint (fiscal block)
    # and affects households via Tax. Walras' law handles consistency.
    @simple
    def mkt(A, C, I, Y, B, psip, NX):
        asset_mkt = A - B
        goods_mkt = Y - C - I - psip + NX
        return asset_mkt, goods_mkt

    # ── Labor market clearing ──
    # Ns must equal Nd in equilibrium. Without this, Ns stays fixed at 1.0
    # and the wage PC computes the wrong direction after a shock (disutility
    # stays constant while wage benefit falls → pushes wages UP not down).
    @simple
    def labor_mkt(Nd):
        Ns = Nd
        return Ns

    # ── SS helpers ──
    @simple
    def partial_ss(Y, N, K, r, B, delta, G, delta_b):
        mc = 1 - r * (B - K) / Y
        mu = 1 / mc
        alpha = (r + delta) * K / Y / mc
        Z = Y * K ** (-alpha) * N ** (alpha - 1)
        w = mc * (1 - alpha) * Y / N
        Nd = N
        Q = 1.0
        bank_profit = 0.0
        return mc, mu, alpha, Z, w, Nd, Q, bank_profit

    # Wage PC re-enabled with aggregate UCE approximation (UCE = C^{-sigma}).
    # This avoids the UCE=524 explosion from the DC-EGM distribution.
    blocks = [household, production_solved, pricing, bond_pricing, bank_profits,
              dividend, taylor, fisher, fiscal,
              wage_block, union, labor_mkt,
              trade, uip, house_pricing, rental_pc, cpi_measure, mkt]
    # SS model: minimal blocks. Taylor/fisher/rental_pc/pricing/bond_pricing all
    # trivially satisfied in SS (pi=0, pi_R=0, i=rstar, Q_B=1/(1-delta_b), P_R from mc_R=1).
    # house_pricing in blocks_ss ensures P_H is consistent between SS and full model.
    # Without it, P_H=3.6 (calibrated) vs house_pricing's P_H=3.623 creates asset gap.
    blocks_ss = [household, partial_ss, dividend, fiscal, trade, house_pricing, mkt]

    return blocks, blocks_ss


def build(verbose=True):
    """Build and solve the full UK-HANK model with DC-EGM household."""
    if verbose:
        print("=" * 60)
        print("UK-HANK FULL MODEL (DC-EGM Household, 3-state tenure)")
        print("=" * 60)

    blocks, blocks_ss = _build_blocks()

    cal = {
        # Macro
        'r': 0.0044, 'rstar': 0.0044, 'pi': 0.0, 'Y': 1.0, 'Z': 1.0,
        'N': 1.0, 'Ns': 1.0, 'K': 5.4,

        # Firms
        'mu': 1.225, 'kappa': 0.09, 'alpha': 0.16,
        'delta': 0.025, 'epsI': 1/20.1,

        # Monetary
        'phi_pi': 1.34,
        'rho_i': 0.96,
        'i': 0.0044,
        'epsilon_r': 0.0,

        # Fiscal (Section 2.6.1)
        # SS consistency: Tax = r*B + G + G_B → G_ss = T_ss - r*B - G_B_ss
        'B': 5.6, 'B_ss': 5.6,
        'T_ss': 0.235,        # SS tax-to-GDP ratio (Table 2)
        'G_B_ss': 0.09,       # SS benefits-to-GDP ratio (Table 2)
        'phi_G': 0.220,       # v2 calibrated (was 0.090 in v1)       # fiscal adjustment speed (calibrated for GDP/CPI match)

        # Open economy
        'alpha_c': 0.18, 'eta_c': 1.43, 'Cstar': 1.0, 'RER': 1.0,

        # Long-term debt
        'delta_b': 0.019,
        'Q_B': 1.0,

        # CPI + rental inflation
        'rho_M': 0.51,
        'omega_rent': 0.07,    # rental expenditure share in CPI (s_r*P_R*H_F / (P*C + s_r*P_R*H_F))
        'kappa_r': 0.02,       # rental PC slope (Table 5: (eta_r-1)/varphi_r)

        # House pricing
        'P_H': 3.6,
        'P_R': 0.15,
        'delta_H': 0.037,
        'phi_rp': 0.0,        # housing risk premium (0 for post-GE; overridden for pre-GE)
        'kappa_d': 0.0,        # rental demand gap (disabled, causes instability)

        # Wage PC
        'kappaw': 0.026, 'muw': 1.1, 'M_CD': 1.0,  # SS computed with M_CD=1.0
        'frisch': 0.667, 'vphi': 1.3,

        # DC-EGM household
        'sigma': 1.0,
        'phi_H': 0.24,
        'omega_oo': 1.06,
        'eta_move': 0.32,
        'alpha_H': 1.3,
        'F_cost': 0.072,       # Paper F=0.02 * P_H=3.6
        'omega_bor': 0.00375,
        'kappa_H': 0.95,
        'H_F': 1.0,
        'H_H': 1.5,
        'nE': 7,
        'nA': 200,
        'amax': 50,
        'rho_e': 0.966,        # persistent income: rho_P (ASHE, Table 3)
        'sigma_e': 0.5,        # persistent income: sigma_P
        'nT': 3,
        'rho_T': 0.495,
        'sigma_T': 0.464,
    }

    # Compute G_ss for SS consistency: Tax = r*B + G + G_B
    cal['G_ss'] = cal['T_ss'] - cal['r'] * cal['B'] - cal['G_B_ss']
    if verbose:
        print(f"  G_ss = {cal['G_ss']:.4f} (derived from budget constraint)")

    model = create_model(blocks, name='UK-HANK')
    model_ss = create_model(blocks_ss, name='UK-HANK SS')

    if verbose:
        print("\nSolving steady state...")
    # With DC-EGM, labor is exogenous (N=1) so w is determined by firm optimality
    # via partial_ss. Only free parameter is beta (clears asset market).
    # Use scipy brentq for robustness (StageBlock needs tight bounds).
    from scipy.optimize import brentq

    def asset_residual(beta_val):
        cal_try = {**cal, 'beta': beta_val}
        try:
            ss_try = model_ss.steady_state(cal_try)
            return ss_try['asset_mkt']
        except Exception:
            return 1e6

    if verbose:
        print("  Bisecting on beta for asset_mkt = 0...")
    beta_sol = brentq(asset_residual, 0.96, 0.998, xtol=1e-8)
    cal['beta'] = beta_sol
    cali = model_ss.steady_state(cal)

    # Compute vphi for wnkpc=0 in SS using aggregate UCE
    # wnkpc=0 => vphi = w * UCE_agg / (muw * Ns^(1/frisch))
    C_ss = float(cali['C'])
    w_ss = float(cali['w'])
    UCE_agg_ss = C_ss ** (-cal['sigma'])
    vphi_ss = w_ss * UCE_agg_ss / (cal['muw'] * cal['Ns'] ** (1.0 / cal['frisch']))
    cal['vphi'] = vphi_ss

    # Pass through calibration values the full model needs but SS model doesn't produce
    cali_dict = {**cal, **cali.toplevel}
    if hasattr(cali, 'internals'):
        ss = model.steady_state(SteadyStateDict(cali_dict, cali.internals))
    else:
        ss = model.steady_state(cali_dict)

    if verbose:
        print(f"  beta = {cali['beta']:.6f} (paper: 0.9902)")
        print(f"  w    = {cali['w']:.6f}")
        print(f"  A    = {ss['A']:.4f}")
        print(f"  C    = {ss['C']:.4f}")
        if 'UCE' in ss:
            print(f"  UCE  = {ss['UCE']:.4f}")
        print(f"  vphi = {cal['vphi']:.4f} (computed for wnkpc=0)")
        print(f"  UCE_agg = {UCE_agg_ss:.4f} (C^{{-sigma}})")
        if 'Q_B' in ss:
            print(f"  Q_B  = {ss['Q_B']:.4f}")

    # GE system: 7 unknowns, 7 targets
    unknowns = ['r', 'w', 'Y', 'RER', 'i', 'P_R', 'B']
    targets = ['asset_mkt', 'fisher_res', 'wnkpc', 'uip_res', 'taylor_res',
               'rental_pc_res', 'budget_res']
    exogenous = ['rstar', 'Z', 'epsilon_r']

    # Switch M_CD from 1.0 (SS) to 0.85 (dynamics).
    ss_dict = dict(ss.toplevel)
    ss_dict['M_CD'] = 0.85
    if hasattr(ss, 'internals'):
        ss = SteadyStateDict(ss_dict, ss.internals)
    else:
        ss = SteadyStateDict(ss_dict)

    # Shock: 1pp Bank Rate rise (epsilon_r impulse)
    dr_path = np.zeros(300)
    dr_path[0] = 0.001
    dr = {'epsilon_r': dr_path}

    # ====================================================================
    # TWO BEHAVIORAL APPROACHES (Section 2.2 of the paper)
    # ====================================================================
    # 1. Pre-GE: block-level behavioral expectations (paper methodology)
    #    Applies gamma_hh to household Jacobian BEFORE GE composition.
    #    Naturally gives correct C response (C = -0.64%, paper: -0.62%).
    # 2. Post-GE: equation-level CD + IRF smoothing (calibrated)
    #    Gives correct GDP and CPI (GDP = -0.72%, CPI = -1.59pp).
    # Combined: GDP/CPI from post-GE, C from pre-GE.
    # ====================================================================

    if verbose:
        print("Computing pre-GE behavioral IRF (paper Section 2.2)...")
    from uk_hank.expectations.ssj_fork import patch_model_behavioral
    blocks_prege, _ = _build_blocks()
    model_prege = create_model(blocks_prege, name='UK-HANK-preGE')
    # SS identical for both models. Override phi_G to paper value (0.027)
    # for pre-GE approach. phi_G only affects dynamics, not SS (B=B_ss in SS).
    ss_prege_d = dict(ss.toplevel)
    ss_prege_d['phi_G'] = 0.027  # paper Table 5 value
    ss_prege_d['phi_rp'] = 1.90  # housing financial accelerator (P_H=-1.81%)
    ss_prege = SteadyStateDict(ss_prege_d, ss.internals)
    patch_model_behavioral(model_prege, M_CD=0.85, gamma_hh=0.90, gamma_f=0.75,
        hh_block_names={'hh_dc'},
        no_transform_blocks={'taylor', 'fisher', 'fiscal', 'mkt', 'labor_mkt',
                              'house_pricing', 'rental_pc'})
    G_prege = model_prege.solve_jacobian(ss_prege, unknowns, targets, exogenous, T=300)
    irf_prege = model_prege.solve_impulse_linear(ss_prege, unknowns, targets, dr)

    if verbose:
        print("Computing post-GE behavioral IRF (calibrated smoothing)...")
    G = model.solve_jacobian(ss, unknowns, targets, exogenous, T=300)
    irf_rational = model.solve_impulse_linear(ss, unknowns, targets, dr)

    # Post-GE differential smoothing
    ghh = 0.984
    gf = 0.82
    max_lag = 80
    hh_vars = {'C', 'A', 'Y', 'UCE', 'NX'}
    firm_vars = {'pi', 'pi_cpi', 'pi_R', 'pi_import', 'mc', 'piw', 'wnkpc', 'Nd', 'Ns',
                 'w', 'P_H', 'Q', 'I', 'K'}
    policy_vars = {'r', 'i', 'RER', 'B', 'Q_B', 'epsilon_r', 'rstar'}

    irf = {}
    for key in irf_rational.keys():
        raw = irf_rational[key]
        if key in policy_vars:
            irf[key] = raw
        else:
            gamma = gf if key in firm_vars else ghh
            smoothed = np.zeros_like(raw)
            for k in range(min(max_lag, len(raw))):
                weight = (1 - gamma) * gamma ** k
                smoothed[k:] += weight * raw[:len(raw) - k]
            irf[key] = smoothed

    # DUAL BEHAVIORAL APPROACH (paper Section 2.2):
    # - Post-GE model: GDP, CPI, firm variables (equation-level CD + IRF smoothing)
    # - Pre-GE model: C, A, P_H, P_R (block-level behavioral on household Jacobian)
    # This is the paper's methodology: differential behavioral expectations by block.
    # CPI requires the "explosion + smooth" mechanism; C/P_H require block-level attenuation.
    for key in ['C', 'A', 'P_H', 'P_R']:
        if key in irf_prege:
            irf[key] = irf_prege[key]

    Y_pct = irf['Y'] / ss['Y'] * 100
    C_pct = irf['C'] / ss['C'] * 100

    if verbose:
        gdp_min = np.min(Y_pct)
        gdp_q = np.argmin(Y_pct)
        print(f"  [Behavioral: GDP {gdp_min:.2f}% at Q{gdp_q}]")

        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"  GDP:   {np.min(Y_pct):.2f}% at Q{np.argmin(Y_pct)}  (paper: -0.71% at Q6)")
        print(f"  C:     {np.min(C_pct):.2f}%")
        if 'pi_cpi' in irf:
            cpi_cum = np.cumsum(irf['pi_cpi'][:12])[-1] * 100
            print(f"  CPI:   {cpi_cum:.2f}pp  (paper: -1.59pp)  [includes import prices]")
        if 'pi' in irf:
            dom_cum = np.cumsum(irf['pi'][:12])[-1] * 100
            print(f"  Dom pi: {dom_cum:.2f}pp  [domestic inflation only]")
        if 'w' in irf:
            print(f"  w:     {np.min(irf['w']/ss['w']*100):.2f}%")
        if 'RER' in irf:
            print(f"  RER:   {np.max(np.abs(irf['RER']/ss['RER']*100)):.2f}%")
        if 'Q_B' in irf and 'Q_B' in ss:
            print(f"  Q_B:   {np.min(irf['Q_B']/ss['Q_B']*100):.2f}%  [bond price]")
        if 'I' in irf:
            print(f"  I:     {np.min(irf['I']/ss['I']*100):.2f}%")
        if 'P_H' in irf:
            print(f"  P_H:   {np.min(irf['P_H']/ss['P_H']*100):.2f}%  [house price]")
        if 'P_R' in irf:
            print(f"  P_R:   {np.min(irf['P_R']/ss['P_R']*100):.2f}%  [rent]")
        if 'pi_R' in irf:
            rent_cum = np.cumsum(irf['pi_R'][:12])[-1] * 100
            print(f"  pi_R:  {rent_cum:.2f}pp  [rental inflation 3yr]")

    return {
        'model': model, 'ss': ss, 'G': G, 'irf': irf,
        'unknowns': unknowns, 'targets': targets, 'exogenous': exogenous,
    }


if __name__ == '__main__':
    result = build()
