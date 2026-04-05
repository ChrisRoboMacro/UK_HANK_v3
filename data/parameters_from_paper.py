"""
UK-HANK Model Parameters
========================
All parameter values from Bank of England Macro Technical Paper No. 7 (March 2026)
"A UK-HANK model" by Albuquerque, Hill, Lavender, Lenney and Polo.

Source: /Users/beast1/RoboMacro/a-uk-hank-model.pdf

Tables referenced:
  - Table 2 (p.20): Calibrated parameters (external + internal)
  - Table 3 (p.21): Income process parameters
  - Table 5 (p.23): Dynamically estimated parameters (IRF matching)
  - Table B.1 (p.52): Earnings process target moments
"""

from dataclasses import dataclass, field
from typing import Dict


# =============================================================================
# Table 2: Externally Calibrated Parameters
# =============================================================================

@dataclass
class HouseholdParams:
    """Household preference and constraint parameters."""
    sigma: float = 1.0        # Relative risk aversion (Elminejad et al. 2022)
    nu: float = 1.5           # Labour disutility curvature (Frisch elasticity 1/nu = 0.75)
    zeta: float = 7.566       # Labour disutility scale (normalised labour supply = 1/3)
    beta: float = 0.9902      # Discount factor (internally calibrated: net fin wealth / avg income = 0.34)
    phi_H: float = 0.24       # Housing share in utility (CPI-H weight, ONS 2008-2023)


@dataclass
class TaxBenefitParams:
    """UK progressive tax and benefit system parameters."""
    T_ss: float = 0.235       # Steady-state tax-to-GDP ratio (ONS 2001-2023)
    G_B_ss: float = 0.09      # Benefits-to-GDP ratio (ONS 2001-2023)
    lam: float = 0.07         # Tax progressivity (curvature of retention function)
    lam_B0: float = 0.69      # Scale of benefit schedule
    lam_B1: float = -0.65     # Curvature of benefit schedule
    # Retention function: z = tau * (W_tilde*N + div) * e^{1-lambda} + lambda_{B,0} * e^{lambda_{B,1}} * G_B
    # See Appendix Figure B.1 for fit to ONS "Effects of taxes and benefits" data


@dataclass
class FirmParams:
    """Firm and production parameters."""
    eta_x: float = 5.45       # Goods demand elasticity (pure profit share = 18%)
    alpha_k: float = 0.16     # Capital elasticity (labour share = 68%, Barkai 2020)
    eta_w: float = 11.0       # Labour union elasticity (wage mark-up = 1.1, Chan et al. 2024)


@dataclass
class FinancialParams:
    """Financial markets and wealth parameters."""
    r_star: float = 0.0044    # Steady-state quarterly real rate (annual 1.76%, Davis et al. 2024)
    pi_star: float = 0.005    # Steady-state quarterly inflation (annual 2%)
    L_hat_ss: float = 0.24    # NS&I holdings relative to quarterly GDP (NS&I/Gilts = 0.14, ONS)
    B_star_ss: float = -0.54  # Net foreign lending / quarterly GDP (NIIP excl Gilts = -0.33, ONS)
    kappa: float = 0.13       # Share of long-term debt swapped for reserves (ONS)
    delta_bond: float = 0.019 # Share of long-term debt principal repaid per quarter (avg Gilt maturity 13yrs, Andreolli 2021)


@dataclass
class OpenEconomyParams:
    """Small open economy parameters."""
    eta_c: float = 1.43       # Consumer trade elasticity (Huo et al. 2024)
    eta_c_star: float = 1.43  # Foreign consumer trade elasticity
    eta_y: float = 0.89       # Intermediate trade elasticity (Huo et al. 2024)
    eta_y_star: float = 0.89  # Foreign intermediate trade elasticity
    alpha_c: float = 0.18     # Share of foreign goods in consumption (ONS I-O tables 1995-2020)
    alpha_y: float = 0.15     # Share of foreign goods in production (ONS I-O tables 1995-2020)
    kappa_star: float = 0.25  # Foreign ownership share of domestic long-term bonds (ONS)


@dataclass
class HousingParams:
    """Housing market parameters."""
    delta_H: float = 0.037    # Housing maintenance/depreciation (CPI-H 2008-2023)
    F: float = 0.02           # Transaction cost (fraction of P_H; stamp duty + closing costs, Halifax)
    omega_bor: float = 0.00375  # Quarterly mortgage spread (2yr 75% LTV - 2yr Gilt = 1.5% ann, 1997-2023)
    kappa_H: float = 0.95     # Max loan-to-value (95th pct FTB PSD, 2005-2023)
    kappa_y: float = 4.5      # Max loan-to-income (95th pct FTB PSD, 2005-2023)
    # Note: F = 0.02 * P_H in the paper. Transaction cost is 2% of steady-state house price.


# =============================================================================
# Table 2: Internally Calibrated Parameters
# =============================================================================

@dataclass
class InternalCalibrationParams:
    """Parameters solved to match steady-state targets."""
    # These are the RESULTS of the internal calibration (Table 2 bottom section)
    beta: float = 0.9902      # Discount factor → target: net fin wealth / avg annual gross income = 0.34 (WAS & ONS)
    P_H: float = 20.27        # House price → target: housing wealth-to-income = 6.3 (ONS 1997-2023)
    P_R: float = 0.15         # Rental price → target: renter share = 33% (EHS)
    omega_oo: float = 1.06    # Extra utility from ownership → target: mortgagor share = 54% (EHS)
    eta_move: float = 0.32    # Utility cost of moving → target: own-to-rent transition prob = 1% (EHS)
    # Note: eta_move is described as equivalent to 40% of avg quarterly household consumption


# =============================================================================
# Table 3: Income Process Parameters
# =============================================================================

@dataclass
class IncomeProcessParams:
    """
    Kaplan et al. (2018) style jump-shock income process.
    Estimated from ASHE data 1993-2016 to match moments in Table B.1.

    log(e^P_{i,t}) = (1 - J^P_{i,t}) * rho_P * log(e^P_{i,t-1}) + J^P_{i,t} * epsilon^P_{i,t}
    log(e^T_{i,t}) = (1 - J^T_{i,t}) * rho_T * log(e^T_{i,t-1}) + J^T_{i,t} * epsilon^T_{i,t}

    where J^s ~ Bernoulli(xi_s), epsilon^s ~ N(0, sigma_s^2)
    Total productivity: e = e^P + e^T (in logs: additive)
    """
    # Transitory component
    xi_T: float = 0.14        # Probability of transitory shock arriving
    rho_T: float = 0.495      # Persistence of transitory shock
    sigma_T: float = 0.464    # Std dev of transitory shock

    # Persistent component
    xi_P: float = 0.011       # Probability of permanent shock arriving (1.1%)
    rho_P: float = 0.995      # Persistence of permanent shock
    sigma_P: float = 0.825    # Std dev of permanent shock


# =============================================================================
# Table 5: Dynamically Estimated Parameters (IRF Matching)
# =============================================================================

@dataclass
class DynamicParams:
    """
    Parameters estimated via IRF matching (Section 3.3).
    Minimise inverse-variance-weighted distance to empirical IRFs from
    Albuquerque et al. (2025b) SVAR, extended with additional variables.
    """
    # External calibration
    M_CD: float = 0.85        # Cognitive discounting factor (Gabaix 2020)

    # IRF-matched parameters
    gamma_hh: float = 0.13    # Household probability of updating expectations (1-gamma = 0.87 sticky)
    gamma_f: float = 0.16     # Firm probability of updating expectations (1-gamma_f = 0.84 sticky)
    phi_T: float = 0.027      # Fiscal adjustment speed
    varphi_x: float = 64.3    # Scale of price adjustment cost (PC slope = 0.09)
    varphi_r: float = 158.0   # Scale of rental price adjustment cost (PC slope = 0.02)
    varphi_w: float = 385.0   # Scale of wage adjustment cost (PC slope = 0.03)
    rho_i: float = 0.96       # Taylor rule inertia coefficient
    phi_pi: float = 1.34      # Taylor rule inflation coefficient
    phi_y: float = 0.05       # Taylor rule output gap coefficient
    rho_m: float = 0.51       # Final goods price pass-through (import LCP/PCP mix)
    rho_x: float = 0.56       # Foreign goods price pass-through (export LCP/PCP mix)
    varphi_i: float = 20.1    # Scale of investment adjustment cost


# =============================================================================
# Table B.1: Target Moments for Income Process Estimation
# =============================================================================

@dataclass
class IncomeTargetMoments:
    """
    Empirical moments from ASHE 1993-2016 and model fit.
    Used to estimate income process parameters in Table 3.
    All based on households age 25-55 with at least min-wage equivalent earnings.
    """
    # UK data moments
    std_log_earnings_uk: float = 0.70
    std_1yr_change_log_earnings_uk: float = 0.28
    std_5yr_change_log_earnings_uk: float = 0.42
    kurtosis_1yr_change_uk: float = 24.8
    share_1yr_change_lt_10pct_uk: float = 0.60
    share_1yr_change_lt_20pct_uk: float = 0.78
    share_1yr_change_lt_50pct_uk: float = 0.94
    kurtosis_5yr_change_uk: float = 11.5

    # Model fit (discretized process)
    std_log_earnings_model: float = 0.65
    std_1yr_change_log_earnings_model: float = 0.26
    std_5yr_change_log_earnings_model: float = 0.49
    kurtosis_1yr_change_model: float = 24.8
    share_1yr_change_lt_10pct_model: float = 0.53
    share_1yr_change_lt_20pct_model: float = 0.76
    share_1yr_change_lt_50pct_model: float = 0.95
    kurtosis_5yr_change_model: float = 11.8


# =============================================================================
# Table 4: Wealth Distribution Targets
# =============================================================================

@dataclass
class WealthTargets:
    """
    Wealth distribution comparison: model vs WAS waves 1-7 (2007-2020).
    Net financial wealth = all non-pension financial wealth + debt including mortgage.
    Total wealth = financial wealth + housing wealth.
    """
    no_wealth_share_model: float = 0.21
    no_wealth_share_data: float = 0.18
    neg_fin_wealth_share_model: float = 0.49
    neg_fin_wealth_share_data: float = 0.50
    bottom_50_share_model: float = 0.16
    bottom_50_share_data: float = 0.06
    top_10_share_model: float = 0.36
    top_10_share_data: float = 0.48


# =============================================================================
# Housing Tenure States and Transitions (Table 1)
# =============================================================================

# 5 effective tenure states (3 base types with transitions tracking prior state)
TENURE_RENTER = 0
TENURE_OWN_FLAT = 1
TENURE_OWN_HOUSE = 2

# Housing sizes
H_F = 1.0   # Flat size (normalised)
H_H = 1.5   # House size (H_H > H_F, paper says "houses are bigger than flats")

# Transition-specific budget constraints from Table 1 (p.5)
# Each transition h -> h' has:
#   c_H(P_H, P_R, h'): housing expenditure
#   a_lower(h', P_H, z): borrowing limit
#
# Format: (housing_cost_formula, borrowing_limit_formula)
# All expressed in terms of: P_H (house price), P_R (rent), F (transaction cost),
#                            delta_H (maintenance), kappa_H (LTV), kappa_y (LTI), z (income)
TRANSITIONS = {
    # (from, to): description
    ('own_house', 'own_house'):  {'c_H': '-delta_H * H_H',
                                  'a_lower': 'min(a, max(-kappa_H * P_H * H_H, -kappa_y * z))'},
    ('own_flat', 'rent'):        {'c_H': 'P_H * H_F - F - P_R * H_F',
                                  'a_lower': '0'},
    ('rent', 'own_flat'):        {'c_H': '-P_H * H_F - F - delta_H * H_F',
                                  'a_lower': 'max(-kappa_H * P_H * H_F, -kappa_y * z)'},
    ('rent', 'rent'):            {'c_H': '-P_R * H_F',
                                  'a_lower': '0'},
    ('own_house', 'own_flat'):   {'c_H': 'P_H * (H_H - H_F) - 2*F - delta_H * H_H',
                                  'a_lower': 'max(-kappa_H * P_H * H_F, -kappa_y * z)'},
    ('own_flat', 'own_house'):   {'c_H': 'P_H * (H_F - H_H) - 2*F - delta_H * H_H',
                                  'a_lower': 'max(-kappa_H * P_H * H_H, -kappa_y * z)'},
    ('own_flat', 'own_flat'):    {'c_H': '-delta_H * H_F',
                                  'a_lower': 'min(a, max(-kappa_H * P_H * H_F, -kappa_y * z))'},
    ('own_house', 'rent'):       {'c_H': 'P_H * H_H - F - P_R * H_F',
                                  'a_lower': '0'},
}


# =============================================================================
# Aggregate Convenience: All Parameters
# =============================================================================

@dataclass
class UKHANKParams:
    """All UK-HANK parameters in one place."""
    household: HouseholdParams = field(default_factory=HouseholdParams)
    tax_benefit: TaxBenefitParams = field(default_factory=TaxBenefitParams)
    firm: FirmParams = field(default_factory=FirmParams)
    financial: FinancialParams = field(default_factory=FinancialParams)
    open_economy: OpenEconomyParams = field(default_factory=OpenEconomyParams)
    housing: HousingParams = field(default_factory=HousingParams)
    internal: InternalCalibrationParams = field(default_factory=InternalCalibrationParams)
    income: IncomeProcessParams = field(default_factory=IncomeProcessParams)
    dynamic: DynamicParams = field(default_factory=DynamicParams)
    income_moments: IncomeTargetMoments = field(default_factory=IncomeTargetMoments)
    wealth_targets: WealthTargets = field(default_factory=WealthTargets)

    def steady_state_interest_rate(self) -> float:
        """Quarterly nominal interest rate in steady state."""
        return self.financial.r_star + self.financial.pi_star

    def annual_real_rate(self) -> float:
        """Annualised real interest rate."""
        return (1 + self.financial.r_star)**4 - 1

    def annual_inflation(self) -> float:
        """Annualised inflation target."""
        return (1 + self.financial.pi_star)**4 - 1

    def price_phillips_slope(self) -> float:
        """Implied slope of linearised price Phillips curve."""
        return (self.firm.eta_x - 1) / self.dynamic.varphi_x

    def wage_phillips_slope(self) -> float:
        """Implied slope of linearised wage Phillips curve."""
        return (self.firm.eta_w - 1) / self.dynamic.varphi_w

    def rental_phillips_slope(self) -> float:
        """Implied slope of linearised rental Phillips curve."""
        # eta_r not explicitly stated; paper says rental sector has monopolistic competition
        # Using same approach: (eta_r - 1) / varphi_r. Paper reports PC slope = 0.02
        return 0.02


# =============================================================================
# Key Equations Reference (for implementation)
# =============================================================================

EQUATION_REFERENCE = """
Key equilibrium conditions (equation numbers from the paper):

HOUSEHOLD:
  Budget: a' + c + c_H(P_H, P_R, h') = [(1+i_A + omega_bor*1_{a<0})/(1+pi)] * a + z(e, W, N, tau, B_G, div)
  Borrowing: a' >= a_lower(h', P_H, z(.), a)
  Utility: u(c,h) = (c^{1-phi_H} * x(h)^{phi_H})^{1-sigma} / (1-sigma)
  Housing services: x(h) = H(h) * (1 + omega_oo * 1_{owner-occupier})

FIRMS:
  Eq 7: Price Phillips Curve (domestic intermediate goods)
  Eq 8: Real marginal cost
  Eqs 10-11: Capital firms FOCs (Tobin's Q)

HOUSING:
  Eq 3: Housing market equilibrium (H_bar = H_F*(s_r + s_oF) + H_H*s_oH)
  Eq 4: Rental market equilibrium (H_R = s_r * H_F)
  Eq 5: Rental Phillips Curve

WAGES:
  Eq 2: Wage Phillips Curve (Rotemberg)

GOVERNMENT:
  Eq 12: Government budget constraint
  Eq 13: Tax rate determination
  Eq 14: Central bank budget constraint

OPEN ECONOMY:
  Eqs 16-17: Export pricing (LCP/PCP mix)
  Eq 19 (4th): UIP condition

FINANCIAL:
  Eq 19 (1st): Fisher equation
  Eq 19 (3rd): Long-term bond pricing
  Eq 20: Intermediary balance sheet
  Eq 21: Intermediary profits

MARKET CLEARING:
  Eq 22: Asset market
  Eq 24: Domestically-produced goods
  Eq 25: Intermediate goods
  Eq 26: Resource constraint
  Eq 27: Trade balance

NATIONAL ACCOUNTS:
  Eq 28: Housing consumption (imputed rents)
  Eq 29: Total consumption
  Eq 30: GDP definition
"""


if __name__ == '__main__':
    params = UKHANKParams()
    print("UK-HANK Parameters loaded successfully")
    print(f"  Steady-state quarterly nominal rate: {params.steady_state_interest_rate():.4f}")
    print(f"  Annual real rate: {params.annual_real_rate()*100:.2f}%")
    print(f"  Annual inflation target: {params.annual_inflation()*100:.2f}%")
    print(f"  Price Phillips curve slope: {params.price_phillips_slope():.4f}")
    print(f"  Wage Phillips curve slope: {params.wage_phillips_slope():.4f}")
    print(f"  Income process: persistent shock prob = {params.income.xi_P:.3f}, "
          f"transitory shock prob = {params.income.xi_T:.3f}")
    print(f"  Housing: H_F = {H_F}, H_H = {H_H}")
    print(f"  Tenure transitions defined: {len(TRANSITIONS)}")
