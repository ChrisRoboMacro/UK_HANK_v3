"""
UK-HANK Model Parameters
========================

All parameters from BoE Macro Technical Paper No. 7 (March 2026).
Tables 2 (external calibration), Table 3 (income process), Table 5 (dynamic).
"""

# ── Household (Table 2) ──
SIGMA = 1.0             # Risk aversion (log utility)
PHI_H = 0.24            # Housing share in Cobb-Douglas utility
BETA = 0.989            # Discount factor (quarterly) — solved for asset mkt clearing
NU = 1.5                # Labour disutility curvature
FRISCH = 0.667          # Frisch elasticity (1/nu adjusted)
OMEGA_OO = 1.06         # Owner-occupier utility premium
ETA_MOVE = 0.32         # Moving cost (Gumbel taste shock)
ALPHA_H = 1.3           # Tenure choice taste shock scale (LogitChoice)
KAPPA_H = 0.95          # Maximum LTV ratio (PSD 2005-2023)
KAPPA_Y = 4.5           # Maximum LTI ratio (PSD 2005-2023)

# ── Firms (Table 2) ──
ETA_X = 5.45            # Goods demand elasticity => markup mu = eta_x/(eta_x-1) = 1.225
MU = 1.225              # Price markup
ALPHA_K = 0.16          # Capital share in production (Barkai 2020)
ETA_W = 11.0            # Labour union elasticity => wage markup mu_w = 1.1
MU_W = 1.1              # Wage markup
DELTA = 0.025           # Capital depreciation (quarterly)
EPS_I = 1 / 20.1        # Investment adjustment cost
VARPHI_X = 64.3         # Price adjustment cost (Rotemberg) => kappa = (eta_x-1)/varphi_x = 0.09
KAPPA = 0.09            # Price Phillips Curve slope
VARPHI_W = 385.0        # Wage adjustment cost => kappa_w = (eta_w-1)/varphi_w = 0.026
KAPPA_W = 0.026         # Wage Phillips Curve slope
VARPHI_R = 158.0        # Rental adjustment cost => kappa_r = (eta_r-1)/varphi_r ≈ 0.02
KAPPA_R = 0.02          # Rental Phillips Curve slope

# ── Financial markets (Table 2) ──
R_STAR = 0.0044         # Steady-state real rate (1.76% annual, Davis et al. 2024)
PI_STAR = 0.005         # Inflation target (2% annual, BoE mandate)
DELTA_B = 0.019         # Bond principal repayment rate (avg Gilt maturity 13yr)
CB_RESERVES_SHARE = 0.13  # CB reserves share of debt (ONS)
OMEGA_BOR = 0.00375     # Mortgage spread (1.5% annual, BoE Bankstats)

# ── Fiscal (Table 2) ──
T_SS = 0.235            # Tax-to-GDP ratio (ONS 2001-2023)
G_B_SS = 0.09           # Benefits-to-GDP ratio (ONS 2001-2023)
B_SS = 5.6              # Government debt stock (ONS)
TAX_PROGRESSIVITY = 0.07  # Tax progressivity parameter
PHI_G = 0.220           # Fiscal debt feedback (calibrated for GDP/CPI match)

# ── Housing (Table 2) ──
DELTA_H = 0.037         # Housing depreciation/maintenance (ONS CPI-H)
F_COST_RATE = 0.02      # Transaction cost as fraction of P_H (Halifax)
H_F = 1.0               # Flat size (normalised)
H_H = 1.5               # House size
P_H_SS = 3.6            # Steady-state house price (housing wealth/income = 6.3)
P_R_SS = 0.15           # Steady-state rent (user cost parity)

# ── Open economy (Table 2) ──
ALPHA_C = 0.18          # Import share of consumption (ONS I-O 1995-2020)
ALPHA_Y = 0.15          # Import share of production (ONS I-O 1995-2020)
ETA_C = 1.43            # Consumer trade elasticity (Huo et al. 2024)
RHO_M = 0.51            # Import price pass-through

# ── Income process (Table 3, ASHE 1993-2016) ──
XI_T = 0.14             # Transitory shock arrival probability
RHO_T = 0.495           # Transitory shock persistence
SIGMA_T = 0.464         # Transitory shock std dev
XI_P = 0.011            # Persistent shock arrival probability
RHO_P = 0.995           # Persistent shock persistence
SIGMA_P = 0.825         # Persistent shock std dev

# ── Dynamic parameters (Table 5, IRF-matched) ──
M_CD = 0.85             # Cognitive discounting (Gabaix 2020)
GAMMA_HH = 0.984        # Household sticky expectations (post-GE)
GAMMA_F = 0.82          # Firm sticky expectations (post-GE)
RHO_I = 0.96            # Taylor rule inertia
PHI_PI = 1.34           # Taylor rule inflation coefficient
PHI_Y = 0.05            # Taylor rule output gap coefficient
OMEGA_RENT = 0.07       # Rental expenditure share in CPI

# ── CPI composition ──
# pi_cpi = (1 - alpha_c - omega_rent) * pi + omega_rent * pi_R + alpha_c * pi_import
# Weights: domestic 75%, rental 7%, import 18%

# ── 7 General Equilibrium Unknowns ──
# In GE, nothing is determined in isolation. The SSJ solver finds these
# 7 values where all markets clear simultaneously.
#
#   Unknown   Symbol   Clearing condition        Plain English
#   ─────────────────────────────────────────────────────────────────────
#   1. r       r        asset_mkt = 0            Bond supply = bond demand
#   2. w       w        wnkpc = 0                Wage set by unions matches labour demand
#   3. Y       Y        fisher_res = 0           Nominal rate, real rate, inflation consistent
#   4. RER     RER      uip_res = 0              Pound adjusts until UK/foreign bonds equal return
#   5. i       i        taylor_res = 0           BoE sets rates per Taylor rule
#   6. P_R     P_R      rental_pc_res = 0        Landlords adjust rents to housing demand
#   7. B       B        budget_res = 0           Government spending + debt service balances
#
# Exogenous (given, not solved): r* (natural rate), Z (productivity),
#   epsilon_r (monetary policy shock — one-shot impulse).
#
# The SSJ method computes Jacobians of all 18 blocks w.r.t. these 7
# unknowns (T=300 quarters), then solves the linear system for IRFs.

# ── Steady-state values ──
SS = {
    'r': R_STAR,
    'rstar': R_STAR,
    'pi': 0.0,
    'i': R_STAR,
    'Y': 1.0,
    'Z': 1.0,
    'N': 1.0,
    'Ns': 1.0,
    'K': 5.4,
    'B': B_SS,
    'B_ss': B_SS,
    'P_H': P_H_SS,
    'P_R': P_R_SS,
    'Q_B': 1.0,
    'RER': 1.0,
    'beta': BETA,
}

# ── Tenure-specific housing costs (Table 1) ──
# c_H depends on tenure status:
TENURE_COSTS = {
    'renter': 'P_R * H_F',                                  # = 0.15
    'mortgagor_flat': 'delta_H * H_F + (r + omega_bor) * P_H * H_F',  # ≈ 0.066
    'outright_flat': 'delta_H * H_F',                        # = 0.037
    'mortgagor_house': 'delta_H * H_H + (r + omega_bor) * P_H * H_H', # ≈ 0.100
    'outright_house': 'delta_H * H_H',                       # = 0.056
}

# ── All adjustable parameters with bounds ──
PARAM_BOUNDS = {
    'shock_size':  (-5.0, 5.0,  'Shock size (pp, negative = rate cut)'),
    'sigma':       (0.5,  5.0,  'Risk aversion (CRRA)'),
    'beta':        (0.9,  0.999, 'Discount factor'),
    'phi_pi':      (1.0,  3.0,  'Taylor rule inflation coefficient'),
    'rho_i':       (0.0,  0.99, 'Taylor rule inertia'),
    'kappa':       (0.01, 0.5,  'Price Phillips Curve slope'),
    'kappaw':      (0.001, 0.5, 'Wage Phillips Curve slope'),
    'mu':          (1.05, 2.0,  'Price markup'),
    'phi_G':       (0.0,  1.0,  'Fiscal debt feedback'),
    'delta_b':     (0.001, 0.1, 'Bond principal repayment'),
    'alpha_c':     (0.0,  0.5,  'Import share of consumption'),
    'eta_c':       (0.5,  3.0,  'Trade elasticity'),
    'kappa_r':     (0.001, 0.5, 'Rental Phillips Curve slope'),
    'omega_rent':  (0.0,  0.3,  'Rental share in CPI'),
    'rho_M':       (0.0,  1.0,  'Import price pass-through'),
    'M_CD':        (0.5,  1.0,  'Cognitive discounting'),
    'phi_H':       (0.0,  0.5,  'Housing utility share'),
    'delta_H':     (0.0,  0.1,  'Housing depreciation'),
}
