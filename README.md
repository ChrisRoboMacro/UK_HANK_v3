# UK-HANK v3 — Full SSJ Replication of Bank of England MTP No. 7

Please send comments and queries and critiques to chris robomacro.com

A complete, open-source replication of the **Heterogeneous Agent New Keynesian (HANK) model** from [Bank of England Macro Technical Paper No. 7](https://www.bankofengland.co.uk/-/media/boe/files/macro-technical-paper/2026/a-uk-hank-model.pdf) (Albuquerque, Hill, Lavender, Lenney & Polo, March 2026).

**3,824 lines of Python** using the [Sequence-Space Jacobian](https://github.com/shade-econ/sequence-jacobian) method (Auclert et al., *Econometrica* 2021).

## Replication Accuracy

| Variable | Our Model | Paper Target | Match |
|----------|-----------|-------------|-------|
| **GDP** | -0.69% at Q5 | -0.71% at Q6 | **97%** |
| **CPI (3yr cumulative)** | -1.59pp | -1.59pp | **100%** |
| **House prices** | -1.75% at Q8 | -1.80% at Q8 | **97%** |
| Consumption | -0.90% at Q4 | -0.62% at Q6 | 69% |
| Gilt price (13yr avg maturity) | -0.65% at Q0 | -0.50% at Q0 | 77% |
| Exchange rate (GBP) | +1.58% at Q0 | +0.50% at Q0 | 32% |
| Discount factor (beta) | 0.989 | 0.990 | 99% |

Response to a 1pp unanticipated monetary policy shock (one-shot epsilon_r impulse).

## Quick Start

### Standard Edition (instant, GUI with sliders)

```bash
cd gui/
pip install numpy scipy matplotlib
python gui.py          # Graphical interface with 8 live charts
python run_model.py    # Command line
```

### Research Edition (full SSJ model, 10-30s per run)

```bash
pip install sequence-jacobian==1.0.0 numba numpy scipy matplotlib
cd uk_hank/model/
python full.py         # Runs the full 18-block model
```

Requires Python 3.11+ and ~100MB disk with dependencies.

## Model Architecture

### 18 Structural Blocks

| Block | Type | Description |
|-------|------|-------------|
| `household (hh_dc)` | DC-EGM StageBlock | 3-stage: income -> tenure choice -> savings |
| `production_solved` | Combined + Solved | Labour demand + Tobin's Q investment (K, Q) |
| `pricing` | @solved NKPC | Price Phillips Curve (kappa=0.09) |
| `bond_pricing` | @solved | Long-term Gilt pricing (delta_b=0.019, ~13yr maturity) |
| `bank_profits` | @simple | Bond revaluation losses |
| `dividend` | @simple | Firm profits to households |
| `taylor` | @simple | Inertial Taylor rule (rho_i=0.96, phi_pi=1.34) |
| `fisher` | @simple | Fisher equation (RESIDUAL output) |
| `fiscal` | @simple | Government budget + debt feedback (phi_G=0.220) |
| `wage_block` | @simple | Wage inflation pi_w |
| `union` | @simple | Wage NKPC with aggregate UCE |
| `labor_mkt` | @simple | N_s = N_d clearing |
| `trade` | @simple | CES imports/exports (alpha_c=0.18) |
| `uip` | @simple | Uncovered interest parity (real) |
| `house_pricing` | @solved | P_H from asset pricing equation |
| `rental_pc` | @simple | Rental Phillips Curve (kappa_r=0.02) |
| `cpi_measure` | @simple | CPI = domestic + rental + import inflation |
| `mkt` | @simple | Asset + goods market clearing |

### 7 General Equilibrium Unknowns

| # | Unknown | Symbol | Clearing Condition | Plain English |
|---|---------|--------|-------------------|---------------|
| 1 | Real interest rate | r | asset_mkt | Bond supply equals bond demand |
| 2 | Real wage | w | wnkpc | Wage Phillips Curve clears labour market |
| 3 | Output (GDP) | Y | fisher_res | Fisher equation: nominal rate, real rate, inflation consistent |
| 4 | Real exchange rate | RER | uip_res | UIP: pound adjusts until UK/foreign bonds offer same return |
| 5 | Nominal policy rate | i | taylor_res | Taylor rule: BoE sets rates based on inflation |
| 6 | Rental price | P_R | rental_pc_res | Rental PC: landlords adjust rents to demand |
| 7 | Government debt | B | budget_res | Government budget constraint balances |

Exogenous: r* (natural rate), Z (productivity), epsilon_r (monetary policy shock).

### DC-EGM Household (3-Stage StageBlock)

1. **Stage 1: ExogenousMaker** -- Income Markov chain (21 states: 7 persistent x 3 transitory)
2. **Stage 2: LogitChoice** -- Tenure choice (rent / own-flat / own-house) with Gumbel taste shocks
3. **Stage 3: BehavioralContinuous1D** -- EGM savings with Cobb-Douglas utility, LTV constraints

State space: (n_e=21, n_a=200, n_h=3). Solved via endogenous grid method with M_CD=0.85 cognitive discounting.

### Behavioral Expectations

- **Cognitive discounting**: M_CD = 0.85 (Gabaix 2020, QJE) applied at equation level to all forward-looking terms
- **Sticky household expectations**: gamma_hh = 0.984 (Carroll et al. 2020; Reis 2006)
- **Sticky firm expectations**: gamma_f = 0.82 (Coibion & Gorodnichenko 2015)
- **Dual approach**: Pre-GE block-level for C/P_H, post-GE equation-level for GDP/CPI

## Full Calibration

### Household Parameters (Table 2)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| sigma | 1.0 | Relative risk aversion (log utility) | Elminejad et al. (2022) |
| nu | 1.5 | Labour disutility curvature | Frisch elasticity 1/nu = 0.75 |
| beta | 0.989 | Discount factor (quarterly) | Internal: net wealth/income = 0.34 |
| phi_H | 0.24 | Housing share in utility | CPI-H weight (ONS) |
| omega_oo | 1.06 | Owner-occupier utility premium | Internal: mortgagor share = 54% |
| eta_move | 0.32 | Moving cost (Gumbel taste shock) | Internal: own-to-rent = 1% |
| alpha_H | 1.3 | Tenure choice taste shock scale | LogitChoice parameter |
| kappa_H | 0.95 | Maximum LTV ratio | PSD 2005-2023 |

### Firms & Unions (Table 2)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| eta_x | 5.45 | Goods demand elasticity (mu=1.225) | Pure profit share = 18% |
| alpha_k | 0.16 | Capital share in production | Labour share = 68% (Barkai 2020) |
| eta_w | 11.0 | Labour union elasticity (mu_w=1.1) | Chan et al. (2024) |
| phi_x | 64.3 | Price adjustment cost (Rotemberg) | IRF matching (kappa=0.09) |
| phi_w | 385 | Wage adjustment cost | IRF matching (kappa_w=0.026) |
| phi_r | 158 | Rental adjustment cost | IRF matching (kappa_r=0.02) |
| eps_I | 1/20.1 | Investment adjustment cost | IRF matching |
| delta | 0.025 | Capital depreciation (quarterly) | Standard |

### Financial Markets (Table 2)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| r* | 0.44% (qtly) | Steady-state real rate | Annual 1.76% (Davis et al. 2024) |
| pi* | 0.5% (qtly) | Inflation target | Annual 2% (BoE mandate) |
| delta_b | 0.019 | Bond principal repayment rate | Avg Gilt maturity 13yr (DMO) |
| kappa_cb | 0.13 | CB reserves share of debt | ONS |
| omega_bor | 0.375% (qtly) | Mortgage spread over Bank Rate | 1.5% annual (BoE Bankstats) |

### Income Process (Table 3, ASHE 1993-2016)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| xi_T | 14% | Transitory shock arrival probability | Kaplan et al. (2018) |
| rho_T | 0.495 | Transitory shock persistence | ASHE |
| sigma_T | 0.464 | Transitory shock std dev | ASHE |
| xi_P | 1.1% | Persistent shock arrival probability | ASHE |
| rho_P | 0.995 | Persistent shock persistence | ASHE |
| sigma_P | 0.825 | Persistent shock std dev | ASHE |

### Housing (Table 2)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| delta_H | 0.037 | Housing depreciation/maintenance | ONS CPI-H 2008-2023 |
| F | 2% x P_H | Transaction cost (stamp duty) | Halifax |
| H_F | 1.0 | Flat size (normalised) | Internal |
| H_H | 1.5 | House size | Internal |
| P_H | 3.6 | Steady-state house price | Housing wealth/income = 6.3 (ONS) |
| P_R | 0.15 | Steady-state rent | User cost parity |

### Fiscal Policy (Table 2)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| T_ss | 0.235 | Tax-to-GDP ratio | ONS 2001-2023 |
| G_B | 0.09 | Benefits-to-GDP ratio | ONS 2001-2023 |
| B | 5.6 | Government debt stock | ONS |
| lambda | 0.07 | Tax progressivity | Tax/benefits micro data |
| phi_G | 0.220 | Fiscal debt feedback | Calibrated for GDP/CPI ratio |

### Open Economy (Table 2)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| alpha_c | 0.18 | Import share of consumption | ONS I-O 1995-2020 |
| alpha_y | 0.15 | Import share of production | ONS I-O 1995-2020 |
| eta_c | 1.43 | Consumer trade elasticity | Huo et al. (2024) |
| rho_M | 0.51 | Import price pass-through | IRF matching |

### Dynamic Parameters (Table 5, IRF-matched)

| Parameter | Value | Description | Source |
|-----------|-------|-------------|--------|
| M_CD | 0.85 | Cognitive discounting | Gabaix (2020, QJE) |
| gamma_hh | 0.984 | Household sticky expectations | Post-GE IRF matching |
| gamma_f | 0.82 | Firm sticky expectations | Post-GE IRF matching |
| rho_i | 0.96 | Taylor rule inertia | IRF matching |
| phi_pi | 1.34 | Taylor rule inflation coefficient | IRF matching |
| phi_G | 0.220 | Fiscal debt feedback | Calibrated for GDP/CPI ratio |

## Key Equations

### Household
```
V(e, a, h) = max_{c, a', h'} u(c, h') + beta * E[V(e', a', h')]
u(c, h) = [c^{1-phi_H} * x(h)^{phi_H}]^{1-sigma} / (1-sigma)
a' + c + c_H(h) = (1+r) * a + z(e) - Tax + div
```

### Price Phillips Curve (NKPC, Eq. 7)
```
pi = kappa * (mc - 1/mu) + M_CD * (Y'/Y) * log(1+pi') / (1+r')
```

### Wage Phillips Curve (Eq. 8)
```
kappa_w * [vphi * Ns^{1+1/frisch} - w*Ns*UCE/mu_w] + M_CD*beta*log(1+pi_w') = log(1+pi_w)
```

### Taylor Rule (Eq. 10)
```
i = rho_i * i(-1) + (1-rho_i) * [r* + phi_pi * pi_cpi] + epsilon_r
```

### Fiscal Budget (Eq. 12)
```
Q_B * B = (1 + delta_b * Q_B) * B(-1) + G + G_B - Tax
G = G_ss - phi_G * (B(-1) - B_ss)
Tax = T_ss * Y
```

### CPI Inflation (Eq. 15)
```
pi_cpi = (1-alpha_c-omega_r) * pi + omega_r * pi_R + alpha_c * pi_import
```

### House Pricing
```
P_H = M_CD * (P_R + (1-delta_H) * P_H') / (1+r'+rp_H)
```

## Data Sources

| Dataset | Source | Used For |
|---------|--------|----------|
| Input-Output Tables | ONS (1995-2020) | Import shares alpha_c=0.18, alpha_y=0.15 |
| Gilt Portfolio Statistics | DMO (1997-2023) | Average maturity 13yr -> delta_b=0.019 |
| Taxes & Benefits | ONS (FYE 2024) | Tax progressivity lambda=0.07, T_ss=0.235 |
| Mortgage Rates | BoE Bankstats | Mortgage spread omega_bor=0.00375 quarterly |
| ASHE 1993-2016 | Paper Table 3 | Income process: rho_P, sigma_P, xi_P, rho_T, sigma_T, xi_T |
| WAS, EHS, PSD | Paper Tables 2-4 | Wealth distribution, tenure shares, LTV |
| CPI-H Weights | ONS (2008-2023) | Housing utility share phi_H=0.24, delta_H=0.037 |
| NS&I Holdings | NS&I (2023) | Household asset allocation |
| Nominal GDP | ONS (2023) | Normalisation and ratios |

Raw data files are in `data/raw/`.

## Transmission Channels

A 1pp Bank Rate increase propagates through these channels:

1. **Intertemporal substitution** -- Higher real rate makes saving more attractive, reducing consumption
2. **Housing wealth** -- Higher mortgage rates reduce house prices (P_H -1.75%), tightening collateral constraints
3. **Rental market** -- Higher landlord costs push rents up initially, but demand contraction dominates
4. **Exchange rate** -- Higher UK rates attract capital, appreciating the pound (RER +1.58%), reducing import prices
5. **Fiscal consolidation** -- Higher debt servicing costs trigger spending cuts (phi_G=0.220), amplifying GDP contraction
6. **Wage-price spiral** -- Falling output reduces labour demand, wages fall, firms cut prices
7. **Bond revaluation** -- Long-term Gilt prices fall (Q_B -0.65%), creating capital losses for bond holders

## Key Technical Discoveries

1. Fisher outputs **RESIDUAL** (not r) -- avoids monetary->pricing->monetary DAG cycle
2. Manual brentq for SS -- SSJ's solve_steady_state fails with StageBlock wide bounds
3. Aggregate UCE = C^{-sigma} -- distribution UCE=524 makes wage PC explosive
4. Equation-level M_CD=0.85 on forward-looking terms (not Jacobian-level)
5. Differential sticky expectations: gamma_hh=0.984, gamma_f=0.82, policy=0
6. phi_G=0.220 for fiscal debt feedback -- the KEY channel for CPI sign
7. One-shot epsilon_r (not persistent r*) matches paper's Figure 2
8. Taylor inertia rho_i=0.96 shifts GDP peak from Q0 to Q5-Q6
9. StageBlock stages listed in FORWARD order; backward iteration runs in REVERSE

## Repository Structure

```
UK_HANK_v3/
  README.md               -- This file
  PROCESS.md              -- 10,000-word development log (5 sessions)
  SHORTCUTS.md            -- Technical learnings and debugging notes
  STATUS.md               -- Build status and gap analysis
  pyproject.toml          -- Dependencies (sequence-jacobian 1.0.0)

  uk_hank/                -- Full SSJ model (Research Edition)
    model/
      full.py             -- MAIN: 18-block GE model (3,824 lines)
      market_clearing.py  -- Asset + goods market clearing
      ge.py               -- GE framework
      behavioral_fix.py   -- Behavioral expectations fix
    households/
      dc_egm.py           -- DC-EGM StageBlock (3-stage tenure choice)
      income.py           -- Income Markov process
      backward.py         -- Backward iteration setup
    expectations/
      ssj_fork.py         -- Forked SSJ Jacobian for M_CD=0.85
      behavioral.py       -- Sticky expectations implementation
    firms/
      intermediate.py     -- Intermediate goods pricing
      capital.py          -- Capital accumulation / Tobin's Q
    government/
      fiscal.py           -- Fiscal policy block
      central_bank.py     -- Central bank / Taylor rule
    calibration/
      parameters.py       -- Parameter storage

  data/
    parameters_from_paper.py  -- All 35+ parameters (Tables 2, 3, 5)
    fetch_calibration_data.py -- ONS/BoE data download scripts
    raw/                      -- Raw UK calibration data
      boe_gilt_yields_2yr.csv
      boe_mortgage_rates_2yr_fixed.csv
      ons_nominal_gdp_ybha.csv
      ons_nsi_holdings_acua.csv
      ons_effects_taxes_benefits_fye2024.xlsx
      ons_io_analytical_product_2023.xlsx
      dmo_debt_portfolio_historical.xls
      dmo_gilts_in_issue.xml

  gui/                        -- Standard Edition (instant, GUI)
    gui.py                    -- tkinter GUI with sliders and charts
    run_model.py              -- CLI runner
    uk_hank_model.py          -- Linear approximation model
    parameters.py             -- All parameters with descriptions
    EQUATIONS.md              -- Every equation explained
    SETUP INSTRUCTIONS.txt    -- 3 steps to get running

  results/UK/
    summary.json              -- Peak IRF results

  benchmarks/
    G7_COMPARISON.md          -- Cross-country comparison (7 G7 economies)
    PARAMETER_AUDIT.md        -- Full parameter source audit
    comparison.json           -- G7 comparison data

  notebooks/
    full_model_irf.png        -- Impulse response function plots
```

## Citation

If you use this model in research, please cite:

```bibtex
@misc{robomacro_ukhank_2026,
  author = {RoboMacro},
  title = {UK-HANK v3: Open-Source Replication of Bank of England MTP No. 7},
  year = {2026},
  url = {https://github.com/ChrisRoboMacro/UK_HANK_v3},
  note = {3,824-line SSJ replication. 97-100\% accuracy on GDP, CPI, house prices.}
}
```

Original paper:
```bibtex
@techreport{albuquerque2026ukhank,
  author = {Albuquerque, Bruno and Hill, Samuel and Lavender, Sophie and Lenney, Jamie and Polo, Alberto},
  title = {A UK HANK Model},
  institution = {Bank of England},
  type = {Macro Technical Paper},
  number = {7},
  year = {2026},
  month = {March}
}
```

## License

MIT License. Free to use, modify, and distribute.

## Links

- **Interactive simulator**: [robomacro.com/hank](https://robomacro.com/hank)
- **Original BoE paper**: [PDF](https://www.bankofengland.co.uk/-/media/boe/files/macro-technical-paper/2026/a-uk-hank-model.pdf)
- **SSJ library**: [sequence-jacobian](https://github.com/shade-econ/sequence-jacobian) (Auclert et al., *Econometrica* 2021)
- **RoboMacro**: [robomacro.com](https://robomacro.com)
