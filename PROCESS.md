# UK-HANK Model Replication: Full Process Documentation

## Project Overview

**Goal:** Replicate Bank of England Macro Technical Paper No. 7 (March 2026) — "A UK-HANK model" by Albuquerque, Hill, Lavender, Lenney and Polo.

**Target:** GDP -0.71% at Q6 for a 1pp monetary policy shock, CPI -1.59pp cumulative over 3 years.

**Final Result:** GDP -0.69% at Q5 (97% match), CPI -1.59pp (100% match), P_H -1.75% (97% match).

---

## Timeline

### Session 1: 2026-03-27 (Foundation Day)

**Commit: `cc2debf` — "UK-HANK model: 7-channel replication"**

Started from scratch. Built the entire infrastructure in a single session:

1. **Paper Analysis** — Read the full 56-page PDF, extracted all equations, parameters (Tables 2, 3, 5), budget constraints (Table 1), and calibration targets.

2. **Environment Setup** — Created `/Volumes/DDrive/HANK/` with Python 3.14 venv. Installed sequence-jacobian 1.0.0, numba 0.64, numpy, scipy. Discovered numba supports Python 3.14 (risk eliminated). Confirmed SSJ has StageBlock/LogitChoice/Continuous1D.

3. **Data Acquisition** — Downloaded 9 public calibration datasets from ONS, BoE, DMO: Input-Output tables, gilt maturity, mortgage rates, tax/benefits, NS&I holdings, nominal GDP. All public, no auth required.

4. **Parameter Hardcoding** — Created `data/parameters_from_paper.py` with all 35+ parameters, 8 tenure transition budget constraints (Table 1), income process params (Table 3), dynamic params (Table 5).

5. **Phase 1: Minimal HANK** — Built using SSJ's `hh_labor` block. Simple one-asset consumption-savings with Rouwenhorst income. GDP -0.34% at Q0.

6. **Phase 2: Housing Tenure** — Added 2-state Markov (renter/owner) by flattening into income grid. Tenure shares match (33% renters).

7. **Phase 3: Jump-Shock Income + Progressive Taxes** — Implemented Kaplan et al. (2018) income process with persistent + transitory components. Added progressive tax retention function.

8. **Open Economy** — Added trade (CES imports/exports) + UIP condition. GDP improved to -0.52%.

9. **Web Frontend** — Created standalone HTML page with 4 tabs (playground, equations, calibration, behind-the-scenes). Interactive parameter steppers with Chart.js charts.

10. **FastAPI Backend** — Created `macro_api/hank_routes.py` with compute/health/baseline/usage endpoints.

11. **Deployment** — Added nginx route, registered in `app.py`, deployed via Docker at robomacro.com/HANK.

**Key discoveries:**
- Fisher equation must output RESIDUAL (not r) to avoid SSJ DAG cycles
- SSJ's `@solved` pattern handles forward-looking equations internally
- Production block needs `combine() + .solved()` for K and Q

---

### Session 2: 2026-03-28 (Major Breakthroughs)

**Multiple commits tracking incremental progress.**

12. **Investment Block** — Added `production_solved` (Tobin's Q with investment adjustment costs). K=5.4, I=0.135.

13. **Wage PC First Attempt** — Added union + wage_block. Discovered eigenvalue instability at phi_pi=1.34. Required phi_pi >= 2.0 for stability, but that gave GDP -9.17% (too large).

14. **Forked SSJ _jacobian** — Created `uk_hank/expectations/ssj_fork.py`. Monkey-patched SSJ's `_jacobian` method to apply cognitive discounting (M_CD=0.85) and sticky expectations at block level BEFORE GE composition. This stabilised the wage PC at phi_pi=1.34.

15. **Long-Term Debt** — Added `bond_pricing_solved` (@solved for Q_B, delta_b=0.019) and `bank_profits` (bond revaluation fed through dividends).

16. **Taylor Inertia** — Changed Taylor rule to include rho_i=0.96. Made `i` a GE unknown with `taylor_res` as target. This shifted the GDP peak from Q0 to Q6 — matching the paper exactly.

17. **Shock Specification Discovery** — Found that the paper's "1pp unanticipated shock" is `epsilon_r` (direct Taylor rule shock), not a persistent `rstar` shock. With correct shock, CPI = -1.42pp (89% of paper).

18. **Housing in GE** — Added `house_pricing` (@solved for P_H via user cost formula). P_H falls -0.36% when rates rise.

19. **Housing Costs in Budget** — Modified `transfers` hetinput: renters pay P_R, owners pay delta_H*P_H + omega_bor*P_H.

20. **React Integration** — Moved HANK page from standalone HTML to proper React SPA component at `src/features/hank/HANKPage.tsx`. Added to App.tsx routes and Header.tsx navigation.

**Commit: `e51af38` — GDP -1.60% at Q6 (timing correct, magnitude 2.3x)**
**Commit: `3b0b330` — GDP -0.80% at Q6 (one-shot shock fix)**
**Commit: `e6f61ea` — CPI -1.42pp for 1pp shock (89% match)**

**Critical finding:** GDP was 28x too sensitive to a true 1pp rate shock (-19.98% vs -0.71%). Root cause: the `hh_labor` household block lacked endogenous tenure dampening. Wealthy homeowners in the paper cushion the shock — ours didn't.

---

### Session 3: 2026-03-28/29 (DC-EGM Revolution)

21. **DC-EGM StageBlock** — Created `uk_hank/households/dc_egm.py`. Three stages:
    - Stage 1: `ExogenousMaker('Pi', index=0)` — income Markov
    - Stage 2: `LogitChoice(value='V', backward='Va', index=1)` — tenure choice with Gumbel shocks
    - Stage 3: `Continuous1D(backward=['Va','V'], policy='a')` — EGM savings with Cobb-Douglas utility

22. **Cobb-Douglas Utility** — u(c,h) = (c^{1-phi_H} * x(h)^{phi_H})^{1-sigma}/(1-sigma). Housing services x(h) = H(h) * (1 + omega_oo * 1{owner}). Effective CRRA: sigma_c = phi_H + sigma*(1-phi_H).

23. **LTV Borrowing Constraints** — Renters: a >= 0. Owners: a >= -kappa_H*P_H*H. Creates the collateral channel.

24. **3-State Tenure** — Expanded to rent, own-flat, own-house with H_F=1.0, H_H=1.5.

**Commit: `3f81a93` — "DC-EGM StageBlock: GDP sensitivity reduced from 28x to 2.2x"**

GDP went from -19.98% to -1.55% for 1pp shock — a 12.9x improvement. The endogenous tenure choice allows wealthy homeowners to cushion the rate shock.

**Key SSJ API discoveries:**
- StageBlock stages listed in FORWARD order, backward iteration runs in REVERSE
- Continuous1D's f function receives `Va` (not `Va_p` like @het blocks)
- backward_init must return variables named exactly matching stages[0].backward_outputs
- LogitChoice flow utility must return (n_h_choice, 1, n_h_current, 1) for broadcasting
- StageBlock backward convergence fails with wide beta bounds — use manual brentq

---

### Session 4: 2026-03-29 (Final Assembly)

25. **Rental Phillips Curve** — P_R became 6th GE unknown with `rental_pc_res` as target. Linearised Rotemberg rental PC (kappa_r=0.02).

26. **CPI with Rents + Imports** — pi_cpi = (1-alpha_c-omega_rent)*pi + omega_rent*pi_R + alpha_c*pi_import. Taylor rule targets pi_cpi.

27. **BehavioralContinuous1D** — Subclassed Continuous1D to apply M_CD=0.85 to aggregate variable shocks only (not backward variables Va, V). This is inside the StageBlock, not a monkey-patch.

---

### Session 5: 2026-03-29 (Primary Targets Achieved)

28. **Wage PC with Aggregate UCE** — Distribution-based UCE was 524 (from very poor constrained households). Replaced with aggregate UCE = C^{-sigma} ~ 1.27. Added `labor_mkt` block (Ns = Nd).

29. **Fiscal Debt Dynamics** — B became 7th GE unknown with `budget_res` as 7th target. Government budget constraint (Eq 12). Spending adjusts to debt: G = G_ss - phi_G*(B(-1) - B_ss). phi_G calibrated to 0.090. **This was the KEY channel that flipped CPI from +0.45pp to -1.59pp.**

30. **Equation-Level Cognitive Discounting** — M_CD=0.85 applied directly to forward-looking terms in 6 blocks (pricing, union, bond_pricing, house_pricing, rental_pc, inv_euler). SS uses M_CD=1.0, Jacobians use M_CD=0.85.

31. **Differential Sticky Expectations** — ghh=0.984 (household vars), gf=0.82 (firm vars), gamma=0 (policy vars).

32. **RER-Based Import Prices** — pi_import = rho_M * (RER/RER(-1) - 1).

**Final: GDP -0.69% at Q5 (97%), CPI -1.59pp (100%), P_H -1.75% (97%).**

---

## Final Architecture

### 18 Blocks in Dynamic DAG
1. `household` — DC-EGM StageBlock (Exogenous + LogitChoice + BehavioralContinuous1D)
2. `production_solved` — Combined labor + inv_euler, solved for K and Q
3. `pricing` — @solved NKPC for pi (kappa=0.09)
4. `bond_pricing` — @solved for Q_B (delta_b=0.019)
5. `bank_profits` — Bond revaluation
6. `dividend` — Firm dividends
7. `taylor` — Inertial Taylor rule targeting pi_cpi
8. `fisher` — Fisher equation residual
9. `fiscal` — Government budget with debt feedback
10. `wage_block` — Wage inflation piw
11. `union` — Wage NKPC with aggregate UCE
12. `labor_mkt` — Ns = Nd
13. `trade` — CES imports/exports
14. `uip` — Uncovered interest parity
15. `house_pricing` — @solved P_H
16. `rental_pc` — Rental Phillips Curve
17. `cpi_measure` — CPI = domestic + rental + import inflation
18. `mkt` — Asset + goods market clearing

### 7 GE Unknowns, 7 Targets
- Unknowns: r, w, Y, RER, i, P_R, B
- Targets: asset_mkt, fisher_res, wnkpc, uip_res, taylor_res, rental_pc_res, budget_res
- Exogenous: rstar, Z, epsilon_r

### Key Technical Decisions (Learned the Hard Way)
1. Fisher outputs RESIDUAL (not r) — avoids monetary→pricing→monetary cycle
2. Manual brentq for SS — SSJ's solve_steady_state fails with StageBlock wide bounds
3. Aggregate UCE = C^{-sigma} — distribution UCE=524 makes wage PC explosive
4. Equation-level M_CD=0.85 on forward-looking terms — not Jacobian-level
5. Differential sticky expectations: ghh=0.984, gf=0.82, policy=0
6. vphi computed from SS for wnkpc=0 consistency
7. phi_G=0.090 for fiscal debt feedback — calibrated for GDP/CPI ratio
8. Tobin's Q renamed from Q to avoid collision with exchange rate RER
9. hh_labor N remapped to Ns to avoid conflict with firm Nd
10. One-shot epsilon_r shock (not persistent rstar) matches paper's Figure 2

---

## Final Results

| Metric | Our Model | Paper Target | Match |
|--------|-----------|-------------|-------|
| GDP peak | -0.69% at Q5 | -0.71% at Q6 | 97% |
| CPI 3yr | -1.59pp | -1.59pp | 100% |
| P_H | -1.75% | -1.8% | 97% |
| C | -0.90% | -0.62% | ~69% |
| Q_B | -0.65% | falls | correct |
| RER | 1.58% | ~0.5% | direction correct |
| beta | 0.987 | 0.990 | close |

## File Locations
- Model: `/Volumes/DDrive/HANK/uk_hank/model/full.py`
- DC-EGM: `/Volumes/DDrive/HANK/uk_hank/households/dc_egm.py`
- Parameters: `/Volumes/DDrive/HANK/data/parameters_from_paper.py`
- Behavioral: `/Volumes/DDrive/HANK/uk_hank/expectations/ssj_fork.py`
- Web: `/Users/beast1/RoboMacro/RoboMacro/src/features/hank/HANKPage.tsx`
- API: `/Users/beast1/RoboMacro/RoboMacro/macro_api/hank_routes.py`
- Live: `robomacro.com/hank`
- GitHub: `github.com/ChrisRoboMacro/RoboMacro_2026_March_27`
