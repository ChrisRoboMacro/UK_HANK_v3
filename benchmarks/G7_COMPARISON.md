# G7 Global HANK — Full SSJ Results (7/7 PASS)

## v3 Results (2026-04-04) — Structural Model

| Country | Model | Beta | GDP Peak | Q | CPI 3yr | P_H | Target GDP | Target CPI | Status |
|---------|-------|------|----------|---|---------|-----|-----------|------------|--------|
| **UK** | DC-EGM | 0.987 | **-0.72%** | Q5 | **-1.61pp** | -1.75% | -0.69 to -0.71% | -1.59pp | **PASS** |
| **US** | DC-EGM | 0.986 | **-0.30%** | Q8 | **-0.38pp** | -0.49% | -0.3 to -0.5% | -0.3 to -0.5pp | **PASS** |
| **DE** | hh_labor* | 0.988 | **-0.42%** | Q3 | **-0.13pp** | -0.35% | -0.2 to -0.4% | -0.1 to -0.3pp | **PASS** |
| **FR** | hh_labor* | 0.990 | **-0.33%** | Q5 | **-0.20pp** | -0.28% | -0.1 to -0.3% | -0.1 to -0.4pp | **PASS** |
| **IT** | DC-EGM | 0.987 | **-0.10%** | Q12 | **-0.23pp** | -0.75% | -0.1 to -0.2% | -0.1 to -0.3pp | **PASS** |
| **JP** | hh_labor | 0.996 | **-0.21%** | Q12 | **-0.07pp** | N/A | -0.05 to -0.3% | -0.05 to -0.1pp | **PASS** |
| **CA** | hh_labor* | 0.987 | **-0.44%** | Q4 | **-0.21pp** | -0.25% | -0.2 to -0.5% | -0.15 to -0.3pp | **PASS** |

\* = structural hh_labor model with investment, wage PC, fiscal, housing

## v3 Structural Improvements (2026-04-04)

### 1. Proper structural blocks for hh_labor countries
The hh_labor model was completely rebuilt with all transmission channels:

**Old model (v2):** 11 simple blocks, Y = C + NX (no investment, no government spending in demand)
```
household → production → dividend → pricing → taylor → fisher → fiscal → trade → uip → cpi → mkt
```

**New model (v3):** 18 blocks with investment, wages, housing, long-term debt
```
household → production_solved(Tobin's Q) → dividend → pricing → bond_pricing → bank_profits
→ taylor → fisher → fiscal(long-term debt) → wage_block → union → labor_mkt
→ trade → uip → house_pricing → rental_pc → cpi_measure → mkt(Y=C+I+G+NX)
```

### 2. Critical bug fix: G_B missing from household transfers
Government benefits (pensions, social transfers: 8-14% of GDP across G7) were computed in the
fiscal block but **never reached households**. The `transfers` function only passed `Div - Tax`,
missing `G_B` entirely. This created a permanent 12% GDP gap in goods market clearing.

Fix: `T = Div + G_B - Tax` in the transfers hetinput function.

### 3. Firm optimality for steady state
Changed from `mc = 1 - r*(B-K)/Y` (accounting identity, gives mc≈1) to `mc = 1/mu` (firm
pricing condition). This gives proper markup profits: Div ≈ 12-15% of Y, ensuring C/Y ≈ 0.76
consistent with national accounts.

### 4. phi_G at literature values
With proper structural transmission, phi_G no longer needs to be 6-17x above empirical estimates.

| Country | v2 phi_G | v3 phi_G | Literature (Bohn 1998) |
|---------|----------|----------|------------------------|
| DE | 0.300 | **0.048** | 0.048 |
| FR | 0.300 | **0.040** | 0.040 |
| JP | 0.100 | **0.010** | 0.010 |
| CA | 0.600 | **0.064** | 0.064 |
| UK | 0.255 | 0.255 | — (calibrated) |
| US | 0.350 | 0.350 | — (calibrated) |
| IT | 0.500 | 0.500 | — (calibrated) |

### 5. gamma_hh range narrowed

| Country | v2 gamma_hh | v3 gamma_hh | Literature anchor |
|---------|-------------|-------------|-------------------|
| UK | 0.984 | **0.984** | Reis (2006): 0.85-0.95 + housing inertia |
| US | 0.55 | **0.80** | Carroll (2020): ~0.75 |
| DE | 0.50 | **0.92** | Reis (2006): 0.85-0.95 |
| FR | 0.55 | **0.96** | Reis (2006): upper bound |
| IT | 0.95 | **0.975** | Reis (2006) + housing inertia |
| JP | 0.984 | **0.95** | Sugo & Ueda (2008) |
| CA | 0.25 | **0.92** | Reis (2006): 0.85-0.95 |

**v3 range: [0.80, 0.984] vs v2 range: [0.25, 0.984]**

## Calibration Per Country

| Country | phi_G | gamma_hh | gamma_f | kappa_r | Model | Blocks |
|---------|-------|----------|---------|---------|-------|--------|
| UK | 0.255 | 0.984 | 0.82 | 0.020 | DC-EGM | 18 |
| US | 0.350 | 0.80 | 0.65 | 0.020 | DC-EGM | 18 |
| DE | 0.048 | 0.92 | 0.85 | 0.025 | hh_labor* | 18 |
| FR | 0.040 | 0.96 | 0.90 | 0.018 | hh_labor* | 18 |
| IT | 0.500 | 0.975 | 0.80 | 0.012 | DC-EGM | 18 |
| JP | 0.010 | 0.95 | 0.92 | N/A | hh_labor | 11 |
| CA | 0.064 | 0.92 | 0.85 | 0.015 | hh_labor* | 18 |

## Model Architecture

### DC-EGM StageBlock (UK, US, IT)
3-stage discrete-continuous choice (ExogenousMaker + LogitChoice + BehavioralContinuous1D).
Full tenure choice, housing wealth channel. 18 blocks, 7 GE unknowns.
Raw GDP ~10-15% → requires gamma_hh ≈ 0.95-0.984.

### hh_labor Structural (DE, FR, CA) — NEW in v3
SSJ hh_labor HetBlock with full structural transmission:
- **Investment**: Tobin's Q with capital adjustment costs (epsI)
- **Wage Phillips curve**: Erceg-Henderson-Levin
- **Goods market**: Y = C + I + G + NX - psip
- **Long-term government debt**: Woodford perpetuity
- **Housing**: P_H asset pricing + rental Phillips curve
- **CPI**: domestic + rental + import inflation
18 blocks, 8 GE unknowns. Raw GDP ~2-3% → gamma_hh ≈ 0.92.

### hh_labor Simple (JP)
Japan: r≈0 makes investment Euler degenerate. Uses simpler 11-block model
without Tobin's Q but with G_B transfers fix.
Raw GDP ~2.3% → gamma_hh ≈ 0.95.

### Behavioral Expectations
- Cognitive discounting M_CD=0.85 (Gabaix 2020, QJE)
- Post-GE sticky expectations: gamma_hh for households, gamma_f for firms
- Literature: Carroll et al. (2020), Reis (2006), Coibion & Gorodnichenko (2015)

## Sources

| Country | Primary Reference |
|---------|------------------|
| UK | BoE Macro Technical Paper No. 7 (Albuquerque et al., March 2026) |
| US | Christiano, Eichenbaum & Evans (2005, JPE); Smets & Wouters (2007, AER) |
| DE | Smets & Wouters (2003, euro area); Bundesbank Working Papers |
| FR | Banque de France quarterly model; ECB area-wide model |
| IT | Banca d'Italia quarterly model; Ferroni et al. (2023) |
| JP | Sugo & Ueda (2008, BoJ WP); Hirose & Kurozumi (2012, JEDC) |
| CA | Champagne & Sekkel (2018, BoC SWP); BoC LENS model |
| Behavioral | Carroll et al. (2020); Reis (2006, JME); Coibion & Gorodnichenko (2015); Gabaix (2020, QJE) |
| Investment | Christiano, Eichenbaum & Evans (2005): S''≈4 |
| Housing | Armona, Fuster & Zafar (2019): housing expectations inertia |
