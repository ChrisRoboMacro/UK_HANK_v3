# UK-HANK Model Build Status

## Last updated: 2026-03-27

## Working Model
- Location: `/Volumes/DDrive/HANK/uk_hank/model/full.py`
- Run: `cd /Volumes/DDrive/HANK && source .venv/bin/activate && python -m uk_hank.model.full`
- Web UI: `robomacro.com/HANK` (static HTML at `RoboMacro/public/HANK/index.html`)

## Current Results (v3, 4 active channels)

| Variable | Smoothed (γ=0.60) | Raw (no smooth) | Paper target |
|----------|-------------------|------------------|-------------|
| GDP peak | **-0.26% at Q1** | -0.56% at Q0 | -0.71% at Q6 |
| Consumption | -0.15% at Q1 | -0.33% at Q0 | -0.62% at Q6 |
| CPI (3yr) | -0.32pp | -0.33pp | -1.59pp |
| Exchange rate | 0.12% | 0.25% | ~0.5% |
| Net exports | 0.11% GDP | 0.23% GDP | ~0.15% GDP |
| beta | 0.9859 | 0.9859 | 0.9902 |

## Channels Active
1. Direct interest rate / savings (household EGM with jump-shock income) - WORKING
2. Open economy / exchange rate (UIP + trade with CES demand) - WORKING
3. Behavioral expectations (IRF-level exponential smoothing γ=0.60) - WORKING
4. Price Phillips Curve (Rotemberg NKPC, κ=0.09) - WORKING

## Channels Still Missing (see SHORTCUTS.md for details)
5. Cost of capital / investment with Tobin's Q (needs @solved block in DAG)
6. Wage Phillips Curve (needs DAG restructure: union determines w)
7. House price channel (needs P_H as equilibrium variable)
8. Rental channel (needs rental Phillips Curve)
9. Financial income / bond revaluation (needs intermediary block)
10. Cash-flow redistribution (needs heterogeneous returns by tenure)
11. Long-term government debt (needs geometric maturity structure)

## Gap Analysis
- Raw GDP is 79% of paper target (-0.56 vs -0.71%)
- CPI is only 21% of paper target (-0.33 vs -1.59pp) — mostly missing import price channel
- Timing: Q0-Q1 peak vs paper's Q6 — behavioral expectations at block level would fix this
- The remaining channels compound through GE feedback, so adding them is not simply additive

## Environment
- Python 3.14.3, numba 0.64.0, sequence-jacobian 1.0.0
- 42 exogenous states (2 tenure × 7 persistent × 3 transitory) × 100 asset points
- Jacobians: T=300, takes ~10 seconds to compute
