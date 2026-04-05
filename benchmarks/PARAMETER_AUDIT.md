# G7 Global HANK — Parameter Audit

## Full Parameter Matrix

```
Parameter          UK        US        DE        FR        IT        JP        CA
═══════════════════════════════════════════════════════════════════════════════════

--- Monetary Policy ---
phi_pi            1.340     1.500     1.390     1.390     1.390     1.100     1.800
rho_i             0.960     0.800     0.870     0.870     0.870     0.970     0.820
kappa             0.090     0.090     0.085     0.065     0.055     0.010     0.050
kappaw            0.026     0.030     0.035     0.025     0.020     0.005     0.035

--- Fiscal ---
phi_G             0.255     0.350     0.300†    0.300†    0.500     0.100†    0.600†
delta_b           0.019     0.038     0.032     0.030     0.036     0.028     0.042
T_ss              0.235     0.270     0.240     0.280     0.260     0.200     0.240
G_B_ss            0.090     0.070     0.120     0.140     0.110     0.080     0.080
B (debt stock)    5.600     4.800     2.600     4.400     5.600     5.200     4.200

--- Household ---
sigma             1.000     1.000     1.000     1.000     1.000     1.000     1.000
beta (config)     0.989     0.994     0.998     0.998     0.998     0.999     0.994
phi_H             0.240     0.180     0.200     0.220     0.180     0.150     0.300
omega_oo          1.060     1.100     0.900     1.000     1.150     0.950     1.200
eta_move          0.320     0.280     0.250     0.300     0.350     0.500     0.300

--- Prices & Markups ---
mu                1.225     1.200     1.150     1.200     1.250     1.150     1.200
kappa_r           0.020     0.020     0.015     0.018     0.012†    0.080‡    0.015
omega_rent        0.070     0.080     0.076     0.073     0.030     0.150     0.080

--- Open Economy ---
alpha_c           0.180     0.110     0.380     0.310     0.270     0.180     0.330
eta_c             1.430     1.500     1.300     1.250     1.200     0.800     1.500
rho_M             0.510     0.300     0.550     0.500     0.450     0.650     0.700

--- Housing ---
delta_H           0.037     0.012§    0.030     0.032     0.028     0.065     0.035
P_H               3.600     4.200     3.000     3.800     3.200     4.000     5.000
P_R               0.150     0.180     0.200     0.170     0.100     0.120     0.200

--- Behavioral (calibration levers, v2 2026-04-04) ---
gamma_hh          0.984     0.550     0.500     0.550     0.950     0.984     0.250
gamma_f           0.820     0.650     0.800     0.800     0.650     0.996     0.550

--- Income Process ---
rho_e             0.966     0.970     0.966     0.966     0.966     0.966     0.966
sigma_e           0.500     0.520     0.480     0.500     0.520     0.350     0.500

--- Grid/Solver ---
Model type        DC-EGM    DC-EGM    hh_labor  hh_labor  DC-EGM    hh_labor  hh_labor
nA                200       250       200       200       200       300       250
amax              50        80        60        55        50        120       70
backward_maxit    5000      15000     15000     15000     15000     30000     15000
```

## Notes on Flagged Values

### † phi_G (Fiscal feedback)
All phi_G values are CALIBRATED to match country-specific CPI evidence.
Literature values (Bohn 1998: 0.027-0.064) are too low for HANK fiscal channel.
DE=0.300, FR=0.300 (hh_labor, moderate fiscal amplification).
JP=0.100 (near-zero effect in linearised model with gamma_f=0.996).
CA=0.600 (compensates for low kappa=0.050 and aggressive phi_pi=1.80).
UK=0.255, US=0.350, IT=0.500 (DC-EGM, housing channel provides additional amplification).
**Transparent: phi_G is a calibration target, not purely empirical.**

### † IT kappa_r = 0.012 (reduced from 0.030)
Justified by Italian rental regulation:
- Cedolare secca (flat tax encouraging long leases)
- Canone concordato (agreed-rate contracts in major cities)
- 4+4 year standard lease (Legge 431/1998)
Value is between UK (0.020, moderate regulation) and minimum.

### ‡ JP kappa_r = 0.080
Japan has a genuinely flexible rental market (no rent control, 2-year leases).
**However, this value is UNUSED** because Japan uses the hh_labor model which
has no rental Phillips Curve.

### § US delta_H = 0.012 (corrected from 0.006)
Original 0.006 was pure BEA physical depreciation only.
UK's 0.037 includes maintenance + insurance + depreciation (ONS CPI-H).
Corrected to 0.012 = BEA depreciation (2.3%) + BLS maintenance (1.5%) + insurance (0.5%)
= 4.3% annual ≈ 0.011 quarterly, rounded to 0.012.

## Classification of Parameters

### Empirical (from data/literature, should not be changed)
sigma, kappa, kappaw, delta_b, T_ss, G_B_ss, B, alpha_c, eta_c, rho_M,
mu, omega_rent, delta_H, P_H, P_R, r, K, phi_pi, rho_i

### Calibration targets (tuned to match IRF evidence)
phi_G, gamma_hh, gamma_f, kappa_r, rho_e, sigma_e

### Structural (same everywhere)
sigma=1.0, M_CD=0.85

### Model choice (per-country decision)
DC-EGM vs hh_labor, nA, amax, backward_maxit
