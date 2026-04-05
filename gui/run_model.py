#!/usr/bin/env python3
"""
UK-HANK Model Runner
====================

Run the UK-HANK model with default or custom parameters.

Usage:
  python run_model.py                     # Default 1pp shock
  python run_model.py --shock 2.0         # 2pp shock
  python run_model.py --interactive       # Interactive parameter adjustment
  python run_model.py --export out.csv    # Export to CSV
  python run_model.py --plot              # Plot IRFs
"""

import argparse
import sys

from uk_hank_model import compute_irf, print_results, export_csv, plot_irfs
from parameters import PARAM_BOUNDS


def interactive_mode():
    """Interactive parameter adjustment mode."""
    print("\n" + "=" * 60)
    print("UK-HANK MODEL — INTERACTIVE MODE")
    print("=" * 60)
    print("Adjust any parameter. Press Enter to keep default.")
    print("Type 'q' to quit, 'run' to compute with current params.")
    print("Type 'reset' to restore defaults.\n")

    # Current params (start with defaults)
    params = {k: v[0] + (v[1] - v[0]) * 0.5 for k, v in PARAM_BOUNDS.items()}
    # Override with actual defaults
    import parameters as P
    params = {
        'shock_size': 1.0, 'sigma': P.SIGMA, 'beta': P.BETA, 'phi_pi': P.PHI_PI,
        'rho_i': P.RHO_I, 'kappa': P.KAPPA, 'kappaw': P.KAPPA_W, 'mu': P.MU,
        'phi_G': P.PHI_G, 'delta_b': P.DELTA_B, 'alpha_c': P.ALPHA_C, 'eta_c': P.ETA_C,
        'kappa_r': P.KAPPA_R, 'omega_rent': P.OMEGA_RENT, 'rho_M': P.RHO_M,
        'M_CD': P.M_CD, 'phi_H': P.PHI_H, 'delta_H': P.DELTA_H,
    }

    while True:
        print("\n--- Current Parameters ---")
        for i, (k, v) in enumerate(params.items()):
            bounds = PARAM_BOUNDS.get(k, (0, 1, ''))
            desc = bounds[2] if len(bounds) > 2 else ''
            print(f"  [{i+1:2d}] {k:15s} = {v:8.4f}  ({desc})")

        print(f"\nCommands: [number] to edit, 'run', 'plot', 'export <file>', 'reset', 'q'")
        cmd = input("> ").strip()

        if cmd.lower() == 'q':
            break
        elif cmd.lower() == 'reset':
            params = {
                'shock_size': 1.0, 'sigma': P.SIGMA, 'beta': P.BETA, 'phi_pi': P.PHI_PI,
                'rho_i': P.RHO_I, 'kappa': P.KAPPA, 'kappaw': P.KAPPA_W, 'mu': P.MU,
                'phi_G': P.PHI_G, 'delta_b': P.DELTA_B, 'alpha_c': P.ALPHA_C, 'eta_c': P.ETA_C,
                'kappa_r': P.KAPPA_R, 'omega_rent': P.OMEGA_RENT, 'rho_M': P.RHO_M,
                'M_CD': P.M_CD, 'phi_H': P.PHI_H, 'delta_H': P.DELTA_H,
            }
            print("Parameters reset to defaults.")
        elif cmd.lower() == 'run':
            result = compute_irf(**params)
            print_results(result)
        elif cmd.lower() == 'plot':
            result = compute_irf(**params)
            print_results(result)
            plot_irfs(result)
        elif cmd.lower().startswith('export'):
            parts = cmd.split()
            fname = parts[1] if len(parts) > 1 else 'uk_hank_results.csv'
            result = compute_irf(**params)
            export_csv(result, fname)
        else:
            try:
                idx = int(cmd) - 1
                keys = list(params.keys())
                if 0 <= idx < len(keys):
                    key = keys[idx]
                    bounds = PARAM_BOUNDS.get(key, (0, 1, ''))
                    lo, hi = bounds[0], bounds[1]
                    val = input(f"  {key} [{lo:.3f} - {hi:.3f}] = ").strip()
                    if val:
                        v = float(val)
                        if lo <= v <= hi:
                            params[key] = v
                        else:
                            print(f"  Out of bounds [{lo}, {hi}]")
                else:
                    print("  Invalid index.")
            except ValueError:
                print("  Unknown command. Try a number, 'run', 'plot', 'export', 'reset', or 'q'.")


def main():
    parser = argparse.ArgumentParser(description='UK-HANK Model Runner')
    parser.add_argument('--shock', type=float, default=1.0, help='Shock size in pp (default: 1.0)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive parameter mode')
    parser.add_argument('--export', type=str, help='Export results to CSV file')
    parser.add_argument('--plot', action='store_true', help='Plot impulse response functions')
    parser.add_argument('--quarters', '-T', type=int, default=40, help='Number of quarters (default: 40)')

    # Allow setting any parameter via command line
    for k, (lo, hi, desc) in PARAM_BOUNDS.items():
        if k != 'shock_size':
            parser.add_argument(f'--{k}', type=float, default=None, help=f'{desc} [{lo}-{hi}]')

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    # Build kwargs from args
    kwargs = {'shock_size': args.shock, 'T': args.quarters}
    for k in PARAM_BOUNDS:
        if k != 'shock_size':
            val = getattr(args, k, None)
            if val is not None:
                kwargs[k] = val

    result = compute_irf(**kwargs)
    print_results(result)

    if args.export:
        export_csv(result, args.export)

    if args.plot:
        plot_irfs(result)


if __name__ == '__main__':
    main()
