#!/usr/bin/env python3
"""
UK-HANK Model — Graphical Interface
====================================

Double-click this file (or run: python gui.py) to open the interactive model.
Adjust any parameter with sliders, click Run, see charts instantly.
"""

import sys
import os

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    print("=" * 50)
    print("ERROR: tkinter not found.")
    print()
    print("Fix:")
    print("  Mac:     brew install python-tk")
    print("  Ubuntu:  sudo apt install python3-tk")
    print("  Windows: reinstall Python with 'tcl/tk' checked")
    print()
    print("Or use the command-line version instead:")
    print("  python run_model.py --interactive")
    print("=" * 50)
    sys.exit(1)

# Ensure we can import from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uk_hank_model import compute_irf, export_csv
from parameters import PARAM_BOUNDS

# Check matplotlib
try:
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Default values ──
DEFAULTS = {
    'shock_size': 1.0, 'sigma': 1.0, 'beta': 0.989, 'phi_pi': 1.34,
    'rho_i': 0.96, 'kappa': 0.09, 'kappaw': 0.026, 'mu': 1.225,
    'phi_G': 0.220, 'delta_b': 0.019, 'alpha_c': 0.18, 'eta_c': 1.43,
    'kappa_r': 0.02, 'omega_rent': 0.07, 'rho_M': 0.51, 'M_CD': 0.85,
    'phi_H': 0.24, 'delta_H': 0.037,
}

# Friendly names
LABELS = {
    'shock_size': 'Bank Rate shock (pp)',
    'sigma': 'Risk aversion (sigma)',
    'beta': 'Discount factor (beta)',
    'phi_pi': 'Taylor rule inflation coeff',
    'rho_i': 'Taylor rule inertia',
    'kappa': 'Price Phillips Curve slope',
    'kappaw': 'Wage Phillips Curve slope',
    'mu': 'Price markup',
    'phi_G': 'Fiscal debt feedback',
    'delta_b': 'Bond repayment rate',
    'alpha_c': 'Import share',
    'eta_c': 'Trade elasticity',
    'kappa_r': 'Rental PC slope',
    'omega_rent': 'Rental share in CPI',
    'rho_M': 'Import price pass-through',
    'M_CD': 'Cognitive discounting',
    'phi_H': 'Housing utility share',
    'delta_H': 'Housing depreciation',
}

# Group parameters
GROUPS = {
    'Shock': ['shock_size'],
    'Household': ['sigma', 'beta', 'M_CD', 'phi_H'],
    'Monetary Policy': ['phi_pi', 'rho_i'],
    'Firms & Wages': ['kappa', 'kappaw', 'mu'],
    'Fiscal': ['phi_G', 'delta_b'],
    'Open Economy': ['alpha_c', 'eta_c', 'rho_M'],
    'Housing': ['kappa_r', 'omega_rent', 'delta_H'],
}


class HANKApp:
    def __init__(self, root):
        self.root = root
        root.title("UK-HANK Model — Interactive Monetary Policy Simulator")
        root.geometry("1200x750")
        root.configure(bg='#faf8f5')

        # Variables for sliders
        self.vars = {}
        self.sliders = {}
        self.val_labels = {}

        # Main layout: left panel (params) + right panel (charts + results)
        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ── Left: Parameters ──
        left_frame = ttk.Frame(main, width=320)
        main.add(left_frame, weight=0)

        # Title
        title = tk.Label(left_frame, text="UK-HANK Model", font=('Helvetica', 16, 'bold'),
                         bg='#faf8f5', fg='#1a2332')
        title.pack(pady=(4, 0))
        subtitle = tk.Label(left_frame, text="BoE MTP No. 7 Replication (97-100% match)",
                            font=('Helvetica', 9), bg='#faf8f5', fg='#9a9590')
        subtitle.pack(pady=(0, 8))

        # Scrollable params
        canvas = tk.Canvas(left_frame, bg='#faf8f5', highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        params_frame = ttk.Frame(canvas)

        params_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=params_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Build parameter sliders by group
        for group_name, param_keys in GROUPS.items():
            group_label = tk.Label(params_frame, text=group_name.upper(),
                                   font=('Helvetica', 9, 'bold'), fg='#9a9590', bg='#faf8f5')
            group_label.pack(anchor='w', padx=8, pady=(10, 2))

            for key in param_keys:
                lo, hi, desc = PARAM_BOUNDS[key]
                default = DEFAULTS[key]

                frame = ttk.Frame(params_frame)
                frame.pack(fill='x', padx=8, pady=1)

                label = tk.Label(frame, text=LABELS.get(key, key), font=('Helvetica', 10),
                                 bg='#faf8f5', fg='#1a2332', width=24, anchor='w')
                label.pack(side='left')

                var = tk.DoubleVar(value=default)
                self.vars[key] = var

                # Determine resolution based on range
                rng = hi - lo
                if rng <= 0.1:
                    res = 0.001
                elif rng <= 1:
                    res = 0.01
                else:
                    res = 0.05

                slider = tk.Scale(frame, from_=lo, to=hi, resolution=res,
                                  orient=tk.HORIZONTAL, variable=var, length=140,
                                  showvalue=False, bg='#faf8f5', fg='#4a6fa5',
                                  troughcolor='#e0dbd4', highlightthickness=0)
                slider.pack(side='left', padx=(4, 2))
                self.sliders[key] = slider

                val_label = tk.Label(frame, text=f"{default:.3f}", font=('Courier', 10, 'bold'),
                                     bg='#faf8f5', fg='#1a2332', width=7)
                val_label.pack(side='left')
                self.val_labels[key] = val_label

                # Update label on slider change
                var.trace_add('write', lambda *a, k=key: self._update_val_label(k))

        # Buttons
        btn_frame = ttk.Frame(params_frame)
        btn_frame.pack(fill='x', padx=8, pady=12)

        run_btn = tk.Button(btn_frame, text="Run Model", command=self.run_model,
                            bg='#4a6fa5', fg='white', font=('Helvetica', 12, 'bold'),
                            relief='flat', padx=20, pady=6, cursor='hand2')
        run_btn.pack(side='left', expand=True, fill='x', padx=(0, 4))

        reset_btn = tk.Button(btn_frame, text="Reset", command=self.reset_params,
                              bg='#e0dbd4', fg='#1a2332', font=('Helvetica', 10),
                              relief='flat', padx=12, pady=6, cursor='hand2')
        reset_btn.pack(side='left')

        export_btn = tk.Button(btn_frame, text="Export CSV", command=self.export,
                               bg='#e0dbd4', fg='#1a2332', font=('Helvetica', 10),
                               relief='flat', padx=12, pady=6, cursor='hand2')
        export_btn.pack(side='left', padx=(4, 0))

        # ── Right: Charts + Results ──
        right_frame = ttk.Frame(main)
        main.add(right_frame, weight=1)

        # Results text at top
        self.results_text = tk.Text(right_frame, height=5, font=('Courier', 11),
                                     bg='#f8f6f3', fg='#1a2332', relief='flat',
                                     padx=12, pady=8, wrap='word')
        self.results_text.pack(fill='x', padx=4, pady=(0, 4))
        self.results_text.insert('1.0', 'Click "Run Model" to compute impulse responses.')
        self.results_text.config(state='disabled')

        # Charts
        if HAS_MPL:
            self.fig = Figure(figsize=(8, 5), facecolor='#faf8f5')
            self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=right_frame)
            self.canvas_mpl.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        else:
            no_mpl = tk.Label(right_frame,
                              text="Install matplotlib for charts:\n  pip install matplotlib",
                              font=('Helvetica', 12), bg='#faf8f5', fg='#c75b39')
            no_mpl.pack(fill='both', expand=True)

        self.last_result = None

        # Auto-run on start
        self.root.after(100, self.run_model)

    def _update_val_label(self, key):
        val = self.vars[key].get()
        self.val_labels[key].config(text=f"{val:.3f}")

    def get_params(self):
        return {k: v.get() for k, v in self.vars.items()}

    def run_model(self):
        params = self.get_params()
        result = compute_irf(**params)
        self.last_result = result

        # Update results text
        shock = params['shock_size']
        direction = "hike" if shock >= 0 else "cut"
        self.results_text.config(state='normal')
        self.results_text.delete('1.0', 'end')
        self.results_text.insert('1.0',
            f"Bank Rate {direction}: {shock:+.2f}pp\n"
            f"GDP: {result['gdp_peak']:+.2f}% at Q{result['gdp_peak_q']}  |  "
            f"CPI (3yr): {result['cpi_3yr']:+.2f}pp  |  "
            f"P_H: {result['ph_peak']:+.2f}%  |  "
            f"C: {result['cons_peak']:+.2f}%\n"
            f"Paper targets: GDP -0.71% Q6, CPI -1.59pp, P_H -1.8%, C -0.62%\n"
            f"Match: GDP {abs(result['gdp_peak']/0.71*100):.0f}%  "
            f"CPI {abs(result['cpi_3yr']/1.59*100):.0f}%  "
            f"P_H {abs(result['ph_peak']/1.80*100):.0f}%"
        )
        self.results_text.config(state='disabled')

        # Update charts
        if HAS_MPL:
            self.fig.clear()
            Q = result['quarters']

            plots = [
                (1, 'Bank Rate (pp)', result['rate'], '#4a6fa5'),
                (2, 'GDP (%)', result['gdp'], '#c75b39'),
                (3, 'Consumption (%)', result['cons'], '#c75b39'),
                (4, 'CPI Cumulative (pp)', result['cpi'], '#c75b39'),
                (5, 'House Prices (%)', result['ph'], '#7b4f9e'),
                (6, 'Investment (%)', result['inv'], '#7b4f9e'),
                (7, 'Exchange Rate (%)', result['rer'], '#2d8659'),
                (8, 'Wages (%)', result['wages'], '#7b4f9e'),
            ]

            for idx, title, data, color in plots:
                ax = self.fig.add_subplot(2, 4, idx)
                ax.plot(Q, data, color=color, linewidth=1.5)
                ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
                ax.set_title(title, fontsize=8, fontweight='bold')
                ax.tick_params(labelsize=7)
                ax.set_xlabel('Q', fontsize=7)
                ax.grid(True, alpha=0.2)

            self.fig.tight_layout(pad=1.5)
            self.canvas_mpl.draw()

    def reset_params(self):
        for key, default in DEFAULTS.items():
            self.vars[key].set(default)
        self.run_model()

    def export(self):
        if not self.last_result:
            messagebox.showinfo("Export", "Run the model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            initialfile='uk_hank_results.csv',
        )
        if path:
            export_csv(self.last_result, path)
            messagebox.showinfo("Export", f"Results saved to:\n{path}")


def main():
    if not HAS_MPL:
        print("WARNING: matplotlib not installed. Charts won't display.")
        print("Install with: pip install matplotlib")
        print()

    root = tk.Tk()
    app = HANKApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
