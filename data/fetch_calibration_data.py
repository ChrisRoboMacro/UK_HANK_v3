"""
Fetch all public calibration data for UK-HANK model.

All sources are public UK government data — no authentication required.

Data sources:
1. ONS Input-Output Analytical Tables (import shares by industry)
2. DMO Gilt Portfolio Statistics (average maturity of government debt)
3. ONS "Effects of Taxes and Benefits" (tax/benefit progressivity by decile)
4. Bank of England Quoted Mortgage Rates (2yr fixed, various LTVs)
5. ONS NS&I Holdings (National Savings deposits, ACUA series)

Run: python data/fetch_calibration_data.py
"""

import os
import sys
import time
import urllib.request
import urllib.error

RAW_DIR = os.path.join(os.path.dirname(__file__), 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def download(url, filename, description):
    """Download a file with progress reporting."""
    filepath = os.path.join(RAW_DIR, filename)
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / 1e6
        print(f"  [SKIP] {description} — already exists ({size_mb:.1f} MB)")
        return filepath

    print(f"  [FETCH] {description}")
    print(f"          {url}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'UK-HANK-Model/0.1'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(filepath, 'wb') as f:
            f.write(data)
        size_mb = len(data) / 1e6
        print(f"          OK ({size_mb:.1f} MB)")
        return filepath
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"          FAILED: {e}")
        return None


def fetch_all():
    """Fetch all calibration datasets."""
    results = {}

    # =========================================================================
    # 1. ONS Input-Output Analytical Tables
    # =========================================================================
    print("\n1. ONS Input-Output Analytical Tables")
    print("   Purpose: Import shares α_c=0.18 (consumption), α_y=0.15 (production)")

    # Product-by-product tables (contains "Imports use PxI" sheet)
    results['io_product'] = download(
        'https://www.ons.gov.uk/file?uri=/economy/nationalaccounts/supplyandusetables/'
        'datasets/ukinputoutputanalyticaltablesdetailed/2023/iot2023product.xlsx',
        'ons_io_analytical_product_2023.xlsx',
        'ONS IO Analytical Tables - Product by Product (2023)'
    )

    # Supply-Use tables (Blue Book 2025, 1997-2023)
    results['io_supply_use'] = download(
        'https://www.ons.gov.uk/file?uri=/economy/nationalaccounts/supplyandusetables/'
        'datasets/inputoutputsupplyandusetables/current/supublicationtablesbb25.xlsx',
        'ons_supply_use_bb25.xlsx',
        'ONS Supply-Use Tables (Blue Book 2025, 1997-2023)'
    )

    # =========================================================================
    # 2. DMO Gilt Average Maturity
    # =========================================================================
    print("\n2. DMO Gilt Portfolio Statistics")
    print("   Purpose: Average maturity=13yrs → δ=0.019 quarterly principal repayment")

    results['dmo_portfolio'] = download(
        'https://www.dmo.gov.uk/media/jjrpbplt/debt-portfolio-statistics-historical-070922.xls',
        'dmo_debt_portfolio_historical.xls',
        'DMO Debt Portfolio Statistics (historical)'
    )

    # Also fetch gilts in issue for current maturity profile
    results['dmo_gilts_in_issue'] = download(
        'https://www.dmo.gov.uk/data/XmlDataReport?reportCode=D1A',
        'dmo_gilts_in_issue.xml',
        'DMO Gilts in Issue (current, XML)'
    )

    # =========================================================================
    # 3. ONS Effects of Taxes and Benefits
    # =========================================================================
    print("\n3. ONS Effects of Taxes and Benefits on Household Income")
    print("   Purpose: Tax progressivity λ=0.07, benefit schedule λ_{B,0}=0.69, λ_{B,1}=-0.65")

    results['tax_benefits'] = download(
        'https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/'
        'personalandhouseholdfinances/incomeandwealth/datasets/'
        'theeffectsoftaxesandbenefitsonhouseholdincomefinancialyearending2014/'
        'financialyearending2024/etbreferencetablesfye202324final.xlsx',
        'ons_effects_taxes_benefits_fye2024.xlsx',
        'ONS Effects of Taxes and Benefits (FYE 2024)'
    )

    # =========================================================================
    # 4. Bank of England Mortgage Rates
    # =========================================================================
    print("\n4. Bank of England Quoted Mortgage Rates")
    print("   Purpose: Mortgage spread ω_bor = 2yr fixed rate - 2yr Gilt = ~1.5% annual")

    # 2yr fixed mortgage rates at various LTVs (monthly, 1995-present)
    boe_series = {
        'IUMZICQ': '2yr_fixed_60LTV',
        'IUMBV34': '2yr_fixed_75LTV',
        'IUMZICR': '2yr_fixed_85LTV',
        'IUMB482': '2yr_fixed_90LTV',
        'IUM2WTL': '2yr_fixed_95LTV',
    }
    codes = ','.join(boe_series.keys())
    results['boe_mortgage'] = download(
        f'https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?'
        f'csv.x=yes&Datefrom=01/Jan/1995&Dateto=27/Mar/2026&SeriesCodes={codes}'
        f'&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N',
        'boe_mortgage_rates_2yr_fixed.csv',
        f'BoE 2yr Fixed Mortgage Rates ({", ".join(boe_series.values())})'
    )

    # Also fetch 2yr gilt yields for computing the spread
    results['boe_gilt_2yr'] = download(
        'https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?'
        'csv.x=yes&Datefrom=01/Jan/1995&Dateto=27/Mar/2026&SeriesCodes=IUMALNZC'
        '&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N',
        'boe_gilt_yields_2yr.csv',
        'BoE 2yr Gilt Yield (IUMALNZC)'
    )

    # =========================================================================
    # 5. ONS NS&I Holdings
    # =========================================================================
    print("\n5. ONS NS&I Holdings (National Savings)")
    print("   Purpose: L̂_ss = 24% of GDP (NS&I / Gilts ratio = 0.14)")

    results['nsi_holdings'] = download(
        'https://www.ons.gov.uk/generator?format=csv&uri=/economy/'
        'governmentpublicsectorandtaxes/publicsectorfinance/timeseries/acua/pusf',
        'ons_nsi_holdings_acua.csv',
        'ONS NS&I Holdings (ACUA series, GBP millions)'
    )

    # Also get nominal GDP for computing ratios
    results['nominal_gdp'] = download(
        'https://www.ons.gov.uk/generator?format=csv&uri=/economy/'
        'grossdomesticproductgdp/timeseries/ybha/pn2',
        'ons_nominal_gdp_ybha.csv',
        'ONS Nominal GDP (YBHA series, GBP millions)'
    )

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    success = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"  {success}/{total} files downloaded successfully")

    for name, path in results.items():
        status = "OK" if path else "FAILED"
        print(f"  [{status}] {name}")

    if success < total:
        print("\n  WARNING: Some downloads failed. The model can still run")
        print("  using hardcoded parameter values from the paper.")
        print("  Re-run this script to retry failed downloads.")

    return results


if __name__ == '__main__':
    fetch_all()
