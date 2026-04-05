"""
Parameter management for UK-HANK model.
Imports from data/parameters_from_paper.py and provides convenience access.
"""

import sys
import os

# Add project root to path so we can import from data/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from data.parameters_from_paper import (
    UKHANKParams,
    HouseholdParams,
    TaxBenefitParams,
    FirmParams,
    FinancialParams,
    OpenEconomyParams,
    HousingParams,
    InternalCalibrationParams,
    IncomeProcessParams,
    DynamicParams,
    IncomeTargetMoments,
    WealthTargets,
    TRANSITIONS,
    H_F, H_H,
    TENURE_RENTER, TENURE_OWN_FLAT, TENURE_OWN_HOUSE,
)

# Default parameter set
DEFAULT_PARAMS = UKHANKParams()


def get_params(**overrides) -> UKHANKParams:
    """Get parameters with optional overrides for experimentation."""
    params = UKHANKParams()
    for key, val in overrides.items():
        parts = key.split('.')
        if len(parts) == 2:
            group, param = parts
            setattr(getattr(params, group), param, val)
        else:
            raise ValueError(f"Override key must be 'group.param', got '{key}'")
    return params
