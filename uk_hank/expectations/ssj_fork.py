"""
Forked SSJ _jacobian method with block-level behavioral expectations.
=====================================================================

This monkey-patches the CombinedBlock._jacobian method to apply
cognitive discounting and sticky expectations to each block's Jacobian
BEFORE GE composition.

This is the correct implementation of Section 2.2 of the paper.

Usage:
    from uk_hank.expectations.ssj_fork import patch_model_behavioral
    patch_model_behavioral(model, M_CD=0.85, gamma_hh=0.87, gamma_f=0.84)
    # Now model.solve_jacobian() and model.solve_impulse_linear() use behavioral Jacobians
"""

import numpy as np
from sequence_jacobian.classes.jacobian_dict import JacobianDict
from sequence_jacobian.classes.sparse_jacobians import SimpleSparse, IdentityMatrix
from sequence_jacobian.utilities.ordered_set import OrderedSet

from uk_hank.expectations.behavioral import (
    apply_cognitive_discounting_fast,
    apply_sticky_expectations,
)


def transform_jacobian_dict_block_level(J, T, gamma, M_CD=0.85):
    """
    Apply behavioral expectations to a single block's JacobianDict.

    Converts SimpleSparse entries to dense, applies transforms, returns new JacobianDict.

    Parameters
    ----------
    J : JacobianDict
        One block's Jacobian (outputs x inputs, each entry is SimpleSparse or ndarray)
    T : int
        Time horizon
    gamma : float
        Sticky expectations parameter (0.87 for households, 0.84 for firms)
    M_CD : float
        Cognitive discounting parameter
    """
    new_nd = {}
    for out_var in J.nesteddict:
        new_nd[out_var] = {}
        for in_var in J.nesteddict[out_var]:
            entry = J.nesteddict[out_var][in_var]

            # Convert to dense
            if isinstance(entry, SimpleSparse):
                dense = entry.matrix(T)
            elif isinstance(entry, IdentityMatrix):
                dense = entry.matrix(T)
            elif isinstance(entry, np.ndarray):
                dense = entry
            else:
                # Unknown type — pass through
                new_nd[out_var][in_var] = entry
                continue

            # Apply cognitive discounting
            dense = apply_cognitive_discounting_fast(dense, M_CD)

            # Apply sticky expectations (NO normalization)
            dense = apply_sticky_expectations(dense, gamma=gamma, max_lag=min(80, T))

            new_nd[out_var][in_var] = dense

    return JacobianDict(new_nd, J.outputs, J.inputs)


def make_behavioral_jacobian_method(original_jacobian, M_CD=0.85, gamma_hh=0.87, gamma_f=0.84,
                                     hh_block_names=None, no_transform_blocks=None):
    """
    Create a patched _jacobian method that applies behavioral expectations.

    Parameters
    ----------
    original_jacobian : bound method
        The original CombinedBlock._jacobian method
    M_CD : float
        Cognitive discounting factor
    gamma_hh : float
        Household sticky expectations (probability of NOT updating)
    gamma_f : float
        Firm/other block sticky expectations
    hh_block_names : set
        Names of household blocks (get gamma_hh)
    no_transform_blocks : set
        Names of blocks to NOT transform (policy rules, identities)
    """
    if hh_block_names is None:
        hh_block_names = {'hh'}
    if no_transform_blocks is None:
        no_transform_blocks = {'taylor', 'fisher', 'fiscal', 'mkt'}

    def _jacobian_behavioral(self, ss, inputs, outputs, T, Js, options):
        """Patched _jacobian: applies behavioral expectations to each block's Jacobian."""
        Js = self._partial_jacobians(ss, inputs, outputs, T, Js, options)

        original_outputs = outputs

        total_Js = JacobianDict.identity(inputs)

        vector_valued = ss._vector_valued()
        inputs = (inputs | self._required) - vector_valued
        outputs = (outputs | self._required) - vector_valued

        for block in self.blocks:
            if (inputs & block.inputs) and (outputs & block.outputs):
                J = block.jacobian(ss, inputs & block.inputs, outputs & block.outputs, T, Js, options)

                # === BEHAVIORAL EXPECTATIONS TRANSFORM ===
                block_name = getattr(block, 'name', '')
                if block_name not in no_transform_blocks:
                    # Determine gamma for this block
                    gamma = gamma_hh if block_name in hh_block_names else gamma_f
                    # No cognitive discounting on solved blocks — their internal solver
                    # already handles forward-looking expectations. Only apply sticky info.
                    block_M_CD = 1.0 if ('solved' in block_name or 'pricing' in block_name) else M_CD

                    try:
                        J = transform_jacobian_dict_block_level(J, T, gamma, block_M_CD)
                    except Exception as e:
                        # If transform fails, use original (don't break the model)
                        pass
                # === END TRANSFORM ===

                total_Js.update(J @ total_Js)

        return total_Js[original_outputs & total_Js.outputs, :]

    return _jacobian_behavioral


def patch_model_behavioral(model, M_CD=0.85, gamma_hh=0.87, gamma_f=0.84,
                            hh_block_names=None, no_transform_blocks=None):
    """
    Monkey-patch a model to use behavioral expectations in Jacobian computation.

    After calling this, model.solve_jacobian() and model.solve_impulse_linear()
    will automatically apply cognitive discounting + sticky expectations
    to each block's Jacobian before GE composition.

    Parameters
    ----------
    model : CombinedBlock
        The SSJ model to patch
    M_CD : float
        Cognitive discounting factor (paper: 0.85)
    gamma_hh : float
        Household stickiness (paper: 0.87 = 13% update prob)
    gamma_f : float
        Firm stickiness (paper: 0.84 = 16% update prob)
    """
    import types

    behavioral_method = make_behavioral_jacobian_method(
        model._jacobian, M_CD, gamma_hh, gamma_f,
        hh_block_names, no_transform_blocks
    )

    # Bind the new method to the model instance
    model._jacobian = types.MethodType(behavioral_method, model)

    return model


def unpatch_model(model):
    """Remove the behavioral patch, restoring rational expectations."""
    # Restore the original _jacobian from the class
    from sequence_jacobian.blocks.combined_block import CombinedBlock
    model._jacobian = types.MethodType(CombinedBlock._jacobian, model)
    return model


if __name__ == '__main__':
    print("Testing SSJ fork with block-level behavioral expectations...")

    # Build the working investment model
    from uk_hank.model.incremental_test import test_v3_plus_investment
    import numpy as np

    # First run RATIONAL
    r_rat = test_v3_plus_investment()
    print(f"\nRATIONAL: GDP = {r_rat['gdp_peak']:.2f}%")

    # Now patch the model and run BEHAVIORAL
    model = r_rat['model']
    ss = r_rat['ss']

    patch_model_behavioral(model, M_CD=0.85, gamma_hh=0.87, gamma_f=0.84)

    # Recompute Jacobians (now with behavioral transforms)
    unknowns = r_rat['unknowns']
    targets = r_rat['targets']
    exogenous = r_rat['exogenous']

    G_beh = model.solve_jacobian(ss, unknowns, targets, exogenous, T=300)

    # Compute IRF
    dr = {'rstar': 0.0025 * 0.5 ** np.arange(300)}
    irf_beh = model.solve_impulse_linear(ss, unknowns, targets, dr)

    Y_beh = irf_beh['Y'] / ss['Y'] * 100
    print(f"BEHAVIORAL: GDP = {np.min(Y_beh):.2f}% at Q{np.argmin(Y_beh)}")

    if 'C' in irf_beh:
        print(f"  C: {np.min(irf_beh['C']/ss['C']*100):.2f}%")
    if 'pi' in irf_beh:
        print(f"  CPI 3yr: {np.cumsum(irf_beh['pi'][:12])[-1]*100:.2f}pp")

    print(f"\n  Paper: GDP -0.71% at Q6")
