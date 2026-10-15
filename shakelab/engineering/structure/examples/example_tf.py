"""
Example: Comparison of Transfer Functions for a 1D Building Model over a Half-Space

This script computes and compares the fixed-base transfer function TF_{u-u}(f) = u_top / u_base
using three different formulation methods available in ShakeLab:
1. `transfer_function_n_layer` (Thomson-Haskell propagator matrix formulation)
2. `sh_transfer_function` (Knopoff implicit layer matrix scheme)
3. `hassh` (Sanchez-Sesma / PSVQ matrix method)

All three methods compute the structural response of a 1D building layer over a half-space,
demonstrating consistent matching across the frequency spectrum.
"""

import numpy as np
import matplotlib.pyplot as plt
from shakelab.engineering.structure import F0BuildingModel, HalfSpaceModel
from shakelab.site.response import transfer_function_n_layer, sh_transfer_function
from shakelab.site.psvq.psvqlib import hassh


def run_example():
    # -------------------------------------------------------------------------
    # 1. Define Model Parameters
    # -------------------------------------------------------------------------
    # Building model parameterized by fundamental frequency f0:
    # - f0 = 6 Hz
    # - thickness (height) = 12 m
    # - damping ratio zeta = 0.05 (converted to Q = 1 / (2 * zeta) = 10)
    # - density rho = 540 kg/m^3
    # - implied shear-wave velocity Vs = 4 * f0 * H = 288 m/s
    building = F0BuildingModel(f0=6, thickness=12, zeta=0.05, rho=540)
    
    # Half-space model (using defaults: vs = 300 m/s, Q = 30, rho = 3000 kg/m^3)
    half_space = HalfSpaceModel()

    # Define frequency range for evaluation (0.05 to 50 Hz)
    freqs = np.linspace(0.05, 50, endpoint=True, num=100)

    # -------------------------------------------------------------------------
    # 2. Method 1: transfer_function_n_layer
    # -------------------------------------------------------------------------
    # Computes fixed-base transfer function 1 / A[0, 0] using Thomson-Haskell
    # layer matrices.
    tf_n_layer = transfer_function_n_layer(freqs, building, half_space)
    amp_n_layer = np.abs(tf_n_layer)

    # Prepare common layered-media parameters for wave propagation methods
    # - vs: S-wave velocities for [building, half_space]
    # - dn: densities for [building, half_space]
    # - qs: quality factors for [building, half_space]
    vs = np.array([building.vs, half_space.vs])
    dn = np.array([building.rho, half_space.rho])
    qs = np.array([building.Q, half_space.Q])

    # -------------------------------------------------------------------------
    # 3. Method 2: sh_transfer_function
    # -------------------------------------------------------------------------
    # Knopoff implicit layer matrix scheme.
    # Notes on parameters:
    # - `hl_sh` requires the half-space thickness to be specified as 0.0.
    # - `depth=[0., building.thickness]` evaluates displacements at the roof (top)
    #   and base (foundation interface), respectively.
    # - The fixed-base building transfer function is the ratio u_top / u_base.
    hl_sh = np.array([building.thickness, 0.])
    disp_sh = sh_transfer_function(
        freqs, hl_sh, vs, dn, qs, iang=0., depth=[0., building.thickness]
    )
    tf_sh = disp_sh[0] / disp_sh[1]  # Ratio of top (z=0) to base (z=H)
    amp_sh = np.abs(tf_sh)

    # -------------------------------------------------------------------------
    # 4. Method 3: hassh (PSVQ library)
    # -------------------------------------------------------------------------
    # Sanchez-Sesma boundary / layer formulation.
    # Notes on parameters:
    # - `H` specifies layer thicknesses above the half-space: [building.thickness].
    # - `hassh` operates on scalar frequencies, so a loop over `freqs` is used.
    # - Returned array `VS`: VS[0] is roof displacement, VS[1] is base displacement.
    # - The fixed-base building transfer function is the ratio VS[0] / VS[1].
    h_layers = [building.thickness]
    tf_hassh_list = [hassh(f, h_layers, vs, dn, qs, GAMMA=0.) for f in freqs]
    tf_hassh = np.array([res[0] / res[1] for res in tf_hassh_list])
    amp_hassh = np.abs(tf_hassh)

    # -------------------------------------------------------------------------
    # 5. Plot and Compare All Transfer Functions
    # -------------------------------------------------------------------------
    # Distinct line widths, styles, and colors are used so overlapping
    # curves can be easily distinguished and validated.
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs, amp_n_layer, label='transfer_function_n_layer', color='blue', linewidth=3, linestyle='-')
    plt.loglog(freqs, amp_sh, label='sh_transfer_function', color='red', linewidth=2, linestyle='--')
    plt.loglog(freqs, amp_hassh, label='hassh', color='green', linewidth=1.5, linestyle=':')
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(r'Amplitude $|TF_{u-u}(f)|$')
    plt.title('Comparison of Transfer Functions $TF_{u-u}(f)$ (Roof/Base)')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('example_tf.png', dpi=300)
    print("Plot saved to example_tf.png")


if __name__ == '__main__':
    run_example()

