import numpy as np
import matplotlib.pyplot as plt
from shakelab.engineering.structure.structure import F0BuildingModel, HalfSpaceModel
from shakelab.site.response import transfer_function_n_layer


def run_example():
    # Build a model with the given parameters
    # height=12, zeta=0.05, density=540, f0=6
    building = F0BuildingModel(f0=6, thickness=12, zeta=0.05, rho=540)
    
    # Create the half-space model (using defaults: vs=300, Q=30, rho=3000)
    half_space = HalfSpaceModel()

    # Define frequencies
    freqs = np.linspace(0.05, 50, endpoint=True, num=100)

    # Compute transfer function
    tf = transfer_function_n_layer(freqs, building, half_space)
    amplitude = np.abs(tf)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.loglog(freqs, amplitude, label='Transfer Function (Amplitude)', color='b', linewidth=2)
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Transfer Function TF_{u-u}(f) for Building Model over Half-Space')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    
    # Save the plot
    plt.savefig('example_tf.png')
    print("Plot saved to example_tf.png")


if __name__ == '__main__':
    run_example()
