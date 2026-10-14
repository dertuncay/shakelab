import numpy as np
import matplotlib.pyplot as plt

from shakelab.signals.base import Stream, Record
from shakelab.signals.fourier import Spectrum
from shakelab.engineering.structure.structure import F0BuildingModel, HalfSpaceModel
from shakelab.site.response import transfer_function_n_layer
from shakelab.libutils.utils import calculate_stream_difference, calc_drift
from shakelab.engineering.structure.damage_state import DamageState


def generate_signal(freq, duration, sampling_rate):
    t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    noise = np.random.normal(0, 0.1, len(t))
    return t, signal + noise


def run_example():
    sampling_rate = 100
    duration = 60
    dt = 1.0 / sampling_rate
    
    t, bottom_data = generate_signal(2, duration, sampling_rate)
    
    # 1. Create Stream for bottom waveform
    rec_bottom = Record(data=bottom_data, delta=dt)
    st_bottom = Stream()
    st_bottom.record.append(rec_bottom)
    
    # 2. Compute FFT (Bottom FFT)
    spec_bottom = Spectrum()
    spec_bottom.fft(data=st_bottom.record[0].data, delta=dt)
    freqs = spec_bottom.frequency_axis
    bottom_fft_amp = spec_bottom.amplitude
    
    # 3. Calculate Transfer Function
    building = F0BuildingModel(f0=6, thickness=12, zeta=0.05, rho=540)
    half_space = HalfSpaceModel()
    
    tf = transfer_function_n_layer(freqs, building, half_space)
    
    # 4. Multiply TF with Bottom FFT to get Top FFT
    spec_top = Spectrum()
    spec_top.set_data(spec_bottom.data * tf, delta=dt, nsamp=len(bottom_data))
    top_fft_amp = spec_top.amplitude
    
    # 5. IFFT to get Top Waveform
    top_data = spec_top.ifft()
    rec_top = Record(data=top_data, delta=dt)
    st_top = Stream()
    st_top.record.append(rec_top)
    
    # 6. Calculate Drift
    st_drift = calculate_stream_difference(st_top, st_bottom)
    drift_data = st_drift.record[0].data
    
    max_drift = calc_drift(st_top, st_bottom)
    
    summary_text = f"Max Drift: {max_drift:.3f}\n"
    summary_text += "Damage States (Scaini):\n"
    
    scenarios = [
        ("URM", 1940, 2),
        ("URM", 1960, 3),
        ("URM", 1990, 4),
        ("URM/MX", 1940, 2),
        ("URM/MX", 1960, 3),
        ("URM/MX", 1990, 4),
        ("RC", 1970, 3),
        ("RC", 2000, 5),
    ]
    
    for mat, year, stories in scenarios:
        ds = DamageState(method="scaini_etal", drift=max_drift, 
                         construction_material=mat, 
                         construction_year=year, 
                         number_of_floor=stories)
        state = ds.get_damage_stage()
        summary_text += f" - {mat} ({year}), {stories} fl: {state}\n"
    
    # Plotting 3x3 grid
    fig, axs = plt.subplots(3, 3, figsize=(20, 12))
    
    # Left column: Waveforms
    axs[0, 0].plot(t, top_data, 'b')
    axs[0, 0].set_title('Top Waveform')
    axs[0, 0].set_ylabel('Amplitude')
    
    axs[1, 0].plot(t, drift_data, 'g')
    axs[1, 0].set_title('Drift (Top - Bottom)')
    axs[1, 0].set_ylabel('Amplitude')
    
    axs[2, 0].plot(t, bottom_data, 'k')
    axs[2, 0].set_title('Bottom Waveform (Input)')
    axs[2, 0].set_ylabel('Amplitude')
    axs[2, 0].set_xlabel('Time (s)')
    
    # Middle column: FFTs & Transfer Function
    valid_idx = freqs > 0
    f_valid = freqs[valid_idx]
    
    axs[0, 1].loglog(f_valid, top_fft_amp[valid_idx], 'b')
    axs[0, 1].set_title('Top FFT')
    axs[0, 1].set_ylabel('Amplitude')
    
    axs[1, 1].loglog(f_valid, np.abs(tf)[valid_idx], 'r')
    axs[1, 1].set_title('Transfer Function')
    axs[1, 1].set_ylabel('Amplitude')
    
    axs[2, 1].loglog(f_valid, bottom_fft_amp[valid_idx], 'k')
    axs[2, 1].set_title('Bottom FFT (Input)')
    axs[2, 1].set_ylabel('Amplitude')
    axs[2, 1].set_xlabel('Frequency (Hz)')
    
    # Right column: Summary Text Box
    for i in range(3):
        axs[i, 2].axis('off')  # Hide all axes in the third column
    
    # Place text in the middle right panel
    axs[1, 2].text(0.1, 0.5, summary_text, transform=axs[1, 2].transAxes, 
                   fontsize=12, verticalalignment='center', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('example_stream.png')
    print("Plot saved to example_stream.png")


if __name__ == '__main__':
    run_example()
