
import torch
import matplotlib.pyplot as plt
import numpy as np

from librosa import resample


def plot_spec(audio, sr, target_sr=16000, n_fft=1024, hop_length=512, vmin=-60, vmax=0, figsize=(6, 4)):
    if sr != target_sr:
        audio = resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    audio = torch.tensor(audio)
    spec = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft), return_complex=True)
    spec_power = torch.pow(torch.abs(spec), 2)
    spec_power_norm = spec_power / torch.abs(spec_power).max()
    spec_dB = 10 * torch.log10(spec_power_norm + 1e-8)

    T_coef = np.arange(spec_dB.shape[1]) * hop_length / sr
    F_coef = np.arange(spec_dB.shape[0]) * sr / n_fft
    left = min(T_coef)
    right = max(T_coef) + n_fft / sr
    lower = min(F_coef) / 1000
    upper = max(F_coef) / 1000
 
    plt.figure(figsize=figsize)
    plt.imshow(spec_dB, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax, extent=[left, right, lower, upper]) 
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [kHz]')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def plot_spec_dual(audio1, sr1, audio2, sr2, target_sr=16000, n_fft=1024, hop_length=512, vmin=-60, vmax=0, figsize=(12, 4)):
    if sr1 != target_sr:
        audio1 = resample(audio1, orig_sr=sr1, target_sr=target_sr)
        sr1 = target_sr
    if sr2 != target_sr:
        audio2 = resample(audio2, orig_sr=sr2, target_sr=target_sr)
        sr2 = target_sr
    
    # Convert audios to tensor
    audio1 = torch.tensor(audio1)
    audio2 = torch.tensor(audio2)
    
    # Compute spectrograms
    spec1 = torch.stft(audio1, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft), return_complex=True)
    spec2 = torch.stft(audio2, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft), return_complex=True)

    # Convert to power and decibels
    spec_power1 = torch.pow(torch.abs(spec1), 2)
    spec_power_norm1 = spec_power1 / torch.abs(spec_power1).max()
    spec_dB1 = 10 * torch.log10(spec_power_norm1 + 1e-8)

    spec_power2 = torch.pow(torch.abs(spec2), 2)
    spec_power_norm2 = spec_power2 / torch.abs(spec_power2).max()
    spec_dB2 = 10 * torch.log10(spec_power_norm2 + 1e-8)
    
    # Time and frequency coefficients
    T_coef1 = np.arange(spec_dB1.shape[1]) * hop_length / sr1
    F_coef1 = np.arange(spec_dB1.shape[0]) * sr1 / n_fft

    T_coef2 = np.arange(spec_dB2.shape[1]) * hop_length / sr2
    F_coef2 = np.arange(spec_dB2.shape[0]) * sr2 / n_fft
    
    left1, right1 = min(T_coef1), max(T_coef1) + n_fft / sr1
    lower1, upper1 = min(F_coef1) / 1000, max(F_coef1) / 1000
    
    left2, right2 = min(T_coef2), max(T_coef2) + n_fft / sr2
    lower2, upper2 = min(F_coef2) / 1000, max(F_coef2) / 1000

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    # Plot first audio
    im1 = ax[0].imshow(spec_dB1, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax, extent=[left1, right1, lower1, upper1])
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Frequency [kHz]')
    ax[0].set_title('Input')

    # Plot second audio
    im2 = ax[1].imshow(spec_dB2, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax, extent=[left2, right2, lower2, upper2])
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Frequency [kHz]')
    ax[1].set_title('Output')

    # Create a single colorbar for both plots
    fig.colorbar(im1, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
    plt.show()

def plot_spec_subplot(audios, path, ncols=4, sr=16000, n_fft=1024, hop_length=512, vmin=-60, vmax=0, figsize=(12, 4)):
    specs = []

    for audio in audios:
        audio = torch.tensor(audio).squeeze()
        spec = torch.stft(torch.tensor(audio), n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft), return_complex=True)
        spec_power = torch.pow(torch.abs(spec), 2)
        spec_power_norm = spec_power / torch.abs(spec_power).max()
        spec_dB = 10 * torch.log10(spec_power_norm + 1e-8)
        specs.append(spec_dB)

    T_coef = np.arange(spec_dB.shape[1]) * hop_length / sr
    F_coef = np.arange(spec_dB.shape[0]) * sr / n_fft

    left1, right1 = min(T_coef), max(T_coef) + n_fft / sr
    lower1, upper1 = min(F_coef) / 1000, max(F_coef) / 1000

    # equaliy distribute with taking the edges into account
    selected_specs = [specs[i] for i in np.round(np.linspace(0, len(specs) - 1, ncols)).astype(int)]

    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)

    for i, spec_dB in enumerate(selected_specs):
        axs[i].imshow(spec_dB, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax, extent=[left1, right1, lower1, upper1])
        axs[i].set_xticks([])  # Remove x ticks
        axs[i].set_yticks([])  # Remove y ticks
        axs[i].axis('off')  
        # axs[i].set_xlabel('Time [s]')
        # axs[i].set_ylabel('Frequency [kHz]')
        # axs[i].set_title(f'Audio {i}')

    # fig.colorbar(axs[1].imshow(spec_dB2, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax, extent=[left1, right1, lower1, upper1]), ax=axs[1])
    plt.subplots_adjust(hspace=1.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()

def plot_audio(audio, sr, title='Audio Recording'):

    # Create a time array for the x-axis
    duration = len(audio) / sr
    time = np.linspace(0, duration, int(duration * sr)) 

    plt.figure(figsize=(10, 3))
    plt.plot(time, audio)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.xlim(0, duration)
    plt.ylim(-1, 1)
    plt.grid(True)
    plt.show()