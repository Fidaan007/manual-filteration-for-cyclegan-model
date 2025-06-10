import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import stft

def generate_separate_stft_plots(
    input_file, 
    output_dir, 
    label, 
    fs=256, 
    nperseg=256, 
    noverlap=128
):
    """
    Compute and plot STFT for each EEG channel in separate subplots (within one figure).
    Saves the figure to output_dir as '{label}_stft_separate_channels.png'.
    
    Args:
        input_file (str): Path to the CSV file containing EEG data.
        output_dir (str): Directory to save the STFT images.
        label (str): Label for the dataset (e.g., 'raw' or 'filtered').
        fs (int): Sampling frequency of the EEG data (default: 256 Hz).
        nperseg (int): Length of each segment for STFT (default: 256 samples).
        noverlap (int): Number of overlapping samples (default: 128 samples).
    """
    # Load data from CSV
    data = pd.read_csv(input_file)
    
    # Preserve original column order, then remove metadata columns
    all_columns = list(data.columns)
    metadata_columns = ['timestamp', 'pkt_num']
    eeg_channels = [col for col in all_columns if col not in metadata_columns]
    
    # Extract EEG data as (num_channels, num_samples)
    eeg_data = data[eeg_channels].values.T
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare subplots: one row per channel
    num_channels = len(eeg_channels)
    fig_height = 2 * num_channels  # adjust as needed

    # Use constrained_layout=True to avoid tight_layout warnings
    fig, axs = plt.subplots(
        num_channels, 1, 
        figsize=(12, fig_height), 
        sharex=True, 
        constrained_layout=True
    )
    
    # If there's only one channel, axs won't be an array; make it iterable
    if num_channels == 1:
        axs = [axs]
    
    # Compute STFT for each channel and plot in its own subplot
    im = None
    for i, ch_name in enumerate(eeg_channels):
        f, t, Zxx = stft(eeg_data[i, :], fs=fs, nperseg=nperseg, noverlap=noverlap)
        
        # Convert magnitude to dB scale
        spectrogram_dB = 10 * np.log10(np.abs(Zxx) + 1e-8)
        
        # Plot the spectrogram
        im = axs[i].imshow(
            spectrogram_dB,
            aspect='auto',
            cmap='jet',
            origin='lower',
            extent=[t[0], t[-1], f[0], f[-1]]
        )
        
        # Label the y-axis as frequency
        axs[i].set_ylabel('Frequency (Hz)')
        # Put the electrode name in the subplot title
        axs[i].set_title(f'Channel: {ch_name}')
    
    # The bottom subplot gets the time label
    axs[-1].set_xlabel('Time (s)')
    
    # Add a single colorbar on the right side
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Power (dB)')
    
    # Give the whole figure a title
    fig.suptitle(f'{label.upper()} - STFT for Each Channel', fontsize=14)
    
    # Save the figure
    output_path = os.path.join(output_dir, f'{label}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"Spectrograms saved to {output_path}")

# -----------------------------
# Example usage:

raw_file = 'ur file.csv'  # Path to the raw EEG CSV file
filtered_file = 'filtered_eeg_all.csv'  # Path to the filtered EEG CSV file

# Output directories for the spectrogram images
output_raw_stft_folder = 'output/raw_stft'
output_filtered_stft_folder = 'output/filtered_stft'

# Process individual raw CSV file
generate_separate_stft_plots(
    input_file=raw_file,
    output_dir=output_raw_stft_folder,
    label='raw'
)

# Process individual filtered CSV file
generate_separate_stft_plots(
    input_file=filtered_file,
    output_dir=output_filtered_stft_folder,
    label='filtered'
)
