import os
import numpy as np
import pandas as pd

def time_shift_small(channel_data, max_shift=1):
    """Circularly shift the channel by up to ±1 sample (very small shift)."""
    shift_amount = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(channel_data, shift_amount)

def amplitude_scale_small(channel_data, scale_range=(0.99, 1.01)):
    """Randomly scale amplitude by ±1% (very small scaling)."""
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    return channel_data * scale_factor

def add_random_noise_small(channel_data, noise_std=0.001):
    """
    Inject Gaussian noise with std = 0.1% of the channel's own std.
    This is a very minimal noise addition.
    """
    channel_std = np.std(channel_data)
    noise = np.random.normal(0, noise_std * channel_std, size=channel_data.shape)
    return channel_data + noise

def create_120sec_surrogates(
    raw_csv_path, 
    output_dir, 
    fs=256,               # sampling rate (Hz)
    n_surrogates=3, 
    replicate_factor=12   # 12 * 10s => 120s
):
    """
    Generate multiple 120-second surrogate EEG CSVs from a CSV containing >= 10s of data.
    
    1) We forcibly slice the first 10s from the CSV (based on fs).
    2) We replicate that 10s chunk 'replicate_factor' times => total 120s.
    3) For each chunk, we apply tiny random augmentations (time shift, scale, noise).
    4) We generate new timestamps from 0..120s (with 'fs' samples per second).
    
    :param raw_csv_path: Path to the original CSV (must have >=10s data).
    :param output_dir: Directory to save the 120s surrogate CSVs.
    :param fs: Sampling rate in Hz (default 256).
    :param n_surrogates: How many separate 120s files to create.
    :param replicate_factor: Replicate 10s chunk N times => 10*N total seconds.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1) Load the raw data
    raw_data = pd.read_csv(raw_csv_path)
    
    # 2) Identify metadata vs. EEG columns
    metadata_cols = ['timestamp', 'pkt_num']
    all_cols = list(raw_data.columns)
    eeg_channels = [c for c in all_cols if c not in metadata_cols]
    
    # 3) Figure out how many samples correspond to 10 seconds
    samples_10s = 10 * fs  # e.g., 2560 for 256 Hz
    
    # Check if the CSV has enough rows
    if len(raw_data) < samples_10s:
        raise ValueError(
            f"CSV has only {len(raw_data)} rows but we need at least {samples_10s} for 10 seconds!"
        )
    
    # 4) Slice the first 10s from the CSV
    raw_data_10s = raw_data.iloc[:samples_10s].copy()
    
    # 5) Extract EEG array (channels, samples)
    eeg_array_10s = raw_data_10s[eeg_channels].to_numpy().T
    num_channels, num_samples_10s = eeg_array_10s.shape  # (Nch, 2560) if fs=256
    
    # 6) We'll create new timestamps from 0..120s
    total_samples_120s = replicate_factor * samples_10s
    new_time = np.linspace(0, replicate_factor * 10, total_samples_120s, endpoint=False)
    
    # We'll replicate the first 10s' pkt_num pattern
    pkt_num_10s = raw_data_10s['pkt_num'].values
    
    # Create the surrogates
    base_name = os.path.splitext(os.path.basename(raw_csv_path))[0]
    
    for s_idx in range(n_surrogates):
        chunk_list = []
        pkt_num_list = []
        
        for rep_idx in range(replicate_factor):
            # Copy the 10s array
            chunk_array = eeg_array_10s.copy()
            
            # Apply very small augmentations
            for ch_idx in range(num_channels):
                channel_data = chunk_array[ch_idx, :]
                channel_data = time_shift_small(channel_data, max_shift=1)
                channel_data = amplitude_scale_small(channel_data, scale_range=(0.99, 1.01))
                channel_data = add_random_noise_small(channel_data, noise_std=0.001)
                chunk_array[ch_idx, :] = channel_data
            
            chunk_list.append(chunk_array)
            pkt_num_list.append(pkt_num_10s)
        
        # Concatenate all chunk arrays along time axis
        big_array = np.hstack(chunk_list)
        
        # shape => (time_samples, channels)
        big_df = pd.DataFrame(big_array.T, columns=eeg_channels)
        
        # Flatten packet numbers
        big_pkt_num = np.concatenate(pkt_num_list, axis=0)
        
        # Insert new_time and pkt_num
        big_df.insert(0, 'timestamp', new_time)
        big_df.insert(1, 'pkt_num', big_pkt_num)
        
        # Save
        out_filename = f"surrogate_{s_idx+1}.csv"
        out_path = os.path.join(output_dir, out_filename)
        big_df.to_csv(out_path, index=False)
        
        print(f"Saved 120s surrogate file: {out_path}")

# ---------------- Example usage ----------------
if __name__ == "__main__":
    raw_csv_file = "s10.csv"        # Must have >=10s data
    output_directory = "surrogate_data"
    
    create_120sec_surrogates(
        raw_csv_path=raw_csv_file,
        output_dir=output_directory,
        fs=256,            # or your actual sampling rate
        n_surrogates=20,    # create 3 big surrogates
        replicate_factor=12
    )
