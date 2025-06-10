import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from mne import create_info
from mne.io import RawArray

# Initialize a global variable for file numbering
n = 1

# Function to extract numeric suffix from a filename
def extract_numeric_suffix(filename):
    base, _ = os.path.splitext(filename)  # Remove file extension
    parts = base.split('_')  # Split filename by underscores
    try:
        return int(parts[-1])  # Return the last part as an integer (if numeric)
    except ValueError:
        return 0  # Return 0 if no numeric suffix is found

# Function to apply a bandpass filter to EEG data
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency (half the sampling rate)
    low = lowcut / nyquist  # Normalize low cutoff frequency
    high = highcut / nyquist  # Normalize high cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Design a Butterworth bandpass filter
    return filtfilt(b, a, data)  # Apply the filter using zero-phase filtering

# Function to apply a notch filter to remove powerline noise
def notch_filter(data, notch_freq, fs, quality_factor=30):
    b, a = iirnotch(notch_freq / (0.5 * fs), quality_factor)  # Design a notch filter
    return filtfilt(b, a, data)  # Apply the filter using zero-phase filtering

# Function to process all EEG data files in the input directory
def process_directory(input_dir, output_dir, fs=256, lowcut=1.0, highcut=60.0, notch_freq=60.0):
    n = 1  # Local variable for numbering output files
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List all CSV files in the input directory
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    
    # Sort files based on their numeric suffix to process them sequentially
    all_files_sorted = sorted(all_files, key=extract_numeric_suffix)

    # Iterate through each CSV file in the sorted list
    for file_name in all_files_sorted:
        input_path = os.path.join(input_dir, file_name)  # Full path to the input file
        
        # Read the CSV file while skipping problematic lines
        eeg_data = pd.read_csv(input_path, delimiter=',', on_bad_lines='skip')

        # Extract all column names
        all_columns = list(eeg_data.columns)
        
        # Define non-EEG columns that should not be filtered
        non_eeg_columns = ['pkt_num', 'timestamp']
        
        # Identify EEG channel columns (all columns except the non-EEG ones)
        eeg_channels = [col for col in all_columns if col not in non_eeg_columns]

        # Convert EEG data to a NumPy array (transpose to match channels x samples)
        eeg_data_numeric = eeg_data[eeg_channels].values.T

        # Apply bandpass and notch filtering to each EEG channel
        filtered_eeg_data = np.array([
            notch_filter(bandpass_filter(ch, lowcut, highcut, fs), notch_freq, fs) 
            for ch in eeg_data_numeric
        ])

        # Convert filtered EEG data back to a DataFrame
        filtered_df = pd.DataFrame(filtered_eeg_data.T, columns=eeg_channels)

        # Add the non-EEG columns back to the DataFrame
        filtered_df['pkt_num'] = eeg_data['pkt_num']
        filtered_df['timestamp'] = eeg_data['timestamp']

        # Ensure the DataFrame maintains the correct column order
        filtered_df = filtered_df[non_eeg_columns + eeg_channels]

        # Define the output file path with a unique number
        output_path = os.path.join(output_dir, f"filtered_{n}.csv")
        n += 1  # Increment file numbering

        # Save the filtered data to a new CSV file
        filtered_df.to_csv(output_path, index=False)
        print(f"Filtered data saved to {output_path}")

# Main execution block
if __name__ == "__main__":
    input_directory = 'surrogate_data'  # Define input directory containing raw EEG CSV files
    output_directory = 'filtered_csv_files'  # Define output directory for processed EEG files
    process_directory(input_directory, output_directory)  # Start processing files
