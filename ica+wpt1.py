import os
import numpy as np
import pywt
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch

# Class to perform Wavelet Packet Transform (WPT) and Independent Component Analysis (ICA) filtering
class WPTFilter:
    def __init__(self, wavelet='dmey', maxlevel=7, n_components=8):
        self.wavelet = wavelet  # Type of wavelet transform used
        self.maxlevel = maxlevel  # Maximum decomposition level
        self.n_components = n_components  # Number of independent components for ICA

    # Function to apply WPT filtering
    def _wpt_filter(self, data: np.ndarray) -> np.ndarray:
        channels, length = data.shape  # Get number of channels and data length

        # Perform wavelet packet decomposition for each channel
        wp = {
            f'c_{c}': pywt.WaveletPacket(
                data=data[c, :],
                wavelet=self.wavelet,
                mode='symmetric',
                maxlevel=self.maxlevel
            )
            for c in range(channels)
        }

        # Get node paths at the maximum decomposition level
        nodes = [node.path for node in wp['c_0'].get_level(self.maxlevel, 'natural')]

        # Compute energy of each node
        wp_nodes_energy = {
            node: np.std(
                np.sum(
                    np.array([wp[f'c_{c}'][node].data for c in range(channels)])**2,
                    axis=0
                )
            )
            for node in nodes
        }

        # Find the node with maximum energy (likely containing noise)
        max_energy_node = max(wp_nodes_energy, key=wp_nodes_energy.get)

        # Reconstruct signal after removing the noisy node
        filtered_data = np.zeros_like(data)
        for c in range(channels):
            del wp[f'c_{c}'][max_energy_node]  # Remove the highest energy node
            filtered_data[c, :] = wp[f'c_{c}'].reconstruct()

        return filtered_data

    # Function to apply WPT followed by ICA filtering
    def wptica_filter(self, data: np.ndarray) -> np.ndarray:
        # Apply WPT filtering
        data_wpt_filtered = self._wpt_filter(data)

        # Apply ICA to extract independent components
        ica = FastICA(
            n_components=self.n_components,
            random_state=0,
            max_iter=1000,  
            tol=1e-3         
        )
        ica_data = ica.fit_transform(data_wpt_filtered.T)  # Perform ICA on transposed data

        # Find the component with maximum standard deviation (likely noise)
        std_devs = np.std(ica_data, axis=0)
        max_std_index = np.argmax(std_devs)
        ica_data[:, max_std_index] = 0  # Remove the highest variance component

        # Reconstruct the cleaned EEG signal
        cleaned_data = ica.inverse_transform(ica_data).T  
        return cleaned_data

# Function to compute Signal-to-Noise Ratio (SNR)
def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    power_signal = np.mean(signal**2)
    power_noise = np.mean(noise**2)
    snr = 10 * np.log10(power_signal / power_noise)  # Compute SNR in dB
    return snr

# Function to extract numeric suffix from filename (for sorting)
def extract_numeric_suffix(filename: str) -> int:
    base, _ = os.path.splitext(filename)  # Remove file extension
    parts = base.split('_')  # Split filename by '_'
    try:
        return int(parts[-1])  # Extract last part as an integer (numeric suffix)
    except ValueError:
        return 0  # Default to 0 if no numeric suffix

# Main function to process all CSV files in the input directory
def process_directory(
    input_dir,
    output_dir,
    wavelet='dmey',
    maxlevel=7,
    n_components=8,
    snr_threshold=7
):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List and sort all CSV files numerically based on suffix
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    all_files_sorted = sorted(all_files, key=extract_numeric_suffix)

    # Initialize the filter processor with given parameters
    filter_processor = WPTFilter(wavelet=wavelet, maxlevel=maxlevel, n_components=n_components)
    
    required_metadata = {'timestamp', 'pkt_num'}  # Required columns

    # Iterate through each file in the sorted list
    for file_name in all_files_sorted:
        input_path = os.path.join(input_dir, file_name)
        raw_data = pd.read_csv(input_path)  # Read CSV file

        # Check if required metadata columns are present
        if required_metadata.issubset(raw_data.columns):
            all_columns = list(raw_data.columns)

            # Extract EEG channel names (excluding metadata columns)
            eeg_channels = [col for col in all_columns if col not in required_metadata]

            # Extract timestamp and packet number columns
            timestamps = raw_data['timestamp']
            pkt_nums = raw_data['pkt_num']

            # Convert EEG data to a NumPy array (channels x samples)
            eeg_data = raw_data[eeg_channels].to_numpy().T

            # Apply WPT and ICA filtering
            filtered_eeg = filter_processor.wptica_filter(eeg_data)

            # Compute SNR after filtering
            raw_signal_total = eeg_data.flatten()
            filtered_signal_total = filtered_eeg.flatten()
            noise_total = raw_signal_total - filtered_signal_total
            snr_value = compute_snr(filtered_signal_total, noise_total)

            # Save filtered data if SNR meets the threshold
            if snr_value >= snr_threshold:
                filtered_df = pd.DataFrame(filtered_eeg.T, columns=eeg_channels)
                filtered_df.insert(0, 'timestamp', timestamps)
                filtered_df.insert(1, 'pkt_num', pkt_nums)

                output_path = os.path.join(output_dir, file_name)
                filtered_df.to_csv(output_path, index=False)
                print(f"File '{file_name}' processed and saved with SNR {snr_value:.2f} dB.")
            else:
                print(f"File '{file_name}' skipped due to low SNR ({snr_value:.2f} dB).")
        else:
            print(f"File '{file_name}' skipped (missing 'timestamp' or 'pkt_num').")

# Main execution block
if __name__ == "__main__":
    input_directory = 'filtered_csv_files'   # Input directory with filtered EEG data
    output_directory = 'ICA_csv_files'       # Output directory for final processed data
    process_directory(input_directory, output_directory)
