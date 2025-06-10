# 🧠 EEG Artifact Removal Using CycleGAN

A final-year B.Tech project by students of Electronics & Communication Engineering at NSS College of Engineering, Palakkad, APJ Abdul Kalam Technological University.

## 📘 Overview

This project explores the use of **CycleGAN**, a deep learning framework, for **automated artifact removal** from EEG signals during **motor imagery tasks**. By leveraging unpaired image translation, we enhance the clarity of EEG spectrograms without requiring clean-noisy training pairs.

## 🔬 Highlights

- 🎯 **Motor Imagery EEG Classification** (Left/Right hand)
- 🧼 **Artifact Removal** using:
  - Bandpass & Notch Filters
  - ICA + Wavelet Packet Transform (WPT)
- 🎨 **STFT-based Spectrogram Generation**
- 🤖 **CycleGAN** for unpaired EEG spectrogram denoising
- 📈 Improved signal quality validated by SNR, SSIM & classification results

## ⚙️ Tech Stack

- Python, NumPy, Pandas, SciPy, Matplotlib
- MNE, PyWavelets, Scikit-learn
- PyTorch (for CycleGAN)
- EEG Hardware: Multi-electrode EEG Cap

## 🧪 Results

- Enhanced EEG clarity for MI classification
- Successful CycleGAN training on unpaired 256x256 spectrograms
- Demonstrated application in BCI and neurorehabilitation

## 👨‍💻 Contributors

- Abhijith S
- Akshaya M K
- Amal M
- Fidaan Hussain P

---

> For academic purposes only. All datasets and tools used are properly cited in the full report.
