1. wg, irm, iam functions:
   These are mask estimation functions:
   - wg: Wiener Gain
   - irm: Ideal Ratio Mask
   - iam: Ideal Amplitude Mask
   They calculate different types of masks based on the clean signal (S) and noisy signal (X).

2. Stft class:
   Implements the Short-Time Fourier Transform (STFT) as a PyTorch module.

3. Istft class:
   Implements the Inverse Short-Time Fourier Transform (ISTFT) as a PyTorch module.

4. MultiResSpecLoss class:
   Implements a multi-resolution spectrogram loss, which computes loss across multiple FFT sizes.

5. SpectralLoss class:
   Implements a spectral loss function, considering both magnitude and complex components of the spectrogram.

6. MaskLoss class:
   Implements a loss function for mask-based speech enhancement, supporting different mask types and loss computations in the ERB domain.

7. MaskSpecLoss class:
   A variant of MaskLoss that applies the mask to the noisy signal before computing the spectral loss.

8. DfAlphaLoss class:
   Implements a loss function to penalize the use of deep filtering in very noisy segments.

9. SiSdr class:
   Implements the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss.

10. SdrLoss and SegSdrLoss classes:
    Wrapper classes for SDR (Signal-to-Distortion Ratio) loss, with SegSdrLoss supporting segmental SDR computation.

11. LocalSnrLoss class:
    Implements a loss based on local Signal-to-Noise Ratio (SNR) estimates.

12. ASRLoss class:
    Implements a loss based on Automatic Speech Recognition (ASR) using the Whisper model.

13. Loss class:
    A wrapper class that combines multiple loss functions, including MaskLoss, SpectralLoss, MultiResSpecLoss, SdrLoss, LocalSnrLoss, and ASRLoss.

14. test_local_snr function:
    A test function to demonstrate the usage of local SNR calculation and visualization.