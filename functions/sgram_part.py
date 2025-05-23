from scipy.signal import spectrogram
import numpy as np


def sgram_part(
    sig,
    fs,
    win_width,
    frs,
    overlap=0,
    window='hann',
):
    f, t, Sxx = spectrogram(
        sig,
        fs=fs,
        nperseg=int(fs * win_width),
        noverlap=int(fs * win_width * overlap / 100),
        window=window,
    )

    if frs[1] < 0:
        f_indices = np.where(f >= frs[0])[0]
    else:
        f_indices = np.where((f >= frs[0]) & (f <= frs[1]))[0]

    f = f[f_indices]
    Sxx = Sxx[f_indices, :]

    PSD = np.mean(Sxx, axis=0)

    return f, t, Sxx, PSD
