import librosa
import numpy as np

def extract_features(path, max_len=16000):
    y, sr = librosa.load(path, sr=16000)

    # Pad/Trim
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1, -1)
