import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def preprocess_vector(v):
    arr = np.array(v).reshape(1, -1)
    return scaler.fit_transform(arr)
