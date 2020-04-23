# %%

from helpers import *
import os, glob, cv2, random
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
# Preview

base_path = 'dataset'

# %%

X, y = read_csv(os.path.join(base_path, 'dataset.csv'))

# Add likelihood 1.0
y = np.hstack((y, np.ones((y.shape[0], 1))))

print(X.shape, y.shape)

# %% Add false dataset

fX = np.random.random((2874, 26, 34, 1)) * 256
fy = np.zeros((2874, 2))
fy[:, 0] = (np.random.rand(2874) > 0.5).astype(int)

X = np.vstack((X, fX))
y = np.vstack((y, fy))
print(X.shape, y.shape)

# %% Preprocessing

n_total = len(X)
X_result = np.empty((n_total, 26, 34, 1))

for i, x in enumerate(X):
    img = x.reshape((26, 34, 1))

    X_result[i] = img

x_train, x_val, y_train, y_val = train_test_split(X_result, y, test_size=0.1)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)

np.save('dataset/x_train.npy', x_train)
np.save('dataset/y_train.npy', y_train)
np.save('dataset/x_val.npy', x_val)
np.save('dataset/y_val.npy', y_val)

# %%
