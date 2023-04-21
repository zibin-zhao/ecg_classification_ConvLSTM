import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.signal


X = np.loadtxt('data_3600/X_MBa_MLII.csv', delimiter=',', skiprows=1).astype(np.float32)
y = np.loadtxt('data_3600/y_MBa_MLII.csv', delimiter=',', dtype=str, skiprows=1)   # n_samples, 1

# lead2 = db.iloc[0:50000, 1].to_numpy()  # extract lead 2 as the sample lead
# lead2 = np.array(lead2).reshape(-1, lead2.shape[0]) # reshape into 
# print(lead2.shape)
# #plt.plot(lead2)

# # data normalizing
# norm_lead2 = preprocessing.normalize(lead2, axis=1)
# norm_lead2 = np.transpose(norm_lead2)
# # print(norm_lead2)
# # print(norm_lead2.shape)
# # print(len(norm_lead2))

# re_signal = scipy.signal.resample(norm_lead2, 3600)
# print(X.shape)
# print(y[1])

plt.plot(X[1,:])
plt.xlabel("Data")
plt.ylabel("Amplitude")
plt.title("N type of Train Data from MITBIH-arrythmia")
plt.show()