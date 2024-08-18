# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 00:14:55 2024

@author: chengjie
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize

#%% Load the  original data
# Load the dataset
file_path = 'C:/Users/chengjie/Desktop/dataset6.csv'
df = pd.read_csv(file_path)
#df_select =  df[df['label 3'] == 4]

# Splitting the dataset into features and labels
features = df.iloc[:, 6:]
    
labels = df.iloc[:, :6]
labels_rest=labels.reset_index(drop=True,)

# Time settings for the x-axis
start_time = 1780  # in ps
time_step = 0.05  # in ps
time_axis = np.arange(start_time, start_time + time_step * features.shape[1], time_step)

# Plotting the spectra
plt.rcParams.update({'font.size': 20})


normalized_data = normalize(features, norm='l1', axis=1)

# Updating the dataset with the normalized data
#data.update(pd.DataFrame(normalized_data, columns=feature_data.columns))
features_l1_df = pd.DataFrame(normalized_data, columns=features.columns)

combined_l1_df = pd.concat([labels_rest, features_l1_df], axis=1)

#%%tsec
# Perform t-SNE with 2 components
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features_l1_df )
tsne_df = pd.DataFrame(tsne_result, columns=[f'PC{i+1}' for i in range(2)])

combined_tsne_df = pd.concat([labels_rest, tsne_df], axis=1)
plt.rcParams.update({'font.size': 45})
# Plotting the t-SNE result
plt.figure(figsize=(30, 25))
sns.scatterplot(data=combined_tsne_df, x='PC1', y='PC2', hue='Scan', style = 'AC', palette='bright',s=200)

plt.title('t-SNE projection of Standard Scale THz-TDS')
plt.xlabel('tsne 1')
plt.ylabel('tsne 2')
plt.show()