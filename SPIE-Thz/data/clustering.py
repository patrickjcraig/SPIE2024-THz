# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:17:47 2024

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
file_path = r'C:\Users\patri\git\SPIE-Thz\data\dataset6.csv'
df = pd.read_csv(file_path)
#df_select =  df[df['label 3'] == 4]

# Splitting the dataset into features and labels
features = df.iloc[:, 6:]
    
labels = df.iloc[:, :6]
labels_rest=labels.reset_index(drop=True,)

# Time settings for the x-axis
start_time = 1790  # in ps
time_step = 0.05  # in ps
time_axis = np.arange(start_time, start_time + time_step * features.shape[1], time_step)

# Plotting the spectra
plt.rcParams.update({'font.size': 20})
#%%
plt.figure(figsize=(12, 8))

# Plotting spectra by category (Authentic and Counterfeit)
for index, row in df.iterrows():
    if row['AC'] == 'Authentic':
        color = 'green'
        label = 'Authentic'
    elif row['AC'] == 'Counterfeit':
        color = 'red'
        label = 'Counterfeit'
    else:
        continue  # Skip rows that are neither Authentic nor Counterfeit
    plt.plot(time_axis, row[6:], color=color, alpha=0.5, label=label)

plt.title('Original THz-TDS Spectral Data ')
plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
# Adding a legend and ensuring each label is only added once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.show()
#%%   # Normalizing the features row-wise
# We transpose the features to treat each row as a 'feature' for the purpose of row-wise normalization
scaler_row = StandardScaler()
features_scaled_row_wise = scaler_row.fit_transform(features.T).T

# Converting back to DataFrame for better visualization
features_scaled_row_wise_df = pd.DataFrame(features_scaled_row_wise, columns=features.columns)

combined_scaled_df = pd.concat([labels_rest, features_scaled_row_wise_df], axis=1)

plt.figure(figsize=(12, 8))

# Plotting spectra by category (Authentic and Counterfeit)
for index, row in combined_scaled_df.iterrows():
    if row['AC'] == 'Authentic':
        color = 'green'
        label = 'Authentic'
    elif row['AC'] == 'Counterfeit':
        color = 'red'
        label = 'Counterfeit'
    else:
        continue  # Skip rows that are neither Authentic nor Counterfeit
    plt.plot(time_axis, row[6:], color=color, alpha=0.5, label=label)

plt.title('Standard Scale THz-TDS Spectral Data ')
plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
# Adding a legend and ensuring each label is only added once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.show()
#%% L1
# Applying L1 normalization to the feature columns for each row
normalized_data = normalize(features, norm='l1', axis=1)

# Updating the dataset with the normalized data
#data.update(pd.DataFrame(normalized_data, columns=feature_data.columns))
features_l1_df = pd.DataFrame(normalized_data, columns=features.columns)

combined_l1_df = pd.concat([labels_rest, features_l1_df], axis=1)

plt.figure(figsize=(12, 8))

# Plotting spectra by category (Authentic and Counterfeit)
for index, row in combined_l1_df.iterrows():
    if row['AC'] == 'Authentic':
        color = 'green'
        label = 'Authentic'
    elif row['AC'] == 'Counterfeit':
        color = 'red'
        label = 'Counterfeit'
    else:
        continue  # Skip rows that are neither Authentic nor Counterfeit
    plt.plot(time_axis, row[6:], color=color, alpha=0.5, label=label)

plt.title('L1 Normalized THz-TDS Spectral Data ')
plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
# Adding a legend and ensuring each label is only added once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.show()

#%% Standard + L1
standard_l1_data = normalize(features_scaled_row_wise_df, norm='l1', axis=1)

# Updating the dataset with the normalized data
#data.update(pd.DataFrame(normalized_data, columns=feature_data.columns))
standard_l1_df = pd.DataFrame(standard_l1_data, columns=features.columns)

combined_standard_l1_data_df = pd.concat([labels_rest, standard_l1_df], axis=1)

plt.figure(figsize=(12, 8))

# Plotting spectra by category (Authentic and Counterfeit)
for index, row in combined_standard_l1_data_df.iterrows():
    if row['AC'] == 'Authentic':
        color = 'green'
        label = 'Authentic'
    elif row['AC'] == 'Counterfeit':
        color = 'red'
        label = 'Counterfeit'
    else:
        continue  # Skip rows that are neither Authentic nor Counterfeit
    plt.plot(time_axis, row[6:], color=color, alpha=0.5, label=label)

plt.title('Standard Scale+L1 Normalized THz-TDS Spectral Data ')
plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
# Adding a legend and ensuring each label is only added once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.show()
#%% PCA
pca = PCA(n_components=10)  # Example: reducing to 2 dimensions for visualization
pca_result = pca.fit_transform(features_scaled_row_wise_df)

# Converting the PCA result to a DataFrame for easier visualization
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(10)])

combined_pca_df = pd.concat([labels_rest, pca_df], axis=1)

plt.figure(figsize=(10, 8))
#sns.scatterplot(data=data, x='PC1', y='PC2', hue='label 4', style='label 3', palette='viridis')
sns.scatterplot(data=combined_pca_df, x='PC1', y='PC2', hue='AC',  palette='bright',s=50)

plt.title('PCA of Standard Scale THz-TDS')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.show()

#%%tsec
# Perform t-SNE with 2 components
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features_scaled_row_wise_df )
tsne_df = pd.DataFrame(tsne_result, columns=[f'PC{i+1}' for i in range(2)])

combined_tsne_df = pd.concat([labels_rest, tsne_df], axis=1)

# Plotting the t-SNE result
plt.figure(figsize=(10, 8))
sns.scatterplot(data=combined_tsne_df, x='PC1', y='PC2', hue='AC',  palette='bright',s=50)

plt.title('t-SNE projection of Standard Scale THz-TDS')
plt.xlabel('tsne 1')
plt.ylabel('tsne 2')
plt.show()

#%% umap

umap_model = umap.UMAP(n_neighbors=15, min_dist=1, n_components=2, random_state=50)
umap_result = umap_model.fit_transform(features_scaled_row_wise_df)
umap_result_df = pd.DataFrame(umap_result,columns=[f'PC{i+1}' for i in range(2)])


combined_umap_df = pd.concat([labels_rest, umap_result_df], axis=1)

# Plotting the results
plt.figure(figsize=(10, 8))
#sns.scatterplot(combined_df.iloc[:, 0], y=umap_result_df.iloc[:, 1], hue=AC, palette='viridis', legend='full')
sns.scatterplot(data=combined_umap_df, x='PC1', y='PC2', hue='AC', palette='bright',s=50)

plt.title('UMAP Projection of L1 Normalizedfeatures_l1_df THz-TDS')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()



#%%min-max
scaler = MinMaxScaler()

# Applying MinMax normalization to the spectral data
minmax_normalized_spectral_data = scaler.fit_transform(features)

features_minmax_row_wise_df = pd.DataFrame(minmax_normalized_spectral_data, columns=features.columns)

combined_minmax_scaled_df = pd.concat([labels_rest, features_minmax_row_wise_df], axis=1)


plt.figure(figsize=(12, 8))

# Plotting spectra by category (Authentic and Counterfeit)
for index, row in combined_minmax_scaled_df.iterrows():
    if row['AC'] == 'Authentic':
        color = 'green'
        label = 'Authentic'
    elif row['AC'] == 'Counterfeit':
        color = 'red'
        label = 'Counterfeit'
    else:
        continue  # Skip rows that are neither Authentic nor Counterfeit
    plt.plot(time_axis, row[6:], color=color, alpha=0.5, label=label)

plt.title('Normalized THz-TDS Spectral Data of Scans')
plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
# Adding a legend and ensuring each label is only added once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.show()


#%%# Taking the first derivative of the spectral data in each row
first_derivative = np.diff(features_scaled_row_wise_df, axis=1)

# Since taking the derivative reduces the number of columns by one, we adjust the column names
derivative_column_names = [str(i) + "-" + str(i+1) for i in range(1, features_scaled_row_wise_df.shape[1])]

# Creating a new dataframe for the first derivative data
first_derivative_dataset = pd.DataFrame(first_derivative, columns=derivative_column_names)

combined_scaled_first_df = pd.concat([labels_rest, first_derivative_dataset ], axis=1)

plt.figure(figsize=(12, 8))

# Plotting spectra by category (Authentic and Counterfeit)
for index, row in combined_scaled_first_df.iterrows():
    if row['AC'] == 'Authentic':
        color = 'green'
        label = 'Authentic'
    elif row['AC'] == 'Counterfeit':
        color = 'red'
        label = 'Counterfeit'
    else:
        continue  # Skip rows that are neither Authentic nor Counterfeit
    plt.plot(row[6:], color=color, alpha=0.5, label=label)

plt.title('L1 + Normalized on THz-TDS  Data ')
#plt.xlabel('Time (ps)')
plt.ylabel('Intensity')
# Adding a legend and ensuring each label is only added once
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())


plt.show()
