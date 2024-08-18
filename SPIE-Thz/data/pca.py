# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:32:16 2024

@author: chengjie
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA


# Load the dataset
file_path = 'C:/Users/chengjie/Desktop/dataset3_downloadable.csv'
df = pd.read_csv(file_path)
df_select =  df[df['label 3'] == 4]

# Splitting the dataset into features and labels
features = df_select.iloc[:, :800]

labels = df_select.iloc[:, 800:]
labels_rest=labels.reset_index(drop=True,)

# Normalizing the features row-wise
# We transpose the features to treat each row as a 'feature' for the purpose of row-wise normalization
scaler_row = StandardScaler()
features_scaled_row_wise = scaler_row.fit_transform(features.T).T

# Converting back to DataFrame for better visualization
features_scaled_row_wise_df = pd.DataFrame(features_scaled_row_wise, columns=features.columns)

#%%
#plt.figure(figsize=(15, 5))
#plt.plot(features.iloc[0, :], label="Normalized Row 1 (All Features)")
#plt.title("Plot of the First Row of Normalized Data")
#plt.xlabel("Feature Index")
#plt.ylabel("Normalized Value")
#plt.legend()
#plt.show()


#%%

#Calcualte first derivative of the data in row  applying a Savitzky-Golay filter
# Applying the Savitzky-Golay filter to calculate the first derivative
# Window length and polynomial order need to be chosen. 
# Example: window length of 5 and a polynomial order of 2
window_length = 5  # must be odd
poly_order = 2

# Calculating the first derivative for every row in the normalized dataset
first_derivatives = savgol_filter(features_scaled_row_wise_df, window_length, poly_order, deriv=1, axis=1)

# Converting the result to a DataFrame for easier handling
first_derivatives_df = pd.DataFrame(first_derivatives)

#%%
pca = PCA(n_components=10)  # Example: reducing to 2 dimensions for visualization
pca_result = pca.fit_transform(first_derivatives_df)

# Converting the PCA result to a DataFrame for easier visualization
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(10)])

#%%
combined_df = pd.concat([labels_rest, pca_df], axis=1)


#%%

plt.figure(figsize=(10, 8))
#sns.scatterplot(data=data, x='PC1', y='PC2', hue='label 4', style='label 3', palette='viridis')
sns.scatterplot(data=combined_df, x='PC1', y='PC2', hue='label 4', style = 'label 2' , palette='viridis',s=200)

plt.title('PCA: PC1 vs PC2 with Labels')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.show()




