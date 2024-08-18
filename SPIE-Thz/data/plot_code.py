# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 08:43:31 2024

@author: chengjie
"""
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns


new_file_path = 'C:/Users/chengjie/Desktop/pca_result_on-normzlized data.csv'
data = pd.read_csv(new_file_path)


plt.figure(figsize=(10, 8))
#sns.scatterplot(data=data, x='PC1', y='PC2', hue='label 4', style='label 3', palette='viridis')
sns.scatterplot(data=data, x='PC1', y='PC2', hue='label 1', palette='viridis')

plt.title('PCA: PC1 vs PC2 with Labels')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.show()