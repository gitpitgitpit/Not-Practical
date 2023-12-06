import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris=load_iris()
data=iris.data
target=iris.target
feature_names=iris.feature_names

df=pd.DataFrame(data,columns=feature_names)
df["target"]=target
print(df.head())
scaler=StandardScaler()
scaled_data=scaler.fit_transform(data)

df2=pd.DataFrame(scaled_data,columns=feature_names)
print(df2.head())

pca=PCA()
pca_result=pca.fit_transform(scaled_data)

explained_variance_ratio=pca.explained_variance_ratio_
print(explained_variance_ratio)

plt.xlabel("no of components")
plt.ylabel("explained variance")
plt.plot(np.cumsum(explained_variance_ratio))
plt.show()

pca=PCA(n_components=2)
reduced_data=pca.fit_transform(scaled_data)

reduced_df=pd.DataFrame(reduced_data,columns=[f'pc{i+1}'for i in range(2)])
reduced_df["target"]=target
print(reduced_df)