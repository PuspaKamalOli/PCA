import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()
print(cancer)
# above loaded data works like dictionary so:
cancer.keys()

print(cancer['DESCR'])
# converting into dataframe
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head(8)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
print(scaled_data.shape, x_pca.shape)
plt.figure(figsize=(8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='coolwarm')
plt.xlabel('first principal component')
plt.ylabel('second principal component')

# hence PCA is used to reduce high dimension to lower like we had 30 features which were reduced to only two
