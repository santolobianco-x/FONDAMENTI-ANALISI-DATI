import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,silhouette_samples


url = "https://gist.githubusercontent.com/pravalliyaram/5c05f43d2351249927b8a3f3cc3e5ecf/raw/8bd6144a87988213693754baaa13fb204933282d/Mall_Customers.csv"
df = pd.read_csv(url)

df = df.rename(columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'SpendingScore'
})

print(df.head())
print("\n---Data Info---")
df.info()


sns.pairplot(df.drop('CustomerID',axis=1), hue='Gender', diag_kind='kde')
plt.show()

X = df[['Income','SpendingScore']].values

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
plt.figure(figsize=(8,6))
sns.scatterplot(x= X_scaled[:,0], y = X_scaled[:,1], s=50)
plt.title('Scaled Customer Data')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.show()

k_values = range(1,11)
inertias = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_values, inertias, '-o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters(K)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_values)
plt.grid(True, linestyle='--', alpha=0.6)

plt.axvline(x=5, color='red', linestyle='--', label= 'Elbow Point (K = 5)')
plt.legend()
plt.show()

k_values_sil = range(2,11)
silhouette_scores = []

for k in k_values_sil:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled,labels)
    silhouette_scores.append(score)


plt.figure(figsize=(8,5))
plt.plot(k_values_sil,silhouette_scores, 'o-')
plt.title('Silhouette Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Silhouette Score')
plt.xticks(k_values_sil)
plt.grid(True, linestyle='--', alpha = 0.6)

best_k_sil = k_values_sil[np.argmax(silhouette_scores)]
plt.axvline(x=best_k_sil, color='red', linestyle='--', label=f'Best K = {best_k_sil}')
plt.legend()
plt.show()



final_kmeans = KMeans(n_clusters=best_k_sil,random_state=42, n_init=10)
final_labels = final_kmeans.fit_predict(X_scaled)

final_centers_scaled = final_kmeans.cluster_centers_

df['Cluster'] = final_labels

plt.figure(figsize=(10,8))
sns.scatterplot(x ='Income', y='SpendingScore', data=df, hue='Cluster',palette='viridis', s= 100, alpha =0.9)


final_centers_original = scaler.inverse_transform(final_centers_scaled)
plt.scatter(final_centers_original[:,0], final_centers_original[:,1], marker='X',s=300, c='red', edgecolors='black', label='Cluster Centers')


plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True, linestyle = '--', alpha =0.5)
plt.show()


cluster_analysis = df.groupby('Cluster')[['Income','SpendingScore']].mean()
print(cluster_analysis)

