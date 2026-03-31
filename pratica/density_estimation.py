import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_curve,auc, roc_auc_score


X_all, y_all = load_digits(return_X_y=True)



X_normal_all = X_all[y_all != 0]
y_normal_all = y_all[y_all != 0]




X_anomaly_all = X_all[y_all == 0]
y_anomaly_all = np.ones(len(X_anomaly_all))


X_train_normal, X_test_normal, _ ,_ = train_test_split(X_normal_all, y_normal_all, test_size=0.3, random_state=42, stratify=y_normal_all) 


y_test_normal = np.zeros(len(X_test_normal))

X_test = np.concatenate((X_test_normal, X_anomaly_all))
y_test = np.concatenate((y_test_normal, y_anomaly_all))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_normal)
X_test_scaled = scaler.transform(X_test)





model_single_gauss = GaussianMixture(n_components=1, random_state=42)
model_single_gauss.fit(X_train_scaled)


model_gmm = GaussianMixture(n_components=4,random_state=42)
model_gmm.fit(X_train_scaled)



model_kde = KernelDensity(kernel='gaussian')
model_kde.fit(X_train_scaled)




scores_single = -model_single_gauss.score_samples(X_test_scaled)
scores_gmm = -model_gmm.score_samples(X_test_scaled)
scores_kde = -model_kde.score_samples(X_test_scaled)




result_df = pd.DataFrame({
    'y_true': y_test,
    'SingleGaussian_NLL': scores_single,
    'GMM_K4_NLL': scores_gmm,
    'KDE_NLL': scores_kde
})

result_df['Label'] = result_df['y_true'].map({0: 'Normal(1-9)', 1: 'Anomaly(0)'})




plt.figure(figsize=(6, 6))


fpr, tpr, _ = roc_curve(result_df['y_true'], result_df['SingleGaussian_NLL'])
auc_single = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, label=f'Single Gaussian (AUC = {auc_single:.3f})')


fpr, tpr, _ = roc_curve(result_df['y_true'], result_df['GMM_K4_NLL'])
auc_gmm = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, label=f'GMM (K=4) (AUC = {auc_gmm:.3f})')


fpr, tpr, _ = roc_curve(result_df['y_true'], result_df['KDE_NLL'])
auc_kde = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, label=f'KDE (AUC = {auc_kde:.3f})')


plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC = 0.500)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison for Anomaly Detection (Finding \'0\'s)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

