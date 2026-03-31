import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier


iris = load_iris()
X_i = iris.data
y_i = iris.target

target_names_i = iris.target_names

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_i,y_i, test_size=0.3,random_state=42, stratify=y_i)



#SOFTMAX

pipe_multi = Pipeline(
    [
        ('scaler',StandardScaler()),
        ('model',LogisticRegression(solver='lbfgs',max_iter=200))
    ]
)

param_grid_multi ={
    'model__C': np.logspace(-3,3,7)
}

print("Running Grid Search for Multinomial(Softmax)...")

search_multi = GridSearchCV(pipe_multi,param_grid_multi,cv=5, scoring='accuracy')
search_multi.fit(X_train_i,y_train_i)
print(f"Best 'C' for Multinomial: {search_multi.best_params_['model__C']}")
print(f"Best CV Accuracy: {search_multi.best_score_:.4f}")


#OVR

pipe_ovr = Pipeline([
    ('scaler', StandardScaler()),
    ('model',OneVsRestClassifier(LogisticRegression(solver='liblinear')))
])

param_grid_ovr = {
    'model__estimator__C': np.logspace(-3,3,7)
}

print("Running Grid Search for One-vs-Rest(OvR)...")
search_ovr = GridSearchCV(pipe_ovr,param_grid_ovr,cv=5, scoring='accuracy')
search_ovr.fit(X_train_i,y_train_i)


print(f"Best 'C' for OvR: {search_ovr.best_params_['model__estimator__C']}")
print(f"Best CV Accuracy: {search_ovr.best_score_:.4f}")



#CONFRONTO

y_predi_multi = search_multi.predict(X_test_i)
acc_multi = accuracy_score(y_test_i,y_predi_multi)

y_predi_ovr = search_ovr.predict(X_test_i)
acc_ovr = accuracy_score(y_test_i,y_predi_ovr)


results = {
    "Multinomial(Softmax)": acc_multi,
    "One-vs-Rest": acc_ovr
}

results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Test Accuracy'])

print("\n--- Multi-Class Model Comparison ---")
print(results_df.sort_values(by='Test Accuracy', ascending=False))

print("\n--- Report for Multinomial (Softmax) Model ---")
print(classification_report(y_test_i, y_predi_multi, target_names=target_names_i))

print("\n--- Report for One-vs-Rest (OvR) Model ---")
print(classification_report(y_test_i, y_predi_ovr, target_names=target_names_i))