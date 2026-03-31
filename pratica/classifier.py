import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report


data = pd.read_csv("http://antoninofurnari.it/downloads/height_weight.csv")


sns.histplot(data= data, x='height', hue='sex',kde=True,bins=30)
plt.title('Distribuzione altezza in base al sesso')
plt.show()



soglia = 170 #impostiamo la soglia


male_pred = (data['height'] >= soglia) #classificazione

male_gt = (data['sex'] == 'M') #veri positivi


cm = confusion_matrix(male_gt,male_pred) #creiamo la matrice di confusione

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
             xticklabels=['Predicted: Female', 'Predicted: Male'],
            yticklabels=['True: Female', 'True: Male'])
plt.title(f'Confusion Matrix (Threshold = {soglia} cm)')
plt.show()
print(classification_report(male_gt, male_pred))
