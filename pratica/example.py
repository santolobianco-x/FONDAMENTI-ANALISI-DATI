import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


link = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"


titanic = pd.read_csv(link,index_col='PassengerId')
pclass = titanic['Pclass'].value_counts(normalize=True).sort_index()

plt.vlines(pclass.index.values, ymin=0, ymax=pclass.values, linestyles='solid')
plt.plot(pclass.index.values, pclass.values, 'o', markersize=8)

plt.xticks(pclass.index.values)

plt.grid(True)
plt.show()





ages = titanic['Age'].dropna()

ages_sorted = np.sort(ages)
ecdf = np.arange(1,len(ages_sorted)+1)/len(ages_sorted)


plt.figure(figsize=(8,5))
plt.plot(ages_sorted,ecdf)
plt.xlabel("Age")
plt.ylabel("ECDF")
plt.grid(True)
plt.show()