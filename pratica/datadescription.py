import pandas as pd
import numpy as np
link = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"
titanic = pd.read_csv(link, index_col='PassengerId')
from matplotlib import pyplot as plt

age = titanic['Age'].dropna()
bins_struges = int(3.3*np.log(len(age)))
bins_rice = int(2*len(age)**(1/3))

plt.figure(figsize=(18,6))

plt.subplot(2,1,1)
_,edges,_ =plt.hist(age,bins=bins_struges) # prende solo edges dei tre valori restituiti
plt.xticks(edges)
plt.title("Struges {}".format(bins_struges))
plt.grid(axis='y', alpha=0.5,linestyle=':')


plt.subplot(2,1,2)
_,edges,_ =plt.hist(age,bins=bins_rice) # prende solo edges dei tre valori restituiti
plt.xticks(edges)
plt.title("Rice {}".format(bins_rice))
plt.grid(axis='y', alpha=0.5,linestyle=':')
plt.show()

titanic.groupby('Sex')['Age'].plot.hist(width=3, alpha=0.5)
plt.legend()
plt.grid()
plt.show()



age = age.dropna()

plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
_,edges,_ = plt.hist(age,bins=bins_struges)
plt.xticks(edges[::2])
plt.title("Struges {}".format(bins_struges))
plt.grid()


plt.subplot(1,2,2)
age.plot.density()
plt.title("Density")
plt.xlim(0,100)
plt.grid()
plt.show()



link1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wines = pd.read_csv(link1,sep=';')
chlorides = wines['chlorides']


plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
bins_struges = (int)(3.33*np.log(len(chlorides)))
_,edges,_ = plt.hist(chlorides,bins=bins_struges)
plt.xticks(edges, rotation=90)

plt.subplot(1,2,2)
chlorides.plot.density()
plt.xlim(0,1)
plt.show()



import seaborn as sbn
q1 = age.quantile(1/4)
q3 = age.quantile(3/4)
iqr = q3-q1

sbn.kdeplot(age, fill=True, color='steelblue',alpha=0.6)
plt.axvline(q1, color='green', linestyle='--', linewidth=2, label=f'Q1: {q1:.1f}')
#AGGIUNGERE DELLE LINEE CHE PERCORRONO TUTTO L'ASSE DELLE Y, DOVE X ASSUME IL VALORE ASSEGNATO
plt.axvline(q3, color='orange', linestyle='--', linewidth=2, label=f'Q3: {q3:.1f}')


plt.legend()
plt.show()


titanic.plot.box(column='Age',by='Sex',figsize=(18,6))
plt.show()

import numpy as np
import pandas as pd


from matplotlib import pyplot as plt
samples = np.random.normal(mean = 0, std=1, size= 100)
samples.plot.density()
plt.show()