import pandas as pd
import numpy as np

link = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"
titanic = pd.read_csv(link,index_col='PassengerId')



from scipy.stats import chi2_contingency

contingency = pd.crosstab(titanic['Pclass'], titanic['Survived']) 
print(chi2_contingency(contingency))#matrice di contingenza dei valori attesi in caso di indipendenza, statistica chi-quadro, p-value, dof



from scipy.stats.contingency import association
print(f"Level of association : {association(contingency):0.2f}") #statistica V di Cramer

print()
print()
print()


from statsmodels.datasets import get_rdataset
data = get_rdataset('Diabetes','heplots').data
print(data.head())



from matplotlib import pyplot as plt
data.plot.scatter(x='glutest',y='glufast')#scatter plot costruito sulle due feature
plt.grid()
plt.show()



import seaborn as sbn
sbn.pairplot(data, height=1.5, aspect=2)
#height serve ad indicare quanto deve essere il grafico in generale
#aspect serve per indicare che la larghezza deve essere il doppio dell'altezza
plt.show()


sbn.pairplot(data, height=1.5, aspect=2, hue='group')
plt.show()


data.plot.hexbin(x='glufast', y='instest', gridsize=15,cmap='viridis')
#gridsize indica che devono essere presenti 15 esagoni lungo l'asse x
#cmap indica la scala di colorazione per la densità
plt.show()


from matplotlib.axes import Axes

fig, axes = plt.subplots(1,2,figsize=(14,6))#subplot ritorna le figure e i subplot da utilizzare
axes : tuple[Axes,Axes]

sbn.kdeplot(data=data,x='sspg',y='glutest', fill='true',cmap='viridis', ax=axes[0])
#fill aggiunge il colore che indica la densità del grafico. solitamente è impostato a false

sbn.kdeplot(data=data, x='sspg', y='glutest', levels=5, color='white', linewidths=0.5, ax=axes[0])
#aggiunge i contorni bianchi per mostrare la separazione netta dei vari grafici

axes[0].set_title('Density Plot with Contours')
axes[0].set_xlabel("sspg")
axes[0].set_ylabel("glutest")

sbn.kdeplot(data=data, x='sspg',y='glutest',cmap='viridis',fill=False,ax=axes[1])
axes[1].set_title("Contours only")
axes[1].set_xlabel("sspg")

plt.tight_layout() #risolve conflitti con le etichette ecc.
plt.show()



covariance_matr = data.drop('group',axis=1).cov()
print(f"{covariance_matr}")import pandas as pd
import numpy as np

link = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"
titanic = pd.read_csv(link,index_col='PassengerId')



from scipy.stats import chi2_contingency

contingency = pd.crosstab(titanic['Pclass'], titanic['Survived']) 
print(chi2_contingency(contingency))#matrice di contingenza dei valori attesi in caso di indipendenza, statistica chi-quadro, p-value, dof



from scipy.stats.contingency import association
print(f"Level of association : {association(contingency):0.2f}") #statistica V di Cramer

print()
print()
print()


from statsmodels.datasets import get_rdataset
data = get_rdataset('Diabetes','heplots').data
print(data.head())



from matplotlib import pyplot as plt
data.plot.scatter(x='glutest',y='glufast')#scatter plot costruito sulle due feature
plt.grid()
plt.show()



import seaborn as sbn
sbn.pairplot(data, height=1.5, aspect=2)
#height serve ad indicare quanto deve essere il grafico in generale
#aspect serve per indicare che la larghezza deve essere il doppio dell'altezza
plt.show()


sbn.pairplot(data, height=1.5, aspect=2, hue='group')
plt.show()


data.plot.hexbin(x='glufast', y='instest', gridsize=15,cmap='viridis')
#gridsize indica che devono essere presenti 15 esagoni lungo l'asse x
#cmap indica la scala di colorazione per la densità
plt.show()


from matplotlib.axes import Axes

fig, axes = plt.subplots(1,2,figsize=(14,6))#subplot ritorna le figure e i subplot da utilizzare
axes : tuple[Axes,Axes]

sbn.kdeplot(data=data,x='sspg',y='glutest', fill='true',cmap='viridis', ax=axes[0])
#fill aggiunge il colore che indica la densità del grafico. solitamente è impostato a false

sbn.kdeplot(data=data, x='sspg', y='glutest', levels=5, color='white', linewidths=0.5, ax=axes[0])
#aggiunge i contorni bianchi per mostrare la separazione netta dei vari grafici

axes[0].set_title('Density Plot with Contours')
axes[0].set_xlabel("sspg")
axes[0].set_ylabel("glutest")

sbn.kdeplot(data=data, x='sspg',y='glutest',cmap='viridis',fill=False,ax=axes[1])
axes[1].set_title("Contours only")
axes[1].set_xlabel("sspg")

plt.tight_layout() #risolve conflitti con le etichette ecc.
plt.show()



covariance_matr = data.drop('group',axis=1).cov()
print(f"{covariance_matr}")


from scipy.stats import pearsonr

print(f"Pearson value between relwt and sspg:\n{pearsonr(data['relwt'],data['sspg'])}")

from scipy.stats import kendalltau
print(f"Kendall value between relwt and sspg:\n{kendalltau(data['relwt'],data['sspg'])}")

from scipy.stats import spearmanr
print(f"Spearman value between relwt and sspg:\n{spearmanr(data['relwt'],data['sspg'])}")



sbn.heatmap(data.drop('group', axis =1).corr(), annot=True)
plt.show()


sbn.heatmap(data.drop('group', axis =1).corr(method='spearman'), annot=True)
plt.show()


sbn.heatmap(data.drop('group', axis =1).corr(method='kendall'), annot=True)
plt.show()
