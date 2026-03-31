import numpy as np
import pandas as pd


from matplotlib import pyplot as plt
samples = np.random.normal(mean:= 0, std:=1, size= 100)
samples = pd.Series(samples)
samples.plot.density()
plt.xlim(-5,5)
plt.show()



samples = np.random.exponential(scale=2, size=100)
samples = pd.Series(samples)
samples.plot.density()
plt.show()
i = 1
for s in samples:
    print(f"{i}^position: {s}")
    i += 1


link = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"

#random sampling
titanic = pd.read_csv(link, index_col='PassengerId')
sample = titanic.sample(100, replace=True) #replace permette il ripescaggio
sample.info()
print(sample.index)


sample = titanic.groupby('Pclass', group_keys=False).apply(lambda x:x.sample(10))


print(f"Name: {sample['Name']}")


from scipy import stats
sample = titanic['Age'].dropna().sample(150).to_numpy()

sample = (sample,)



###OPERAZIONE DI BOOTSTRAPPING
##CREI INTERVALLI DI CONFIDENZA CON UN SOLO CAMPIONE CHE VIENE RICAMPIONATO CON ORDINI DIVERSI
bootstrap_result = stats.bootstrap(sample,np.mean,n_resamples=1000,confidence_level=0.95)


confidence_interval = bootstrap_result.confidence_interval

print(f"real mean: {titanic['Age'].mean()}")
print(f"confidence level interval: [{confidence_interval.low:.2f},{confidence_interval.high:.2f}]")




#VERIFICA SE LA MEDIA È UGUALE A 30 ATTRAVERSO TEST DI IPOTESI
samples = titanic['Age'].sample(500, replace=True).dropna()
null_hypothesis_age = 28
t_stat, p_value = stats.ttest_1samp(samples,null_hypothesis_age)#si calcola la t-student e il p-value



alpha = 0.05
if p_value < alpha :
    print("Ipotesi nulla rigettata")
else:
    print("Impossibile rigettare l'ipotesi nulla")
print(f"t-statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.4f}")

#--------------------------------------------------------



#VERIFICA SE LE MEDIE CAMPIONARIE SONO VICINE OPPURE NO

samplem = titanic[titanic['Sex']== 'male']['Age'].dropna().sample(500,replace=True)
samplef = titanic[titanic['Sex']== 'female']['Age'].dropna().sample(500,replace=True)

t_stat,p_value = stats.ttest_ind(samplem,samplef)

alpha = 0.05

if p_value < alpha :
    print("C'è una differenza significante tra le due medie")
else:
    print("Non c'è una differnza significante tra le due medie")

print(f"Test statistic: {t_stat:0.2f}")
print(f"Significance level: {alpha:0.2f}")
print(f"P-value: {p_value:0.2f}")


from scipy.stats import chi2_contingency

titanic = titanic.dropna()

contingency_table = pd.crosstab(titanic['Survived'], titanic['Pclass'])
print(contingency_table)

from scipy.stats.contingency import association
print(f"Cramer V statistic: {association(contingency_table):0.2f}")

chi2, p,_,_ = chi2_contingency(contingency_table)

alpha = 0.05

if p < alpha:
    print("Non c'è correlazione tra la tabella survived e pclass")
else:
    print("C'è una correlazione tra la tabella survived e pclass")



##VERIFICA SE LA DISTRIBUZIONE È NORMALE
#Se i valori seguono la retta, allora la distribuzione è la normale
from statsmodels.graphics.gofplots import qqplot



qqplot(titanic['Age'], line='45', fit=True)
plt.title("Q-Q Plot dell'età")
plt.grid(True)
plt.show()





#ALTRI MODI PER VERIFICARE SE SEGUE LA GAUSSIANA
#SHAPIRO PER CAMPIONI PICCOLI <2000
from scipy.stats import shapiro
t_stat,p_value = shapiro(titanic['Age'])
alpha = 0.05
if p_value > alpha:
    print("segue la gaussiana con shapiro")
else:
    print("non segue la gaussiana con shapiro")


#D'AGOSTINO PER CAMPIONI GRANDI >50
from scipy.stats import normaltest
t_stat,p_value = normaltest(titanic['Age'])
alpha = 0.05
if p_value > alpha:
    print("segue la gaussiana con d'agostino")
else:
    print("non segue la gaussiana con d'agostino")

