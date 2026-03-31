import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


import statsmodels.api as sm
import statsmodels.formula.api as smf




#dataset standard di sns
mpg = sns.load_dataset('mpg')

print(mpg.head())

mpg = mpg.drop('name', axis=1)


mpg['horsepower'] = pd.to_numeric(mpg['horsepower'],errors='coerce')



mpgclean = mpg.dropna()

#REGRESSIONE LINEARE SINGOLA
#applica il metodo di regressione lineare più usato ovvero Ordinary Least Squares(Minimi quadrati ordinari)
model_slr = smf.ols("mpg ~ horsepower",data=mpgclean).fit()
print(model_slr.summary())


fittedvalues_slr = model_slr.fittedvalues
residuals_slr = model_slr.resid


plt.figure(figsize=(12,5))


ax1 = plt.subplot(1,2,1)
sns.scatterplot(x=fittedvalues_slr,y=residuals_slr,ax=ax1)
ax1.axhline(0, color='red',linestyle='--')
ax1.set_title('Residual vs Fitted(Simple Linear Regression)')
ax1.set_xlabel('Fitted values(predict mpg)')
ax1.set_ylabel('Residuals')


ax2 = plt.subplot(1,2,2)
sm.qqplot(residuals_slr,line='s',ax=ax2)
ax2.set_title('Q-Q Plot of Residuals (SLR)')
plt.tight_layout()
plt.show()



#REGRESSIONE LINEARE MULTIPLA

allfeature = 'mpg ~ cylinders + displacement + horsepower + weight + acceleration + model_year + C(origin)'
model_full = smf.ols(allfeature,data=mpgclean).fit()

print(model_full.summary())



#sistemiamo origin che con il valore di japan è alterato 
allfeature = 'mpg ~ cylinders + displacement + horsepower + weight + acceleration + model_year + C(origin, Treatment(reference="usa"))'
model_full = smf.ols(allfeature,data=mpgclean).fit()

print(model_full.summary())

allfeature = 'mpg ~ cylinders + displacement + horsepower + weight + model_year + C(origin, Treatment(reference="usa"))'
model_full = smf.ols(allfeature,data=mpgclean).fit()

print(model_full.summary())



fittedvalues_full = model_full.fittedvalues
residuals_full = model_full.resid


plt.figure(figsize=(12,5))
ax1 = plt.subplot(1,2,1)
sns.scatterplot(x=fittedvalues_full,y=residuals_full,ax=ax1)
ax1.axhline(0, color='red',linestyle='--')
ax1.set_title('Residual vs Fitted(Multiple Linear Regression)')
ax1.set_xlabel('Fitted values(predict mpg)')
ax1.set_ylabel('Residuals')


ax2 = plt.subplot(1,2,2)
sm.qqplot(residuals_full,line='s',ax=ax2)
ax2.set_title('Q-Q Plot of Residuals (FULL)')
plt.tight_layout()
plt.show()


