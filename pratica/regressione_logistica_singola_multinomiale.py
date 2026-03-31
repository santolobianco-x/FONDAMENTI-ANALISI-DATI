import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

diabetes_data = sm.datasets.get_rdataset("Diabetes","heplots").data
print(diabetes_data.head(5))


diabetes_data2 = diabetes_data.copy()

diabetes_data2['group'] = diabetes_data2['group'].map({
    'Normal': 0,
    'Chemical_Diabetic': 1,
    'Overt_Diabetic': 2
}).astype(int)


model = smf.mnlogit("group ~ glutest",data=diabetes_data2)
result = model.fit()


print(result.summary())

print("\nModel Likelihood Ratio Test:")
print(result.llr, result.llr_pvalue)


#-------------------------------------LOGISTICA BINARIA
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.datasets import get_rdataset
from statsmodels.formula.api import logit

biopsy = get_rdataset('biopsy',package='MASS')

df = pd.DataFrame(biopsy.data)

df['cl'] = df['class'].replace({'benign':0,'malignant':1})


model = logit('cl ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9', df).fit()
print(model.summary())


model = logit('cl ~ V1 + V3 + V4 + V5 + V6 + V7 + V8 + V9', df).fit()
print(model.summary())



model = logit('cl ~ V1 + V3 + V4 + V6 + V7 + V8', df).fit()
print(model.summary())



print(np.exp(model.params))

print()
print("In percentuale:")
for ind,perc in np.exp(model.params).items():
    perc = (perc -1 )
    print(f"{ind} : {perc:.2%}")
