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
