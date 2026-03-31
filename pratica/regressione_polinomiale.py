import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_california_housing

data  = fetch_california_housing()

X = data.data
y = data.target #valore da predire, valore medio di una casa in un quartiere


from sklearn.model_selection import train_test_split
validation_prop = 0.2 #dividiamo in validation set
test_prop = 0.2 #dividiamo in test set 
#il resto sarà training set 

#SEPARIAMO DA TRAINING SET E VALIDATION SET I VALORI UTILIZZATI PER IL SET DI TEST
X_trainval, X_test, y_trainval, y_test = train_test_split(X,y,test_size=test_prop,random_state=42)
#SEPARIAMO DA TRAINING SET I VALORI UTILIZZATI PER IL VALIDATION SET
X_train, X_val, y_train, y_val = train_test_split(X_trainval,y_trainval,test_size=test_prop/(1-test_prop),random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)#apprende media e deviazione standarddel dataset


X_train = scaler.transform(X_train)#applica lo z-scoring a train set
X_val = scaler.transform(X_val)#applica lo z-scoring a test set
X_test = scaler.transform(X_test)#applica lo z-scoring a validation set


from sklearn.linear_model import LinearRegression


linear_regression = LinearRegression()
#DAI DATI CHE HA A DISPOSIZIONE IL MODELLO IMPARA
#CERCA UNA FUNZIONE CHE PERMETTA DI OTTENERE y DA x
linear_regression.fit(X_train,y_train) #VIENE COSTRUITO IL MODELLO


print(f"Coef: {linear_regression.coef_}")
print(f"Intercept: {linear_regression.intercept_}")


y_val_pred = linear_regression.predict(X_val) #DOPO AVER IMPARATO IL MODELLO PREDICE I DATI 

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#METRICHE PER VALUTARE IL NOSTRO MODELLO
mae = mean_absolute_error(y_val,y_val_pred)
mse = mean_squared_error(y_val,y_val_pred)
rmse = np.sqrt(mse)



print(f"mse: {mse} rmse: {rmse} mae: {mae}")


y_train_pred = linear_regression.predict(X_train)


mae_train = mean_absolute_error(y_train,y_train_pred)
mse_train = mean_squared_error(y_train,y_train_pred)
rmse_train = np.sqrt(mse_train)

print(f"mse train: {mse} rmse train: {rmse} mae train: {mae}")
print()
print()
print()


############################################################
#REGRESSIONE POLINOMIALE
print("REGRESSIONE POLINOMIALE")
from sklearn.preprocessing import PolynomialFeatures
def trainval_polynomial(degree):

    pf = PolynomialFeatures(degree)

    pf.fit(X_train)# non addestra il modello, bensì si calcola tutte le combinazioni polinomiali
    X_train_poly = pf.transform(X_train)# trasformazione in polinomi
    X_val_poly = pf.transform(X_val)


    polyreg = LinearRegression()#si utilizza la regressione lineare però con polinomi al posto di feature
    polyreg.fit(X_train_poly,y_train)


    y_poly_train_pred = polyreg.predict(X_train_poly) #si verifica per il training set
    y_poly_val_pred = polyreg.predict(X_val_poly) #si verifica per il validation set

    mae_train = mean_absolute_error(y_train, y_poly_train_pred)
    mse_train = mean_squared_error(y_train, y_poly_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_poly_train_pred))

    mae_val = mean_absolute_error(y_val, y_poly_val_pred)
    mse_val = mean_squared_error(y_val, y_poly_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_poly_val_pred))

    return mae_train, mse_train, rmse_train, mae_val, mse_val, rmse_val


for d in range(1,4):
    print("DEGREE: {} \n      {:>8s} {:>8s} {:>8s}\nTRAIN {:8.2f} {:8.2f} {:8.2f} \nVAL   {:8.2f} {:8.2f} {:8.2f}\n\n".format(d,"MAE", "MSE", "RMSE", *trainval_polynomial(d)))


print()


######################################################################
#REGRESSIONE RIDGE
print("REGRESSIONE RIDGE")
from sklearn.linear_model import Ridge
def trainval_polynomial_ridge(degree,alpha):

    pf = PolynomialFeatures(degree)

    pf.fit(X_train)
    X_train_poly = pf.transform(X_train)
    X_val_poly = pf.transform(X_val)


    polyreg = Ridge(alpha=alpha)#regressione lineare con regolarizzazione Ridge e alpha iper-parametro
    polyreg.fit(X_train_poly,y_train)


    y_poly_train_pred = polyreg.predict(X_train_poly) #si verifica per il training set
    y_poly_val_pred = polyreg.predict(X_val_poly) #si verifica per il validation set

    mae_train = mean_absolute_error(y_train, y_poly_train_pred)
    mse_train = mean_squared_error(y_train, y_poly_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_poly_train_pred))

    mae_val = mean_absolute_error(y_val, y_poly_val_pred)
    mse_val = mean_squared_error(y_val, y_poly_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_poly_val_pred))

    return mae_train, mse_train, rmse_train, mae_val, mse_val, rmse_val


print("RIDGE, DEGREE: 2")
for alpha in [0,100,200,300,400]:
    print("Alpha: {:0.2f} \n      {:>8s} {:>8s} {:>8s}\nTRAIN {:8.2f} {:8.2f} {:8.2f} \nVAL   {:8.2f} {:8.2f} {:8.2f}\n\n".format(alpha,"MAE", "MSE", "RMSE", *trainval_polynomial_ridge(2,alpha)))


##################################################################################
#REGRESSIONE LASSO
from sklearn.linear_model import Lasso
def trainval_polynomial_lasso(degree, alpha):
    pf = PolynomialFeatures(degree)
    # While the model does not have any learnable parameters, the "fit" method here is used to compute the output number of features
    pf.fit(X_train)
    X_train_poly = pf.transform(X_train)
    X_val_poly = pf.transform(X_val)

    polyreg = Lasso(alpha=alpha) # a Polynomial regressor is simply a linear regressor using polynomial features
    polyreg.fit(X_train_poly, y_train)

    y_poly_train_pred = polyreg.predict(X_train_poly)
    y_poly_val_pred = polyreg.predict(X_val_poly)

    mae_train = mean_absolute_error(y_train, y_poly_train_pred)
    mse_train = mean_squared_error(y_train, y_poly_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_poly_train_pred))

    mae_val = mean_absolute_error(y_val, y_poly_val_pred)
    mse_val = mean_squared_error(y_val, y_poly_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_poly_val_pred))

    return mae_train, mse_train, rmse_train, mae_val, mse_val, rmse_val

print("LSSO, DEGREE: 2")
for alpha in [0.02,0.03,0.04,0.05, 0.06]:
    print("Alpha: {:0.2f} \n      {:>8s} {:>8s} {:>8s}\nTRAIN {:8.2f} {:8.2f} {:8.2f} \nVAL   {:8.2f} {:8.2f} {:8.2f}\n\n".format(alpha,"MAE", "MSE", "RMSE", *trainval_polynomial_lasso(2,alpha)))

#####POSSIAMO CERCARE ALPHA E GRADO MIGLIORE ATTRAVERSO GRID SEARCH
#FISSATO UN IPER-PARAMETRO SI CALCOLANO LE MISURE PER OGNI GRADO 



###PROCESSO PIU' AUTOMATIZZATO 

# RICARICHIAMO IL SET E EFFETTUEREMO STANDARDIZZAZIONE, ADDESTRAMENTO MODELLO E PREDIZIONE IN UN UNICO PASSAGGIO
data = fetch_california_housing()
X = data.data 
y = data.target 


val_prop = 0.2
test_prop = 0.2


X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_prop, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=test_prop/(1-test_prop), random_state=42)

from sklearn.pipeline import Pipeline
polynomial_regressor = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomial_expansion', PolynomialFeatures()),
    ('ridge_regression', Ridge())
])


polynomial_regressor.set_params(polynomial_expansion__degree=2, ridge_regression__alpha=300)

polynomial_regressor.fit(X_train, y_train)
y_val_pred = polynomial_regressor.predict(X_val)

mean_absolute_error(y_val, y_val_pred), mean_squared_error(y_val, y_val_pred), np.sqrt(mean_squared_error(y_val, y_val_pred))