import pandas as pd 
import numpy as np
from utils.correcao import correcao_data
from utils.persp2 import persp2_data
import warnings
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

train = pd.read_csv("in/csv/train.csv", sep=';')
test = pd.read_csv("in/csv/test.csv", sep=';')


_INDEX_V = len(train.columns) - 4
_INDEX_X = len(train.columns) - 4
_INDEX_Y = len(train.columns) - 2


X_train = train.iloc[:, 0:_INDEX_V].values
y_train = train.iloc[:, _INDEX_X:_INDEX_Y].values

X_test = test.iloc[:, 0:_INDEX_V].values
y_test = test.iloc[:, _INDEX_X:_INDEX_Y].values



X_train = persp2_data(correcao_data(X_train, 1), score =1)
y_train = persp2_data(correcao_data(y_train, 1), score =1)

X_test = persp2_data(correcao_data(X_test, 1), score =1)
y_test = persp2_data(correcao_data(y_test, 1), score =1)

from utils.utils import shuffle

X_train = shuffle(X_train)


#%% XGBOOST REGRESSOR
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


model = XGBRegressor()
multioutputregressor = MultiOutputRegressor(model).fit(X_train, y_train)

print(' ------- XGBOOST')
print('Train')
result = multioutputregressor.predict(X_train)
print('mean_absolute_error: '+str((mean_absolute_error(y_train, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_train, result))))

print('Test')
result = multioutputregressor.predict(X_test)
print('mean_absolute_error: '+str((mean_absolute_error(y_test, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_test, result))))

print(str(y_test[0]) + '---' + str(result[0]))

#%% MPL REGRESSOR
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X_train)

model = MLPRegressor()
model = model.fit(imp_mean.transform(X_train), y_train)

print(' ------- MPLREGRESSOR')
print('Train')
result = model.predict( imp_mean.transform(X_train) )
print('mean_absolute_error: '+str((mean_absolute_error(y_train, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_train, result))))

print('Test')
result = model.predict( imp_mean.transform(X_test))
print('mean_absolute_error: '+str((mean_absolute_error(y_test, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_test, result))))


#%% Guasiana
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X_train)

kernel = DotProduct() + WhiteKernel()
model = GaussianProcessRegressor(kernel=kernel)
model = model.fit(imp_mean.transform(X_train), y_train)

print(' ------- Guasiana')
print('Train')
result = model.predict( imp_mean.transform(X_train) )
print('mean_absolute_error: '+str((mean_absolute_error(y_train, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_train, result))))

print('Test')
result = model.predict( imp_mean.transform(X_test))
print('mean_absolute_error: '+str((mean_absolute_error(y_test, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_test, result))))


#%% Tirando os Olhos

from utils.utils import convert_to_np, fit

X_train, y_train, i = convert_to_np(train, "label_eyes.csv")
X_test, y_test, i = convert_to_np(test, "label_eyes.csv")

X_train = persp2_data(correcao_data(X_train, 1), score =1)
y_train = persp2_data(correcao_data(y_train, 1), score =1)

X_test = persp2_data(correcao_data(X_test, 1), score =1)
y_test = persp2_data(correcao_data(y_test, 1), score =1)

X_train = shuffle(X_train)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)


fit(multioutputregressor, X_train, y_train, X_test, y_test)

#%% Tirando goleiro

from utils.utils import convert_to_np, fit

X_train, y_train, i = convert_to_np(train, "label_goalkeeper.csv")
X_test, y_test, i = convert_to_np(test, "label_goalkeeper.csv")

X_train = persp2_data(correcao_data(X_train, 1), score =1)
y_train = persp2_data(correcao_data(y_train, 1), score =1)

X_test = persp2_data(correcao_data(X_test, 1), score =1)
y_test = persp2_data(correcao_data(y_test, 1), score =1)

X_train = shuffle(X_train, players=8)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)


fit(multioutputregressor, X_train, y_train, X_test, y_test)

#%% Tirando score

from utils.utils import convert_to_np, fit

X_train, y_train, i = convert_to_np(train, "label_score.csv")
X_test, y_test, i = convert_to_np(test, "label_score.csv")

X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)

X_train = shuffle(X_train)


modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)


fit(multioutputregressor, X_train, y_train, X_test, y_test)

#%% Apenas pés e joelhos

from utils.utils import convert_to_np, fit

X_train, y_train, i = convert_to_np(train, "label_pes.csv")
X_test, y_test, i = convert_to_np(test, "label_pes.csv")

X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)

X_train = shuffle(X_train)
modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)


#model = GaussianProcessRegressor(kernel=kernel)
#imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp_mean.fit(X_train)
#X_train = imp_mean.transform(X_train)
#X_test = imp_mean.transform(X_test)


fit(multioutputregressor, X_train, y_train, X_test, y_test)


#%% Data augmentation 100
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils.utils import criar_dados
from utils.utils import convert_to_np, fit

X_train, y_train, i = convert_to_np(train, "label_score.csv")
X_test, y_test, i = convert_to_np(test, "label_score.csv")
X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)

X_train = shuffle(X_train)
X_train,y_train = criar_dados(25000, X_train,y_train)
#print('100')
modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit(multioutputregressor, X_train, y_train, X_test, y_test)

#%% Data augmentation 10000 


X_train, y_train, i = convert_to_np(train, "label_pes.csv")
X_test, y_test, i = convert_to_np(test, "label_pes.csv")
X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)


X_train,y_train = criar_dados(10000, X_train,y_train)
print('10000')
modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit(multioutputregressor, X_train, y_train, X_test, y_test)

#%% Data augmentation 25000 


X_train, y_train, i = convert_to_np(train, "label_pes.csv")
X_test, y_test, i = convert_to_np(test, "label_pes.csv")
X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)

X_train = shuffle(X_train)
X_train,y_train = criar_dados(25000, X_train,y_train)
print('25000')
modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit(multioutputregressor, X_train, y_train, X_test, y_test)

#%% Data augmentation 35000 


X_train, y_train, i = convert_to_np(train, "label_pes.csv")
X_test, y_test, i = convert_to_np(test, "label_pes.csv")
X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)


X_train = shuffle(X_train)
X_train,y_train = criar_dados(35000, X_train,y_train)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
print('35000')
fit(multioutputregressor, X_train, y_train, X_test, y_test)


#%% Data augmentation 60000 


X_train, y_train, i = convert_to_np(train, "label_pes.csv")
X_test, y_test, i = convert_to_np(test, "label_pes.csv")
X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)


X_train,y_train = criar_dados(60000, X_train,y_train)

modelXGB = XGBRegressor()
print('60000')
multioutputregressor = MultiOutputRegressor(modelXGB)
fit(multioutputregressor, X_train, y_train, X_test, y_test)

#%%
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils.utils import criar_dados
from utils.utils import convert_to_np, fit, shuffle
warnings.filterwarnings(action='ignore', category=UserWarning)

X_train, y_train, i = convert_to_np(train, "label_pes.csv")
X_test, y_test, i = convert_to_np(test, "label_pes.csv")
X_train = persp2_data(correcao_data(X_train, 0), score =0)
y_train = persp2_data(correcao_data(y_train, 0), score =0)

X_test = persp2_data(correcao_data(X_test, 0), score =0)
y_test = persp2_data(correcao_data(y_test, 0), score =0)
print('create data')
X_train,y_train = criar_dados(25000, X_train,y_train)
model = XGBRegressor()

param_grid = {
    'estimator__n_estimators': [50, 100, 200, 300], #100
    'estimator__colsample_bytree': [0.7, 1, 1.2], #1
    'estimator__max_depth': [1,3,5,10], #3
    'estimator__reg_alpha': [0, 0.5,1], #0 
    'estimator__reg_lambda': [0.7, 1, 1.3], #1
    'estimator__subsample': [0.5, 1, 1.5], #1 
    'estimator__learning_rate': [0.05, 0.1, 0.2]  
    
}

gs = RandomizedSearchCV(
        estimator = MultiOutputRegressor(model),
        param_distributions =param_grid, 
        cv=5, 
        n_jobs=-1, 
        scoring='neg_mean_squared_error',
        verbose=5,
        n_iter=5,
        return_train_score=True
    )

fitted_model = gs.fit(X_train, y_train)

print(np.sqrt(-fitted_model.best_score_))
print(fitted_model.best_params_)

from sklearn.metrics import mean_squared_error, mean_absolute_error

result = fitted_model.predict(X_train)
print('mean_absolute_error: '+str((mean_absolute_error(y_train, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_train, result))))

print('Test')
result = fitted_model.predict(X_test)
print('mean_absolute_error: '+str((mean_absolute_error(y_test, result))))
print('mean_squared_error: '+str(np.sqrt(mean_squared_error(y_test, result))))
