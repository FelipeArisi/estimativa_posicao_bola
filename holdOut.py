import pandas as pd 
import numpy as np
from utils.correcao import correcao_data
from utils.persp2 import persp2_data
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.erro_euclideano import erro_euclideano_data

warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

data = pd.read_csv("in/csv/data.csv")

_INDEX_V = len(data.columns) - 4
_INDEX_X = len(data.columns) - 4
_INDEX_Y = len(data.columns) - 2


X = data.iloc[:, 0:_INDEX_V].values
y = data.iloc[:, _INDEX_X:_INDEX_Y].values


X = persp2_data(correcao_data(X, 1), score =1)
y = persp2_data(correcao_data(y, 1), score =1)



abError = {'train': np.zeros(10), 'test': np.zeros(10)}
sbError = {'train': np.zeros(10), 'test': np.zeros(10)}

def fit_hold(model, X, y, augmentation=0):
    X = shuffle(X)
    for i in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=i)  
        X_train,y_train = criar_dados(augmentation, X_train, y_train)
        model = model.fit(X_train, y_train)
        # ---- train
        
        result = model.predict(X_train)
        abError['train'][i] = mean_absolute_error(y_train, result)
        sbError['train'][i] = np.sqrt(mean_squared_error(y_train, result))
        
        # ---- test
        result = model.predict(X_test)
        abError['test'][i] = mean_absolute_error(y_test, result)
        sbError['test'][i] = np.sqrt(mean_squared_error(y_test, result))
        #erro_euclideano_data( np.concatenate((result, y_test), axis=1) )
    
    print('Treino')
    print('Absoulte: '+str(np.mean(abError['train'])) + " - " + str(np.std(abError['train'])))
    print('Sqrt: '+str(np.mean(sbError['train'])) + " - "  + str(np.std(sbError['train'])))
          
    print('Teste')
    print('Absoulte: '+str(np.mean(abError['test'])) + " - " + str( np.std(abError['test'])))
    print('Sqrt: '+str(np.mean(sbError['test'])) + " - " +  str(np.std(sbError['test'])))

#%% XGBOOST REGRESSOR
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

from utils.utils import convert_to_np, fit

model = XGBRegressor()
multioutputregressor = MultiOutputRegressor(model)
fit_hold(multioutputregressor, X,y)

#%%
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)

model = MLPRegressor()

fit_hold(model, imp_mean.transform(X), y)

#%% Guasiana
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean.fit(X)

kernel = DotProduct() + WhiteKernel()
model = GaussianProcessRegressor(kernel=kernel)

fit_hold(model, imp_mean.transform(X), y)



#%% Tirando os Olhos
X, y, i = convert_to_np(data, "label_eyes.csv")

X = persp2_data(correcao_data(X, 1), score =1)
y = persp2_data(correcao_data(y, 1), score =1)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)


fit_hold(multioutputregressor, X,y)

#%% Tirando goleiro

from utils.utils import convert_to_np, fit

X, y, i = convert_to_np(data, "label_goalkeeper.csv")

X = persp2_data(correcao_data(X, 1), score =1)
y = persp2_data(correcao_data(y, 1), score =1)

X = shuffle(X, players=8)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)

fit_hold(multioutputregressor, X,y)

#%% Tirando score

from utils.utils import convert_to_np, fit

X, y, i = convert_to_np(data, "label_score.csv")
X = persp2_data(correcao_data(X, 0), score =0)
y = persp2_data(correcao_data(y, 0), score =0)

X_train = shuffle(X_train)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)

fit_hold(multioutputregressor, X,y)

#%% Apenas pés e joelhos

from utils.utils import convert_to_np, fit

X, y, i = convert_to_np(data, "label_pes.csv")

X = persp2_data(correcao_data(X, 0), score =0)
y = persp2_data(correcao_data(y, 0), score =0)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)


fit_hold(multioutputregressor, X,y)


#%% Data augmentation 1000
from utils.utils import criar_dados

X, y, i = convert_to_np(data, "label_pes.csv")

X = persp2_data(correcao_data(X, 0), score =0)
y = persp2_data(correcao_data(y, 0), score =0)

#X,y = criar_dados(100, X, y)
modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit_hold(multioutputregressor, X,y, 1000)

#%% Data augmentation 10000 

from utils.utils import criar_dados

X, y, i = convert_to_np(data, "label_pes.csv")

X = persp2_data(correcao_data(X, 0), score =0)
y = persp2_data(correcao_data(y, 0), score =0)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit_hold(multioutputregressor, X,y, 10000)

#%% Data augmentation 25000 
from utils.utils import criar_dados

X, y, i = convert_to_np(data, "label_pes.csv")

X = persp2_data(correcao_data(X, 0), score =0)
y = persp2_data(correcao_data(y, 0), score =0)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit_hold(multioutputregressor, X,y, 25000)

#%% Data augmentation 35000 

from utils.utils import criar_dados

X, y, i = convert_to_np(data, "label_pes.csv")

X = persp2_data(correcao_data(X, 0), score =0)
y = persp2_data(correcao_data(y, 0), score =0)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit_hold(multioutputregressor, X,y, 35000)

#%% Data augmentation 50000 


from utils.utils import criar_dados

X, y, i = convert_to_np(data, "label_pes.csv")

X = persp2_data(correcao_data(X, 0), score =0)
y = persp2_data(correcao_data(y, 0), score =0)

modelXGB = XGBRegressor()
multioutputregressor = MultiOutputRegressor(modelXGB)
fit_hold(multioutputregressor, X,y, 50000)
#%%
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from utils.utils import criar_dados
from utils.utils import convert_to_np, fit, shuffle
warnings.filterwarnings(action='ignore', category=UserWarning)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=1)  
X_train,y_train = criar_dados(35000, X_train, y_train)
print('create data')
#%%
model = XGBRegressor()

param_grid = {
    'estimator__n_estimators': [50, 100, 200, 300, 500], #100
    'estimator__colsample_bytree': [0.7, 1, 0.5], #1
    'estimator__max_depth': [1,3,5,10], #3
    'estimator__reg_alpha': [0, 2,10], #0 
    'estimator__reg_lambda': [0.7, 1, 2], #1
    'estimator__subsample': [0.5, 1, 0.3], #1 
    'estimator__learning_rate': [0.05, 0.1, 0.2]  
    
}

gs = RandomizedSearchCV(
        estimator = MultiOutputRegressor(model),
        param_distributions =param_grid, 
        cv=5, 
        n_jobs=-1, 
        scoring='neg_mean_absolute_error',
        verbose=0,
        n_iter=100,
        return_train_score=True
    )

fitted_model = gs.fit(X_train, y_train)

print(np.sqrt(-fitted_model.best_score_))
print(fitted_model.best_params_)

#%%
abError = {'train': np.zeros(1), 'test': np.zeros(1)}
sbError = {'train': np.zeros(1), 'test': np.zeros(1)}

model = fitted_model
result = model.predict(X_train)
abError['train'][0] = mean_absolute_error(y_train, result)
sbError['train'][0] = np.sqrt(mean_squared_error(y_train, result))

# ---- test
result = model.predict(X_test)
abError['test'][0] = mean_absolute_error(y_test, result)
sbError['test'][0] = np.sqrt(mean_squared_error(y_test, result))
erro_euclideano_data( np.concatenate((result, y_test), axis=1) )

print('Treino')
print('Absoulte: '+str(np.mean(abError['train'])) + " - " + str(np.std(abError['train'])))
print('Sqrt: '+str(np.mean(sbError['train'])) + " - "  + str(np.std(sbError['train'])))
  
print('Teste')
print('Absoulte: '+str(np.mean(abError['test'])) + " - " + str( np.std(abError['test'])))
print('Sqrt: '+str(np.mean(sbError['test'])) + " - " +  str(np.std(sbError['test'])))


#%%

import joblib
joblib.dump(gs, 'model_file_name.pkl')