


#%% treino separado em X e Y
"""
data_x = data
data_y = data

label_x = pd.read_csv("csv_files/label_x.csv")
label_x = list(label_x.columns[:])
data_y = data_y.drop(label_x, axis=1, errors='ignore')


label_y = pd.read_csv("csv_files/label_y.csv")
label_y = list(label_y.columns[:])
data_x = data_x.drop(label_y, axis=1, errors='ignore')

"""
#model_x = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1009,
               # max_depth = 32, alpha = 10, n_estimators = 4500)

#model_y = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                #max_depth = 16, alpha = 10, n_estimators = 2500)

model_x = MLPRegressor(max_iter=1000, solver='lbfgs', activation='logistic', alpha=0.0005, power_t=0.1)
model_y = MLPRegressor(max_iter=1000, solver='lbfgs', activation='logistic', alpha=0.0005, power_t=0.1)
kf = KFold(n_splits=10)
y_pred = []

importances_x = np.zeros(np.size(X_x,1))
importances_y = np.zeros(np.size(X_x,1))
i = 0 

test = np.array([], dtype=np.int64).reshape(0,2)
pred = np.array([], dtype=np.int64).reshape(0,2)

#plt.figure()

for index_train, index_test in kf.split(index):
    X_x_train, X_x_test, X_y_train, X_y_test, y_train, y_test = X_x[index_train], X_x[index_test],X_y[index_train], X_y[index_test], y[index_train], y[index_test]

    model_x.fit(X_x_train,y_train[:,0])
    y_pred_x = model_x.predict(X_x_test)
    
    model_y.fit(X_y_train, y_train[:,1])
    y_pred_y = model_y.predict(X_y_test)
    
    y_pred = np.vstack((y_pred_x, y_pred_y))
    y_pred = y_pred.T
    
    test = np.vstack((test,y_test))
    pred = np.vstack((pred,y_pred))
    
   # importances_x = return_importances(model_x, 0) + importances_x
   # importances_y = return_importances(model_y, 0) + importances_y
   # i = i + 1
    
    #plot_ball()

'''
plt.xlim(0, 1920)
plt.ylim(1080, 0)
plt.show() 
plt.title('Result') 
'''
print_error(test, pred)
metrica(test, pred)


#%% MPL REGRESSOR 
regr = MLPRegressor(random_state=1, max_iter=1000, solver='lbfgs', activation='logistic', alpha=0.0005, power_t=0.1)
regr_x = MLPRegressor(max_iter=500, solver='sgd', activation='logistic', alpha=0.0005, power_t=0.1)
#, learning_rate_init= 0.0015
regr_y = MLPRegressor(max_iter=1000, solver='sgd', activation='logistic', alpha=0.0005, power_t=0.1)

plt.figure()
test = np.array([], dtype=np.int64).reshape(0,2)
pred = np.array([], dtype=np.int64).reshape(0,2)

kf = KFold(n_splits=10)
for index_train, index_test in kf.split(index):
    X_train, X_test, y_train, y_test = X[index_train], X[index_test], y[index_train], y[index_test]
   
    regr.fit(X_train,y_train)
    y_pred = regr.predict(X_test)
    
    '''
    
    regr_x.fit(X_train,y_train[:,0])
    y_pred_x = regr_x.predict(X_test)
    
    regr_y.fit(X_train, y_train[:,1])
    y_pred_y = regr_y.predict(X_test)
    
    y_pred = np.vstack((y_pred_x, y_pred_y))
    y_pred = y_pred.T
    '''
    test = np.vstack((test,y_test))
    pred = np.vstack((pred,y_pred))
    
    plot_ball()
    
plt.xlim(0, 1920)
plt.ylim(1080, 0)
plt.show() 
plt.title('Result') 


print_error(test, pred)
metrica(test, pred)

#%%
    
def calcular_erro():
    ax = np.zeros(10)
    count = np.ones(10)
    for i in range(len(y_test)):
        erro = np.sqrt(mean_squared_error(y_test, y_pred))
        if(y_test[i] < 192):
            ax[0] = ax[0] + erro  
            count[0] = count[0] + 1
        elif(y_test[i] < 192*2):
            ax[1] = ax[1] + erro  
            count[1] = count[1] + 1
        elif(y_test[i] < 192*3):
            ax[2] = ax[2] + erro  
            count[2] = count[2] + 1
        elif(y_test[i] < 192*4):
            ax[3] = ax[3] + erro  
            count[3] = count[3] + 1
        elif(y_test[i] < 192*5):
            ax[4] = ax[4] + erro  
            count[4] = count[4] + 1
        elif(y_test[i] < 192*6):
            ax[5] = ax[5] + erro  
            count[5] = count[5] + 1
        elif(y_test[i] < 192*7):
            ax[6] = ax[6] + erro  
            count[6] = count[6] + 1
        elif(y_test[i] < 192*8):
            ax[7] = ax[7] + erro  
            count[7] = count[7] + 1
        elif(y_test[i] < 192*9):
            ax[8] = ax[8] + erro  
            count[8] = count[8] + 1
        elif(y_test[i] < 192*10):
            ax[9] = ax[9] + erro  
            count[9] = count[9] + 1
    
    graf = ax / count
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    langs = ('0', '1', '2', '3', '4', '5','6','7','8','9')
    y_pos = np.arange(len(langs))
    
    ax.bar(y_pos,graf)
    plt.xticks(y_pos, langs)
    plt.xlabel('Segmento de quadra')
    plt.ylabel('Media do Erro')
    plt.show()

