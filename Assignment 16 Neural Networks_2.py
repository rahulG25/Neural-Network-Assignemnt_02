#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.models import Sequential
import keras


# In[2]:


data=pd.read_csv("gas_turbines.csv")


# In[3]:


data


# In[4]:


x=data.iloc[:,0:3]
y=data['TEY']


# In[5]:


from sklearn.preprocessing import StandardScaler
a=StandardScaler()
x_standardized=a.fit_transform(x)


# In[6]:


pd.DataFrame(x_standardized).describe()


# # Tuning of hyperparameters: BatchSize and Epochs

# In[8]:


from sklearn.model_selection import GridSearchCV,KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam


# In[9]:


def create_model():
    model=Sequential()
    model.add(Dense(5,input_dim=3,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(3,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
    adam=Adam(lr=0.01)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# In[10]:


model=KerasClassifier(build_fn=create_model,verbose=0)
batch_size=[10,15,20]
epochs=[30,40,50]
param_grid=dict(batch_size=batch_size,epochs=epochs)
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# In[11]:


print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Tuning of hyperparameters: LearningRate and DropOut

# In[12]:


from keras.layers import Dropout
def create_model(learning_rate,dropout_rate):
    model=Sequential()
    model.add(Dense(8,input_dim=3,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(5,input_dim=3,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[13]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=10,epochs=30)
learning_rate=[0.1,0.2,0.3]
dropout_rate=[0.11,0.22,0.33]
param_grids=dict(learning_rate=learning_rate,dropout_rate=dropout_rate)
grid=GridSearchCV(estimator=model,param_grid=param_grids,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# In[14]:


print('Best : {}, Using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Activation function and Kernel initializer

# In[15]:


def create_model(activation_function,init):
    model=Sequential()
    model.add(Dense(9,input_dim=3,kernel_initializer=init,activation=activation_function))
    model.add(Dropout(0.11))
    model.add(Dense(5,input_dim=3,kernel_initializer=init,activation=activation_function))
    model.add(Dropout(0.11))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=0.1)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[16]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=10,epochs=30)
activation_function=['softmax','relu','tanh','linear']
init=['uniform','normal','zero']
param_grid=dict(activation_function=activation_function,init=init)
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# In[17]:


print('Best : {}, Using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Tuning of hyperparameters: LearningRate and DropOut

# In[18]:


from keras.layers import Dropout
def create_model(learning_rate,dropout_rate):
    model=Sequential()
    model.add(Dense(8,input_dim=3,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(5,input_dim=3,kernel_initializer='normal',activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[19]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=10,epochs=30)
learning_rate=[0.1,0.2,0.3]
dropout_rate=[0.11,0.22,0.33]
param_grids=dict(learning_rate=learning_rate,dropout_rate=dropout_rate)
grid=GridSearchCV(estimator=model,param_grid=param_grids,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# In[20]:


print('Best : {}, Using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Activation function and Kernel initializer

# In[21]:


def create_model(activation_function,init):
    model=Sequential()
    model.add(Dense(9,input_dim=3,kernel_initializer=init,activation=activation_function))
    model.add(Dropout(0.11))
    model.add(Dense(5,input_dim=3,kernel_initializer=init,activation=activation_function))
    model.add(Dropout(0.11))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=0.1)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[22]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=10,epochs=30)
activation_function=['softmax','relu','tanh','linear']
init=['uniform','normal','zero']
param_grid=dict(activation_function=activation_function,init=init)
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# In[23]:


print('Best : {}, Using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Number of neurons in activation layer

# In[24]:


def create_model(neuron1,neuron2):
    model=Sequential()
    model.add(Dense(neuron1,input_dim=3,kernel_initializer='uniform',activation='softmax'))
    model.add(Dropout(0.11))
    model.add(Dense(neuron2,input_dim=3,kernel_initializer='uniform',activation='softmax'))
    model.add(Dropout(0.11))
    model.add(Dense(1,activation='sigmoid'))
    
    
    adam=Adam(lr=0.1)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model


# In[25]:


model=KerasClassifier(build_fn=create_model,verbose=0,batch_size=10,epochs=30)
neuron1=[30,20,10]
neuron2=[10,15,20]
param_grid=dict(neuron1=neuron1,neuron2=neuron2)
grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=KFold(),verbose=10)
grid_result=grid.fit(x_standardized,y)


# In[26]:


print('Best : {}, Using {}'.format(grid_result.best_score_,grid_result.best_params_))
means=grid_result.cv_results_['mean_test_score']
stds=grid_result.cv_results_['std_test_score']
params=grid_result.cv_results_['params']
for mean,stdevs,param in zip (means,stds,params):
    print('{},{} with : {}'.format (mean,stdevs,param))


# # Train model with optimum values of hyperparameter

# In[27]:


model=Sequential()
model.add(Dense(30,input_dim=3,activation='softmax',kernel_initializer='uniform'))
model.add(Dropout(0.11))
model.add(Dense(10,input_dim=3,activation='softmax',kernel_initializer='uniform'))
model.add(Dropout(0.11))
model.add(Dense(1,activation='sigmoid'))

adam=Adam(lr=0.1)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[28]:


model.fit(x_standardized,y,verbose=0,batch_size=10,epochs=30)


# In[31]:


y_predict=model.predict(x_standardized)
cutoff = 0.4                          
y_predict_classes = np.zeros_like(y_predict)  
y_predict_classes[y_predict > cutoff] = 1 

y_classes = np.zeros_like(y_predict)
y_classes[y > cutoff] = 1


# In[33]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_classes,y_predict_classes))

