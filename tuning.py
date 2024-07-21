import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp 
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow_probability import distributions as tfd
from datetime import datetime
import logging
import sys
import os
import optuna
import json
import time
import tensorflow.compat.v2.keras as keras
from datetime import datetime

print("import correctly")
data= pd.read_csv("DE_final.csv", index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S%z') for e in data.index]

N=2160*24
data = data.iloc[:N]

distribution = 'JSU' # JSU
paramcount = {'Normal': 2,
              'JSU': 4,
              'NormalInverseGaussian': 4,
              'GeneralizedNormal': 3,
              'StudentT': 3,                         
              'Point': None
              }
val_multi = 13  # number of times the model will be retrained 
val_window = 364 // val_multi #28: size of each validation window, each validation window will cover 28 days.
INP_SIZE =  269
activations = ['sigmoid', 'relu', 'elu', 'tanh', 'softplus', 'softmax']

binopt = [True, False]

def objective(trial):
    # prepare the input/output dataframes
    Y = np.zeros((1460, 24))
    Yf = np.zeros((365, 24))
    for d in range(1460):
        Y[d, :] = data.loc[data.index[d*24:(d+1)*24], 'Price'].to_numpy()
    # Y = Y[7:, :] # skip first 7 days
    for d in range(365):
        Yf[d, :] = data.loc[data.index[(
            d+1095)*24:(d+1096)*24], 'Price'].to_numpy()
    #
    X = np.zeros((1095+365, INP_SIZE))
    for d in range(7, 1095+365):
        
        X[d, :24] = data.loc[data.index[(d-1)*24:(d)*24], 'Price'].to_numpy() # D-1 price
        X[d, 24:48] = data.loc[data.index[(d-2)*24:(d-1)*24], 'Price'].to_numpy()  # D-2 price
        X[d, 48:72] = data.loc[data.index[(d-3)*24:(d-2)*24], 'Price'].to_numpy()  # D-3 price
        X[d, 72:96] = data.loc[data.index[(d-7)*24:(d-6)*24], 'Price'].to_numpy()  # D-7 price
        
        X[d, 96:120] = data.loc[data.index[(d)*24:(d+1)*24], 'Load_DA'].to_numpy()  # D load forecast
        X[d, 120:144] = data.loc[data.index[(d-1)*24:(d)*24], 'Load_DA'].to_numpy()  # D-1 load forecast
        X[d, 144:168] = data.loc[data.index[(d-7)*24:(d-6)*24], 'Load_DA'].to_numpy()  # D-7 load forecast
        
        X[d, 168:192] = data.loc[data.index[(d)*24:(d+1)*24], 'RES_gen'].to_numpy()  # D RES sum forecast
        X[d, 192:216] = data.loc[data.index[(d-1)*24:(d)*24], 'RES_gen'].to_numpy()  # D-1 RES sum forecast
        
        X[d, 216:240] = data.loc[data.index[(d)*24:(d+1)*24], 'Resid_Load'].to_numpy()  # D residual load
        X[d, 240:264] = data.loc[data.index[(d-1)*24:(d)*24], 'Resid_Load'].to_numpy()  # D-1 residual load
        
        X[d, 264] = data.loc[data.index[(d-2)*24:(d-1)*24:24], 'EUA_fM'].to_numpy()  # D-2 EUA
        X[d, 265] = data.loc[data.index[(d-2)*24:(d-1)*24:24], 'Coal_fM'].to_numpy()  # D-2 Coal
        X[d, 266] = data.loc[data.index[(d-2)*24:(d-1)*24:24], 'Gas_fM'].to_numpy()  # D-2 Gas
        X[d, 267] = data.loc[data.index[(d-2)*24:(d-1)*24:24], 'Oil_fM'].to_numpy()  # D-2 Oil
       
        X[d, 268] = data.index[d].weekday()
        
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if trial.suggest_categorical('price_D-1', binopt): colmask[:24] = [True] * 24
    if trial.suggest_categorical('price_D-2', binopt): colmask[24:48] = [True] * 24
    if trial.suggest_categorical('price_D-3', binopt): colmask[48:72] = [True] * 24
    if trial.suggest_categorical('price_D-7', binopt): colmask[72:96] = [True] * 24
    
    if trial.suggest_categorical('load_D', binopt): colmask[96:120] = [True] * 24
    if trial.suggest_categorical('load_D-1', binopt): colmask[120:144] = [True] * 24
    if trial.suggest_categorical('load_D-7', binopt): colmask[144:168] = [True] * 24
    
    if trial.suggest_categorical('RES_D', binopt): colmask[168:192] = [True] * 24
    if trial.suggest_categorical('RES_D-1', binopt): colmask[192:216] = [True] * 24
    
    if trial.suggest_categorical('resid_D', binopt): colmask[216:240] = [True] * 24
    if trial.suggest_categorical('resid_D-1', binopt): colmask[240:264] = [True] * 24
    
    
    if trial.suggest_categorical('eua_D-2', binopt): colmask[264] = True
    if trial.suggest_categorical('coal_D-2', binopt): colmask[265] = True
    if trial.suggest_categorical('gas_D-2', binopt): colmask[266] = True
    if trial.suggest_categorical('oil_D-2', binopt): colmask[267] = True

    if trial.suggest_categorical('Dummy', binopt): colmask[268] = True
    X = X[:, colmask]
    
    # '''
    Xwhole = X.copy()
    Ywhole = Y.copy()
    Yfwhole = Yf.copy()
    metrics_sub = []
    
    
    for train_no in range(val_multi):
        start = val_window * train_no
        X = Xwhole[start:1095+start, :]
        Xf = Xwhole[1095+start:1095+start+val_window, :]
        Y = Ywhole[start:1095+start, :]
        Yf = Ywhole[1095+start:1095+start+val_window, :]
        X = X[7:1095, :]
        Y = Y[7:1095, :]
        # begin building a model
        # <= INP_SIZE as some columns might have been turned off
        inputs = keras.Input(X.shape[1])
        # batch normalization- normalize the inputs
       
        batchnorm = True
        if batchnorm:
            norm = keras.layers.BatchNormalization()(inputs)
            last_layer = norm
        else:
            last_layer = inputs
        # dropout
        dropout = trial.suggest_categorical('dropout', binopt)
        if dropout:
            rate = trial.suggest_float('dropout_rate', 0, 1)
            drop = keras.layers.Dropout(rate)(last_layer)
            last_layer = drop
            
            
        # regularization of 1st hidden layer,
        # activation - output, kernel - weights/parameters of input
        regularize_h1_activation = trial.suggest_categorical(
            'regularize_h1_activation', binopt)
        regularize_h1_kernel = trial.suggest_categorical(
            'regularize_h1_kernel', binopt)
        h1_activation_rate = (0.0 if not regularize_h1_activation
                              else trial.suggest_float('h1_activation_rate_l1', 1e-5, 1e1, log=True))
        h1_kernel_rate = (0.0 if not regularize_h1_kernel
                          else trial.suggest_float('h1_kernel_rate_l1', 1e-5, 1e1, log=True))
        
        # define 1st hidden layer with regularization
        hidden = keras.layers.Dense(trial.suggest_int('neurons_1', 16, 256, log=False),
                                    activation=trial.suggest_categorical(
                                        'activation_1', activations),
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(
                                        h1_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
        
        
        # regularization of 2nd hidden layer,
        # activation - output, kernel - weights/parameters of input
        regularize_h2_activation = trial.suggest_categorical(
            'regularize_h2_activation', binopt)
        regularize_h2_kernel = trial.suggest_categorical(
            'regularize_h2_kernel', binopt)
        h2_activation_rate = (0.0 if not regularize_h2_activation
                              else trial.suggest_float('h2_activation_rate_l1', 1e-5, 1e1, log=True))
        h2_kernel_rate = (0.0 if not regularize_h2_kernel
                          else trial.suggest_float('h2_kernel_rate_l1', 1e-5, 1e1, log=True))
        # define 2nd hidden layer with regularization
        hidden = keras.layers.Dense(trial.suggest_int('neurons_2', 16, 256, log=False),
                                    activation=trial.suggest_categorical(
                                        'activation_2', activations),
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(
                                        h2_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)
        
        ### DNN
        if paramcount[distribution] is None:
            outputs = keras.layers.Dense(24, activation='linear')(hidden)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                          loss='mae',
                          metrics='mae')
        else:
            # define parameter layers with their regularization
            param_layers = []
            param_names = ["loc", "scale", "tailweight", "skewness"]
            for p in range(paramcount[distribution]):
                regularize_param_kernel = trial.suggest_categorical(
                    'regularize_'+param_names[p], binopt)
                param_kernel_rate = (0.0 if not regularize_param_kernel
                                     else trial.suggest_float(param_names[p]+'_rate_l1', 1e-5, 1e1, log=True))
                param_layers.append(keras.layers.Dense(
                    24, activation='linear',  # kernel_initializer='ones',
                    kernel_regularizer=keras.regularizers.L1(param_kernel_rate))(hidden))
        
        
        # Modeling
            linear = tf.keras.layers.concatenate(param_layers)
            # define outputs
            if distribution == 'Normal':
                outputs = tfp.layers.DistributionLambda(
                    lambda t: tfd.Normal(
                        loc=t[..., :24],
                        scale=1e-3 + 3 * tf.math.softplus(t[..., 24:])))(linear)
            elif distribution == 'StudentT':
                outputs = tfp.layers.DistributionLambda(
                    lambda t: tfd.StudentT(
                        loc=t[..., :24],
                        scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                        df=1 + 3 * tf.math.softplus(t[..., 48:])))(linear)
            elif distribution == 'JSU':
                outputs = tfp.layers.DistributionLambda(
                    lambda t: tfd.JohnsonSU(
                        loc=t[..., :24],
                        scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                        tailweight=1 + 3 * tf.math.softplus(t[..., 48:72]),
                        skewness=t[..., 72:]))(linear)
            else:
                raise ValueError(f'Incorrect distribution {distribution}')
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                          loss=lambda y, rv_y: -rv_y.log_prob(y), #minimize the negative log-likelihood
                          metrics='mae')
        # '''
        
        # define callbacks
        callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
        model.fit(X, Y, epochs=100, validation_data=(Xf, Yf),callbacks=callbacks, batch_size=32, verbose=True)

        # for point its a list of one [loss, MAE]
        metrics = model.evaluate(Xf, Yf)
        metrics_sub.append(metrics[0])
        avg_metric = np.mean(metrics_sub)
        # Update the best parameters if this trial has better performance
        if 'best_metric' not in trial.user_attrs or avg_metric < trial.user_attrs['best_metric']:
            trial.user_attrs['best_metric'] = avg_metric
            trial.user_attrs['best_params'] = trial.params
        # we optimize the returned value, -1 will always take the model with best MAE
    return avg_metric



# Create an Optuna study
study_jsu = optuna.create_study(direction='minimize')

# Optimize the study
study_jsu.optimize(objective, n_trials=100)

# Access the best hyperparameters
best_params_jsu = study_jsu.best_params
print("Best hyperparameters:", best_params_jsu)
