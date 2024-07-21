import pandas as pd
import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import LassoCV

import tensorflow_probability as tfp 
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from tensorflow_probability import distributions as tfd
from datetime import datetime, timedelta
import tensorflow.compat.v2.keras as keras
from datetime import datetime
from sklearn import linear_model
from scipy.stats import norm, johnsonsu
print("import correctly")

data= pd.read_csv("DE_final.csv", index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S%z') for e in data.index]



INP_SIZE =  269
cal=84 # 730, 1095, 1460
# 5 years size of 1460 data point

def lasso(inp):
    cal, dayno = inp
    df = data.iloc[dayno*24:dayno*24+1460*24+24]
    df = df[-(cal+1)*24:]
    
    # prepare the input/output dataframes
    Y = np.zeros((cal, 24))
    
    for d in range(cal):
        Y[d, :] = df.loc[df.index[d*24:(d+1)*24], 'Price'].to_numpy()
    Y = Y[7:, :] # skip first 7 days

    X = np.zeros((cal+1, INP_SIZE))
    for d in range(7, cal+1):
        X[d, :24] = df.loc[df.index[(d-1)*24:(d)*24], 'Price'].to_numpy()  # D-1 price
        X[d, 24:48] = df.loc[df.index[(d-2)*24:(d-1)*24], 'Price'].to_numpy() # D-2 price
        X[d, 48:72] = df.loc[df.index[(d-3)*24:(d-2)*24], 'Price'].to_numpy()  # D-3 price
        X[d, 72:96] = df.loc[df.index[(d-7)*24:(d-6)*24], 'Price'].to_numpy()  # D-7 price
       
        X[d, 96:120] = df.loc[df.index[(d)*24:(d+1)*24], 'Load_DA'].to_numpy()  # D load forecast
        X[d, 120:144] = df.loc[df.index[(d-1)*24:(d)*24], 'Load_DA'].to_numpy()  # D-1 load forecast
        X[d, 144:168] = df.loc[df.index[(d-7)*24:(d-6)*24], 'Load_DA'].to_numpy()  # D-7 load forecast
        
        X[d, 168:192] = df.loc[df.index[(d)*24:(d+1)*24], 'RES_gen'].to_numpy()  # D RES sum forecast
        X[d, 192:216] = df.loc[df.index[(d-1)*24:(d)*24], 'RES_gen'].to_numpy()  # D-1 RES sum forecast
        
        X[d, 216:240] = df.loc[df.index[(d)*24:(d+1)*24], 'Resid_Load'].to_numpy()  # D residual load
        X[d, 240:264] = df.loc[df.index[(d-1)*24:(d)*24], 'Resid_Load'].to_numpy()  # D-1 residual load
        
        X[d, 264] = df.loc[df.index[(d-2)*24:(d-1)*24:24], 'EUA_fM'].to_numpy()[0]  # D-2 EUA
        X[d, 265] = df.loc[df.index[(d-2)*24:(d-1)*24:24], 'Coal_fM'].to_numpy()[0]  # D-2 Coal
        X[d, 266] = df.loc[df.index[(d-2)*24:(d-1)*24:24], 'Gas_fM'].to_numpy()[0]  # D-2 Gas
        X[d, 267] = df.loc[df.index[(d-2)*24:(d-1)*24:24], 'Oil_fM'].to_numpy()[0]  # D-2 Oil
        
        X[d, 268] = df.index[d].weekday()

    Xf = X[-1:, :]
    X = X[7:-1, :]
    
    predDF = pd.DataFrame(index=df.index[-24:])
    predDF['real'] = df.loc[df.index[-24:], 'Price'].to_numpy()
    predDF['forecast'] = np.nan
    
    best_alpha = np.full(24, np.nan)
    best_coefs = np.full((24, INP_SIZE), np.nan)

    for h in range(24):
        model_lasso = linear_model.LassoCV(eps=1e-6, n_alphas=100, cv=7)
        model_lasso.fit(X, Y[:, h])
        pred = model_lasso.predict(Xf)[0]
        predDF.loc[predDF.index[h], 'forecast'] = pred

        # Store the best alpha and coefficients
        best_alpha[h] = model_lasso.alpha_
        best_coefs[h, :] = model_lasso.coef_
        
    print(predDF)
    return predDF, best_alpha, best_coefs

inputlist = [(cal, day) for day in range(len(data) // 24 - 1460)]
print(len(inputlist))

all_forecasts = pd.DataFrame()

for e in inputlist:
    forecast_lasso, best_alpha, best_coefs = lasso(e)
    all_forecasts = pd.concat([all_forecasts, forecast_lasso])

# Save all forecasts to a single CSV file
all_forecasts.to_csv('lasso_84.csv')

final_best_alpha_lasso = best_alpha
final_best_coefs_df_lasso = best_coefs

print(final_best_alpha_lasso)
print(final_best_coefs_df_lasso)
