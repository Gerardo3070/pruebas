import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mpdates
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as LR
import statsmodels.api as sm
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

#Modelo MLR

dataset=pd.read_csv('master_join.csv')
X=dataset.iloc[:,3:].values
y=dataset.iloc[:,2:3].values
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=0)
regresion=LR()
regresion.fit(X_train, y_train)
y_pred_MLR=regresion.predict(X_test)
X_opt=X_test
regresion_OLS = sm.OLS(endog = y_test, exog = X_opt.tolist()).fit()

#Modelo FR

merge_df = pd.read_csv('merge_df.csv')
target = 'estimated_delivery_time'
features = ['freight_value', 
            'product_volume_cm3', 
            'product_weight_g',
            'carrier_delivery_time',
            'distance'
           ]



X_RF = merge_df[features]
y_rf = merge_df[target]

X_train_RF, X_test_RF, y_train_rf, y_test_rf = tts(X_RF, y_rf, test_size=0.2, random_state = 0)

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train_RF, y_train_rf)
y_pred_RF = regressor.predict(X_test_RF)
r2=regressor.score(X_train_RF, y_train_rf)


#Slidebars y predicciones

st.sidebar.header('Especificar parámetros del artículo a enviar')
peso = st.sidebar.slider('peso (g)', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
largo = st.sidebar.slider('largo (cm)', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
alto = st.sidebar.slider('alto (cm)', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
ancho = st.sidebar.slider('ancho (cm)', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
dist = st.sidebar.slider('distancia (factor)', float(X[:,4].min()), float(X[:,4].max()), float(X[:,4].mean()))
st.sidebar.header('Especificar parámetros extra para cálculo de entrega de producto')
tiem = st.sidebar.slider('tiempo de envío a operador (días)', float(0), float(X_RF.carrier_delivery_time.max()), float(X_RF.carrier_delivery_time.mean()))
distance = st.sidebar.slider('Distancia (factor)', float(X_RF.distance.min()), float(X_RF.distance.max()), float(X_RF.distance.mean()))
val_pred=[peso, largo, alto, ancho, dist]
vol=largo*alto*ancho
st.header('Predicción precio de envío')
val_pred=np.array(val_pred).reshape(1,5)
pred_MLR=regresion.predict(val_pred)
st.write(pred_MLR+6.522)
st.write('---')
st.header('Predicción tiempo de envío')
val_pred_RF=[pred_MLR, vol, peso, tiem, distance]
val_pred_RF=np.array(val_pred_RF).reshape(1,5)
pred_RF=regressor.predict(val_pred_RF)
st.write(pred_RF-14)
st.write('---')
st.header('Score Random Forest (R^2 criterion)')
st.write(round(r2, 2))
st.write(regresion_OLS.summary())
st.write('---')

