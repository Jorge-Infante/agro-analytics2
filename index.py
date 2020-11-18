from flask import Flask, render_template, abort
#Permite conectar con Boostrap
from flask_bootstrap import Bootstrap
#permite traer los datos d POST
from flask import request
#Librerias para modelo de regresión
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sodapy import Socrata
import json 
import plotly
import plotly.plotly as py
import plotly.graph_objs as go



#Importar el conjunto de datos
import json
import requests
import pygal
import io 
from flask import Response
mv_data_json = requests.get('https://www.datos.gov.co/resource/2pnw-mmge.json?$limit=500000')
mv_list_recs = json.loads(mv_data_json.text)
cultivos = pd.DataFrame(mv_list_recs)
#mv_list_recs
#En la consola ejecutar para que sea en tiempo real 
# set "FLASK_ENV=development"
#set FLASK_APP=src/index.py

#Eliminamos las columnas no necesarias
cultivos = cultivos.drop(['nombre_cientifico'], axis=1)
cultivos = cultivos.drop(['c_d_mun'], axis=1)
cultivos = cultivos.drop(['municipio'], axis=1)
cultivos = cultivos.drop(['grupo_de_cultivo'], axis=1)
cultivos = cultivos.drop(['desagregaci_n_regional_y'], axis=1)
cultivos = cultivos.drop(['a_o'], axis=1)
cultivos = cultivos.drop(['subgrupo_de_cultivo'], axis=1)
cultivos = cultivos.drop(['estado_fisico_produccion'], axis=1)
#Actualizamos el nombre de las columnas
cultivos = cultivos.rename(columns={'c_d_dep':'COD_DEP','departamento':'DEPARTAMENTO',
                                   'cultivo':'CULTIVO','periodo':'PERIODO','rea_sembrada_ha':'AREA SEMBRADA','rea_cosechada_ha':'AREA COSECHADA','producci_n_t':'PRODUCCION','rendimiento_t_ha':'RENDIMIENTO','ciclo_de_cultivo':'CICLO CULTIVO'})
datosMaiz = cultivos[(cultivos['CULTIVO'] == 'MAIZ')]
datosArroz = cultivos[(cultivos['CULTIVO'] == 'ARROZ')]
datosCafe = cultivos[(cultivos['CULTIVO'] == 'CAFE')]
datosFinales = pd.concat([datosMaiz, datosArroz, datosCafe], ignore_index=True)
dGraficar = pd.concat([datosMaiz, datosArroz, datosCafe], ignore_index=True)

datosFinales['PRODUCCION'] = pd.to_numeric(datosFinales['PRODUCCION'])
datosFinales['AREA SEMBRADA'] = pd.to_numeric(datosFinales['AREA SEMBRADA'])
datosFinales['COD_DEP'] = pd.to_numeric(datosFinales['COD_DEP'])
datosFinales['AREA COSECHADA'] = pd.to_numeric(datosFinales['AREA COSECHADA'])
datosFinales['RENDIMIENTO'] = pd.to_numeric(datosFinales['RENDIMIENTO'])

dGraficar['PRODUCCION'] = pd.to_numeric(dGraficar['PRODUCCION'])
dGraficar['AREA SEMBRADA'] = pd.to_numeric(dGraficar['AREA SEMBRADA'])
dGraficar['COD_DEP'] = pd.to_numeric(dGraficar['COD_DEP'])
dGraficar['AREA COSECHADA'] = pd.to_numeric(dGraficar['AREA COSECHADA'])
dGraficar['RENDIMIENTO'] = pd.to_numeric(dGraficar['RENDIMIENTO'])

#Luego utilizamos la función replace para realizar el cambio
datosFinales['CULTIVO'].replace(to_replace = ['MAIZ','ARROZ','CAFE'], value =[0,1,2], inplace=True)
datosFinales['CICLO CULTIVO'].replace(to_replace = ['ANUAL','PERMANENTE','TRANSITORIO'], value =[0,1,2], inplace=True)

from sklearn.preprocessing  import KBinsDiscretizer
Produccion_dis = KBinsDiscretizer(n_bins=5, encode='ordinal',strategy = "kmeans").fit_transform(datosFinales[['PRODUCCION']])

Produccion_dis = pd.DataFrame(Produccion_dis)
Produccion_dis = Produccion_dis.rename(columns = {0: 'PRODUCCION'})

datosFinales[['PRODUCCION']] = Produccion_dis
#Defino mis variables de entrenamiento
X = datosFinales[['COD_DEP','CULTIVO','AREA SEMBRADA','AREA COSECHADA']]
y = datosFinales['PRODUCCION']



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('modelo.html')

@app.route('/procesamiento')
def procesamiento():
    return render_template('.../Sakila.html')
@app.route('/modelo')
def modelo():
    return render_template('modelo.html')

@app.route('/parametros',methods=['GET', 'POST'])
def parametros():
    
    if request.method == "POST":
        res=0
        cultivo = request.form['cultivo']
        departamento = request.form['departamento']
        Asembrada = request.form['Asembrada']
        Acosechada = request.form['Acosechada']
        if int(Acosechada) > int(Asembrada):
            res = "EL área cosechada no podrá ser mayor al área sembrada"
            
        else:
            #Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            #Naive Bayes
            from sklearn.naive_bayes import GaussianNB
            algoritmo = GaussianNB()
            #Entreno el modelo
            algoritmo.fit(X_train, y_train)
            #Realizo una predicción
            y_pred = algoritmo.predict(X_test)
            resultado = algoritmo.predict([[int(departamento),int(cultivo),int(Asembrada),int(Acosechada)]])
            res =  resultado[0]
            if res == 0:
                res = "La producción del cultivo estará entre 0 y 35000 Toneladas"
            if res == 1:
                res = "La producción del cultivo estará entre 35000 y 70000 Toneladas"
            if res == 2:
                res = "La producción del cultivo estará entre 70000 y 105000 Toneladas"
            if res == 3:
                res = "La producción del cultivo estará entre 105000 y 140000 Toneladas"
            if res == 4:
                res = "La producción del cultivo estará entre 140000 y 210000 Toneladas"
    return render_template('modelo.html', resul = res)

@app.route('/graficar')
def graficar():
    return False

if __name__=='_main_':
    app.run(debug=True)