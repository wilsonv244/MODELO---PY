# %%
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from datetime import datetime, date
import pyodbc
import time
import re
from requests import ReadTimeout
import seaborn as sns
import datetime 
import pyodbc
from  calendar import monthrange

from flask import Flask, request, render_template
app = Flask(__name__)

#from products import products

#Carga de información
#Estadísticas de preferencias
#Importando las librerías
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from datetime import datetime, date
import pyodbc
import time
import re
import seaborn as sns
import datetime 
from  calendar import monthrange

def modeloriesgos(input_table_1,dir):

    
    plt.rcParams.update({'figure.max_open_warning': 0})
    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    d=dir
    os.chdir(d)


    from sklearn.tree import DecisionTreeClassifier,export_graphviz
    from IPython.display import Image as PImage
    from subprocess import check_call
    from PIL import Image, ImageDraw, ImageFont
    from pydotplus import graph_from_dot_data    


    from sklearn import preprocessing
    from sklearn.preprocessing  import StandardScaler
    from sklearn.decomposition import PCA

    from sklearn.model_selection import learning_curve
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    #from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import GridSearchCV
    import pickle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()


    ###prueba


    def fecha_spanish(fecha):
        mes=[]
        if fecha[5:7]=='01':
            mes='Ene'
        elif fecha[5:7]=='02':
            mes='Feb'
        elif fecha[5:7]=='03':
            mes='Mar'
        elif fecha[5:7]=='04':
            mes='Abr'
        elif fecha[5:7]=='05':
            mes='May'
        elif fecha[5:7]=='06':
            mes='Jun'
        elif fecha[5:7]=='07':
            mes='Jul'
        elif fecha[5:7]=='08':
            mes='Ago'
        elif fecha[5:7]=='09':
            mes='Set'
        elif fecha[5:7]=='10':
            mes='Oct'
        elif fecha[5:7]=='11':
            mes='Nov'
        else:
            mes='Dic'

        return mes+fecha[2:4]


    def last_day_of_month(date_value):
        return date_value.replace(day = monthrange(date_value.year, date_value.month)[1])
    
    today = date.today()
    
    
    
    
    import dateutil.relativedelta

  
    ParametroFecha = today - dateutil.relativedelta.relativedelta(months=1)
  
    
    
    
    ParametroFecha=last_day_of_month(last_day_of_month(today)-pd.DateOffset(months=1))
    ParametroFecha=ParametroFecha.strftime("%Y-%m-%d")


    folder='./Files_'+str(fecha_spanish(ParametroFecha))+'/'
    print('---------------el nuevo modelo')
    print(folder)
    #folder='./Files_Ene22/'
    # Example

    os.chdir(folder)













    


    input_table_1['Fechap']=pd.to_datetime(input_table_1['Fechap'])
    
    input_table_1=input_table_1[(input_table_1['Fechap'].apply(lambda x:last_day_of_month(x))==last_day_of_month(today).strftime("%Y-%m-%d")) & (input_table_1['idEstado']==1)]
    #input_table_1=input_table_1[(input_table_1['Fechap'].apply(lambda x:last_day_of_month(x))==last_day_of_month(today).strftime("%Y-%m-%d")) & (input_table_1['idEstado']==1)]

    input_table_1['FechaNacimiento']=pd.to_datetime(input_table_1['FechaNacimiento'])
    input_table_1['Fecha_so']=input_table_1['Fechap']
    input_table_1['plazo']=input_table_1['plazo_sol']
    input_table_1['Edad']=round((input_table_1['Fechap']-  input_table_1['FechaNacimiento'])/np.timedelta64(1,'Y'),0)
    input_table_1['Fecha_so']
    
    td= input_table_1.copy() 
    Result = pd.DataFrame()

    # load the model from disk
    pcafile='pca.sav'
    filename = 'Modelo.sav'
    prep='scaler.sav'
    pca = pickle.load(open(pcafile, 'rb'))
    lr = pickle.load(open(filename, 'rb'))
    scaler = pickle.load(open(prep, 'rb'))
 

    fecha=input_table_1['Fechap']

    final_features=pd.read_csv('features.csv')
    final_features=final_features['Variables'].to_list()

    variables_normalizar=pd.read_csv('variables_normalizar.csv')
    variables_normalizar=variables_normalizar['Variables'].to_list()

    pcas=pd.read_csv('pcas.csv')
    pcas=pcas['pcas'].to_list()

    td=td[[
    'num_solicitud',
    'Edad',
    'Sexo',
    'EstadoCivil',
    'TipoVivienda',
    'NivelInstruccion',
    'Profesion',
    'Departamento',
    'telefono',
    'NroDependientes',
    'AniosResidencia',
    'Fecha_so',
    'Monto_sol',
    'plazo'
    ]]
    
    td['Monto_sol']=td['Monto_sol'].astype('float')
    td['plazo']=td['plazo'].astype('float')
    td['NroDependientes']=td['NroDependientes'].astype('int')
    td['AniosResidencia']=td['AniosResidencia'].astype('int')

    td = td.set_index('num_solicitud')
    



    features_c=td.select_dtypes(include='object').columns.tolist()
    features_n=td.select_dtypes(exclude='object').columns.tolist()

    features_n.remove('Fecha_so')
    features_c.remove('telefono')

    for label in features_c:
        #print(label)
        td[label].fillna(td[label].mode()[0], inplace=True)



    #Tratamiento de los datos

    def TransformVariables2(Rango,valores):
        for label in Rango:
            for categoria in valores:
                td[label+'_'+categoria]=np.where(td[label]==categoria,1,0)
                td[label+'_'+categoria]=td[label+'_'+categoria].astype('O')

        td[label+'_'+'OTROS']=np.where(td[label].apply(lambda x: x not in valores),1,0)        
        td[label+'_'+'OTROS']=td[label+'_'+'OTROS'].astype('O') 
        td.drop([label],axis=1,inplace=True)

    #Creación de la variable Dia de la semana
    td['Fecha_so']=pd.to_datetime(td['Fecha_so'])
    td['DiaSemana'] = td['Fecha_so'].dt.dayofweek
    td['DiaSemana']=np.where(td['DiaSemana']==0,"Otros",np.where(td['DiaSemana']==1,"Martes",np.where(td['DiaSemana']==2,"Miercoles",np.where(td['DiaSemana']==3,"Jueves",np.where(td['DiaSemana']==4,"Viernes",np.where(td['DiaSemana']==5,"Otros","Otros"))))))


    #creación de variables

    #td['Nombre_producto_sol']=np.where(td['Nombre_producto_sol'].str.lower().str.contains('pyme'),'PYME','CONN')

    td['NumReferenciasTel']=td['telefono'].apply(lambda x:x.rstrip().lstrip().count(' ')+x.rstrip().lstrip().count(',')+x.rstrip().lstrip().count(';')+1)



    td['Profesion']=np.where(td['Profesion'].isnull(),td['Profesion'],td['Profesion'].apply(lambda x: x.replace(',','').replace('/','').replace('(','').replace(')','')))

    Rango=['DiaSemana']
    valores=['Martes','Miercoles','Jueves','Viernes','Otros']
    TransformVariables2(Rango,valores)

    Rango=['Sexo']
    valores=['MASCULINO','FEMENINO']
    TransformVariables2(Rango,valores)

    #1 
    Rango=['Profesion']
    valores=['COMERCIANTE  VENDEDOR','OTROS señalar','OBRERO  OPERADOR','ALBAÑIL OBRERO DE CONSTRUCCIÓN','CONDUCTOR CHOFER  TAXISTA','TRANSPORTISTA','AGRICULTOR AGRÓLOGO ARBORICULTOR GEÓGRAFO','GANADERO','ADMINISTRADOR','DOCENTE','AMA DE CASA','CAMPESINO']
    TransformVariables2(Rango,valores)


    #2
    Rango=['NivelInstruccion']
    valores=['SECUNDARIA COMPLETA','SECUNDARIA INCOMPLETA','SECUNDARIA','PRIMARIA COMPLETA','SUPERIOR NO UNIVERSITARIO','SUPERIOR UNIVERSITARIO','PRIMARIA INCOMPLETA','PRIMARIA']
    TransformVariables2(Rango,valores)

    #3
    Rango=['Departamento']
    valores=['HUANUCO','CUSCO','JUNIN','CAJAMARCA','AYACU+CHO','AREQUIPA','AMAZONAS','PUNO','PASCO','APURIMAC','TACNA','HUANCAVELICA']
    TransformVariables2(Rango,valores)

    #4
    Rango=['TipoVivienda']
    valores=['FAMILIAR','PROPIA AUTOFINANCIADO','ALQUILADO','PROPIA HEREDADA']
    TransformVariables2(Rango,valores)

    #5
    Rango=['EstadoCivil']
    valores=['SOLTERO(A)','CONVIVIENTE','CASADO(A)','VIUDO(A)','SEPARADO(A)','DIVORCIADO(A)']
    TransformVariables2(Rango,valores)


    td=td.drop(['telefono','Fecha_so'],axis=1)
    features_c=td.select_dtypes(include='object').columns.tolist()


    for label in features_n:
        td[label].fillna(td[label].median(), inplace=True)

    features_n_originales=features_n
    #print('-----aqui')
    #print(td.describe())   



    for label in features_n:
        trans_1=label+'_log'
        trans_2=label+'_sqrt'
        trans_3=label+'_power2'
        trans_4=label+'_power3'

        td[trans_1]=np.log(td[label]+1)
        td[trans_2]=np.sqrt(td[label])
        td[trans_3]=np.power(td[label],2)
        td[trans_4]=np.power(td[label],3)

    td.drop(['DiaSemana_OTROS','Sexo_OTROS'],axis=1,inplace=True)

    features_n=td.select_dtypes(exclude='object').columns.tolist()

    td=td.reset_index(inplace=False).sort_values(by='num_solicitud',ascending=True).set_index('num_solicitud')
    X= td

    columns=X.columns.tolist()


    #print('Esto')
    #X.to_excel ('validar_completo_auto.xlsx', index = True, header=True)
    #################
    #Split train/test 
    #################


    x_train=X

    #print(final_features)

    #seleccionado las mejores variables transformadas

    x_train_res=x_train[variables_normalizar]
    #x_val=x_val[best_variables]

    from sklearn.preprocessing import StandardScaler 
    x_train_res_scal = scaler.transform(x_train_res)

    x_train_res_scal=pd.DataFrame(x_train_res_scal,columns=x_train_res.columns.tolist())





    x= x_train_res_scal[final_features]

      #Componentes Principales

    principalComponentes = pca.transform(x)
    x_pca=principalComponentes  





    x_train_pca=pd.DataFrame(x_pca,columns=pcas)

    X=x_train_pca

    predicciones=lr.predict_proba(x_train_pca)
    predicciones = pd.DataFrame(predicciones, columns = lr.classes_)
    predicciones=predicciones.loc[:,1:2]    

    y_predict_train= lr.predict(X)
    X['probabilidades']=predicciones
    X['Predict']=np.where(y_predict_train==1,'Malo','Bueno')
    X['Fecha']=today
    X['FechaCarga']=date.today()
    X['num_solicitud']=td.reset_index()['num_solicitud']
    output_table_1=X[['Fecha','Predict','num_solicitud','probabilidades','FechaCarga']]
    
    
    #dirección de la búsqueda#
    
    # import urllib
    # from sqlalchemy import create_engine
    # server = '10.5.5.226'
    # database = 'BD_RIESGOS'
    # params = urllib.parse.quote_plus('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';PORT=1443;DATABASE='+database+';Trusted_Connection=yes;')
    # engine = create_engine("mssql+pyodbc:///?odbc_connect=%s"%params)
    # output_table_1.to_sql('PL_PRUEBA',con=engine, if_exists='append', index=False)#LA BASE SE LLAMA PL_PRUEBA
    # return output_table_1 
    #cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.5.5.226;DATABASE=BD_RIESGOS;UID=wvargas;PWD=Losandes.123')
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.4.12.15;DATABASE=GNormativas;UID=uNormativas;PWD=123456789')
    cursor = cnxn.cursor()

    cursor.execute("INSERT INTO PruebaRiezgo_2(Fecha,Predict,num_solicitud,probabilidades,FechaCarga) values (?, ?, ?, ?, ?)",output_table_1['Fecha'][0],output_table_1['Predict'][0],int(output_table_1['num_solicitud'][0]),output_table_1['probabilidades'][0],output_table_1['FechaCarga'][0])
    cnxn.commit()
    return output_table_1


# Testing Route
@app.route('/prueba')
def index():
    
    return "<h1>Rogger - Index</h1>"

# Testing Route
@app.route('/modeloNoBancarizado', methods=['POST'])
def getModelo():
        
    from urllib.parse import unquote
    #decoded = unquote(t)
    
    # num_solicitud = request.args.get('num_solicitud',0,type=int)
    # Fechap = request.args.get('Fechap','',type=str)
    # idEstado = request.args.get('idEstado',0,type=int)
    # Monto_sol = request.args.get('Monto_sol',0,type=float)
    # plazo_sol = request.args.get('plazo_sol',0,type=int)
    # FechaNacimiento = request.args.get('FechaNacimiento','',type=str)
    # Sexo = request.args.get('Sexo','',type=str)
    # EstadoCivil = request.args.get('EstadoCivil','',type=str)
    # TipoVivienda = request.args.get('TipoVivienda','',type=str)
    # NivelInstruccion = request.args.get('NivelInstruccion','',type=str)
    # Actividad = request.args.get('Actividad','',type=str)
    # Profesion = request.args.get('Profesion','',type=str)
    # Departamento = request.args.get('Departamento','',type=str)
    # telefono = request.args.get('telefono','',type=str)
    # NroDependientes = request.args.get('NroDependientes',0,type=int)
    # AniosResidencia = request.args.get('AniosResidencia',0,type=int)
    num_solicitud = request.json['num_solicitud']
    Fechap = request.json['Fechap']
    idEstado = request.json['idEstado']
    Monto_sol = request.json['Monto_sol']
    plazo_sol = request.json['plazo_sol']
    FechaNacimiento = request.json['FechaNacimiento']
    Sexo = request.json['Sexo']
    EstadoCivil = request.json['EstadoCivil']
    TipoVivienda = request.json['TipoVivienda']
    NivelInstruccion = request.json['NivelInstruccion']
    Actividad = request.json['Actividad']
    Profesion = request.json['Profesion']
    Departamento = request.json['Departamento']
    telefono = request.json['telefono']
    NroDependientes = request.json['NroDependientes']
    AniosResidencia = request.json['AniosResidencia']
    
    var_1=[num_solicitud]
    var_2=[Fechap]
    var_3=[idEstado]
    var_4=[Monto_sol]
    var_5=[plazo_sol]
    var_6=[FechaNacimiento]
    var_7=[Sexo]
    var_8=[EstadoCivil]
    var_9=[TipoVivienda]
    var_10=[NivelInstruccion]
    var_11=[Actividad]
    var_12=[Profesion]
    var_13=[Departamento]
    var_14=[telefono]
    var_15=[NroDependientes]
    var_16=[AniosResidencia]
    
    #se crea una lista
    df=[]
    df.append(var_1)
    df.append(var_2)
    df.append(var_3)
    df.append(var_4)
    df.append(var_5)
    df.append(var_6)
    df.append(var_7)
    df.append(var_8)
    df.append(var_9)
    df.append(var_10)
    df.append(var_11)
    df.append(var_12)
    df.append(var_13)
    df.append(var_14)
    df.append(var_15)
    df.append(var_16)
    
    df = pd.DataFrame (df).transpose()
    df.columns=['num_solicitud','Fechap','idEstado','Monto_sol','plazo_sol',
                'FechaNacimiento','Sexo','EstadoCivil','TipoVivienda','NivelInstruccion',
                'Actividad','Profesion','Departamento','telefono','NroDependientes','AniosResidencia']
    
    modeloriesgos(df,"D:/Modelo Grupo Solidario")
    
    return "<h1>Datos Procesados</h1>"


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=4000)
    

# %%
x = 2 
print(x)



# %%
