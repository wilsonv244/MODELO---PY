# %%
from ast import Try
from types import MethodType
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
from datetime import datetime, date
import pyodbc
import time
import re
#from requests import ReadTimeout
#import seaborn as sns
import datetime 
import pyodbc
from  calendar import monthrange

from flask import Flask, request, render_template
from sklearn.covariance import fast_mcd
from sqlalchemy import false, null, true
from zmq import EVENT_CLOSE_FAILED
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
import datetime
import re
#import seaborn as sns
import datetime 
from  calendar import monthrange



def modeloriesgos(input_table_1,dir):

    
    plt.rcParams.update({'figure.max_open_warning': 0})
    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    d=dir
    #os.chdir(d)


    #from sklearn.tree import DecisionTreeClassifier,export_graphviz
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

    #Resultado de la consulta
    responseMessage = pd.DataFrame(
        {
            'Fechap':[],
            'num_solicitud':[],
            'score':[],
            'mensajeScore':[],
            'Oficina':[],
            'Asesor':[],
            'Edad':[],
            'Sexo':[],
            'EstadoCivil':[],
            'TipoVivienda':[],
            'NivelInstruccion':[],
            'Monto_sol':[],
            'Plazo':[],
            'NroDependientes':[],
            'AniosResidencia':[],
            'Telefono':[],
            'Profesion':[],
            'DiaDeLaSemana':[],
            'Departamento':[],
            'DestinoCredito':[]

        }
    )

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


    folder='/Modelo Grupo Solidario/data/Files_'+str(fecha_spanish(ParametroFecha))+'/'
    print('---------------el nuevo modelo')
    print(folder)
    #folder='./Files_Ene22/'
    # Example

    os.chdir(folder)













    


    input_table_1['Fechap']=pd.to_datetime(input_table_1['Fechap'],dayfirst=True)
    
    input_table_1=input_table_1[(input_table_1['Fechap'].apply(lambda x:last_day_of_month(x))==last_day_of_month(today).strftime("%Y-%m-%d")) & (input_table_1['idEstado']==1)]
    #input_table_1=input_table_1[(input_table_1['Fechap'].apply(lambda x:last_day_of_month(x))==last_day_of_month(today).strftime("%Y-%m-%d")) & (input_table_1['idEstado']==1)]

    input_table_1['FechaNacimiento']=pd.to_datetime(input_table_1['FechaNacimiento'])
    input_table_1['Fecha_so']=input_table_1['Fechap']
    input_table_1['plazo']=input_table_1['plazo_sol']
    input_table_1['Edad']=round((input_table_1['Fechap']-  input_table_1['FechaNacimiento'])/np.timedelta64(1,'Y'),0)
    input_table_1['Fecha_so']

    responseMessage['Fechap']= pd.to_datetime(responseMessage['Fechap'])
    responseMessage['Fechap']= input_table_1['Fechap']
    responseMessage['num_solicitud']= input_table_1['num_solicitud']
    responseMessage['Edad'] = input_table_1['Edad']
    responseMessage['Sexo'] = input_table_1['Sexo']
    responseMessage['EstadoCivil'] = input_table_1['EstadoCivil']
    responseMessage['TipoVivienda'] = input_table_1['TipoVivienda']
    responseMessage['NivelInstruccion'] = input_table_1['NivelInstruccion']
    responseMessage['Monto_sol'] = input_table_1['Monto_sol']
    responseMessage['Plazo'] = input_table_1['plazo_sol']
    responseMessage['NroDependientes'] = input_table_1['NroDependientes']
    responseMessage['AniosResidencia'] = input_table_1['AniosResidencia']
    responseMessage['Telefono'] = input_table_1['Telefono']
    responseMessage['Profesion'] = input_table_1['Profesion']
    responseMessage['Departamento'] = input_table_1['Departamento']
    responseMessage['DestinoCredito'] = input_table_1['DestinoCredito']
    responseMessage['Oficina'] = input_table_1['Oficina']
    responseMessage['Asesor'] = input_table_1['Asesor']

    
    responseMessage['DiaDeLaSemana']=pd.to_datetime(responseMessage['DiaDeLaSemana'])
    responseMessage['DiaDeLaSemana'] = input_table_1['Fechap'].dt.dayofweek
    responseMessage['DiaDeLaSemana']=np.where(responseMessage['DiaDeLaSemana']==0,"Otros",np.where(responseMessage['DiaDeLaSemana']==1,"Martes",np.where(responseMessage['DiaDeLaSemana']==2,"Miercoles",np.where(responseMessage['DiaDeLaSemana']==3,"Jueves",np.where(responseMessage['DiaDeLaSemana']==4,"Viernes",np.where(responseMessage['DiaDeLaSemana']==5,"Otros","Otros"))))))

    

    
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
    'Telefono',
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
    features_c.remove('Telefono')

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
    #td['Fecha_so'] = input_table_1['Fechap']
    td['DiaSemana'] = td['Fecha_so'].dt.dayofweek
    print(td['DiaSemana'])
    td['DiaSemana']=np.where(td['DiaSemana']==0,"Otros",np.where(td['DiaSemana']==1,"Martes",np.where(td['DiaSemana']==2,"Miercoles",np.where(td['DiaSemana']==3,"Jueves",np.where(td['DiaSemana']==4,"Viernes",np.where(td['DiaSemana']==5,"Otros","Otros"))))))
    
    #responseMessage['DiaDeLaSemana']=td['DiaSemana']

    #creación de variables

    #td['Nombre_producto_sol']=np.where(td['Nombre_producto_sol'].str.lower().str.contains('pyme'),'PYME','CONN')

    td['NumReferenciasTel']=td['Telefono'].apply(lambda x:x.rstrip().lstrip().count(' ')+x.rstrip().lstrip().count(',')+x.rstrip().lstrip().count(';')+1)



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


    td=td.drop(['Telefono','Fecha_so'],axis=1)
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

    responseMessage['score']= output_table_1['probabilidades'] 
    responseMessage['mensajeScore']= output_table_1['Predict']
    
    
    
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
    cursor.execute("INSERT INTO PruebaRiezgo_3(dFechaConsulta,cNumeroSolicitud,nResultadoScore,cResultadoMensaje,cOficina,cAsesor,nEdad,cSexo,cEstadoCivil,cTipoVivienda,cNivelDeInstruccion,nMontoSolicitado,nPlazo,nNroDependientes,nAniosResidencia,cTelefono,cProfesion,cDiaSemana,cDepartamento,cDestinoCredito) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    output_table_1['Fecha'][0],int(output_table_1['num_solicitud'][0]),output_table_1['probabilidades'][0],responseMessage['mensajeScore'][0],responseMessage['Oficina'][0],responseMessage['Asesor'][0],responseMessage['Edad'][0],responseMessage['Sexo'][0],
                    responseMessage['EstadoCivil'][0],responseMessage['TipoVivienda'][0],responseMessage['NivelInstruccion'][0],responseMessage['Monto_sol'][0],responseMessage['Plazo'][0],responseMessage['NroDependientes'][0],responseMessage['AniosResidencia'][0],
                    responseMessage['Telefono'][0],responseMessage['Profesion'][0],responseMessage['DiaDeLaSemana'][0],responseMessage['Departamento'][0],responseMessage['DestinoCredito'][0])
    cnxn.commit()
    return responseMessage


# Testing Route
@app.route('/prueba')
def index():
    
    return "<h1>Rogger - Index</h1>"
# Testing Route
@app.route('/ObtenerExcepcion', methods=['POST'])
def getErrors():
    errores = {

    }
    # try:
    #     num_solicitud = request.json['num_solicitud']
    # except:
    #     errores = {'codigo':1, 'mensaje':'Numero de solicitud es olbigatorio'}
    # try:
    #     Fechap = request.json['Fechap']
    # except:
    #     errores = {'codigo':2, 'mensaje':'el campo de fecha es olbigatorio'}

    num_solicitud = request.json['num_solicitud']
    Fechap = request.json['Fechap']
    Monto_sol = request.json['Monto_sol']
    plazo_sol = request.json['plazo_sol']
    MotivoRechazo = request.json['MotivoRechazo']

    obtenerExcepcion = pd.DataFrame(
    {
        "num_solicitud" : [num_solicitud],
        "Fechap" : [Fechap],
        "Monto_sol" : [Monto_sol],
        "plazo_sol" : [plazo_sol],
        "MotivoRechazo" : [MotivoRechazo]
    })
    obtenerExcepcion['Fechap'] = pd.to_datetime(obtenerExcepcion['Fechap'], dayfirst=True)
    respuesta = getExcepcion(obtenerExcepcion)
    return ({'Mensaje':respuesta})


def getExcepcion(getExcepcion):
    cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=10.4.12.15;DATABASE=GNormativas;UID=uNormativas;PWD=123456789')
    cursor = cnxn.cursor()
    query = "SELECT TOP 1 * FROM dbo.PruebaRiezgo_3 WHERE"+ " cNumeroSolicitud =" + str(getExcepcion['num_solicitud'][0]) + " AND dFechaConsulta ="+"'"+str(getExcepcion['Fechap'][0])+"'" +" ORDER BY idRegistroConsultaModeloNoBancarizado DESC"
    cursor.execute(query)
    row = cursor.fetchone()
    aceptar = false
    if(row!=null):
        while row:
            num_solicitud = row[2],
            Monto_sol = row[13],
            plazo_sol = row[14]
            row = cursor.fetchone()
            aceptar=true
    else:
        return 'Error 10: No hay datos que actualizar'
    
    if(aceptar != false):
        queryUpdate = "UPDATE dbo.PruebaRiezgo_3 SET cMotivoRechazo ="+ "'"+getExcepcion['MotivoRechazo'][0]+"'"+ "WHERE cNumeroSolicitud =" + str(getExcepcion['num_solicitud'][0]) + " AND nPlazo ="+ str(plazo_sol)+ "AND nMontoSolicitado ="+ str(getExcepcion['Monto_sol'][0])+ "AND dFechaConsulta ="+ "'"+str(getExcepcion['Fechap'][0])+"'"
        cursor.execute(queryUpdate)
        cnxn.commit()
        return 'Actualizado correctamente'
    else:
        return 'Error 10: No hay datos que actualizar'

 
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
    # Telefono = request.args.get('Telefono','',type=str)
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
    Telefono = request.json['Telefono']
    NroDependientes = request.json['NroDependientes']
    AniosResidencia = request.json['AniosResidencia']

    Oficina = request.json['Oficina']
    Asesor = request.json['Asesor']
    DestinoCredito = request.json['DestinoCredito']
    
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
    var_14=[Telefono]
    var_15=[NroDependientes]
    var_16=[AniosResidencia]

    var_17=[Oficina]
    var_18=[Asesor]
    var_19=[DestinoCredito]
    
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
    df.append(var_17)
    df.append(var_18)
    df.append(var_19)
    
    df = pd.DataFrame (df).transpose()
    df.columns=['num_solicitud','Fechap','idEstado','Monto_sol','plazo_sol',
                'FechaNacimiento','Sexo','EstadoCivil','TipoVivienda','NivelInstruccion',
                'Actividad','Profesion','Departamento','Telefono','NroDependientes','AniosResidencia','Oficina','Asesor','DestinoCredito']
    
    response = modeloriesgos(df,"D:/Modelo Grupo Solidario")
    
    return ({'Fechap': response['Fechap'][0],
              'num_solictud': response['num_solicitud'][0],
              'score': response['score'][0],
              'mensajeScore': response['mensajeScore'][0],
              'Oficina': response['Oficina'][0],
              'Asesor': response['Asesor'][0],
              'Edad':response['Edad'][0],
              'Sexo': response['Sexo'][0],
              'EstadoCivil': response['EstadoCivil'][0],
              'TipoVivienda': response['TipoVivienda'][0],
              'NivelInstruccion': response['NivelInstruccion'][0],
              'Monto_sol': response['Monto_sol'][0],
              'Plazo_sol': response['Plazo'][0],
              'NroDependientes': response['NroDependientes'][0],
              'AniosResidencia': response['AniosResidencia'][0],
              'Telefono': response['Telefono'][0],
              'Profesion': response['Profesion'][0],
              'Departamento': response['Departamento'][0],
              'DiaDeLaSemana': response['DiaDeLaSemana'][0],
              'DestinoCredito': response['DestinoCredito'][0],

            })

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=4000)
    



# %%
