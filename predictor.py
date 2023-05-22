from matplotlib import pyplot as plt
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import numpy as np
from datetime import datetime
import sklearn
import sklearn.model_selection



time = 3600 #Tiempo entre intervalos en segundos
Clase = 1 # 1->sentada; 2->tumbada; 3->depie
path ='runs/detect/datos_predictor/pruebalog.txt' #ruta del log

class Model(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super().__init__()
    self.layer1 = nn.Linear(input_size, hidden_size)
    self.layer2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x = self.layer1(x)
    x = nn.Sigmoid()(x)
    x = self.layer2(x)
    return x


def get_info(dataframe,Class,time):
    lista_values=[]
    lista_results = []
    lista_m=[]
    lista_d =[]
    lista_H = []
    lista_M = []
    lista_s = []

    lista_values.append(dataframe.iloc[0,Class])
    #sacar medias de cada hora
    i = 1
    fecha1 = datetime.strptime(dataframe.iloc[0,0],'%Y:%m:%d %H:%M:%S').timestamp() #2.3 cojo la primera
    while i < len(dataframe):#2.1recorro la tabla
        fecha2 = datetime.strptime(dataframe.iloc[i,0],'%Y:%m:%d %H:%M:%S').timestamp() #2.4 voy comparando con el resto
        #2.4.1 si la diferencia es menor a time: guardo el valor y paso al siguiente
        if (fecha2-fecha1) < (time):
            lista_values.append(dataframe.iloc[i,Class])
            i+=1
        #2.4.2 si la diferencia en mayor a time: calculo la media de los valores guardados y actualizo el valor 
        else:
            result = 0
            for dato in lista_values:
                result = result+dato
            if len(lista_values)>0:
                result = result/len(lista_values) # calculo la media en el tiempo dado
            else: 
                result=result #si no se guardo ningun valor, se guarda un 0

            lista_values.clear()
            lista_results.append(result) #guardamos el resultado
            fecha1 = fecha2 # actualizamos la fecha de la siguiente iteraccion
            lista_m.append(datetime.utcfromtimestamp(fecha1).month)
            lista_d.append(datetime.utcfromtimestamp(fecha1).day)
            lista_H.append(datetime.utcfromtimestamp(fecha1).hour)
            lista_M.append(datetime.utcfromtimestamp(fecha1).minute)
            lista_s.append(datetime.utcfromtimestamp(fecha1).second)
            i+=1
    if i == len(dataframe): #si no hay mas valores, saco la ultima media
        result = 0
        for dato in lista_values:
            result = result+dato
        if len(lista_values)>0:
                result = result/len(lista_values) # calculo la media en el tiempo dado
        else: 
            result=result #si no se guardo ningun valor, se guarda un 0
        lista_values.clear()
        lista_results.append(result) 
        fecha1 = fecha2
        lista_m.append(datetime.utcfromtimestamp(fecha1).month)
        lista_d.append(datetime.utcfromtimestamp(fecha1).day)
        lista_H.append(datetime.utcfromtimestamp(fecha1).hour)
        lista_M.append(datetime.utcfromtimestamp(fecha1).minute)
        lista_s.append(datetime.utcfromtimestamp(fecha1).second)
        i+=1

    X = np.array(lista_results[0:len(lista_results)-1])
    X2 = np.array(lista_m[1:])
    X3 = np.array(lista_d[1:])
    X4 = np.array(lista_H[1:])
    X5 = np.array(lista_M[1:])
    X6 = np.array(lista_s[1:])
    y =np.array(lista_results[1:])
    return X,X2,X3,X4,X5,X6,y

#recoger los datos(fecha,sentada,tumbada, depie)
df=pandas.read_csv(path,sep=';',names=['Date', 'P_sen', 'P_tumb', 'P_Depie'],header=None)
df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8, random_state=1 )#divido los datos de entrenamiento y test

#ordeno los valores de los datos
df_train = df_train.sort_values(by='Date',kind='quicksort')
#print(df_train)
df_test = df_test.sort_values(by='Date',kind='quicksort')
#print(df_test)

#obtengo las matrices necesarias para mi modelo
X1train, X2train,X3train, X4train,X5train, X6train, y_train = get_info(df_train,Clase,time)
X1test, X2test,X3test, X4test,X5test, X6test, y_test = get_info(df_test,Clase,time)

   
X1train = torch.from_numpy(X1train).float()
X2train = torch.from_numpy(X2train).float()
X3train = torch.from_numpy(X3train).float()
X4train = torch.from_numpy(X4train).float()
X5train = torch.from_numpy(X5train).float()
X6train = torch.from_numpy(X6train).float()
y_train = torch.from_numpy(y_train).float()

tensor_2d = torch.stack((X1train, X2train,X3train, X4train,X5train, X6train), dim=1)

train_ds = TensorDataset(tensor_2d, y_train)

train_dl = DataLoader(train_ds, 4, shuffle=False)

# X1test = torch.from_numpy(X1test).float()
# X2test = torch.from_numpy(X2test).float()
X1test = torch.from_numpy(X1test).float()
X2test = torch.from_numpy(X2test).float()
X3test = torch.from_numpy(X3test).float()
X4test = torch.from_numpy(X4test).float()
X5test = torch.from_numpy(X5test).float()
X6test = torch.from_numpy(X6test).float()
y_test = torch.from_numpy(y_test).float()

tensor_2d_test = torch.stack((X1test, X2test,X3test, X4test,X5test,X6test), dim=1)

input_size = 6
hidden_size = 12
output_size = 1
model = Model(input_size, hidden_size, output_size)

# Definición de la función de pérdida y el optimizador
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00003) 

num_epochs = 200
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs
# Entrenamiento del modelo
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        y_pred = model(x_batch)
        y_batch = y_batch.unsqueeze(1)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_hist[epoch] += loss.item()*y_batch.size(0)
        is_correct = (torch.argmax(y_pred, dim=1) == y_batch).float()

        accuracy_hist[epoch] += is_correct.sum()

    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)

#Resultados del modelo
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(loss_hist, lw=3)

ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(1, 2, 2)
ax.plot(accuracy_hist, lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()

with torch.no_grad():
 pred = model(tensor_2d_test)
 y_test = y_test.unsqueeze(1)
 loss = loss_fn(pred, y_test)
 print(f'Test MSE: {loss.item():.4f}')
 print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')
  