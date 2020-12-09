import pandas as pd
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def neuralNet(listVar,nomArch,D_in,H,epochs,muestra=False):
    #df3 = pd.read_csv('data/softdefect.csv')#, index_col=[0]) #idex col para omitir la primer columna con indices 
    #df3.drop(['id'],axis=1,inplace=True)
    try: 
        df3 = pd.read_csv('data/'+ str(nomArch) +'.csv', index_col=0)  
    except:
        print('Nombre incorrecto, se usa el Data frame por defecto')
        df3= pd.read_csv('data/softdefect.csv', index_col=0)
    
    columDF = df3.columns
    
    for i in range(len(columDF)-1):
        if listVar[i] == 0:
            df3 = df3.drop([columDF[i]], axis=1)
    
    #Escoge las X del modelo
    df4 = df3.iloc[:, 0:-1]
    X, Y = df4, df3["defects"]

    #se separa el 20% para prueba
    ochenta = math.ceil(len(df4) * (1-0.2))
    X_train, X_test, y_train, y_test = X[:ochenta], X[ochenta:], Y[:ochenta], Y[ochenta:]
    X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
    
    X_t = torch.from_numpy(X_train).float()
    Y_t = torch.from_numpy(y_train).float()

    #D_in, H, D_out = 10, 8, 1
    D_out = 1
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid()
    )

    def evaluate(x):
        model.eval()
        y_pred = model(x)
        y_probas = softmax(y_pred)
        return torch.argmax(y_probas, axis=1)
    
    
    criterion = torch.nn.MSELoss()#CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.8)

    #epochs = 100
    log_each = 10
    l = []
    model.train()
    iteraVec=[]
    vecerror =[]

    for e in range(1, epochs+1): 
        iteraVec.append(e)
        # forward
        y_pred = model(X_t)

        # loss
        loss = criterion(y_pred, Y_t)
        l.append(loss.item())
        
        vecerror.append(np.mean(l))

        # ponemos a cero los gradientes
        optimizer.zero_grad()

        # Backprop (calculamos todos los gradientes automáticamente)
        loss.backward()

        # update de los pesos
        optimizer.step()
        """
        if (not e % log_each) and (not muestra):
            print(f"Epoch {e}/{epochs} Loss {np.mean(l):.5f}")
        """

    y_pred = evaluate(torch.from_numpy(X_test).float()) #.cuda())
    accuracy_score(y_test, y_pred.cpu().numpy())


    
    
    fig = retFig(iteraVec,vecerror,muestra)
    return vecerror[-1], df4.columns,fig

def retFig(iteraVec,vecerror,muestra=False):
    fig = plt.figure()
    plt.title('MSE red neuronal')
    a = plt.plot(iteraVec,vecerror)
    #plt.plot(iteraVec,vecerror)
    if muestra:
        plt.show()
    return fig

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1,keepdims=True)

#listVar= [0,1,1,1,0,0,0,1,0,1,1,1,1,0,1,0,0,0,0,0,1] para puebas locales rápiadas

