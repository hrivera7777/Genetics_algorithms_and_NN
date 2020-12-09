import pandas as pd

"""
  entra = []
  col = df.columns
  for f in col:
    if  df[f].quantile(0.25) == df[f].quantile(0.75):
      entra.append(False)
    else:
      entra.append(True)
  """

def prese(nomArch,ColY):#nombre del archivo y nombre de la columna de los Y real
  try: 
    df3 = pd.read_csv('data/'+ str(nomArch) +'.csv', index_col=0)
    df3 = df3.drop([ColY],axis=1)
    
  except:
    print('Nombre incorrecto, se usa el Data frame por defecto')
    df3= pd.read_csv('data/softdefect.csv', index_col=0)
    df3 = df3.drop(['defects'],axis=1)
  entra = []
  col = df3.columns
  for f in col:
    mx = df3[f].max()
    mn= df3[f].min()
    rango= mx-mn 
    mitad = str(df3[f].quantile(0.50))
    conteo=0
    for i in mitad:
      if i.isdigit():
        conteo+=1
      else:
        break
    if df3[f].quantile(0.75) -  df3[f].quantile(0.25) > rango * (1/(10**conteo)) :
      entra.append(False)
    else:
      entra.append(True)
    mx = 0
    mn = 0
  return entra