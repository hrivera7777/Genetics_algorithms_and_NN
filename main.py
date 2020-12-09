from matplotlib.backends.backend_pdf import PdfPages
import preselec 
import AgProyecto
import nn

'''
nomArch = nombre del archivo (csv) que se desea analizar 
ColY = nombre de la columna de los valores de Y real
cantidadCol = cantidad de variables que desea en la red neuronal y que el algoritmo genetico entregará (1/2 de la cantidad total de columnas del df)
Pmuta = probabilidad de mutacion del algoritmo genético (recomendado 0,1)
Pcruce = probabilidad de cruce del algoritmo genético (recomendado 0,95)
iteraGenetico = número de iteraciones que desea correr el algoritmo 
capasOcultasNN = cantidad de neuronas en la capa oculta de la red neuronal 
epochs = cantidad de iteraciones o épocas que desea realizar en la red neuronal
cantidadPruebas = cantidad de muestras de que entrega este archivo
muestra = cambie el valor a False para que no se muestren los plt en cada época
'''

nomArch= 'softdefect'
ColY = 'defects'

cantidadCol=10
Pmuta =0.1
Pcruce = 0.95
iteraGenetico = 30

capasOcultasNN=8
epochs = 100 
muestra = False
cantidadPruebas = 5

entra =[]



#métodos
def pruebas(cantidadPruebas):
    mPerdidas = []
    mColumnas = []
    varSelec = []
    mFig=[]
    for i in range(cantidadPruebas):
        perdida, columnas = 0 , []
        varSelec = AgProyecto.aGenetico(entra,cantidadCol,Pmuta,Pcruce,iteraGenetico)
        perdida, columnas,fig = nn.neuralNet(varSelec,nomArch,cantidadCol,capasOcultasNN,epochs,muestra)
        mPerdidas.append(perdida)
        mColumnas.append(columnas)
        mFig.append(fig)
    return mPerdidas,mColumnas,mFig

def mejor(mPerdidas,mColumnas):
    mejor = mPerdidas[0]
    pos = 0
    for i in range(len(mPerdidas)):
        if mPerdidas[i] <= mejor :
            mejor = mPerdidas[i]
            pos = i
    print('\n el mejor error es',"{0:.3f}".format(mejor) ,' con las columnas',mColumnas[pos])

def table(mPerdidas,mColumnas):
    print ("\n",'Tabla:',"\n")
    print('Prueba'," ",'Error',"  ","                     ", 'Columnas')
    for j in range(len(mPerdidas)):
        print('  ',[j+1],"        ","{0:.3f}".format(mPerdidas[j]),"            ",mColumnas[j])
        print ("\n")

def saveFig(mfig):
    pp = PdfPages('graf.pdf')
    for i in range(len(mfig)):
        pp.savefig(mfig[i])
    pp.close()    

#datos que se pasan entre algoritmos
entra = preselec.prese(nomArch,ColY)
mperd, mcol,mfig = pruebas(cantidadPruebas)

#muestra los datos y guarda los gráficos
table(mperd,mcol)
mejor(mperd,mcol)
saveFig(mfig)