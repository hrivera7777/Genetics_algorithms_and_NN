## Algoritmo genetico proyecto final Sistemas inteligentes 
## John Eder Posada Zapata
## Jorge Luis Ordoñez Ospina
## Hans Rivera Londoño

import random
import numpy as np
import math 
import random


##### FUNCIONES PARA OPERADORES

def evalua(n,x,poblIt,varDec,listaProb): #agregar utilidad si es necesario
    # vec[i] = xi * 0,3
    #  x2 -> sig(1/1+e-algo) -> algo mean percentil  -> min y max == 1 
    # algo esta entre -10 y 10 (-10 da el porcentaje más bajo)

    # x1 x5 x8 -> 
    
    #total+= 1/suma # se hace la inversa para minimizar para que el valor mas pequeño sea el importante o el numero mas grande
    # print(fitness,'fitness en eva', total ,'total en eva')
  fitness= np.empty((n))
  suma=0
  total=0
  # -> 1 * 0,32 + 0* 0,2 + .......... 
  for p in range(len(poblIt)):  
    for j in range(len(listaProb)):
      suma += (poblIt[p][j] * listaProb[j]) 
    fitness[p]=suma
    total+=suma
    suma=0
  return fitness,total

def probaDeVar(entra): #determina la probabilidad para entregarle a la funcion evaluacion de cada variable 
  if entra:
    proba= random.uniform(0,10)
  else:
    proba= random.uniform(-10, 0)
  
  return 1/(1+(math.e)**(-(proba)))


def porcentajesIndiv(n,total,fitness): #entrega las probabilidades y el acumulado
  acumula=0
  acumulado= np.empty((n))
  for i in range(0, n):
    probab=fitness[i]/total # para problemas de max
    #probab=(1/fitness[i])/total # MIN ---> QUE TAN APTO ES EL INDIVIDUO se hace la inversa para minimizar para que el valor mas pequeño sea el importante o el numero mas grande
    acumula+=probab
    acumulado[i]=acumula
  return acumulado


def seleccion(acumulado,n,poblIt):
  escoje=np.random.rand()
  #print("escoje:      ", escoje)
  for i in range(0,n):
    if acumulado[i]>escoje:
      #print('indiv ', i+1)
      return poblIt[i] # retorna el padre escogido      
  return poblIt[0]  

def cruce(p1,p2,Pcruce,x): # se modifica el punto de corte para hacerlo aleatorio.
  a1=np.random.rand()

  if a1<= Pcruce:
    #print("Mas grande", Pcruce, "que ", a1, "-> Si Cruzan")
    proCorte = np.random.rand()#probabilidad donde se hace el corte
    puntoCorte = 1/(x-1) # nos entrega el porcentaje de corte entre gen
    # cada 1/(x-1) hay punto de corte 
    i=1
    pos=0
    #print('\n donde se debe hacer el corte ',proCorte)
    for i in range(x):
      if i*puntoCorte > proCorte:
        pos=i
        break    
    #print('\n el corte es en la posicion', pos)
    temp1=p1[0:pos] #[i:j] corta desde [i a j)
    temp2=p1[pos:x]
    #print(temp1,temp2)
    temp3=p2[0:pos]
    temp4=p2[pos:x]
    #print(temp3,temp4)
    hijo1 = list(temp1)
    hijo1.extend(list(temp4))
    hijo2 = list(temp3)
    hijo2.extend(list(temp2))

  else:
    #print("Menor", Pcruce, "que ", a1, "-> NO Cruzan")
    hijo1=p1
    hijo2=p2
  
  return hijo1,hijo2

def mutacion(h1,h2,Pmuta,x): # h1 = al hijo que entra para verificar si muta uno de los genes
  for i in range(x): 
    a1=np.random.rand()
    if a1<= Pmuta:
      #print("Mas grande", Pmuta, "que ", a1, "-> Si muta el gen", i+1, 'de ', h)
      if h1[i] == 0:
         h1[i] = 1
      else:
        h1[i] = 0
    a2=np.random.rand()
    if a2<= Pmuta:
      #print("Mas grande", Pmuta, "que ", a1, "-> Si muta el gen", i+1, 'de ', h)
      if h2[i] == 0:
         h2[i] = 1
      else:
        h2[i] = 0
  
  return h1,h2

#si se requiere que un individuo cumpla con condiciones especiales
def factible(h,cantVarNN):
  sumfac = 0 
  mult = 0
  for i in range(len(h)):#x es el numero de genes 
    sumfac += h[i] 
  return sumfac == cantVarNN


#encuentra el mejor individuo 
def mejorFit(fitness):
  mejor= fitness[0]

  for i in fitness:
    if i > mejor:
      mejor=i
  return mejor

#encuetra el promedio (media aritmetica) de la funcion fitness de todos los individuos 
def promFit(fitness):
  prom=0
  for i in fitness:
    prom += i
  
  return prom/len(fitness)

def imprimeTabla(numIter,mejorFit,promedioFit,totalFit): #mientras 
    #Tabla de evaluación de la Población
    acumula=0
    print ("\n",'Tabla:',"\n")
    print('iteración'," ",'mejor Fitness',"  ","promedio Fitness","  ", 'Fitness total')
    for i in range(0, len(mejorFit)):
      print('  ',[i+1],"        ","{0:.3f}".format(mejorFit[i]),"            ","{0:.3f}".format(promedioFit[i]),"           ","{0:.3f}".format(totalFit[i]))


def listaCompletaDecim(listCont):
  listaDecem=[]
  for i in listCont:
    listaDecem.append(bin2dec(i))

  return listaDecem

def bin2dec(b):
  number = 0
  for idx, num in enumerate(b[::-1]): # Iterating through b in reverse order
    number += int(num)*(2**idx)
  return number

def concatene(indiv, genPorVarible):
  listConcat = []
  concat = ''
  for i in range(0,len(indiv),genPorVarible):
    concat = ''
    for j in range(genPorVarible):
      concat +=str(indiv[i+j])
    listConcat.append(concat)
  return listConcat #esta de esta forma # [1,0,1,0,0,0,1] -> [01,00,11,01] 

def xi (valorQVarDecesion,listaDecim,x):
  listaXi = []
  
  for i in range(len(listaDecim)):
    listaXi.append(min(valorQVarDecesion) + listaDecim[i] * (max(valorQVarDecesion) - min(valorQVarDecesion))/(2**x - 1))
  #asi seria cada xi ---> xi = min(valorQVarDecesion) + decimal * (max(valorQVarDecesion) - min(valorQVarDecesion))/(2**x - 1)
  #print(listaXi, 'lista xi') 
  return listaXi

#consigue el mejor individuo para el elitismo de ser necesario##########
def mejorPosicFit(fitness):
  mayor= fitness[0]
  pos = 0
  for i in range(len(fitness)):
    if fitness[i] > mayor:
      pos = i
      mayor = fitness[i]
  return pos

def indivElite(pos,poblIt): #posicion del fitness y poblacion de la iteracion
  return poblIt[pos]
###############################


#################################################################
#funcion principal 

'''
cantVarNN = cantidad de variables que desea en la red neuronal (1/2 de la cantidad total de columnas del df)
Pmuta = probabilidad de mutacion del algoritmo genético (recomendado 0,1)
Pcruce = probabilidad de cruce del algoritmo genético (recomendado 0,95)
numIter = numero de iteraciones que desea correr el algoritmo 
'''
def aGenetico(entra,cantVarNN,Pmuta,Pcruce,numIter): # cantVarNN


  #### Parametros #####
  #entra =[]
  # se llama la funcion para la preselección de las variables


  #cantVarNN = 10 # cantidad de variables pensandas para entregarle a la NN
  
  varDec = len(entra) #numero de variables de decision - Elementos diferentes (columnas del dataFrame)

  valorQVarDecesion = [0,1] # valor que puede tomar las variables de decision
  numDecimPrecisi = 0

  dentro = 1 + (max(valorQVarDecesion) - min(valorQVarDecesion))* (10**numDecimPrecisi)
  base = 2
  genPorVarible = math.ceil(np.log(dentro) / np.log(base)) # = [3, 4]

  x= varDec * genPorVarible

  n=np.random.randint(1,3)*x  #numero de individuos en la poblacion - cromosomas: n
  #Pcruce=0.90  #Probabilidad de Cruce
  #Pmuta=0.1   #Probabilidad de Mutación
  #pesoMax = 60 # peso maximo que puede cargar la mochila

  fitness= np.empty((n))
  acumulado= np.empty((n))
  suma=0
  total=0

  """
  ran = np.random.randint(0, 2, (varDec))
  for f in range(varDec):
    
    if ran[f] == 1:
      entra.append(True)
    else:
      entra.append(False)
  """


  listaProb = np.zeros(shape=(varDec)) 
  for q in range(len(listaProb)):
    listaProb[q] = probaDeVar(entra[q])

  #Individuos, soluciones o cromosomas 
  poblInicial = np.random.randint(0, 2, (n, x)) # aleatorios (n por x) enteros entre [0 y2)


  for i in range(n):
    while not factible(poblInicial[i],cantVarNN):
      poblInicial[i] = np.random.randint(0, 2, x) # aleatorios (n por x) enteros entre [0 y2)


  """ muestra la poblacion inicial
  print("Poblacion inicial Aleatoria:","\n", poblInicial)
  print("\n","Utilidad:", utilidad) 
  print("\n","Pesos", pesos )  
  """

  poblIt=poblInicial

  fitnessInic,totalInc=evalua(n,x,poblIt,varDec,listaProb) # se agrega utilidad de ser necesario poblIt en estos momentos = a poblInicial
  acumulado = porcentajesIndiv(n,totalInc,fitnessInic)

  ######  FIN DE LOS DATOS INICIALES


  ##### ***************************************
  ##### ***************************************

  # Inicia Iteraciones

  # Crear matriz de 5x2 vacio  a = numpy.zeros(shape=(5,2))
  a = np.zeros(shape=(n,x))
  poblintermedia = a.astype(int) # convierte a enteros todos los 0  de la matriz

  pos=0 #inidice o posicion del vector que esta llenando con los hijos aptos.
  '''
  try:
    numIter = int(input('ingrese el número de iteraciones que desea: '))
  except:
    print('por favor ingrese un valor entero ejm 2 , 10')
  '''
  mejorInv=[]
  promedioFit=[]
  totalFit=[] 


  #realiza las iteraciones de cada generación (todos los individuos son factibles)
  for iter in range(numIter):
    #print("\n","Iteración ", iter+1)
    pos=0
    a = np.zeros(shape=(n,x))
    poblintermedia = a.astype(int) # convierte a enteros todos los 0  de la matriz
    hijoA=[]
    hijoB=[]
    
    hijoElite = []
    posElit = 0

    if iter == 0: # se usa para elitismo
      posElit = mejorPosicFit(fitnessInic)
      hijoElite= indivElite(posElit,poblIt)
    else:
      posElit = mejorPosicFit(fitness)
      hijoElite= indivElite(posElit,poblIt)
    
    while pos < n: # para generar los hijos en todas las posiciones del vector 
      
      papa1=seleccion(acumulado,n,poblIt) # Padre 1 con la posicion que ocupa 
      #print("padre 1:", papa1)
      papa2=seleccion(acumulado,n,poblIt) # Padre 2 con la posicion que ocupa 
      #print("padre 2:", papa2)
      
      hijoA,hijoB=cruce(papa1,papa2,Pcruce,x)
    
      interme1,interme2= mutacion(hijoA,hijoB,Pmuta,x)
      
      
      if pos == 0:# se usa para elitismo
        poblintermedia[pos]=hijoElite # individuo elite
        #print(hijoElite,'individuo elite',fitness[posElit])
        pos+=1
      
      elif factible(interme1,cantVarNN) and factible(interme2,cantVarNN) and pos < n-1: #seria elif cuando es elitsmo

        #print(interme1,'esto se agrega a IF inter1')
        poblintermedia[pos]=interme1
        
        pos+=1
        #print(interme2,'esto se agrega a IF inter2')
        poblintermedia[pos]=interme2
        pos+=1  # se aumenta en 1 porque se agrego un hijo a la matriz
        #print("hijo2: IF", hijoB,'pos', pos)

      elif factible(interme1,cantVarNN) and pos<n:
        #print(interme1,'esto se agrega a elIF inter1')
        poblintermedia[pos]=interme1
        hijoB=[]
        pos+=1

      elif factible(interme2,cantVarNN) and pos<n:
        #print(interme2,'esto se agrega a elIF inter')
        poblintermedia[pos]=interme2
        hijoA=[]
        pos+=1

      else:
        hijoA=[]
        hijoB=[]
        interme2 = []
        interme2 = []
        #print('ningun hijo es factible se continua...')
      #print('asi va poblIt', poblintermedia, 'en pos ', pos,"\n")

    
    poblIt=poblintermedia
      
    #print("\n","Poblacion Iteración ", iter+1,"\n", poblIt)
    fitness,total=evalua(n,x,poblIt,varDec,listaProb) # se agrega utilidad de ser necesario
    acumulado = porcentajesIndiv(n,total,fitness)
    mejorInv.append(mejorFit(fitness))
    promedioFit.append(promFit(fitness))
    totalFit.append(total)
  
  posEleg=0
  indEleg = []
  posEleg = mejorPosicFit(fitness)
  indEleg= indivElite(posEleg,poblIt)
  #print(indEleg,'elegido', fitness[posEleg]) 
  return indEleg 
  #imprimeTabla(numIter,mejorInv,promedioFit,totalFit)



 
""" se puede eliminar 
def imprime(n,total,fitness,poblIt):
    #Tabla de evaluación de la Población
    acumula=0
    print ("\n",'Tabla Iteración:',"\n")
    for i in range(0, n):
      probab=fitness[i]/total
      acumula+=probab
      print([i+1]," ",poblIt[i],"  ",fitness[i]," ","{0:.3f}".format(probab)," ","{0:.3f}".format(acumula))
      acumulado[i]=acumula
    print("Total Fitness:      ", total)
    return acumulado
"""
