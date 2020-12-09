import numpy as np
import math
import random
sumFil = [0]*3

indice=0
salto=1
x= 0
pos=0

while indice < 9:
    
    
    for i in range(indice,3*salto):#x es el numero de genes 
        print('algo', i)
    
    indice= 3*salto
    
    salto +=1

    print('indice',indice)
    print('salto',salto)
    """
    for indice in range(salto,9, 3):#x es el numero de genes 
        print('algo', indice)




    print(np.random.randint(0, 2, 9))
    salto +=1
    indice+=1
    print('indice',indice)
    print('salto',salto)
    """
""" 
print(sumFil)
print(np.random.randint(1,3))
"""
sumFil = [0]*3
matri0 = [sumFil]*3
print(matri0)


base = 2
exponent = np.log(3) / np.log(base)  # = [3, 4]

#print(exponent)





varDec = 2 #numero de variables de decision - Elementos diferentes: x
#x= (log 2 (1+ Xmax-Xmin )*10**n ))  #tamaño del individuo
valorQVarDecesion = [-5,5]
numDecimPrecisi = 1

dentro = 1 + (max(valorQVarDecesion) - min(valorQVarDecesion))* (10**numDecimPrecisi)
base = 2
genPorVarible = math.ceil(np.log(dentro) / np.log(base)) # = [3, 4]

print(genPorVarible,"gen")
x= varDec * genPorVarible

print('\n',x, 'Lind\n\n')


def concatene(indiv, genPorVarible):
  listConcat = []
  concat = ''
  for i in range(0,len(indiv),genPorVarible):
    concat = ''
    for j in range(genPorVarible):
      concat +=str(indiv[i+j])
    listConcat.append(concat)
    
  return listConcat #esta de esta forma # [1,0,1,0,0,0,1] -> [01,00,11,01] 

print(concatene([0,0,1,1,1,1,0,1],2),'concatene')



#######
valorQVarDecesion = [0,1]
print('decimal',min(valorQVarDecesion) + 116 * (max(valorQVarDecesion) - min(valorQVarDecesion))/(2**7 - 1))

a = [2,4,5,1,1,1,0,1]
print(a[0:3])


def prime(x):
  return x > 0
listaX = np.zeros(shape=(82)) 

print(prime(3),'\n')
a=random.uniform (-10, 0)
print(listaX,'ram')



def probaDeVar(entra): #determina la probabilidad para entregarle a la funcion evaluacion de cada variable 
  if entra:
    proba= random.uniform(0,10)
  else:
    proba= random.uniform(-10, 0)
  
  return 1/(1+(math.e)**(-(proba)))

print(probaDeVar(False),'proba')


entra =[]
ran = np.random.randint(0, 2, (20))
for f in range(20):
  
  if ran[f] == 1:
    entra.append(True)
  else:
    entra.append(False)

print(entra)


listaProb = np.zeros(shape=(20)) 
for q in range(len(listaProb)): 
  listaProb[q] = probaDeVar(entra[q])
print(listaProb)  