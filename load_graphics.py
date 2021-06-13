import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import numpy as np
import os
import errno
import random


try:
    os.mkdir('Results/Comparation')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
        
total_classes = 100
num_classes = 5
iterable_list = np.arange(0,total_classes, num_classes) 

path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/Results/Text/'     #Directori on guardar la imatge

# Fitxers a combinar
fname1 = 'iCIFAR100pretrained6transfcomoentrada2'       
fname2 = 'iCIFAR100pretrained6sinrepeticion'   
#fname3 = 'iCIFAR100pretrained7transfexp3'
#fname4 = 'iCIFAR100pretrained6transfcomoentrada2'      
#fname5 = 'iCIFAR100pretrained7transfcomoentrada1'
#fname6 = 'iCIFAR100pretrained6besttodos'
#fname7 = 'iCIFAR100pretrained8wider1000'

name1 = path + fname1 + '.txt'
name2 = path + fname2 + '.txt'
#name3 = path + fname3 + '.txt'
#name4 = path + fname4 + '.txt'
#name5 = path + fname5 + '.txt'
#name6 = path + fname6 + '.txt'
#name7 = path + fname7 + '.txt'

# Creació de la matriu de dades a representar
data1 = np.empty((20,20))
data2 = np.empty((20,20))
data3 = np.empty((20,20))
data4 = np.empty((20,20))
data5 = np.empty((20,20))
data6 = np.empty((20,20))
#data7 = np.empty((20,20))

# Lectura del fitxer txt
with open(name1,'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data1[i] = line.split('  ')
        
with open(name2,'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data2[i] = line.split('  ')

"""
with open(name3,'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data3[i] = line.split('  ')

with open(name4,'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data4[i] = line.split('  ')


with open(name5,'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data5[i] = line.split('  ')

with open(name6,'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data6[i] = line.split('  ')

with open(name7,'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        data7[i] = line.split('  ')
"""

# Canvi del tipus de int a float i limitació de la grandaria de l'eix y
path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/Results/Comparation/'
plt.figure(13)
plt.xlabel('Class number')
plt.ylabel('Eval Accuracy [%]')
plt.ylim(0, 100)
data1 = data1.astype(np.float)
data2 = data2.astype(np.float)
#data3 = data3.astype(np.float)
#data4 = data4.astype(np.float)
#data5 = data5.astype(np.float)
#data6 = data6.astype(np.float)
#data7 = data7.astype(np.float)

# Promig de precisions de les tasques conegudes en cada iteració 
lista_dividir = np.arange(1,21)
data1_sum = np.sum(data1, axis = 0)/lista_dividir
data2_sum = np.sum (data2, axis = 0)/lista_dividir
#data3_sum = np.sum (data3, axis = 0)/lista_dividir
#data4_sum = np.sum (data4, axis = 0)/lista_dividir
#data5_sum = np.sum (data5, axis = 0)/lista_dividir
#data6_sum = np.sum (data6, axis = 0)/lista_dividir
#data7_sum = np.sum (data7, axis = 0)/lista_dividir


global_name = fname1 + fname2 


plt.plot(iterable_list, data1_sum, label='Amb repetició')
plt.legend()
plt.plot(iterable_list, data2_sum, label='Sense repetició')
plt.legend()
#plt.plot(iterable_list, data3_sum, label='Exp3')
#plt.legend()
#plt.plot(iterable_list, data4_sum, label='Combinació de tots separadament')
#plt.legend()
#plt.plot(iterable_list, data5_sum, label='Combinació de tots conjuntament')
#plt.legend()
#plt.plot(iterable_list, data6_sum, label='Resultat anterior')
#plt.legend()
#plt.plot(iterable_list, data7_sum, label='1000')
#plt.legend()
        
        
plt.savefig(path + global_name + '.png')
            
        
