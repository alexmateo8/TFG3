import torch
print (torch.cuda.is_available())
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import numpy as np

from data_loader2 import iCIFAR10
from data_loader2 import iCIFAR100
#from model import iCaRLNet
from iCaRL import iCaRLNet
import math
import save_load
import plots
import metrica
import random

print (torch.cuda.is_available())
path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/saved_models/best_model4'

# Híper paràmetres
total_classes = 100      #Número total de classes de la base de dades
num_classes = 5          #Número de classes per tasca
num_epochs = 60          #Número de epochs per realitzar l'entrenament

# Inicializació dels seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Data augmentation sobre les dades d'entrada. Realitza un retallat, rotació i normalització
transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


# Inicialització de la CNN
k = 500*num_classes  #2000                            #Espai de memòria en número d'exemples
#k = 10000
icarl = iCaRLNet(2048, 1, num_epochs)                 #Creació del model, ResNet amb una sortida de 2048 neurones, capa de sortida separada de tamany 1 inicialment
icarl.cuda()

# Per la contabilització del nombre total de paràmetres
#pytorch_total_params = sum(p.numel() for p in icarl.parameters())
#print (pytorch_total_params)

# Llistes per l'emmagatzematge de les precisions noves, antigues i totals en avaluació i precisió
tr_accs_old = []
tr_accs_new = []
te_accs_old = []
te_accs_new = []
tr_accs_total = []
te_accs_total = []

mem_old = []                                            #Número de mostres per classe en la tasca anterior 


iterable_list = np.arange(0,total_classes, num_classes) #Lista amb les tasques a iterar

# Matrius per l'emmagatzematge de les precisions 
matrix_accuracies = torch.zeros(len(iterable_list), len(iterable_list))
matrix_accuracies_train = torch.zeros(len(iterable_list), len(iterable_list))

#Llista amb les pèrdues totals en entrenament i avaluació
loss_total = np.zeros(shape = (len(iterable_list), num_epochs))
loss_distilation = np.zeros(shape = (len(iterable_list), num_epochs))
loss_classification = np.zeros(shape = (len(iterable_list), num_epochs))
loss_total_eval = np.zeros(shape = (len(iterable_list), num_epochs))
loss_distilation_eval = np.zeros(shape = (len(iterable_list), num_epochs))
loss_classification_eval = np.zeros(shape = (len(iterable_list), num_epochs))

count = 0

for s in range(0, total_classes, num_classes):                                     #Iterador del número de tasques
    print ("number of s: %d" %(s))
    print ("Loading training examples for classes", range(s,s+num_classes))
    
    #Depenent del transform triat, s'aplica o no data augmentation 
    train_set = iCIFAR10(root='./data',
                         train=True,
                         classes=range(s,(s+num_classes)) if s != (total_classes-1) else s,
                         download=True,
                         transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100,
                                               shuffle=True, num_workers=1)
                                               
    
    test_set = iCIFAR10(root='./data',
                         train=False,
                         classes=range(0,(s+num_classes)) if s != (total_classes-1) else s,
                         download=True,
                         transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                               shuffle=True, num_workers=1)  #Normalmente batch size en 100, evitamos problemas con cuda out of memory


    # Entrenament mitjançant back propagation
    loss_total[count], loss_classification[count], loss_distilation[count],loss_total_eval[count], loss_classification_eval[count], loss_distilation_eval[count] = icarl.update_representation(train_set, test_set)
    
   
    ##############Diferents opcions de la distribució de memòria entre classes###################   
    """
    #OPCIÓN 1
    m = math.floor(k / icarl.n_classes)
    res = (2500-m*icarl.n_classes)/5
    tot = int(m+res)
    """
    
    """
    #OPCIÓN 2
    
    m = math.floor(k / icarl.n_classes)
    #mem_list = np.flip(np.arange(0.5, 1.5, 1/icarl.n_classes))
    mem_list = np.arange(0.5, 1.5, 1/icarl.n_classes)
    mem_list = [int(mem_list[i]*m) for i in range(len(mem_list))]
    res = (2500-sum(mem_list))/5
    new_classes = [icarl.n_classes -5, icarl.n_classes -4, icarl.n_classes -3, icarl.n_classes -2, icarl.n_classes -1]
    for i in new_classes:
        mem_list[i] = int(mem_list[i] + res)
    res2 = (2500-sum(mem_list))
    mem_list[icarl.n_classes-1] += res2
    """
    
    #OPCIÓN 3 TAMBIÉN CONTINUA MÁS ABAJO
    if (s>0):
        m = math.floor(k / icarl.n_classes)
        print (matrix_accuracies[:class_group, count-1])
        if (s==5):
            acc_list = np.argsort(matrix_accuracies[:class_group, count-1])
        else:
            #acc_list = np.flip(np.argsort(matrix_accuracies[:class_group, count-1].cpu().detach().numpy()))#A mayor probabilidad mas muestras
            acc_list = np.argsort(matrix_accuracies[:class_group, count-1].cpu().detach().numpy())  #A menor probabilidad mas muestras
        mem_list = np.flip(np.arange(0.5, 1.5, 5/icarl.n_known)) #Solo las clases antiguas
        mem_list_aux=np.zeros(mem_list.shape)
        for i,index in enumerate(acc_list):
            if((mem_old[index]) < (int(mem_list[i]*m))):
                mem_list_aux[index] = mem_old[index]
            else:
                mem_list_aux[index] = int(mem_list[i]*m)
        mem_old = []
        for i in mem_list_aux:
            mem_old.append(i)
        aux = 0
        mem_list2 = np.zeros(icarl.n_classes)
        for i in range(s):
            if (i%5 == 0 and i>0):
                aux+=1
            mem_list2[i] = int(mem_list_aux[aux])
        res = (2500-sum(mem_list2))/5
        new_classes = [icarl.n_classes-5, icarl.n_classes-4, icarl.n_classes-3, icarl.n_classes-2, icarl.n_classes-1]
        for i in new_classes:
            mem_list2[i] = int(mem_list2[i]+res)
        mem_old.append(mem_list2[icarl.n_classes-1])
      
        
    else:
        m = math.floor(k / icarl.n_classes)
        res = (2500-m*icarl.n_classes)/5
        tot = int(m+res)
        mem_list2 = np.zeros(icarl.n_classes)
        new_classes = [icarl.n_classes-5, icarl.n_classes-4, icarl.n_classes-3, icarl.n_classes-2, icarl.n_classes-1]
        for i in new_classes:
            mem_list2[i] = tot
        mem_old.append(tot)
        
         
    
    # Reducció de les dades de memòria per les classes ja conegudes
    #icarl.reduce_exemplar_sets(m)
    icarl.reduce_exemplar_sets(mem_list2)
    
    icarl.compute_mean(transform_test, True)

    # Construct exemplar sets for new classes
    for y in range(s, s+num_classes):
        print ("Constructing exemplar set for class-%d..." %(y))
        images = train_set.get_image_class(y)                             #Conjunt de mostres d'entrenament d'aquella classe
        
        #icarl.construct_exemplar_set(images, tot, transform_test, y)     #Selecció de les més representatives per emmagatzemar-les
        icarl.construct_exemplar_set(images, mem_list2[y], transform_test, y)
        print ("Done")

    for y, P_y in enumerate(icarl.exemplar_sets):
        print ("Exemplar set for class-%d:" % (y), P_y.shape)

    icarl.n_known = icarl.n_classes
    print ("iCaRL classes: %d" % icarl.n_known)
    
    icarl = save_load.load_model(icarl, path) 
    
    """
    #####MEDICIÓ DEL NÚMERO DE PARÀMETReS########
    pytorch_total_params = sum(p.numel() for p in icarl.parameters())
    print("##################")
    print("Total parameters in the model")
    print (pytorch_total_params)
    print("##################")
    """
    #Mètrica
    print ("Computing metrica....")  
    
    seen_classes = s+num_classes
    class_group = 0
    for i in range(0, seen_classes, num_classes):
        test_set = iCIFAR10(root='./data',
                            train=False,
                            classes=range(i,(i+num_classes)) if i != (total_classes-1) else i,
                            download=True,
                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)
        matrix_accuracies[class_group,count] = metrica.test_accuracy(test_loader, icarl, transform_test)
        class_group += 1
    
    class_group = 0
    for i in range(0, seen_classes, num_classes):
        train_set = iCIFAR10(root='./data',
                         train=True,
                         classes=range(i,(i+num_classes)) if i != (total_classes-1) else i,
                         download=True,
                         transform=transform_test)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100,
                                                     shuffle=False, num_workers=1)
        matrix_accuracies_train[class_group,count] = metrica.train_accuracy(train_loader, icarl, transform_test)
        class_group += 1
    
        
    acc_list = matrix_accuracies[:class_group, count].cpu().detach().numpy()
      
     
    print ("\n ##########Train metrics###########")
    tr_accs_new.append(matrix_accuracies_train[count, count])
    if (count > 0):
        tr_accs_old.append((sum(matrix_accuracies_train[:count,count]))/count)
        print('Old classes Accuracy: %f %%' % tr_accs_old[count-1])
    tr_accs_total.append(sum(matrix_accuracies_train[:,count])/(count + 1))
    
    
    print('New classes Accuracy: %f %%' % tr_accs_new[count])
    print('Train total Accuracy: %f %%' % tr_accs_total[count])
    
    
    print ("\n ###########Test metrics###########")
    te_accs_new.append(matrix_accuracies[count, count])
    if (count > 0):
        te_accs_old.append((sum(matrix_accuracies[:count,count]))/count)
        print('Old classes Accuracy: %f %%' % te_accs_old[count-1])
    te_accs_total.append(sum(matrix_accuracies[:,count])/(count + 1))
    
    print('New classes Accuracy: %f %%' % te_accs_new[count])
    print('Test total Accuracy: %f %% \n' % te_accs_total[count])
    
    count += 1  
    

print ("\n ##########Accuracies matrix train##########")
print (matrix_accuracies_train)

print ("\n ##########Accuracies matrix test##############")
print (matrix_accuracies)


if (type(train_set) is iCIFAR10):
    fname = 'iCIFAR10pretrained' + str(num_classes) + str(0)
    fname_loss = 'iCIFAR10pretrained_loss' + str(num_classes) + str(0)

aux = True
if (len(te_accs_old) == 0):
    aux = False

plots.save_graphic_evaluation (iterable_list, te_accs_new, te_accs_old, te_accs_total, fname, aux, True)
plots.save_graphic_evaluation (iterable_list, tr_accs_new, tr_accs_old, tr_accs_total, fname, aux, False)
plots.save_loss (iterable_list, num_epochs, loss_total, loss_classification, loss_distilation, fname_loss, True)
plots.save_loss (iterable_list, num_epochs, loss_total_eval, loss_classification_eval, loss_distilation_eval, fname_loss, False)
plots.save_matrix(matrix_accuracies, fname)
