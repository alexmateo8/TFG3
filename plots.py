import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import numpy as np
import os
import errno
import random

def save_graphic_evaluation(iterable_list, te_accs_new, te_accs_old, te_accs_total, fname, aux, eval_boolean):
    """Permet guardar les gràfiques de precisió d'entrenament i d'evaluació
    Args:
        iterable_list: llista de tasques que han estat entrenades
        te_accs_new: llista amb les precisions de les classes noves
        te_accs_old: llista amb les precisions de les classes antigues
        te_accs_total: llista amb les precisions totals de cada tasca
        fname: nom del fitxer amb el que es guardarà
        aux: si és la primera tasca, permetrà que no s'imprimeixi cap valor de precisions antigues
        eval_boolean: permet saber si es tracten de dades d'avaluació o d'entrenament
    Resultat:
        Emmagatzemament de les gràfiques de precisió, tant d'entrenament com d'avaluació
    """    
    
    if (eval_boolean):
        try:
            os.mkdir('Results/Eval')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
        path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/Results/Eval/'
        plt.figure(1)
        plt.ylim(0, 100)
        plt.plot(iterable_list, te_accs_new, label='eval_new')
        plt.legend()
 
        if (aux):
            plt.ylim(0, 100)
            plt.plot(iterable_list[1:], te_accs_old, label='eval_old')
            plt.legend()
       
        plt.ylim(0, 100)
        plt.plot(iterable_list, te_accs_total, label='eval_total')
        plt.legend()
                
        while (os.path.isfile(path + fname + 'total' + '.png')):
            letra = int(fname[19])
            letra = letra + 1
            fname = fname[:19] + str(letra)
            
        plt.savefig(path + fname + 'total' + '.png')
    

    else:
        try:
            os.mkdir('Results/Train')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
        path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/Results/Train/'
        
        plt.figure(4)
        plt.ylim(0, 100)
        plt.plot(iterable_list, te_accs_new, label='train_new')
        plt.legend()
        
        if (aux):
            plt.ylim(0, 100)
            plt.plot(iterable_list[1:], te_accs_old, label='train_old')
            plt.legend()
            
        plt.xlabel('Class number')
        plt.ylabel('Train Accuracy [%]')
        plt.ylim(0, 100)
        plt.plot(iterable_list, te_accs_total, label='train_total')
        plt.legend()
 
        while (os.path.isfile(path + fname + 'total' + '.png')):
            letra = int(fname[19])
            letra = letra + 1
            fname = fname[:19] + str(letra)
            
        plt.savefig(path + fname + 'total' + '.png')

    

def save_loss(iterable_list, num_epochs, loss_total, loss_classification, loss_distilation, fname, train):
    """Permet guardar les gràfiques de pèrdues d'entrenament i d'evaluació
    Args:
        iterable_list: llista de tasques que han estat entrenades
        num_epochs: epochs utilitzades durant l'entrenament
        loss_classification: llista amb les pèrdues obtenides amb la funció de pèrdua per classificació
        loss_distillation: llista amb les pèrdues obtenides amb la funció de pèrdua per destil·lació
        loss_total: llista amb el valor de la pèrdua total obtenides
        fname: nom del fitxer amb el que es guardarà
        train: permet saber si es tracten de dades d'avaluació o d'entrenament
    Resultat:
        Emmagatzemament de les gràfiques de pèrdua, tant d'entrenament com d'avaluació
    """   
    if (train):
        try:
            os.mkdir('Results/Loss/Train')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
                
        path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/Results/Loss/Train/'
        plt.figure(7)
        plt.xlabel('Num_epochs')
        plt.ylabel('Loss_total')
        dim = loss_total.shape
        epochs_axis = np.arange(0, num_epochs, 1)
        for i in range(dim[0]):
            plt.plot(epochs_axis, loss_total[i,:], label = str(iterable_list[i]))
    
        plt.legend()
        
        while (os.path.isfile(path + fname + 'total' + '.png')):
            letra = int(fname[25])
            letra = letra + 1
            fname = fname[:25] + str(letra)
            
        plt.savefig(path + fname + 'total' + '.png')
        
        plt.figure(8)
        plt.xlabel('Num_epochs')
        plt.ylabel('Loss_classification')
        dim = loss_total.shape
        epochs_axis = np.arange(0, num_epochs, 1)
        for i in range(dim[0]):
            plt.plot(epochs_axis, loss_classification[i,:], label = str(iterable_list[i]))
        
        plt.legend()
        
        while (os.path.isfile(path + fname + 'classification' + '.png')):
            letra = int(fname[25])
            letra = letra + 1
            fname = fname[:25] + str(letra)
            
        plt.savefig(path + fname + 'classification' + '.png')
        
        
        
        plt.figure(9)
        plt.xlabel('Num_epochs')
        plt.ylabel('Loss_distillation')
        dim = loss_total.shape
        epochs_axis = np.arange(0, num_epochs, 1)
        for i in range(dim[0]):
            plt.plot(epochs_axis, loss_distilation[i,:], label = str(iterable_list[i]))
      
        plt.legend()
        
        while (os.path.isfile(path + fname + 'distilation' + '.png')):
            letra = int(fname[25])
            letra = letra + 1
            fname = fname[:25] + str(letra)
            
        plt.savefig(path + fname + 'distilation' + '.png')
    
    else:
        try:
            os.mkdir('Results/Loss/Eval')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
 
        
        path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/Results/Loss/Eval/'
        plt.figure(10)
        plt.xlabel('Num_epochs')
        plt.ylabel('Loss_total')
        dim = loss_total.shape
        epochs_axis = np.arange(0, num_epochs, 1)
        for i in range(dim[0]):
            plt.plot(epochs_axis, loss_total[i,:], label = str(iterable_list[i]))
  
        plt.legend()
        
        while (os.path.isfile(path + fname + 'total' + '.png')):
            letra = int(fname[25])
            letra = letra + 1
            fname = fname[:25] + str(letra)
            
        plt.savefig(path + fname + 'total' + '.png')
        
        plt.figure(11)
        plt.xlabel('Num_epochs')
        plt.ylabel('Loss_classification')
        dim = loss_total.shape
        epochs_axis = np.arange(0, num_epochs, 1)
        for i in range(dim[0]):
            plt.plot(epochs_axis, loss_classification[i,:], label = str(iterable_list[i]))

        plt.legend()
        
        while (os.path.isfile(path + fname + 'classification' + '.png')):
            letra = int(fname[25])
            letra = letra + 1
            fname = fname[:25] + str(letra)
            
        plt.savefig(path + fname + 'classification' + '.png')
        
        
        
        plt.figure(12)
        plt.xlabel('Num_epochs')
        plt.ylabel('Loss_distillation')
        dim = loss_total.shape
        epochs_axis = np.arange(0, num_epochs, 1)
        for i in range(dim[0]):
            plt.plot(epochs_axis, loss_distilation[i,:], label = str(iterable_list[i]))

        plt.legend()
        
        while (os.path.isfile(path + fname + 'distilation' + '.png')):
            letra = int(fname[25])
            letra = letra + 1
            fname = fname[:25] + str(letra)
            
        plt.savefig(path + fname + 'distilation' + '.png')
        

    
def save_matrix(matrix_accuracies, fname):
    """Emmagatzematge de la matriu de precisions de totes les tasques
    Args:
        matrix_accuracies: matriu de precisions per cada tasca
        fname: nom del fitxer amb el que es guardarà
    Resultat:
        Matriu de precisions emmagatzemat en un fitxer txt per posteriorment realitzar comparacions
    """
    try:
            os.mkdir('Results/Text')
    except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/Results/Text/'
    while (os.path.isfile(path + fname + '.txt')):
        letra = int(fname[19])
        letra = letra + 1
        fname = fname[:19] + str(letra)
   
    matrix_accuracy = np.matrix(matrix_accuracies)
    filename = path + fname + '.txt'
    with open(filename,'wb') as f:
        for line in matrix_accuracy:
            np.savetxt(f, line, fmt='%1.1e', delimiter='  ')
        