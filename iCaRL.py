import torch
import torch.nn as nn
import torch.nn.functional as F
from pretrained_resnet import resnet18
from pretrained_resnet import resnet50
from pretrained_resnet import resnet101
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

import random

import save_load

from scipy import ndimage

# Hyper paràmetres
batch_size = 100                #Tamany de cada batch
learning_rate = 0.000001        #Learning rate per defecte
T =1.5                          #Temperatura per defecte
weight_losses = 0.5             #Weight losses per defecte per CIFAR 100
momentum = 0.2                  #Se li assigna més importància als exemples guardats ja que normalment més nous hi seran a l'entrada

path = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/saved_models/model2'              #Directori on guardar el model
path2 = '/home/usuaris/imatge/alex.mateo/Downloads/icarl/saved_models/best_model2'        #Directori on guardar el millor model

class iCaRLNet(nn.Module):
    def __init__(self, feature_size, n_classes, num_epochs):
        
        #Elements de la xarxa 
        super(iCaRLNet, self).__init__()
        self.feature_extractor = resnet50(pretrained = True)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.fc = nn.Linear(feature_size, n_classes, bias=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.ReLU = torch.nn.ReLU()
    
        #Híper paràmetres
        self.num_epochs = num_epochs

        #Comptadors
        self.n_classes = n_classes    #Número de classes vistes en entrenament
        self.n_known = 0              #Número de classes ja emmagatzemades, s'ha realitzat tot el procés.
        self.last_classes = []        #Últim conjunt de classes ja emmagatzemats

        #Llista de memòria. Conté les imatges en memòria amb la forma (N,C,H,W) 
        self.exemplar_sets = []                
       
        #Funcions de pèrdua i optimitzadors
        self.cls_loss = nn.CrossEntropyLoss()          #Pèrdua de classificació utilitzada
        self.dist_loss = nn.KLDivLoss()
        self.criterion = nn.BCELoss()                  #Pèrdua de destil·lació utilitzada
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.00001)   #Optimitzador
        
        # Media de los ejemplos
        self.compute_means = True
        self.exemplar_means = []      #Vector amb la mitjana dels exemples emmagatzemats de cada classe
        
        #Net2Net
        self.neurons_added = 1500
        self.channels_added = 0
    
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)        
        return x
    
          
    def increment_classes(self, n):
        """Afegeix n classes a la última capa i realitza el net2net progressiu.
        Args:
            n: número de classes noves en l'entrada
        Resultat: 
            Model més ample i adaptat a les classes ja entrenades
        """
        
        #OPCIÓ 1: aplicació del net2net
        self.wider_net_conv_progressive()
        self.wider_net_progressive(n)
          
        #Opció 2: mètode clàssic sense net2net, només s'amplia la capa de sortida    
        """
        in_features2 = self.feature_extractor.fc.out_features
        out_features2 = self.fc.out_features
        weight2 = self.fc.weight.data     
        self.fc = nn.Linear(in_features2, out_features2+n, bias=False)
        self.fc.weight.data[:out_features2] = weight2
        self.optimizer = optim.Adam([
                                    {'params': self.feature_extractor.parameters()},
                                    {'params': self.fc.parameters(), 'lr': 0.00001}
                            ], lr=learning_rate, weight_decay=0.00001)
        #self.cuda()
        """
        
        self.n_classes += n
        #self.unfreeze_layers()    #Mètode de descongelació de capes que no ha resultat efectiu. No implementat
       
    def wider_net_conv_progressive(self):
        """Augmenta l'amplada de les capes indicades de la ResNet
        Args:
        Result: 
            Model amb més capacitat        
        """
        #CAPA 1
        out_channels, in_channels, height, width = (self.feature_extractor.layer4[2].conv2.weight.shape)
        weight1 = self.feature_extractor.layer4[2].conv2.weight.data
        self.feature_extractor.layer4[2].conv2 =  nn.Conv2d(in_channels, out_channels + self.channels_added, kernel_size=3, stride=1, padding=1, bias=False)
        
        #Opció 1 de nous valors del filtre: aleatoris
        noise_size1 = torch.Size([self.channels_added, in_channels, height, width])
        weight1_noise = torch.normal(0, 0.01, size = noise_size1).cuda()
        weight1 = torch.cat((weight1, weight1_noise), 0)
        self.feature_extractor.layer4[2].conv2.weight.data = weight1
        
        #Opció 2 de nous valors dels filtres: valor copiat aleatoriament d'altres filtres d'aquesta capa
        #activations_units = torch.randperm(out_channels)
        #activations_units = activations_units[:self.channels_added]
        #weights_new = torch.zeros([self.channels_added, in_channels, height, width])
        #for count, i in enumerate(activations_units):
        #    weights_new[count, :, :, :] = weight1[i, :, :, :]
        #weight1 = torch.cat((weight1, weights_new.cuda()), 0)
        #self.feature_extractor.layer4[2].conv2.weight[:out_channels].data = weight1  
        
        
        """
        #CAPA 2
        out_channels3, in_channels3, height3, width3 = (self.feature_extractor.layer3[5].conv2.weight.shape)
        weight3 = self.feature_extractor.layer3[5].conv2.weight.data
        self.feature_extractor.layer3[5].conv2 =  nn.Conv2d(in_channels3, out_channels3 + self.channels_added, kernel_size=3, stride=1, padding=1, bias=False)
        #self.feature_extractor.layer3[5].conv2.weight[:out_channels].data = weight3
        activations_units3 = torch.randperm(out_channels3)
        activations_units3 = activations_units3[:self.channels_added]
        weights_new3 = torch.zeros([self.channels_added, in_channels3, height3, width3])
        for count, i in enumerate(activations_units3):
            weights_new3[count, :, :, :] = weight3[i, :, :, :]
        weight3 = torch.cat((weight3, weights_new3.cuda()), 0)
        self.feature_extractor.layer3[2].conv2.weight[:out_channels3].data = weight3  
        """
        
        #CAPA1
        self.feature_extractor.layer4[2].bn2 = nn.BatchNorm2d(out_channels + self.channels_added)
        #CAPA2
        #self.feature_extractor.layer3[5].bn2 = nn.BatchNorm2d(out_channels3 + self.channels_added)
        
        
        #CAPA1
        out_channels2, in_channels2, height2, width2 = (self.feature_extractor.layer4[2].conv3.weight.shape)
        weight2 = self.feature_extractor.layer4[2].conv3.weight.data
        
        #Opció 1: valors dels nous aleatoris
        noise_size = torch.Size([out_channels2, self.channels_added, height2, width2])
        weight2_noise = torch.normal(0, 0.01, size = noise_size).cuda()
        weight2 = torch.cat((weight2, weight2_noise), 1)
        
        #Opció 2: valors copiats dels ja existents
        #weights_new2 = torch.zeros([out_channels2, self.channels_added, height2, width2])
        #for count, i in enumerate (activations_units):
        #    weight2[:,i, :, :] = weight2[:,i, :, :]/2      
        #    weights_new2[:, count, :, :] = weight2[:, i, :, :]
        #weight2 = torch.cat((weight2, weights_new2.cuda()), 1)       
        
        self.feature_extractor.layer4[2].conv3 =  nn.Conv2d(in_channels2 + self.channels_added, out_channels2, kernel_size=1, stride=1, bias=False)
        self.feature_extractor.layer4[2].conv3.weight.data = weight2
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.00001)
        self.channels_added += 4
        self.cuda()
        
    def wider_net_progressive(self, n):
        """Augment de l'amplada de la última capa fc de la ResNet i increment de la capa de sortida en funció de n
        Args:
            n: número de classes noves de l'entrada.
        Resultat:
            Augment de la capacitat del model i adaptació a les classes ja observades.
        """
        
        #CAPA1
        in_features1 = self.feature_extractor.fc.in_features
        out_features1 = self.feature_extractor.fc.out_features
        weight1 = self.feature_extractor.fc.weight.data  
        bias1 = self.feature_extractor.fc.bias.data
        div_factor = (out_features1+self.neurons_added)/out_features1
     
        
        self.feature_extractor.fc = nn.Linear(in_features1, out_features1+self.neurons_added, bias=True)
        #Opció 1: nous valors inicialitzats aleatoriament
        self.feature_extractor.fc.weight[:out_features1].data = weight1
        self.feature_extractor.fc.bias[:out_features1].data = bias1
        #noise_size_1 = torch.Size([self.neurons_added, in_features1])
        #weight1_noise_1 = torch.normal(0, 0.01*weight1.std(), size = noise_size_1).cuda()
        #weight1 = torch.cat((weight1, weight1_noise_1), 0)
        #bias1_noise = torch.normal (0, 0.01, size = torch.Size([self.neurons_added])).cuda()
        #bias1 = torch.nn.Parameter(torch.cat((bias1, bias1_noise), 0))
        
        
        #Opció 2: valors inicialitzats copiant als ja existents
        #activations_units = torch.randperm(out_features1)
        #activations_units = activations_units[:self.neurons_added]
        #weights_new = torch.zeros([self.neurons_added, in_features1])
        #bias_new = torch.zeros(self.neurons_added)
        #for count, i in enumerate(activations_units):
        #    weights_new[count, :] = weight1[i, :]
        #    bias_new[count] = bias1[i]
        #weight1 = torch.cat((weight1, weights_new.cuda()), 0)
        #bias1 = torch.cat((bias1, bias_new.cuda()), 0) 

        #self.feature_extractor.fc.weight.data = weight1
        #self.feature_extractor.fc.bias.data = bias1
      
        

        #CAPA FC
        in_features2 = self.feature_extractor.fc.out_features
        out_features2 = self.fc.out_features
        weight2 = self.fc.weight.data
        
        #Opció 1: valors inicialitzats aleatoriament
        noise_size = torch.Size([out_features2, self.neurons_added])
        weight2_noise = torch.normal(0, 0.01*weight2.std(), size = noise_size).cuda()
        weight2 = torch.cat((weight2, weight2_noise), 1)
        
        #Opció 2: valors inicialitzats als copiats ja existents
        #weights_new2 = torch.zeros([out_features2, self.neurons_added])
        #for count, i in enumerate (activations_units):
        #    weight2[:,i] = weight2[:,i]/2      
        #    weights_new2[:, count] = weight2[:, i]
        #weight2 = torch.cat((weight2, weights_new2.cuda()), 1)
        
        
        self.fc = nn.Linear(in_features2, out_features2+n, bias=False)
        self.fc.weight.data[:out_features2] = weight2
        self.optimizer = optim.Adam([
                                    {'params': self.feature_extractor.parameters()},
                                    {'params': self.fc.parameters(), 'lr': 0.00001}
                            ], lr=learning_rate, weight_decay=0.00001)
        #self.neurons_added += 120
        self.cuda()    
        
        
    def deeper_net(self):
        """Augment de la profunditat del model afegit una "fully connected" a la sortida de la ResNet
        Result: augment de la capacitat del model
        Avís: no implementat en el projecte degut a que els resultats no han estat els esperats
        """
        
        in_features1 = self.feature_extractor.fc.in_features
        out_features1 = self.feature_extractor.fc.out_features
        fc = self.feature_extractor.fc
        
        #Inicialització aleatòria
        fc_deeper = nn.Linear(out_features1, out_features1, bias=True)
        
        #Inicialització a la matriu identitat
        #weight2 = torch.eye(out_features1, out_features1) 
        #weight2 += (torch.normal(0, 0.1, size = weight2.shape))
        #weight2 = weight2/torch.max(weight2)
        #bias1 = torch.nn.Parameter(torch.zeros(out_features1))
        #fc_deeper.weight.data = weight2
        #fc_deeper.bias = bias1
        
        self.feature_extractor.fc = nn.Sequential(fc,self.ReLU,fc_deeper)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.00001)
        
        
    def wider_net(self, n):
        """Implementació del wider net, no de manera progressiva sinó que de cop.
        Args: 
            n: número de classes noves a l'entrada.
        Resultat: 
            Model més ample.
        Avís: algoritme no implementat, superar pel progressiu

        """
        in_features1 = self.feature_extractor.fc.in_features
        out_features1 = self.feature_extractor.fc.out_features
        weight1 = self.feature_extractor.fc.weight.data  
        bias1 = self.feature_extractor.fc.bias
        
        self.feature_extractor.fc = nn.Linear(in_features1, out_features1*2, bias=True)
        
        #Opció 1: inicialització als pesos ja existents, tenim el doble de sortides amb els resultats obtinguts prèviament dos cops.
        weight1 = torch.cat((weight1, weight1), 0)
        bias1 = torch.nn.Parameter(torch.cat((bias1, bias1), 0))
        
        self.feature_extractor.fc.weight.data = weight1
        self.feature_extractor.fc.bias = bias1
        
        """En la primera capa no es dividirà entre dos ja que es tindrà els mateixos valors que hi havia en les 1024 neurones en les 2048. Serà en la següent capa on es dividirà entr          els pesos i s'afegirà soroll per trencar la simetria. La sortida hauria de donar igual, s'han doblat les neurones de la capa i s'ha dividit els pesos entre 2, la                       combinació lineal deuria de ser la mateixa.
        """
        
        in_features2 = self.feature_extractor.fc.out_features
        out_features2 = self.fc.out_features
        weight2 = self.fc.weight.data
        
        #Inicialització dels nous pesos copiats als d'abans, afegit una petita quantitat de soroll per trencar la simetria.
        weight2 = torch.div(weight2, 2)
        weight2_noise = weight2 + torch.normal(0, 0.001*weight2.std(), size = weight2.shape).cuda()
        weight2 = torch.cat((weight2, weight2), 1)
        
        self.fc = nn.Linear(in_features2, out_features2+n, bias=False)
        self.fc.weight.data[:out_features2] = weight2
        
        self.optimizer = optim.Adam([
                                    {'params': self.feature_extractor.parameters()},
                                    {'params': self.fc.parameters(), 'lr': 0.00001}
                            ], lr=learning_rate, weight_decay=0.00001)
      
     
    def deeper_net_conv(self):
        """Implementació del deeper net2net afegint una etapa de la ResNet al final de la etapa 4
        Args:
        Resultat:
            Model més profund que aporta més capacitat.
        Avís: mètode no implementat, els resultats no han estat els esperats.
        """
        out_channels, in_channels, height, width = (self.feature_extractor.layer3[5].conv3.weight.shape)
        weight4 = torch.zeros((out_channels, in_channels, 1, 1))
        weight42 = torch.zeros((in_channels, in_channels, 3, 3))
        weight43 = torch.zeros((in_channels, out_channels, 1, 1))
        center_h = int((height-1)/2)
        center_w = int((width-1)/2)
        
        #Opció 1: inicialització dels valors dels filtres a la identitat
        #weights4
        for i in range(in_channels):
            tmp = torch.zeros((in_channels, 1, 1))
            tmp[i,0, 0] = 1
        for i in range(out_channels):
            weight4[i, :, :, :] = tmp
        
        #weights42    
        for i in range(in_channels):
            tmp = torch.zeros((in_channels, 3, 3))
            tmp[i,1, 1] = 1
        for i in range(in_channels):
            weight42[i, :, :, :] = tmp
        
        #weights43
        for i in range(out_channels):
            tmp = torch.zeros((out_channels, 1, 1))
            tmp[i,0, 0] = 1
        for i in range(in_channels):
            weight43[i, :, :, :] = tmp
            
        weight3 = torch.nn.Parameter(self.feature_extractor.layer3[5].conv3.weight)  
        conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        conv3.weight = weight3
        
        bn3 = nn.BatchNorm2d(out_channels)
        conv4 = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        conv4.weights = weight4
       
       
        bn4 = nn.BatchNorm2d(in_channels)
        conv42 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        conv42.weight = weight42
                     
        bn42 = nn.BatchNorm2d(in_channels)
        conv43 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)         
        conv43.weight = weight43             
                    
       
        
        self.feature_extractor.layer3[5].conv3 = nn.Sequential(conv3, bn3, conv4, bn4, conv42, bn42, conv43)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.00001)
        
     
    def freeze_layers(self):          
    #Congelació d'algunes capes de la ResNet amb la intenció de reduir els paràmetres del model i millorar l'entrenament.
    #Args:
    #Resultat:
    #    Menys paràmetres a optimitzar.
    #Avís: mètode no implementat, no ha aportat una millora significativa.
    
    
        for param in self.feature_extractor.conv1.parameters():
                param.requires_grad = False
        
        for param in self.feature_extractor.bn1.parameters():
                param.requires_grad = False
         
        for param in self.feature_extractor.layer1.parameters():
                param.requires_grad = False
              
        
        
    def one_hot_encoding(self, labels):
    #Codificació de one hot encoding en les etiquetes
    #Args: 
    #    Labels: etiquetat de l'exemple en un moment donat.
    #Return:
    #    Out2: tensor amb el label codificat amb one-hot encoding
    
      ns = len(labels)
      out = torch.zeros(ns, self.n_classes).cuda()   #Matriu de ns (número de mostres del batch) x n_classes (número de clases observades)
      for sample_idx in range(ns):
          out[sample_idx, labels[sample_idx]] = 1    #Para cada mostra, es fica un 1 en la classe predita.
      out2 = out.type(torch.cuda.LongTensor)
      return out2  
      


    def compute_mean(self, transform, old):
    #Calcula la mitjana dels exemples en memòria, sempre que es tinguin noves dades
    #Args:
    #    transform: transformacions realitzades sobre la imatge per calcular la mitjana
    #Resultat:
    #    Càlcul de la mitjana de les mostres d'una classe necessari per l'obtenciño de les mostres més representatives.
        
        
        print ("Computing mean of exemplars...")
        exemplar_means = []
        for P_y in self.exemplar_sets:                      #Càlcul de la mitjana a partir dels exemples emmagatzemats en memòria                                
            features = []
            # S'extrau les característiques de les imatges
            for ex in P_y:
                with torch.no_grad():
                  ex = Variable(transform(Image.fromarray(ex))).cuda()
                  feature = self.feature_extractor(ex.unsqueeze(0))
                  feature = feature.squeeze()
                  feature.data = feature.data / feature.data.norm() # Normalització
                  features.append(feature)
            features = torch.stack(features)
            mu_y = features.mean(0).squeeze()
            mu_y.data = mu_y.data / mu_y.data.norm()
            exemplar_means.append(mu_y)
        self.exemplar_means = exemplar_means
        if (old):
            self.compute_means = True
        else:
            self.compute_means = False
        print ("Done")
    
    def classify(self, x, transform):
        """Clasifica las imatges pel prototype més proper
        Args:
            x: batch de imatges en el input
        Returns:
            preds: Tensor de tamany (batch_size,).
        """
        
        
        batch_size = x.size(0)

        if self.compute_means:
            self.compute_mean(transform, False)

        #Per tal de comparar ha de tenir les mateixes dimensions
        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means)       # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2)             # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x)       # (batch_size, feature_size)
        
        for i in range(feature.size(0)):               #Es normalitzen les característiques per poder-les comparar amb les ja calculades
            feature.data[i] = feature.data[i]/feature.data[i].norm()  
   
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
        _, preds = dists.min(1)  

        return preds


    def momentum_means(self, num_class, transform, images, old_images, m):
        """Actualització de les mostres emmagatzemades en memòria en el cas de que hi hagin de classes antigues en l'entrada
        Args:
            num_class: classe de les mostres en l'entrada
            transform: transformació realitzada a les imatges emmagatzemades
            images: total d'imatges d'aquella classe a tractar per ser emmagatzemades
            old_images: total d'imatges emmagatzemades prèviament
            m: tamany de memòria
        Resultat:
            Nou conjunt d'imatges emmagatzemades
        """
        
        features = []
        features_old = []
        
        for img in images:
            with torch.no_grad():
                x = Variable(transform(Image.fromarray(img))).cuda()             
                feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()  
                feature = feature / np.linalg.norm(feature) 
                features.append(feature[0])                

        features = np.array(features)
        class_mean = np.mean(features, axis=0)             
        class_mean = class_mean / np.linalg.norm(class_mean)
        
        for img in old_images:
            with torch.no_grad():
                x = Variable(transform(Image.fromarray(img))).cuda()                
                feature_old = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()  
                feature_old = feature_old / np.linalg.norm(feature_old) 
                features_old.append(feature_old[0])                

        features_old = np.array(features_old)
        class_mean_old = np.mean(features_old, axis=0)             
        class_mean_old = class_mean_old / np.linalg.norm(class_mean_old)
        
        class_mean = momentum*class_mean + (1-momentum)*class_mean_old
        features = np.concatenate ((features, features_old), axis = 0)
        images = np.concatenate ((images, old_images), axis=0)
        
        exemplar_set = []
        exemplar_features = []                        
        for k in range(m):                                  # Selecció de les més representatives
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            i = np.argmin(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1)))
            
            exemplar_set.append(images[i])
            exemplar_features.append(features[i])
    
        self.exemplar_sets[num_class] = np.array(exemplar_set)
        
    
    def classify_in_construct(self, images, label, transform):
        """Retorna els índex del conjunt d'imatges que es classifiquen correctament
        Args: 
            images: np.array amb les imatges per classe
            label: conjunt d'etiquetats de les respectives imatges
            transform: transformació a realitzar a les imatges que seran emmagatzemades
        Return:
            indices_correctos: índex de les imatges classificades correctament
            indices_nocorrectos: índex de les imatges classificades incorrectament
       
        """
        indices_correctos = []
        indices_nocorrectos = []
        for i, img in enumerate(images):
            x = Variable(transform(Image.fromarray(img))).cuda()                
            g = self.softmax(self.forward(x.unsqueeze(0))).data.cpu().numpy()
            pred = np.argmax(g, axis=1)
            if (pred == label):
                indices_correctos.append(i)
            else:
                indices_nocorrectos.append(i)
       
        return indices_correctos, indices_nocorrectos

        

    def construct_exemplar_set(self, images, m, transform, num_class):                          
        """Decideix el conjunt d'exemples a guardar en memòria
        Args:
            images: np.array amb les imatges per classe
            m: tamany de memòria disponible per aquella classe
            transform: transformació a realitzar a les imatges que seran emmagatzemades
            num_class: classe que representa al conjunt images
        Resultat: 
            Emmagtzemament del conjunt de mostres de la classe donat el tamany disponible m
        """
        
        if (num_class < self.n_known):     #Cas en el que arriben classes ja conegudes, es tornen a recalcular els protips i els exemples guardats en memòria
            print ("Recomputing examples saved")
            self.momentum_means(num_class, transform, images, self.exemplar_sets[num_class], m)
            
        else:     
        ###########Construct exemplar set sense processament previ######################                         
            """
            print ("Computing examples to save")
            features = []
            for img in images:
                with torch.no_grad():
                    x = Variable(transform(Image.fromarray(img))).cuda()                 #Imatge en x
                    feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()  #Es passa pel feature extractor per obtenir les features. El unsqueeze per elimniar les dimensions 1 i tenir una llista amb totes les features 
                    feature = feature / np.linalg.norm(feature) # L2 normalització
                    features.append(feature[0])                 # Ens guardem el vector de features --> finalment es tindrà una matriu amb les features de cada imatge
    
            features = np.array(features)
            class_mean = np.mean(features, axis=0)                # Mitjana en l'eix vertical   
            class_mean = class_mean / np.linalg.norm(class_mean)  #Normalització (ho indica el paper del iCaRL)
            """
            
            
            """
            #Opció 1: per una inicialització aleatòria
            exemplar_set = []
            random_indexes = random.sample(range(len(images)), int(m))
            for k in random_indexes:
                exemplar_set.append(images[k])
            self.exemplar_sets.append(np.array(exemplar_set))
            """
            
           
           
            """
            #Opció 2: inicialització clàssica com dicta el paper
            exemplar_set = []
            exemplar_features = []                              
            for k in range(int(m)):                                 
                S = np.sum(exemplar_features, axis=0)
                phi = features
                mu = class_mean
                #i = np.argmax(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1)))
                i = np.argmin(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1))) #De normal aquest
    
                exemplar_set.append(images[i])
                exemplar_features.append(features[i])
    
            self.exemplar_sets.append(np.array(exemplar_set))
            
            """
            
            ##############Construct exemplar set en función de si se clasifican correctamente############
            indices_correctos, indices_nocorrectos = self.classify_in_construct(images, num_class, transform)
            
            #OPCIÓ 1: aleatoriament, primer els classificats correctament i després els que no            
            
            exemplar_set = []
            if(len(indices_correctos) < m):
                random_indexes_correct = np.random.choice(indices_correctos, len(indices_correctos), replace = False)
                random_indexes_incorrect = np.random.choice(indices_nocorrectos, int(m-(len(indices_correctos))), replace = False)
                random_indexes = np.concatenate((random_indexes_correct, random_indexes_incorrect), axis = 0)
            else:
                random_indexes = np.random.choice(indices_correctos, int(m), replace = False)
                
            for k in random_indexes:
                exemplar_set.append(images[int(k)])
            self.exemplar_sets.append(np.array(exemplar_set))
            
            
            """
            #OPCION 2: utilitzant el mètode clàssic, primer els índex classificats correctament i després els que no
            features = []
            features_correct = []
            features_incorrect = []
            for i,img in enumerate(images):
                with torch.no_grad():
                    x = Variable(transform(Image.fromarray(img))).cuda()                
                    feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy() 
                    feature = feature / np.linalg.norm(feature) 
                    features.append(feature[0])                 
                    if (i in indices_correctos):
                        features_correct.append(feature[0])
                    else:
                        features_incorrect.append(feature[0])
    
            features = np.array(features)
            features_correct = np.array(features_correct)
            features_incorrect = np.array(features_incorrect)
            class_mean = np.mean(features, axis=0)              
            class_mean = class_mean / np.linalg.norm(class_mean)
             
            exemplar_set = []
            exemplar_features = []                              
            
            if (len(indices_correctos)>m):
                for k in range(int(m)):                                 
                    S = np.sum(exemplar_features, axis=0)
                    phi = features_correct
                    mu = class_mean
                    #i = np.argmax(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1)))
                    i = np.argmin(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1))) #De normal este
                    
                    exemplar_set.append(images[indices_correctos[i]])    #Está bien?
                    exemplar_features.append(features_correct[i])
            else:
                for k in range(len(indices_correctos)):                                  
                    S = np.sum(exemplar_features, axis=0)
                    phi = features_correct
                    mu = class_mean
                    #i = np.argmax(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1)))
                    i = np.argmin(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1))) #De normal este
                    
                    exemplar_set.append(images[indices_correctos[i]])   
                    exemplar_features.append(features_correct[i]) 
                    
                for k in range (int(m-len(indices_correctos))):
                    S = np.sum(exemplar_features, axis=0)
                    phi = features_incorrect
                    mu = class_mean
                    #i = np.argmax(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1)))
                    i = np.argmin(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1))) #De normal este
                    
                    exemplar_set.append(images[indices_nocorrectos[i]])    
                    exemplar_features.append(features_incorrect[i]) 
                    
            self.exemplar_sets.append(np.array(exemplar_set))
            """
            """
            #OPCIÓN 2.1: igual que la 2 però amb una distribució més àmplia. NO VIABLE
            features = []
            features_correct = []
            features_incorrect = []
            for i,img in enumerate(images):
                with torch.no_grad():
                    x = Variable(transform(Image.fromarray(img))).cuda()                
                    feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy() 
                    feature = feature / np.linalg.norm(feature) 
                    features.append(feature[0])                 
                    if (i in indices_correctos):
                        features_correct.append(feature[0])
                    else:
                        features_incorrect.append(feature[0])
    
            features = np.array(features)
            features_correct = np.array(features_correct)
            features_incorrect = np.array(features_incorrect)
            class_mean = np.mean(features, axis=0)              
            class_mean = class_mean / np.linalg.norm(class_mean)
             
            exemplar_set = []
            exemplar_features = []                              
            
            if (len(indices_correctos)>m):
                value = np.sqrt(np.sum((class_mean - (features_correct)) ** 2, axis=1))
                index_to_save = np.argsort(value)
                
                for k in range(int(m)):                                 
                    exemplar_set.append(images[indices_correctos[index_to_save[k]]])    #Está bien?
                    exemplar_features.append(features_correct[index_to_save[k]])
            else:
                value = np.sqrt(np.sum((class_mean - (features_correct)) ** 2, axis=1))
                index_to_save = np.argsort(value)
                for k in range(len(indices_correctos)):                                 
                    exemplar_set.append(images[indices_correctos[index_to_save[k]]])    #Está bien?
                    exemplar_features.append(features_correct[index_to_save[k]])
                
                if(len(features_incorrect) > 0):
                    value = np.sqrt(np.sum((class_mean - (features_incorrect)) ** 2, axis=1))
                    index_to_save = np.argsort(value)    
                    for k in range (int(m-len(indices_correctos))):
                        exemplar_set.append(images[indices_nocorrectos[index_to_save[k]]])    
                        exemplar_features.append(features_incorrect[index_to_save[k]]) 
                    
            self.exemplar_sets.append(np.array(exemplar_set))
            """
            
            #OPCIÓN 3: a partir d'aquells que estan mes o menys a prop dels altres prototips
            #NO ÉS VIABLE
            #self.compute_mean(transform, True)
            """
            print(len(indices_correctos))
            features = []
            features_correct = []
            features_incorrect = []
            for i,img in enumerate(images):
                with torch.no_grad():
                    x = Variable(transform(Image.fromarray(img))).cuda()                 
                    feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy() 
                    feature = feature / np.linalg.norm(feature) 
                    features.append(feature[0])                 
                    if (i in indices_correctos):
                        features_correct.append(feature[0])
                    else:
                        features_incorrect.append(feature[0])
    
            features = np.array(features)
            features_correct = np.array(features_correct)
            features_incorrect = np.array(features_incorrect)
            class_mean = np.mean(features, axis=0)              
            class_mean = class_mean / np.linalg.norm(class_mean)
            
            exemplar_set = []
            exemplar_features = []
            """
            
            """
            if (len(indices_correctos)>m):
                for k in range(int(m)):                                 
                    S = np.sum(exemplar_features, axis=0)
                    phi = features_correct
                    feature_aux = 1000000
                    
                    for class_index in range(len(self.exemplar_means)):
                        value = np.sqrt(np.sum((self.exemplar_means[class_index].data.cpu().numpy() - (phi)) ** 2, axis=1))
                        index = np.argmin(value)                        
                        if(value[i]<feature_aux):
                            feature_aux = value[i]
                            i = index
                    exemplar_set.append(images[indices_correctos[i]])    #Está bien?
                    exemplar_features.append(features_correct[i])
            else:
                for k in range(len(indices_correctos)):                                  
                    S = np.sum(exemplar_features, axis=0)
                    phi = features_correct
                    feature_aux = 100000
                    for class_index in range(len(self.exemplar_means)):
                        value = np.sqrt(np.sum((self.exemplar_means[class_index].data.cpu().numpy() - (phi)) ** 2, axis=1))
                        index = np.argmin(value)                        
                        if(value[i]<feature_aux):
                            feature_aux = value[i]
                            i = index               
                    exemplar_set.append(images[indices_correctos[i]])   
                    exemplar_features.append(features_correct[i]) 
                    
                    for k in range (int(m-len(indices_correctos))):
                        S = np.sum(exemplar_features, axis=0)
                        phi = features_incorrect
                        feature_aux = 100000
                        for class_index in range(len(self.exemplar_means)):
                            value = np.sqrt(np.sum((self.exemplar_means[class_index].data.cpu().numpy() - (phi)) ** 2, axis=1))
                        index = np.argmin(value)                        
                        if(value[i]<feature_aux):
                            feature_aux = value[i]
                            i = index
                        exemplar_set.append(images[indices_nocorrectos[i]])    
                        exemplar_features.append(features_incorrect[i]) 
                        
            self.exemplar_sets.append(np.array(exemplar_set))
            """
            """
            if (len(indices_correctos)>m):
                for k in range(int(m)):                                 
                    S = np.sum(exemplar_features, axis=0)
                    phi = features_correct
                    feature_aux = 100000                
                    for class_index in range(len(self.exemplar_means)):
                        value = np.sqrt(np.sum((self.exemplar_means[class_index].data.cpu().numpy() - (phi)) ** 2, axis=1))
                        index = np.argmin(value)                   
                        if(value[index]<feature_aux):
                            feature_aux = value[index]
                            i = index
                    exemplar_set.append(images[indices_correctos[i]])    #Está bien?
                    exemplar_features.append(features_correct[i])
            else:
                if (len(self.exemplar_means) == 0):
                    for k in range(len(indices_correctos)):                                  
                        S = np.sum(exemplar_features, axis=0)
                        phi = features_correct
                        mu = class_mean
                        #i = np.argmax(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1)))
                        i = np.argmin(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1))) #De normal este
                        
                        exemplar_set.append(images[indices_correctos[i]])   
                        exemplar_features.append(features_correct[i]) 
                    
                    for k in range (int(m-len(indices_correctos))):
                        S = np.sum(exemplar_features, axis=0)
                        phi = features_incorrect
                        mu = class_mean
                        #i = np.argmax(np.sqrt(np.sum((mu - (phi)) ** 2, axis=1)))
                        i = np.argmin(np.sqrt(np.sum((mu - 1.0/(k+1) * (phi + S)) ** 2, axis=1))) #De normal este
                        
                        exemplar_set.append(images[indices_nocorrectos[i]])    
                        exemplar_features.append(features_incorrect[i])
                else:
                    for k in range(len(indices_correctos)):                                  
                        S = np.sum(exemplar_features, axis=0)
                        phi = features_correct
                        feature_aux = 100000
                        for class_index in range(len(self.exemplar_means)):
                            value = np.sqrt(np.sum((self.exemplar_means[class_index].data.cpu().numpy() - (phi)) ** 2, axis=1))
                            index = np.argmin(value)                        
                            if(value[index]<feature_aux):
                                feature_aux = value[index]
                                i = index

                    exemplar_set.append(images[indices_correctos[i]])   
                    exemplar_features.append(features_correct[i]) 
                    
                    for k in range (int(m-len(indices_correctos))):
                        
                        S = np.sum(exemplar_features, axis=0)
                        phi = features_incorrect
                        feature_aux = 100000
                        for class_index in range(len(self.exemplar_means)):
                            value = np.sqrt(np.sum((self.exemplar_means[class_index].data.cpu().numpy() - (phi)) ** 2, axis=1))
                            index = np.argmin(value)                        
                            if(value[index]<feature_aux):
                                feature_aux = value[index]
                                i = index
      
                        exemplar_set.append(images[indices_nocorrectos[i]])    
                        exemplar_features.append(features_incorrect[i]) 
                         
                  
            self.exemplar_sets.append(np.array(exemplar_set))
            """         

    def reduce_exemplar_sets(self, m):
        """Només es guarden els primers m exemples de la classe determinada. Cal recordar que estan guardades en funció de l'ordre d'importància 
        Args:
            m: tamany de memòria de la classe determinada
        Resultat: 
            Tamany d e memòria reduït a l'indicat en m
        """
       
        """
        #Opció 1: memòria equidistribuida entre totes les classes
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:m]
        """
        #Opció 2: memòria per classe diferent
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:int(m[y])]
        

    def combine_dataset_with_exemplars(self, dataset):
        """Construcció d'un dataset amb les mostres de memòria i les de l'entrada. Aplicació del data augmentation en les de memòria
        Args: 
            dataset: conjunt de mostres de l'entrada
        Resultat:
            Augment del dataset amb les de memòria i les de memòria transformades amb el data augmentation
        """
        for y, P_y in enumerate(self.exemplar_sets):
            #exemplar_images = P_y
            exemplar_images = self.data_augmentation(P_y)
            exemplar_labels = [y] * len(exemplar_images)
            dataset.append(exemplar_images, exemplar_labels)
            
      
    def data_augmentation(self, P_y):
    #Aplicació del data augmentation sobre les mostres de memòria
    #Args:
    #    P_y: mostres de memòria
    #Return:
    #    P_y: mostres de memòria transformades y no transformades

        flipUD = np.flipud(P_y)
        #flipLR = np.fliplr(P_y)
        """
        transformation = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)])
        """
        transformation1 = transforms.RandomCrop(32, padding=4)
        transformation2 = transforms.RandomHorizontalFlip()
        transformation3 = transforms.RandomRotation(15)
        transf1 = (np.empty(P_y.shape)).astype(np.uint8)
        transf2 = (np.empty(P_y.shape)).astype(np.uint8)
        transf3 = (np.empty(P_y.shape)).astype(np.uint8)
        
        for i,img in enumerate(P_y):
            with torch.no_grad():
                x = Image.fromarray(img)
                x1 = transformation1(x)
                transf1[i] = np.array(x1)
                
                x2 = transformation2(x)
                transf2[i] = np.array(x2)
                
                x3 = transformation3(x)
                transf3[i] = np.array(x3)
        
        #P_y = np.concatenate((P_y, flipUD), axis = 0)
        #P_y = np.concatenate((P_y, transf), axis = 0)
        P_y = np.concatenate((P_y, transf1, transf2, transf3), axis = 0)
        return P_y
         
    
    
    def update_representation(self, dataset, test_set):
        """Entrenament del model
        Args:
            dataset: conjunt de mostres de l'entrada d'entrenament
            test_set: conjunt de mostres de test
        Return:
            train_losses: pèrdues d'entrenament totals
            classification_losses: pèrdua calculada només en la funció de classificació amb dades d'entrenament
            distilation_losses: pèrdua calculada només en la funció de destil·lació amb dades d'entrenament
            eval_losses: pèrdues d'avaluació totals
            eval_cls_losses: pèrdua calculada només en la funció de classificació amb dades d'avaluació
            eval_dist_losses: pèrdua calculada només en la funció de destil·lació amb dades d'avaluació
        """

        self.compute_means = True
        
        
        classes = list(set(dataset.targets))                      # Classes a l'entrada
        new_classes = [cls for cls in classes if cls >= self.n_classes - 1]  #Classes noves no conegudes
        print ("longitud_dataset: %d" %len(dataset))
        if (len(classes) == 0 & len(self.last_classes)!= 0):      # Cas en el que no hi ha classes noves a l'entrada
            new_classes = last_classes
        self.last_classes = new_classes 
                 
         
        self.combine_dataset_with_exemplars(dataset)              # Combinació de les dades de memòria amb les de l'entrada i aplicació data augmentation
        print ("Longitud del nuevo dataset: %d" %len(dataset))

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=1)
        
        loader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                               shuffle=False, num_workers=1)
        
        # Per tal d'aplicar la destil·lació, es guarden les probabilitats prèvies en entrenament
        q = torch.zeros(len(dataset), self.n_known).cuda()
        for indices, images, labels in loader:
            images = Variable(images).cuda()
            indices = indices.cuda()
            g = self.forward(images)
            g_t = self.softmax(g/T)
            q[indices] = g_t.data
        q = Variable(q).cuda()
        
        
        #Per tal d'aplicar la destil·lació, es guarden les probabilitats prèvies en test
        q_test = torch.zeros(len(test_set), self.n_known).cuda()
        for indices, images, labels in loader_test:
            images = Variable(images).cuda()
            indices = indices.cuda()
            g = self.forward(images)
            g_t = self.softmax(g/T)
            q_test[indices] = g_t.data
        q_test = Variable(q_test).cuda()
        
        # Increment de les neurones a la capa de sortida, adaptant-se a les classes observades
        n = len(new_classes)
        if (0 in classes):
            n = len(new_classes)-1
        self.increment_classes(n)
        self.cuda()

        # Definició de les funcions de pèrudes i optimitzador
        dist_function= self.dist_loss
        cls_function = self.cls_loss        
        optimizer = self.optimizer
        criterion = self.criterion
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)  # Definició del learning rate decay
        
        
        train_losses, classification_losses, distilation_losses = [], [], []
        eval_losses, eval_cls_losses, eval_dist_losses = [], [], []
        train_accuracies, val_accuracies = [], []
        
        train_loss = AverageMeter()
        classification_loss = AverageMeter()
        distilation_loss = AverageMeter()
        
        eval_loss = AverageMeter()
        eval_cls_loss = AverageMeter()
        eval_dist_loss = AverageMeter()
        
        train_accuracy = AverageMeter()
        val_accuracy = AverageMeter()
        
        epoch_losses = 1000      #Per tal de guardar el millor model, serveix com a loss inicial
        
        #Weight losses variable. Avís: no està implementat ja que no s'han observat els resultats esperats
        #weight_losses = 0.5
        #global weight_losses
        #weight_losses -= 0.01 
        
        #Sobre entrenament de la primera tasca. S'ha observat en les learning curves que era necessari
        if (0 in new_classes):
            for epoch in range(30):
                self.train()  
                for i, (indices, images, labels) in enumerate(loader):
                    images = Variable(images).cuda()
                    labels = Variable(labels).cuda()
                    indices = indices.cuda()
                    
                    optimizer.zero_grad()
                    g = self.forward(images)
                    loss = 0
                    # Pèrdua per classificació per noves classes
                    if (n>0):
                        loss = weight_losses * cls_function(g, labels)
    
                    # Pèrdua per destil·lació per classes antigues
                    if (self.n_known > 0):
                        q_i = q[indices]              
                        g_1 = self.softmax(g/T)    
                        loss += (1-weight_losses)*sum(criterion(g_1[:,y], q_i[:,y]) for y in range(self.n_known))                  
                    loss.backward()
                    optimizer.step()
                    
        #Pre entrenament necessari en el cas de que es vulgui implementar el deeper net2net en la classe 50
        """
        if (50 in new_classes):
            for epoch in range(50):
                self.train()  
                for i, (indices, images, labels) in enumerate(loader):
                    images = Variable(images).cuda()
                    labels = Variable(labels).cuda()
                    indices = indices.cuda()
                    
                    optimizer.zero_grad()
                    g = self.forward(images)
                    loss = 0
                    # Pèrdua per classificació per noves classes
                    #if (n>0):
                        #loss = weight_losses * cls_function(g, labels)

                    # Pèrdua per destil·lació per classes antigues
                    if (self.n_known > 0):
                        q_i = q[indices]              
                        g_1 = self.softmax(g/T)    
                        loss += (1-weight_losses)*sum(criterion(g_1[:,y], q_i[:,y]) for y in range(self.n_known))                  
                    loss.backward()
                    optimizer.step()      
        """
        
        for epoch in range(self.num_epochs):
            self.train()
            train_loss.reset()
            classification_loss.reset()
            distilation_loss.reset()
            
    
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                indices = indices.cuda()

                optimizer.zero_grad()
                g = self.forward(images) 
                
                loss = 0
                # Pèrdua per classificació per noves classes
                if (n>0):
                    #g = self.softmax(g)
                    #loss = weight_losses *sum(criterion(g[:,y], (labels==y).type(torch.cuda.FloatTensor)) for y in range(self.n_known, self.n_classes))
                    loss = weight_losses * cls_function(g, labels) 
                    classification_loss.update(loss.item(), n=len(labels))
                
                    
                
                # Pèrdua per destil·lació per classes antigues
                if self.n_known > 0:
                    g_1 = self.softmax(g/T)
                    q_i = q[indices]              
                    loss += (1-weight_losses)*sum(criterion(g_1[:,y], q_i[:,y]) for y in range(self.n_known))
                    #loss2 = -(1-weight_losses)*dist_function(g_1[:,:self.n_known], q_i)
                    dist_loss = (1-weight_losses)*sum(criterion(g_1[:,y], q_i[:,y]) for y in range(self.n_known))
                    #dist_loss = -(1-weight_losses)*dist_function(g_1[:,:self.n_known], q_i)
                    distilation_loss.update(dist_loss.item(), n=len(labels))                    
                                            
               
                loss.backward()
                optimizer.step()
 
                train_loss.update(loss.item(), n=len(labels))
                if (i+1) % 10 == 0:
                    print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                           %(epoch+1, self.num_epochs, i+1, len(dataset)//batch_size, loss.item()))
            
            
            train_losses.append(train_loss.avg)
            classification_losses.append(classification_loss.avg)
            distilation_losses.append(distilation_loss.avg)
            scheduler.step(train_loss.avg)
            
            save_load.save_model(self, path)
            
            self.eval()
            eval_loss.reset()
            eval_dist_loss.reset()
            eval_cls_loss.reset()            
            
            with torch.no_grad():
                for i, (indices, images, labels) in enumerate(loader_test):
                    images = Variable(images).cuda()
                    labels = Variable(labels).cuda()
                    indices = indices.cuda()
                    
                    
                    g = self.forward(images)
                    loss = 0
                    
                    if (n>0):
                        #g = self.softmax(g)
                        #loss = weight_losses * sum(criterion(g[:,y], (labels==y).type(torch.cuda.FloatTensor)) for y in range(self.n_known, self.n_classes))
                        loss = weight_losses * cls_function(g, labels)                    
                        eval_cls_loss.update(loss.item(), n=len(labels))
                   
                    if self.n_known > 0:
                        g_1 = self.softmax(g/T)
                        q_i_test = q_test[indices]
                        loss += (1-weight_losses)*sum(criterion(g_1[:,y], q_i_test[:,y]) for y in range(self.n_known))
                        #loss -= (1-weight_losses)*dist_function(g_1[:,:self.n_known], q_i)
                        dist_loss = (1-weight_losses)*sum(criterion(g_1[:,y], q_i_test[:,y]) for y in range(self.n_known))
                        #dist_loss = (1-weight_losses)*dist_function(g_1[:,:self.n_known], q_i)
                        eval_dist_loss.update(dist_loss.item(), n=len(labels))
                    
                    eval_loss.update(loss.item(), n=len(labels))
                        
            
            eval_losses.append(eval_loss.avg)
            eval_cls_losses.append(eval_cls_loss.avg)
            eval_dist_losses.append(eval_dist_loss.avg)            
            
            if (eval_loss.avg < epoch_losses):
                save_load.save_model(self, path2)
                epoch_losses = eval_loss.avg

        return train_losses, classification_losses, distilation_losses, eval_losses, eval_cls_losses, eval_dist_losses



class AverageMeter(object):
    """Calcula i guarda el promig i el valor actual"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        

