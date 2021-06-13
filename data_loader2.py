from torchvision.datasets import CIFAR10
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

class iCIFAR10(CIFAR10):
    #Inicialització de la classe amb els paràmetres necessaris
    def __init__(self, root,
                 train=True,
                 classes = range(100),
                 transform=None,
                 target_transform=None,
                 download=False):
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        #Selecció del subconjunt de classes
        if self.train:
            train_data = []
            train_labels = []

            if (type(classes) is int):
              for i in range(len(self.data)):
                  if self.targets[i] == classes:
                      train_data.append(self.data[i])
                      train_labels.append(self.targets[i])
            else:
              for i in range(len(self.data)):
                  if self.targets[i] in classes:
                      train_data.append(self.data[i])
                      train_labels.append(self.targets[i])
        
            self.data = np.array(train_data)
            self.targets = train_labels

        
        else:
            test_data = []
            test_labels = []
            
            if (type(classes) is int):
                for i in range(len(self.data)):
                    if (self.targets[i] == classes):
                        test_data.append(self.data[i])
                        test_labels.append(self.targets[i])
            else:
                for i in range(len(self.data)):
                    if self.targets[i] in classes:
                        test_data.append(self.data[i])
                        test_labels.append(self.targets[i])

            self.data = np.array(test_data)
            self.targets = test_labels
        
    
    
    def __getitem__ (self, index):        
    #Retorna l'índex, la imatge i el label
    #Args: 
    #    index: índex amb el qual les dades estan ordenades
    #Return
    #    index: índex amb el qual les dades estan ordenades
    #    img: dades amb aquell índex
    #    target: etiqueta de la imatge amb aquell índex
    
    
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)   
        if self.transform is not None:      
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return index, img, target
    
    def __len__(self):
        if self.train:
            return len(self.data)
        else:
            return len(self.data)

    def get_image_class(self, label):
    #Retorna un array amb totes les dades amb aquella label
    #Args: 
    #    label: etiquetat
    #Return:
    #    Les dades amb aquell etiquetat
    
        return self.data[np.array(self.targets) == label]       
        

    def append(self, images, labels):
        """Afegeix el dataset amb les imatges i labels pàsades com a paràmetres
        Args:
            images: Tensor de tamany (N, C, H, W)
            labels: llista d'etiquetats
        """

        self.data = np.concatenate((self.data, images), axis=0)
        self.targets = self.targets + labels 
    

# Per treballar amb aquesta base de dades simplement es canvia la url i l'origne d'on agafàvem les dades
class iCIFAR100(iCIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }