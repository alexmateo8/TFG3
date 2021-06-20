# TFG3

El principal objectiu del projecte consisteix en l'estudi teòric, comprensió i desenvolupament d’un algoritme basat en la tècnica de “machine learning” anomenada aprenentatge incremental la qual pot oferir un conjunt ampli de possibilitats en el món del “machine learning”.  
Aquest GitHub mostra el codi utilitzat durant el desenvolupament del projecte. El codi proporcionat proporcionen els millors resultats presentats en la tesis. També hi ha diferents opcions d'implementació en algunes seccions de l'algoritme, permet a l'usuari provar diferents opcions que l'autor ha rebutjat per millor funcionament d'unes altres.

Experiments realitzats amb CIFAR 10 i CIFAR 100.

L'algoritme està dividit en diverses parts. Depenent de la base de dades amb la qual es vol realitzar l'experiment:
  - Per realitzar un experiment amb CIFAR 10, executar el fitxer de phython main
  - Per realitzar un experiment amb CIFAR 100, executar el fitxer de phython main_cifar_100

El desenvolupament de l'algoritme està implementat en la secció iCaRL.
Finalment es poden trobar altres fitxers com poden ser data_loader que permet carregar les bases de dades. Metrica proporciona els resultats de precisió i plots per guardar les corbes d'aprenentatge i els gràfics de precisió. A més, save_load permet emmagatzemar el millor model i pretrained_resnet permet carregar la ResNet 50 preentrenada amb ImageNet.

