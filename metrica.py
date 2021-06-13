import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from data_loader import iCIFAR10, iCIFAR100
from model import iCaRLNet
import math



def train_accuracy(train_loader, icarl, transform_test):
    total = 10e-6
    correct = 0.0
    for indices, images, labels in train_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()

    Accuracy = 100* correct/total
    return Accuracy



def test_accuracy(test_loader, icarl, transform_test):
    total = 10e-6
    correct = 0.0
    for indices, images, labels in test_loader:
        images = Variable(images).cuda()
        preds = icarl.classify(images, transform_test)
        total += labels.size(0)
        correct += (preds.data.cpu() == labels).sum()
    return (100 * correct / total)
    