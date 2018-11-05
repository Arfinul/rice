import os
import numpy as np
import time
from time import time as now
import sys
import random

import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
import torch.nn.functional as F

from models import Net


def trans_test(x):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    transformList = []
    transformList.append(transforms.ToPILImage())
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)

    transformSequence = transforms.Compose(transformList)

    y = torch.zeros_like(x)

    for i in range(x.shape[0]):
        y[i] = transformSequence(x[i])

    return y


def test(pathModel, nnClassCount, testTensor, trBatchSize):
    print("\n\n\n")
    print("Inside test funtion")

    CLASS_NAMES = ['Broken', 'Normal']
    # cudnn.benchmark = True
    model = Net()
    # model = model.cuda()
    # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
    print("Is model==None:", model is None)
    print("Is pathModel==None:", pathModel is None)
    # cudnn.benchmark = True

    # if pathModel!=None:
    #     model = Trainer.loadModel(nnArchitecture, nnClassCount, nnIsTrained)
    #     #model = torch.nn.DataParallel(model)
    #     model.to(device)

    if os.path.isfile(pathModel):
        print("=> loading checkpoint: ", pathModel)
        modelCheckpoint = torch.load(pathModel, map_location='cpu')
        model.load_state_dict(modelCheckpoint['state_dict'], strict=False)
        print("=> loaded checkpoint: ", pathModel)
    else:
        print("=> no checkpoint found: ")

    print("\n============================ Loading data into RAM ======================================== ")

    testImage = testTensor
    testSize = testImage.size()[0]

    print("============================= Evaluation of model starts ====================================")
    model.eval()

    broken = 0
    normal = 0
    batchID = 1

    with torch.no_grad():

        for i in range(0, testSize, trBatchSize):

            if (batchID % 1) == 0:
                print("batchID:" + str(batchID) + '/' + str(testImage.size()[0] / trBatchSize))

            if i + trBatchSize >= testSize:
                input = testImage[i:]
            else:
                input = testImage[i:i + trBatchSize]

            input = trans_test(input)
            input = input.type(torch.FloatTensor)

            varInput = torch.autograd.Variable(input)

            out = model(varInput)

            _, predicted = torch.max(out.data, 1)

            print(predicted)

            if i + trBatchSize <= testSize:

                for k in range(trBatchSize):
                    if (predicted[k] == 1):
                        normal += 1
                    elif (predicted[k] == 0):
                        broken += 1
            else:
                for k in range(testSize % trBatchSize):
                    if (predicted[k] == 1):
                        normal += 1
                    elif (predicted[k] == 0):
                        broken += 1

            batchID += 1

    print(' Number of broken grains in sample : ', broken)

    print(' Number of normal grains in sample :', normal)
    # a = [broken, normal]
    # return a
