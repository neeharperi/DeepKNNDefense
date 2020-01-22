import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm
from pprint import pprint
import pandas as pd
import argparse

from Models import *
import dataLoader
import Utilities
import pdb


BATCHSIZE = 512
WORKERS = 8
IMGSIZE = 32

checkPointDirectory = "./modelCheckPoints"
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
validationAugmentation = transforms.Compose([transforms.Resize((IMGSIZE, IMGSIZE)), transforms.ToTensor(), normalize])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]

for modelName in architecture:
    print("Extracting Features for " + modelName)

    if modelName == "DenseNet121":
        model = DenseNet121()
    elif modelName == "DPN92":
        model = DPN92()
    elif modelName == "GoogLeNet":
        model = GoogLeNet()
    elif modelName == "MobileNetV2":
        model = MobileNetV2()
    elif modelName == "ResNet18":
        model = ResNet18()
    elif modelName == "ResNet50":
        model = ResNet50()
    elif modelName == "ResNeXt29_2x64d":
        model = ResNeXt29_2x64d()
    elif modelName == "SENet18":
        model = SENet18()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    CheckPoint = torch.load(checkPointDirectory + "/" + modelName + ".pth")
    model.load_state_dict(CheckPoint["net"])
    model.to(device)

    files = ["./CIFAR10/DataSplit/trainFile.txt", "./CIFAR10/DataSplit/testFile.txt",
             "./CIFAR10/DataSplit/fineTuneFile.txt", "./CIFAR10/DataSplit/" + modelName + "_Poison.txt"]

    extractedFeatures = {}
    for testFile in files:
        CIFAR = dataLoader.ConvexPolytopeFeatureExtraction_DataLoader(testFile, validationAugmentation)
        testData = DataLoader(CIFAR, batch_size=BATCHSIZE, shuffle=False, num_workers=WORKERS)

        featureVector, ID, imgLocation = Utilities.featureExtraction(model, device, testData)

        for data in zip(featureVector, imgLocation):
            FV, IL = data
            extractedFeatures[IL] = FV


    torch.save(extractedFeatures, "./CIFAR10/Features/" + modelName + "_CIFAR10_Features.pth")





