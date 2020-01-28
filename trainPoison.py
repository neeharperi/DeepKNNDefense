import argparse
import itertools
import os
import pdb
import sys
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from Models import *
import dataLoader
import Utilities

def evaluateModel(args, model, device, trainData, validationData, target, targetClass):
    trainAccuracy = Utilities.classificationAccuracy(model, device, trainData)
    print("Train Accuracy: " + str(trainAccuracy))
    Utilities.writeLog(args.logFileLocation, "Train Accuracy: " + str(trainAccuracy))

    testAccuracy = Utilities.classificationAccuracy(model, device, validationData)
    print("Test Accuracy: " + str(testAccuracy))
    Utilities.writeLog(args.logFileLocation, "Test Accuracy: " + str(testAccuracy))

    targetSuccess = Utilities.testTarget(model, device, target, targetClass)
    print("Poison Success on Target Image: " + str(targetSuccess))
    Utilities.writeLog(args.logFileLocation, "Poison Success on Target Image: " + str(targetSuccess))

def train(args):
    modelName = args.architecture
    checkPointDirectory = args.checkPointDirectory
    dataSplitDirectory = args.dataSplitDirectory
    poisonIndex = args.poisonIndex
    CUDA = args.CUDA
    BATCHSIZE = args.batchSize
    EPOCH = args.numEpoch
    WORKERS = args.workers
    LR = args.learningRate
    WEIGHTDECAY = args.weightDecay
    IMGSIZE = args.imageSize
    targetClass = 8 #Ships

    print("Using Parameters:")
    pprint(vars(args))

    START = 0

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    dataAugmentation = transforms.Compose([transforms.Resize((IMGSIZE, IMGSIZE)), transforms.ToTensor(), normalize])

    try:
        Utilities.writeLog(args.logFileLocation, "Model Architecture: " + str(modelName))
        Utilities.writeLog(args.logFileLocation, "Poison Index: " + str(poisonIndex))
        Utilities.writeLog(args.logFileLocation, "K Value: " + str(args.K))
        Utilities.writeLog(args.logFileLocation, "classBalance: " + str(args.classBalance))
        Utilities.writeLog(args.logFileLocation, "Replicate Imbalance: " + str(args.replicateImbalance))

        print("Poison Index: " + str(poisonIndex))
        TrainDataset = dataLoader.ConvexPolytopeFineTune_DataLoader(args, poisonIndex, dataAugmentation)
        trainData = DataLoader(TrainDataset, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)

        ValidationDataset = dataLoader.ConvexPolytopeEvaluation_DataLoader(dataSplitDirectory + "testFile.txt", dataAugmentation)
        validationData = DataLoader(ValidationDataset, batch_size=BATCHSIZE, shuffle=True, num_workers=WORKERS)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        trainableParameters = model.get_penultimate_params_list()
        optimizer = optim.Adam(trainableParameters, lr=LR, weight_decay=WEIGHTDECAY)

        CEL = nn.CrossEntropyLoss()

        if CUDA and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        CheckPoint = torch.load(checkPointDirectory + "/" + modelName + ".pth")
        model.load_state_dict(CheckPoint["net"])
        model.to(device)

        if CUDA and torch.cuda.device_count() > 1:
            trainableParameters = model.module.get_penultimate_params_list()
            optimizer = optim.Adam(trainableParameters, lr=LR, weight_decay=WEIGHTDECAY)

        model.to(device)

        for STEP in range(START, EPOCH):
            epochLoss = 0
            model.train()

            for batchCount, data in enumerate(trainData, 1):
                img = data[0].to(device)
                classID = data[1].to(device)

                classification = model(img)

                optimizer.zero_grad()
                Loss = CEL(classification, classID)
                epochLoss = epochLoss + Loss.item()

                Loss.backward()
                optimizer.step()

            print("Epoch: " + str(STEP + 1) + " of " + str(EPOCH) + " | " + "Epoch Loss: " + str(epochLoss / batchCount))

        targetFile = Utilities.parseTargetIndex(dataSplitDirectory + "targetImageFile.txt", poisonIndex)
        target = dataAugmentation(Image.open(targetFile))
        evaluateModel(args, model, device, trainData, validationData, target, targetClass)

        print("Status: Experiment Completed Successfully")
        Utilities.writeLog(args.logFileLocation, "Status: Experiment Completed Successfully")

    except Exception as e:
        print("Status: " + str(e))
        Utilities.writeLog(args.logFileLocation, "Status: " + str(e))

    print("\n")

parser = argparse.ArgumentParser()
parser.add_argument("--logFileLocation", required=True)
parser.add_argument("--architecture", required=True)
parser.add_argument("--checkPointDirectory", required=True)
parser.add_argument("--poisonIndex", required=True, type=int)
parser.add_argument("--poisonImageDirectory", required=True)
parser.add_argument("--dataSplitDirectory", required=True)
parser.add_argument("--featureDirectory", required=True)
parser.add_argument("--CUDA", dest="CUDA", action="store_true")
parser.add_argument("--CPU", dest="CUDA", action="store_false")
parser.add_argument("--replicateImbalance", dest="replicateImbalance", action="store_true")
parser.add_argument("--classBalance", default=[50, 50, 50, 50, 50, 50, 50, 50, 50, 50], nargs="+", type=int)
parser.add_argument("--batchSize", default=64, type=int)
parser.add_argument("--numEpoch", default=60, type=int)
parser.add_argument("--workers", default=0, type=int)
parser.add_argument("--learningRate", default=0.1, type=float)
parser.add_argument("--K", default=101, type=int)
parser.add_argument("--weightDecay", default=0, type=float)
parser.add_argument("--imageSize", default=32, type=int)
parser.set_defaults(CUDA=True)
parser.set_defaults(replicateImbalance=False)
args = parser.parse_args()

if __name__ == "__main__":
    train(args)

        

