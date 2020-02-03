import os
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from PIL import Image
from tqdm import tqdm
import math
import copy

import Utilities
import pdb

class ConvexPolytopeFineTune_DataLoader(Dataset):
    def __init__(self, args, poisonIndex, transform=None):
        self.fineTuneFile = open(args.dataSplitDirectory + "fineTuneFile.txt")
        self.poisonFile = open(args.dataSplitDirectory + args.architecture + "_Poison.txt")
        self.poisonIndex = poisonIndex
        self.transform = transform
        self.classBalance = copy.deepcopy(args.classBalance)
        self.examples = copy.deepcopy(args.classBalance)
        self.K = args.K
        self.replicateImbalance = args.replicateImbalance
        self.extractedFeatures = torch.load(args.featureDirectory + args.architecture + "_CIFAR10_Features.pth")

        self.imageFiles = {}
        self.balancedImages = []
        self.filteredImages = []

        self.convexPolytopePoison = []

        self.addIndex = set()

        for line in self.poisonFile:
            imgLocation, ID = line.split()
            index = imgLocation.split("/")[-1]
            ID = int(ID.strip("\n"))

            if ID not in self.imageFiles.keys():
                self.imageFiles[ID] = []

            if args.poisonImageDirectory + str(self.poisonIndex) in imgLocation:
                self.addIndex.add(index)
                self.imageFiles[ID].append((imgLocation, ID))
                self.classBalance[ID] = self.classBalance[ID] - 1
                self.convexPolytopePoison.append(imgLocation)

        for line in self.fineTuneFile:
            imgLocation, ID = line.split()
            index = imgLocation.split("/")[-1]
            ID = int(ID.strip("\n"))

            if ID not in self.imageFiles.keys():
                self.imageFiles[ID] = []

            if index not in self.addIndex and self.classBalance[ID] > 0:
                self.addIndex.add(index)
                self.imageFiles[ID].append((imgLocation, ID))
                self.classBalance[ID] = self.classBalance[ID] - 1

        if self.replicateImbalance:
            maxClass = max(self.examples)
            classWeight = []

            for i in self.examples:
                classWeight.append(math.ceil(maxClass / i))

            for key in self.imageFiles:
                self.balancedImages = self.balancedImages + (classWeight[key] * self.imageFiles[key])[0:maxClass]
        else:
            for key in self.imageFiles:
                self.balancedImages = self.balancedImages + self.imageFiles[key]

        KNN = KNeighborsClassifier(algorithm='brute', n_neighbors=self.K)

        trainFeatures = []
        trainLabels = []

        for data in self.balancedImages:
            imgLocation, ID = data
            FV = self.extractedFeatures[imgLocation].cpu().numpy()
            trainFeatures.append(FV)
            trainLabels.append(ID)

        KNN.fit(trainFeatures, trainLabels)
        KNNLabels = KNN.predict(trainFeatures)

        cleanImages = np.equal(KNNLabels, trainLabels)

        TP, FP, TN, FN = 0, 0, 0, 0

        for data, valid in zip(self.balancedImages, cleanImages):
            imgLocation, ID = data
            if valid:
                self.filteredImages.append((imgLocation, ID))
                if imgLocation not in self.convexPolytopePoison:
                    TP = TP + 1
                else:
                    FP = FP + 1
            else:
                if imgLocation in self.convexPolytopePoison:
                    TN = TN + 1
                else:
                    FN = FN + 1

        try: MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        except: MCC = None

        print("True Positive: " + str(TP) + " | " + "True Negative: " + str(TN) + " | " + "False Positive: " + str(FP) + " | " + "False Negative: " + str(FN))
        print("Matthews Correlation Coefficient: " + str(MCC))

        Utilities.writeLog(args.logFileLocation, "True Positive: " + str(TP) + " | " + "True Negative: " + str(TN) + " | " + "False Positive: " + str(FP) + " | " + "False Negative: " + str(FN))
        Utilities.writeLog(args.logFileLocation, "Matthews Correlation Coefficient: " + str(MCC))

    def __len__(self):
        return len(self.filteredImages)

    def __getitem__(self, index):
        imgLocation, classID = self.filteredImages[index]
        img = Image.open(imgLocation)

        if self.transform:
            img = self.transform(img)

        return img, classID

class ConvexPolytopeEvaluation_DataLoader(Dataset):
    def __init__(self, imgLocationFile, transform=None):
        self.imgLocationFile = open(imgLocationFile)
        self.transform = transform
        self.imageFiles = []

        for line in self.imgLocationFile:
            imgLocation, classID = line.strip("\n").split()
            self.imageFiles.append((imgLocation, int(classID)))

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        imgLocation, classID = self.imageFiles[index]
        img = Image.open(imgLocation)

        if self.transform:
            img = self.transform(img)

        return img, classID

class ConvexPolytopeFeatureExtraction_DataLoader(Dataset):
    def __init__(self, imgLocationFile, transform=None):
        self.imgLocationFile = open(imgLocationFile)
        self.transform = transform
        self.imageFiles = []

        for line in self.imgLocationFile:
            imgLocation, classID = line.strip("\n").split()
            self.imageFiles.append((imgLocation, int(classID)))

    def __len__(self):
        return len(self.imageFiles)

    def __getitem__(self, index):
        imgLocation, classID = self.imageFiles[index]
        img = Image.open(imgLocation)

        if self.transform:
            img = self.transform(img)

        return img, classID, imgLocation