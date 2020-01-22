import torch
import pandas as pd
from skimage.transform import warp
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

import pdb

def writeLog(logFile, line):
    log = open(logFile, "a")
    log.write(line + "\n")
    log.close()

def parsePoisonIndex(fileLocation):
    file = open(fileLocation)
    allPoisonIndex = []

    for line in file:
        allPoisonIndex.append(int(line.strip("\n")))

    return allPoisonIndex


def featureExtraction(model, device, testData):
    model.eval()

    allFeatureVector = []
    allID = []
    allFileNames = []

    with torch.no_grad():
        for img, ID, fileName in tqdm(testData):
            img = img.to(device)

            featureVector = model.module.penultimate(img)
            allFeatureVector = allFeatureVector + [vector for vector in featureVector]
            allID = allID + [i for i in ID]
            allFileNames = allFileNames + [file for file in fileName]

    return allFeatureVector, allID, allFileNames

def classificationAccuracy(model, device, testData):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for img, ID in tqdm(testData):
            img = img.to(device)
            ID = ID.to(device)

            predictedClassification = model(img)

            predictedClass = torch.max(predictedClassification, dim=1)[1]
            correct = correct + torch.sum(predictedClass == ID)
            total = total + predictedClassification.shape[0]

    return correct.item() / float(total)