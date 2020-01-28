import torch
import pandas as pd
from skimage.transform import warp
import torch
import torch.nn as nn

import numpy as np
from tqdm import tqdm

import pdb

def parseLogFile(logFile):
    log = open(logFile)

    data = {"Model Architecture" : None,
            "Poison Index" : None,
            "K Value" : None,
            "Class Balance" : None,
            "Replicate Imbalance" : None,
            "True Positive" : None,
            "True Negative" : None,
            "False Positive" : None,
            "False Negative" : None,
            "Matthews Correlation Coefficient" : None,
            "Train Accuracy" : None,
            "Test Accuracy" : None,
            "Poison Success on Target Image" : None,
            "Status" : None}

    for item in log:
        item = item.strip("\n")

        if "Model Architecture" in item:
            data["Model Architecture"] = item.replace("Model Architecture: ", "")

        elif "Poison Index" in item:
            data["Poison Index"] = int(item.replace("Poison Index: ", ""))

        elif "K Value" in item:
            data["K Value"] = int(item.replace("K Value: ", ""),)

        elif "classBalance" in item:
            data["Class Balance"] = item.replace("classBalance: ", "")

        elif "Replicate Imbalance" in item:
            data["Replicate Imbalance"] = True if item.replace("Replicate Imbalance: ", "") == "True" else False

        elif "|" in item:
            metrics = item.split("|")
            data["True Positive"] = int(metrics[0].replace("True Positive: ", ""))
            data["True Negative"] = int(metrics[1].replace("True Negative: ", ""))
            data["False Positive"] = int(metrics[2].replace("False Positive: ", ""))
            data["False Negative"] = int(metrics[3].replace("False Negative: ", ""))

        elif "Train Accuracy" in item:
            data["Train Accuracy"] = float(item.replace("Train Accuracy: ", ""))

        elif "Test Accuracy" in item:
            data["Test Accuracy"] = float(item.replace("Test Accuracy: ", ""))

        elif "Poison Success on Target Image" in item:
            data["Poison Success on Target Image"] = True if item.replace("Poison Success on Target Image: ", "") == "True" else False

        elif "Matthews Correlation Coefficient" in item:
            data["Matthews Correlation Coefficient"] = float(item.replace("Matthews Correlation Coefficient: ", ""))

        elif "Status" in item:
            data["Status"] = item.replace("Status: ", "")

    return data

def writeLog(logFile, line):
    log = open(logFile, "a")
    log.write(line + "\n")
    log.close()

def clearLog(logFile):
    log = open(logFile, "w")
    log.close()

def parsePoisonIndex(fileLocation):
    file = open(fileLocation)
    allPoisonIndex = []

    for line in file:
        allPoisonIndex.append(int(line.strip("\n")))

    return allPoisonIndex

def parseTargetIndex(fileLocation, poisonIndex):
    file = open(fileLocation)

    for line in file:
        imgLocation, targetID = line.split()

        if int(targetID) == poisonIndex:
            return imgLocation

def testTarget(model, device, target, targetClass):
    model.eval()

    classification = model(target.unsqueeze(0).to(device))
    index = torch.argmax(classification).item()

    if index == targetClass:
        return True

    return False

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