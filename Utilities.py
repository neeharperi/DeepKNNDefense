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
            "True Positive Rate" : None,
            "True Negative Rate" : None,
            "False Positive Rate" : None,
            "False Negative Rate" : None,
            "Negative Predictive Value" : None,
            "False Discovery Rate" : None,
            "False Omission Rate" : None,
            "Critical Success Index" : None,
            "Matthews Correlation Coefficient" : None,
            "Precision" : None,
            "Recall" : None,
            "F1" : None,
            "Train Accuracy" : None,
            "Test Accuracy" : None,
            "Poisoning Successful on Target Image" : None,
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
            if len(metrics) == 4:
                data["True Positive"] = int(metrics[0].replace("True Positive: ", ""))
                data["True Negative"] = int(metrics[1].replace("True Negative: ", ""))
                data["False Positive"] = int(metrics[2].replace("False Positive: ", ""))
                data["False Negative"] = int(metrics[3].replace("False Negative: ", ""))
            elif len(metrics) == 3:
                data["Precision"] = float(metrics[0].replace("Precision: ", ""))
                data["Recall"] = float(metrics[1].replace("Recall: ", ""))
                data["F1"] = float(metrics[2].replace("F1: ", ""))

        elif "Train Accuracy" in item:
            data["Train Accuracy"] = float(item.replace("Train Accuracy: ", ""))

        elif "Test Accuracy" in item:
            data["Test Accuracy"] = float(item.replace("Test Accuracy: ", ""))

        elif "Poisoning Successful on Target Image" in item:
            data["Poisoning Successful on Target Image"] = True if item.replace("Poisoning Successful on Target Image: ", "") == "True" else False

        elif "Status" in item:
            data["Status"] = item.replace("Status: ", "")


    try: data["True Positive Rate"] = data["True Positive"] / float(data["True Positive"] + data["False Negative"])
    except: data["True Positive Rate"] = None

    try: data["True Negative Rate"] = data["True Negative"] / float(data["True Negative"] + data["False Positive"])
    except: data["True Negative Rate"] = None

    try: data["False Positive Rate"] = 1 - data["True Negative Rate"]
    except: data["False Positive Rate"] = None

    try: data["False Negative Rate"] = 1 - data["True Positive Rate"]
    except: data["False Negative Rate"] = None

    try: data["Negative Predictive Value"] = data["True Negative"] / float(data["True Negative"] + data["False Negative"])
    except: data["Negative Predictive Value"] = None

    try: data["False Discovery Rate"] = 1 - data["Precision"]
    except: data["False Discovery Rate"] = None

    try: data["False Omission Rate"] = 1 - data["Negative Predictive Value"]
    except: data["False Omission Rate"] = None

    try: data["Critical Success Index"] = data["True Positive"] / float(data["True Positive"] + data["False Negative"] + data["False Positive"])
    except: data["Critical Success Index"] = None

    try: data["Matthews Correlation Coefficient"] = ((data["True Positive"] * data["True Negative"]) - (data["False Positive"] * data["False Negative"])) / np.sqrt((data["True Positive"] + data["False Positive"]) * (data["True Positive"] + data["False Negative"]) * (data["True Negative"] + data["False Positive"]) * (data["True Negative"] + data["False Negative"]))
    except: data["Matthews Correlation Coefficient"] = None

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