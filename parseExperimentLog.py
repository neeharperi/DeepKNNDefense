import os
import pandas as pd
from tqdm import tqdm
import pdb

logFileLocation = "classImbalance_ConvexPolytopePoison_kNNDefense.txt"
saveExcel = "classImbalance_ExperimentSummary.xlsx"
logFile = open(logFileLocation).read()
experiments = logFile.split("\n\n\n")[1:]

dataFrame = {"Model Architecture" : [],
             "Poison Index" : [],
             "K Value" : [],
             "Class Balance" : [],
             "Replicate Imbalance" : [],
             "True Positive" : [],
             "True Negative" : [],
             "False Positive" : [],
             "False Negative" : [],
             "Precision" : [],
             "Recall" : [],
             "F1" : [],
             "Train Accuracy" : [],
             "Test Accuracy" : [],
             "Status" : []}

for experiment in tqdm(experiments):
    items = experiment.split("\n")

    data = {"Model Architecture" : None,
            "Poison Index" : None,
            "K Value" : None,
            "Class Balance" : None,
            "Replicate Imbalance" : None,
            "True Positive" : None,
            "True Negative" : None,
            "False Positive" : None,
            "False Negative" : None,
            "Precision" : None,
            "Recall" : None,
            "F1" : None,
            "Train Accuracy" : None,
            "Test Accuracy" : None,
            "Status" : None}

    for item in items:
        if "Model Architecture" in item:
            data["Model Architecture"] = item.strip("Model Architecture: ")

        elif "Poison Index" in item:
            data["Poison Index"] = int(item.strip("Poison Index: "))

        elif "K Value" in item:
            data["K Value"] = int(item.strip("K Value: "))

        elif "classBalance" in item:
            data["Class Balance"] = item.strip("classBalance: ")

        elif "Replicate Imbalance" in item:
            data["Replicate Imbalance"] = True if item.strip("classBalance: ") == "True" else False

        elif "|" in item:
            metrics = item.split("|")
            if len(metrics) == 4:
                data["True Positive"] = int(metrics[0].strip("True Positive: "))
                data["True Negative"] = int(metrics[1].strip("True Negative: "))
                data["False Positive"] = int(metrics[2].strip("False Positive: "))
                data["False Negative"] = int(metrics[3].strip("False Negative: "))
            elif len(metrics) == 3:
                data["Precision"] = float(metrics[0].strip("Precision: "))
                data["Recall"] = float(metrics[1].strip("Recall: "))
                data["F1"] = float(metrics[2].strip("F1: "))

        elif "Train Accuracy" in item:
            data["Train Accuracy"] = float(item.strip("Train Accuracy: "))

        elif "Test Accuracy" in item:
            data["Test Accuracy"] = float(item.strip("Test Accuracy: "))

        elif "Status" in item:
            data["Status"] = item.strip("Status: ")

        else:
            pass

    for key in data.keys():
        dataFrame[key].append(data[key])

pd.DataFrame.from_dict(dataFrame).to_csv(saveExcel)
