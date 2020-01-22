import os
import pandas as pd
from tqdm import tqdm
import Utilities
import pdb

logFileDirectory = "./Logs/"
poisonIndexDirectory = "./CIFAR10/PoisonIndex/"
saveExcel = "./Logs/experimentSummary.xlsx"

architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]
replicateImbalance = [True, False]
classBalance = ["5", "10", "25", "50"]
KValues = ["1", "2", "5", "10", "20", "50", "100", "200"]


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

for modelName in architecture:
    allPoisonIndex = Utilities.parsePoisonIndex(poisonIndexDirectory + modelName + "_PoisonIndex.txt")
    for targetWeight in classBalance:
        for K in KValues:
            for poisonIndex in allPoisonIndex:
                for replicate in replicateImbalance:

                    experimentName = modelName + "_" + str(targetWeight) + "_" + str(K) + "_" + str(poisonIndex) + "_" + str(replicate) + ".txt"

                    if os.path.isfile(logFileDirectory + experimentName):
                        logFile = open(logFileDirectory + experimentName)

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

                        for item in logFile:
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
                                data["Replicate Imbalance"] = True if item.replace("classBalance: ", "") == "True" else False

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

                            elif "Status" in item:
                                data["Status"] = item.replace("Status: ", "")

                        for key in data.keys():
                            dataFrame[key].append(data[key])

                    else:
                        print("File Does Not Exist: " + logFileDirectory + experimentName)

pd.DataFrame.from_dict(dataFrame).to_csv(saveExcel, index=False)