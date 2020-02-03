import os
import pandas as pd
from tqdm import tqdm
import Utilities
import pdb

logFileDirectory = "./Logs/"
poisonIndexDirectory = "./CIFAR10/PoisonIndex/"
saveExcel = "experimentSummary.xlsx"

architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]
replicateImbalance = [True, False]
classBalance = ["2", "5", "7", "10", "12", "15", "20", "25", "35", "50"]
KValues = ["2", "5", "10", "20", "50", "75", "90", "100", "110", "125", "200", "400"]

dataFrame = {"Model Architecture" : [],
             "Poison Index" : [],
             "K Value" : [],
             "Class Balance" : [],
             "Replicate Imbalance" : [],
             "True Positive" : [],
             "True Negative" : [],
             "False Positive" : [],
             "False Negative" : [],
             "Train Accuracy" : [],
             "Test Accuracy" : [],
             "Matthews Correlation Coefficient" : [],
             "Poison Success on Target Image" : [],
             "Status" : []}

for modelName in architecture:
    allPoisonIndex = Utilities.parsePoisonIndex(poisonIndexDirectory + modelName + "_PoisonIndex.txt")
    for targetWeight in classBalance:
        for K in KValues:
            for poisonIndex in allPoisonIndex:
                for replicate in replicateImbalance:

                    experimentName = modelName + "_" + str(targetWeight) + "_" + str(K) + "_" + str(poisonIndex) + "_" + str(replicate) + ".txt"

                    if os.path.isfile(logFileDirectory + experimentName):
                        data = Utilities.parseLogFile(logFileDirectory + experimentName)

                        for key in data.keys():
                            dataFrame[key].append(data[key])

                    else:
                        print("File Does Not Exist: " + logFileDirectory + experimentName)

pd.DataFrame.from_dict(dataFrame).to_excel(saveExcel, index=False)