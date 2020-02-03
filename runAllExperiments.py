import os
import argparse
import random

import Utilities
import pdb

architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]
replicateImbalance = [True, False]
classBalance = ["2", "5","7", "10", "12", "15", "20", "25", "35", "50"]
KValues = ["2", "5", "10", "20", "50", "75", "90", "100", "110", "125", "200", "400"]

random.shuffle(architecture)
random.shuffle(replicateImbalance)
random.shuffle(classBalance)
random.shuffle(KValues)

logFileDirectory = "./Logs/"
poisonIndexDirectory = "./CIFAR10/PoisonIndex/"

for modelName in architecture:
    allPoisonIndex = Utilities.parsePoisonIndex(poisonIndexDirectory + modelName + "_PoisonIndex.txt")
    for targetWeight in classBalance:
        for K in KValues:
            for poisonIndex in allPoisonIndex:
                for replicate in replicateImbalance:
                    experimentDetails = modelName + "_" + str(targetWeight) + "_" + str(K) + "_" + str(poisonIndex) + "_" + str(replicate) + ".txt"

                    if experimentDetails not in os.listdir(logFileDirectory):
                        Utilities.clearLog(logFileDirectory + experimentDetails)

                        if replicate: os.system("python trainPoison.py --logFileLocation {0} --checkPointDirectory ./modelCheckPoints/ --dataSplitDirectory ./CIFAR10/DataSplit/ --poisonIndex {1} --poisonImageDirectory ./CIFAR10/ConvexPolytopePoisons/ --featureDirectory ./CIFAR10/Features/ --architecture {2} --replicateImbalance --classBalance 50 50 50 50 50 50 {3} 50 50 50  --K {4}".format(logFileDirectory + experimentDetails, poisonIndex, modelName, targetWeight, K))
                        else: os.system("python trainPoison.py --logFileLocation {0} --checkPointDirectory ./modelCheckPoints/ --dataSplitDirectory ./CIFAR10/DataSplit/ --poisonIndex {1} --poisonImageDirectory ./CIFAR10/ConvexPolytopePoisons/ --featureDirectory ./CIFAR10/Features/ --architecture {2} --classBalance 50 50 50 50 50 50 {3} 50 50 50  --K {4}".format(logFileDirectory + experimentDetails, poisonIndex, modelName, targetWeight, K))