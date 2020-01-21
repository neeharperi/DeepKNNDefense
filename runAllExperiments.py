import os
import Utilities
import pdb

architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]
replicateImbalance = [True, False]
classBalance = ["5", "10", "25", "50"]
KValues = ["1", "2", "5", "10", "20", "50", "100", "200", "500"]

logFileLocation = "classImbalance_ConvexPolytopePoison_kNNDefense.txt"
poisonIndexDirectory = "./CIFAR10/PoisonIndex/"
Utilities.clearLog(logFileLocation)

Utilities.writeLog(logFileLocation, "Experiment Parameters")
Utilities.writeLog(logFileLocation, "Architecture: " + str(architecture))
Utilities.writeLog(logFileLocation, "Replicate Imbalance: " + str(replicateImbalance))
Utilities.writeLog(logFileLocation, "Class Balance: " + str(classBalance))
Utilities.writeLog(logFileLocation, "K Values: " + str(KValues))

os.system("export CUDA_VISIBLE_DEVICES=0,1")
for modelName in architecture:
    allPoisonIndex = Utilities.parsePoisonIndex(poisonIndexDirectory + modelName + "_PoisonIndex.txt")
    for targetWeight in classBalance:
        for K in KValues:
            for poisonIndex in allPoisonIndex:
                for replicate in replicateImbalance:
                    if replicate:
                        os.system("python trainPoison.py --logFileLocation classImbalance_ConvexPolytopePoison_kNNDefense.txt --checkPointDirectory ./modelCheckPoints/ --dataSplitDirectory ./CIFAR10/DataSplit/ --poisonIndex {0} --poisonImageDirectory ./CIFAR10/ConvexPolytopePoisons/ --featureDirectory ./CIFAR10/Features/ --architecture {1} --replicateImbalance --classBalance 50 50 50 50 50 50 {2} 50 50 50  --K {3}".format(poisonIndex, modelName, targetWeight, K))
                    else:
                        os.system("python trainPoison.py --logFileLocation classImbalance_ConvexPolytopePoison_kNNDefense.txt --checkPointDirectory ./modelCheckPoints/ --dataSplitDirectory ./CIFAR10/DataSplit/ --poisonIndex {0} --poisonImageDirectory ./CIFAR10/ConvexPolytopePoisons/ --featureDirectory ./CIFAR10/Features/ --architecture {1} --classBalance 50 50 50 50 50 50 {2} 50 50 50  --K {3}".format(poisonIndex, modelName, targetWeight, K))
