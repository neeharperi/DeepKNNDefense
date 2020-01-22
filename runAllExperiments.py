import os
import Utilities
import argparse
import pdb


parser = argparse.ArgumentParser()
parser.add_argument("--restartExperiments", default=True, action="store_false")
args = parser.parse_args()

resumeExperiments = args.restartExperiments
architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]
replicateImbalance = [True, False]
classBalance = ["5", "10", "25", "50"]
KValues = ["1", "2", "5", "10", "20", "50", "100", "200"]

logFileLocation = "/fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/classImbalance_ConvexPolytopePoison_kNNDefense.txt"
completedExperimentsLocation = "/fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/completedExperiments.txt"
poisonIndexDirectory = "/fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/CIFAR10/PoisonIndex/"

completedExperiments = set()
if not resumeExperiments:
    Utilities.clearLog(logFileLocation)
    Utilities.clearLog(completedExperimentsLocation)

    Utilities.writeLog(logFileLocation, "Experiment Parameters")
    Utilities.writeLog(logFileLocation, "Architecture: " + str(architecture))
    Utilities.writeLog(logFileLocation, "Replicate Imbalance: " + str(replicateImbalance))
    Utilities.writeLog(logFileLocation, "Class Balance: " + str(classBalance))
    Utilities.writeLog(logFileLocation, "K Values: " + str(KValues))
else:
    completedExperiments = Utilities.loadExperimentCheckPoint(completedExperimentsLocation)

for modelName in architecture:
    allPoisonIndex = Utilities.parsePoisonIndex(poisonIndexDirectory + modelName + "_PoisonIndex.txt")
    for targetWeight in classBalance:
        for K in KValues:
            for poisonIndex in allPoisonIndex:
                for replicate in replicateImbalance:
                    experimentDetails = modelName + ", " + str(targetWeight) + ", " + str(K) + ", " + str(poisonIndex) + ", " + str(replicate)
                    if experimentDetails not in completedExperiments:

                        if replicate:
                            os.system("python trainPoison.py --logFileLocation /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/classImbalance_ConvexPolytopePoison_kNNDefense.txt --checkPointDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/modelCheckPoints/ --dataSplitDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/CIFAR10/DataSplit/ --poisonIndex {0} --poisonImageDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/CIFAR10/ConvexPolytopePoisons/ --featureDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/CIFAR10/Features/ --architecture {1} --replicateImbalance --classBalance 50 50 50 50 50 50 {2} 50 50 50  --K {3}".format(poisonIndex, modelName, targetWeight, K))
                        else:
                            os.system("python trainPoison.py --logFileLocation /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/classImbalance_ConvexPolytopePoison_kNNDefense.txt --checkPointDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/modelCheckPoints/ --dataSplitDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/CIFAR10/DataSplit/ --poisonIndex {0} --poisonImageDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/CIFAR10/ConvexPolytopePoisons/ --featureDirectory /fs/diva-scratch/peri/CleanLabelPoisons/kNN_Class_Imbalance/CIFAR10/Features/ --architecture {1} --classBalance 50 50 50 50 50 50 {2} 50 50 50  --K {3}".format(poisonIndex, modelName, targetWeight, K))

                        completedExperiments.add(experimentDetails)
                        Utilities.writeLog(completedExperimentsLocation, experimentDetails)