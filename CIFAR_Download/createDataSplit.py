import os
import pdb

trainDirectory = "./CIFAR10/TrainSplit"
trainFile = open("./CIFAR10/DataSplit/trainFile.txt", "w")
fineTuneFile = open("./CIFAR10/DataSplit/fineTuneFile.txt", "w")
targetImageFile = open("./CIFAR10/DataSplit/targetImageFile.txt", "w")
targetClass = "6"

testDirectory = "./CIFAR10/TestSplit"
testFile = open("./CIFAR10/DataSplit/testFile.txt", "w")

for ID in os.listdir(trainDirectory):
    imgList = sorted([int(file.split(".")[0]) for file in os.listdir(trainDirectory + "/" + ID)])
    train = imgList[0:4800]
    fineTune = imgList[4800:4850]

    for img in train:
        trainFile.write(trainDirectory + "/" + ID + "/" + str(img) + ".png" + " " + ID + "\n")

    for img in fineTune:
        fineTuneFile.write(trainDirectory + "/" + ID + "/" + str(img) + ".png" + " " + ID + "\n")

    if ID == targetClass:
        target = imgList[4850:4900]
        for i, img in enumerate(target):
            targetImageFile.write(trainDirectory + "/" + ID + "/" + str(img) + ".png" + " " + str(i) + "\n")

for ID in os.listdir(testDirectory):
    for img in os.listdir(testDirectory + "/" + ID):
        testFile.write(testDirectory + "/" + ID + "/" + str(img) + " " + ID + "\n")


poisonDirectory = "./CIFAR10/PoisonIndex/"
architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]


for model in architecture:
    successfulPoisons = open(poisonDirectory + model + "_PoisonIndex.txt")
    poisonFile = open("./CIFAR10/DataSplit/" + model + "_Poison.txt", "w")

    for line in successfulPoisons:
        i = line.strip("\n")

        for img in os.listdir("./CIFAR10/ConvexPolytopePoisons/" + i + "/8/"):
            poisonFile.write("./CIFAR10/ConvexPolytopePoisons/" + i + "/8/" + img + " " + "8" + "\n")

    poisonFile.close()
           

trainFile.close()
fineTuneFile.close()
testFile.close()