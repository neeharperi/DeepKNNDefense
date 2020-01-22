import os

os.mkdir("./CIFAR10/ConvexPolytopePoisons")
os.mkdir("./CIFAR10/DataSplit")
os.mkdir("./CIFAR10/Features")
os.mkdir("./CIFAR10/TestSplit")
os.mkdir("./CIFAR10/TrainSplit")

os.system("python ./CIFAR_Download/parseCIFAR.py")
os.system("python ./CIFAR_Download/parsePoison.py")
os.system("python ./CIFAR_Download/createDataSplit.py")
os.system("python ./extractFeatures.py")