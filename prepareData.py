import os

os.system("cat modelCheckPointPartition* | tar xzpvf -")

os.mkdir("./CIFAR10/ConvexPolytopePoisons")
os.mkdir("./CIFAR10/DataSplit")
os.mkdir("./CIFAR10/Features")
os.mkdir("./CIFAR10/TestSplit")
os.mkdir("./CIFAR10/TrainSplit")

print("Extracting CIFAR10 Dataset")
os.system("python ./CIFAR_Download/parseCIFAR.py")
print("Parsing Convex Polytope Poisons")
os.system("python ./CIFAR_Download/parsePoison.py")
print("Creating Data Split Files")
os.system("python ./CIFAR_Download/createDataSplit.py")
print("Extracting Features from Pre-trained Models")
os.system("python ./extractFeatures.py")