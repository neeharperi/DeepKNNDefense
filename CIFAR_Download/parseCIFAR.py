import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm
import pdb

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


indexMultiplier = [0, 1, 2, 3, 4]
CIFARTrain = ["./CIFAR_Download/data_batch_1",
              "./CIFAR_Download/data_batch_2",
              "./CIFAR_Download/data_batch_3",
              "./CIFAR_Download/data_batch_4",
              "./CIFAR_Download/data_batch_5"]
CIFARTest = ["./CIFAR_Download/test_batch"]

for multiplier, path in zip(indexMultiplier, CIFARTrain):
    file = unpickle(path)

    for index, data in tqdm(enumerate(zip(file['data'], file['labels'])), total=1e4):
        img, label = data

        red = img[0:1024].reshape((32, 32))
        green = img[1024:2048].reshape((32, 32))
        blue = img[2048:3072].reshape((32, 32))

        img = np.stack((blue, green, red), axis=2)

        if not os.path.isdir("./CIFAR10/TrainSplit/" + str(label)):
            os.mkdir("./CIFAR10/TrainSplit/" + str(label))

        cv2.imwrite("./CIFAR10/TrainSplit/" + str(label) + "/" + str(int(index + 1e4 * multiplier)) + ".png", img)

for path in CIFARTest:
    file = unpickle(path)

    for index, data in tqdm(enumerate(zip(file['data'], file['labels'])), total=1e4):
        img, label = data

        red = img[0:1024].reshape((32, 32))
        green = img[1024:2048].reshape((32, 32))
        blue = img[2048:3072].reshape((32, 32))

        img = np.stack((blue, green, red), axis=2)

        if not os.path.isdir("./CIFAR10/TestSplit/" + str(label)):
            os.mkdir("./CIFAR10/TestSplit/" + str(label))

        cv2.imwrite("./CIFAR10/TestSplit/" + str(label) + "/" + str(int(index)) + ".png", img)
