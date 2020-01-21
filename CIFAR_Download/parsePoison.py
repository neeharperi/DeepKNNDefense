import os
import torch
import torchvision.transforms as transforms
from PIL import Image

import pdb

class NormalizeInverse(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

normalizeInverse = transforms.Compose([NormalizeInverse(MEAN, STD), transforms.ToPILImage()])

rootDirectory = "./CIFAR10/GeneratedPoisons/"
poison = [str(i) for i in range(0,50)]

for i in poison:
    pkl = torch.load(rootDirectory + "/" + i + "/" + "poison_tuple_list.pth")
    imgs, indicies = pkl

    for data in zip(imgs, indicies):
        img, ID = data[0]
        index = int(data[1] + 48000)

        if not os.path.isdir("./CIFAR10/ConvexPolytopePoisons/" + i):
            os.mkdir("./CIFAR10/ConvexPolytopePoisons/" + i)

        if not os.path.isdir("./CIFAR10/ConvexPolytopePoisons/" + i + "/" + str(ID)):
            os.mkdir("./CIFAR10/ConvexPolytopePoisons/" + i + "/" + str(ID))

        img = normalizeInverse(img)
        img.save("./CIFAR10/ConvexPolytopePoisons/" + i + "/" + str(ID) + "/" + str(index) + ".png")
