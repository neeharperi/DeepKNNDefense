# poisoin-end-to-end-training-CIFAR10
This is a poisoning attack on the CIFAR10 data set using end-to-end training on a scaled down AlexNet architecture.

The script is super messy at this point, sorry in advance - will clean up at the earliest convinience

## Data
We have provided all but the X_train.npy data data in the data folder. The X_train.npy was over the 100mb limit. But you can download the data from the [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) website. You just need to vstack the training batches and put all of the image data for training in X_train.npy and save it in the ./Data/ directory. The format should be similar to the provided X_test.npy.

## Running the model
For random target attacks simply run the multiPoison_cifar10.py file.

