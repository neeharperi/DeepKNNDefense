# Deep-KNNDefense
Deep KNN Defense Against Clean Label Poisoning Attacks

## Getting Started
Install required python packages in the requirements.txt file before continuing. Next, set up the environment by executing:

```bash
python prepareData.py
```

concatenate all modelCheckPoints0X.tar into a single file and extract the combined tarball into a folder "modelCheckPoints"

```bash
cat modelCheckPointParition* > modelCheckPoints.tar.gz
tar -xzf modelCheckPoints.tar.gz
```

NOTE: This code was written on a multi-GPU setup with nn.DataParallel. You may need to manually adjust the keys of the state dictionaries to load them on your machine.

## Running All Experiments
In order to use the pre-trained model files, stitch the modelCheckPointXX files and unzip it into the same root directory. The KNN defense is implemented directly in the dataloader. To test the performance of the KNN defense against pre-generated convex polytope adversarial examples, run:

```bash
python runAllExperiments.py
```

If you have sufficient hardware, you can run multiple instances of runAllExperiments.py to speed up evaluation on the parameter sweep. Note that there are no guarantees that experiments will complete in a certain order. If using multiple instances of runAllExperiments.py, make sure run deleteAllIncompleteExperiments.py after the script has finished executing on all instances and re-run runAllExperiments.py on a single instance to prevent data-race conditions.

## Export Log to Excel
After running all experiments, export the results into an excel file by executing:

```bash
python parseExperimentLog.py
```

## Generate Figures
Create plots showing summary statistics comparing the effect of different K values and class balance strategies on several metrics by running
```bash
python generateFigures.py
```
