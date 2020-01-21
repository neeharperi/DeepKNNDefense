# KNNDefense
Strong Baseline Defenses Against Clean Label Poisoning Attacks

## Getting Started
Install required python packages in the requirements.txt file before continuing. Next, set up the environment by executing:

```bash
python prepareData.py
```

## Running All Experiments
The KNN defense is implemented directly in the dataloader. To test the performance of the KNN defense against pre-generated convex polytope adversarial examples, run:

```bash
python runAllExperiments.py
```

## Export Log to Excel
After running all experiments, export the results into an excel file by executing:

```bash
python parseExperimentLog.py
```
