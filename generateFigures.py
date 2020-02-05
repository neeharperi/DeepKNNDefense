import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pdb

figureDirectory = "./Figures/"
excelFile = "experimentSummary.xlsx"
dataFrame = pd.read_excel(excelFile)

if not os.path.isdir(figureDirectory):
    os.mkdir(figureDirectory)

architecture = ["DenseNet121", "DPN92", "GoogLeNet", "MobileNetV2", "ResNet18", "ResNet50", "ResNeXt29_2x64d", "SENet18"]
replicateImbalance = [True, False]
classBalance = [2, 5, 7, 10, 12, 15, 20, 25, 35, 50]
KValues = [2, 5, 10, 20, 50, 75, 90, 100, 110, 125, 200, 400]

fixedClassBalance = 50
fixedK = 100

for rowTable in dataFrame.iterrows():
    CB = rowTable[1]["Class Balance"]
    targetWeight = CB.replace("[50, 50, 50, 50, 50, 50, ", "").replace(", 50, 50, 50]", "")
    dataFrame.at[rowTable[0], "Class Balance"] = int(targetWeight)

#Q1: (Fixed Class Balance) Is the effectiveness of KNN Defense Model Specific?
Q1_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["Model Architecture", "K Value", "Poison Success on Target Image"]]
Q1_Statistics = {"Model Architecture" : [],
                 "Normalized-K Ratio" : [],
                 "Attack Success Rate" : []}

for modelName in architecture:
    for K in KValues:
        Q1_Statistics["Model Architecture"].append(modelName)
        Q1_Statistics["Normalized-K Ratio"].append((K / fixedClassBalance))
        tempDataFrame = Q1_dataFrame[(Q1_dataFrame["Model Architecture"] == modelName) & (Q1_dataFrame["K Value"] == K)]
        Q1_Statistics["Attack Success Rate"].append(np.sum(tempDataFrame["Poison Success on Target Image"] == True) / float(tempDataFrame.shape[0]))

Q1_Plot = pd.DataFrame.from_dict(Q1_Statistics)
ax = sns.pointplot(x="Normalized-K Ratio", y="Attack Success Rate", hue="Model Architecture", data=Q1_Plot)
ax.legend(loc='upper right', ncol=2)
plt.title("Transfer Convex Polytope Attack Success (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_ModelAgnostic.pdf")
plt.clf()

#Q2: (Fixed Class Balance) Matthews Correlation Coefficient of All Models?
Q2_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "Matthews Correlation Coefficient"]]
Q2_Statistics = {"Model Architecture" : [],
                 "Normalized-K Ratio" : [],
                 "Matthews Correlation Coefficient" : []}

for modelName in architecture:
    for K in KValues:
        Q2_Statistics["Model Architecture"].append(modelName)
        Q2_Statistics["Normalized-K Ratio"].append((K / fixedClassBalance))
        tempDataFrame = Q2_dataFrame[(Q2_dataFrame["Model Architecture"] == modelName) & (Q2_dataFrame["K Value"] == K)]
        Q2_Statistics["Matthews Correlation Coefficient"].append(np.mean(tempDataFrame["Matthews Correlation Coefficient"]))


Q2_Plot = pd.DataFrame(Q2_Statistics)
ax = sns.pointplot(x="Normalized-K Ratio", y="Matthews Correlation Coefficient", hue="Model Architecture", ci=None, data=Q2_Plot)
ax.get_legend().set_visible(False)
plt.title("Deep-KNN Poison Filtering Success (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_CorrelationCoefficient.pdf")
plt.clf()

#Q3: (Fixed Class Balance) Test Accuracy of All Models?
Q3_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "Test Accuracy"]]
Q3_Statistics = {"Model Architecture" : [],
                 "Normalized-K Ratio" : [],
                 "Test Accuracy" : []}

for modelName in architecture:
    for K in KValues:
        Q3_Statistics["Model Architecture"].append(modelName)
        Q3_Statistics["Normalized-K Ratio"].append((K / fixedClassBalance))
        tempDataFrame = Q3_dataFrame[(Q3_dataFrame["Model Architecture"] == modelName) & (Q3_dataFrame["K Value"] == K)]
        Q3_Statistics["Test Accuracy"].append(np.mean(tempDataFrame["Test Accuracy"]))

Q3_Plot = pd.DataFrame(Q3_Statistics)
ax = sns.pointplot(x="Normalized-K Ratio", y="Test Accuracy", hue="Model Architecture", ci=None, data=Q3_Plot)
ax.get_legend().set_visible(False)
plt.title("Deep-KNN Test Accuracy (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_TestAccuracy.pdf")
plt.clf()

#Q4: (Fixed K) Is the effectiveness of KNN Defense Model Specific?
Q4_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Model Architecture", "Class Balance", "Replicate Imbalance", "Poison Success on Target Image"]]
Q4_Statistics = {"Model Architecture" : [],
                 "Class Balance Ratio" : [],
                 "Replicate Imbalance" : [],
                 "Attack Success Rate" : []}

for modelName in architecture:
    for targetWeight in classBalance:
        for replicate in replicateImbalance:
            Q4_Statistics["Model Architecture"].append(modelName)
            Q4_Statistics["Class Balance Ratio"].append(targetWeight / fixedClassBalance)
            Q4_Statistics["Replicate Imbalance"].append(replicate)
            tempDataFrame = Q4_dataFrame[(Q4_dataFrame["Model Architecture"] == modelName) & (Q4_dataFrame["Replicate Imbalance"] == replicate) & (Q4_dataFrame["Class Balance"] == targetWeight)]
            Q4_Statistics["Attack Success Rate"].append(np.sum(tempDataFrame["Poison Success on Target Image"] == True) / float(tempDataFrame.shape[0]))

Q4_Plot = pd.DataFrame.from_dict(Q4_Statistics)
ax = sns.catplot(x="Class Balance Ratio", y="Attack Success Rate", hue="Model Architecture",  data=Q4_Plot, col="Replicate Imbalance", order=[i / fixedClassBalance for i in classBalance], kind="point", legend=False)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Transfer Convex Polytope Attack Success (K Value = {})".format(fixedK))
plt.legend(loc='upper right', ncol=2)
plt.savefig(figureDirectory + "fixedK_modelAgnostic.pdf")


#Q5: (Fixed K) Matthews Correlation Coefficient of All Models?
Q5_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "Matthews Correlation Coefficient"]]
Q5_Statistics = {"Model Architecture" : [],
                 "Class Balance Ratio" : [],
                 "Replicate Imbalance" : [],
                 "Matthews Correlation Coefficient" : []}

for modelName in architecture:
    for targetWeight in classBalance:
        for replicate in replicateImbalance:
            Q5_Statistics["Model Architecture"].append(modelName)
            Q5_Statistics["Class Balance Ratio"].append(targetWeight / fixedClassBalance)
            Q5_Statistics["Replicate Imbalance"].append(replicate)
            tempDataFrame = Q5_dataFrame[(Q5_dataFrame["Model Architecture"] == modelName) & (Q5_dataFrame["Replicate Imbalance"] == replicate) & (Q5_dataFrame["Class Balance"] == targetWeight)]
            Q5_Statistics["Matthews Correlation Coefficient"].append(np.mean(tempDataFrame["Matthews Correlation Coefficient"]))

Q5_Plot = pd.DataFrame(Q5_Statistics)
ax = sns.catplot(x="Class Balance Ratio", y="Matthews Correlation Coefficient", hue="Model Architecture",  data=Q5_Plot, col="Replicate Imbalance", order=[i / fixedClassBalance for i in classBalance], kind="point", ci=None, legend=False)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Deep-KNN Poison Filtering Success (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_CorrelationCoefficient.pdf")
plt.clf()

#Q6: (Fixed K Value) Test Accuracy of All Models?
Q6_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "Test Accuracy"]]
Q6_Statistics = {"Model Architecture" : [],
                 "Class Balance Ratio" : [],
                 "Replicate Imbalance" : [],
                 "Test Accuracy" : []}

for modelName in architecture:
    for targetWeight in classBalance:
        for replicate in replicateImbalance:
            Q6_Statistics["Model Architecture"].append(modelName)
            Q6_Statistics["Class Balance Ratio"].append(targetWeight / fixedClassBalance)
            Q6_Statistics["Replicate Imbalance"].append(replicate)
            tempDataFrame = Q6_dataFrame[(Q6_dataFrame["Model Architecture"] == modelName) & (Q6_dataFrame["Replicate Imbalance"] == replicate) & (Q6_dataFrame["Class Balance"] == targetWeight)]
            Q6_Statistics["Test Accuracy"].append(np.mean(tempDataFrame["Test Accuracy"]))

Q6_Plot = pd.DataFrame(Q6_Statistics)
ax = sns.catplot(x="Class Balance Ratio", y="Test Accuracy", hue="Model Architecture",  data=Q6_Plot, col="Replicate Imbalance", order=[i / fixedClassBalance for i in classBalance], kind="point", ci=None, legend=False)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Deep-KNN Test Accuracy (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_TestAccuracy.pdf")
plt.clf()
