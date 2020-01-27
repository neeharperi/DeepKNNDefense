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
classBalance = [5, 10, 25, 50]
KValues = [1, 2, 5, 10, 20, 50, 100, 200]

fixedClassBalance = 50
fixedK = 100

for rowTable in dataFrame.iterrows():
    CB = rowTable[1]["Class Balance"]
    targetWeight = CB.replace("[50, 50, 50, 50, 50, 50, ", "").replace(", 50, 50, 50]", "")
    dataFrame.at[rowTable[0], "Class Balance"] = int(targetWeight)

#Q1: (Fixed Class Balance) Is the effectiveness of KNN Defense Model Specific?
Q1_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["Model Architecture", "K Value", "Poisoning Successful on Target Image"]]
Q1_Statistics = {"Model Architecture" : [],
                 "K Value" : [],
                 "KNN Defense Success" : []}

for modelName in architecture:
    for K in KValues:
        Q1_Statistics["Model Architecture"].append(modelName)
        Q1_Statistics["K Value"].append(K)
        tempDataFrame = Q1_dataFrame[(Q1_dataFrame["Model Architecture"] == modelName) & (Q1_dataFrame["K Value"] == K)]
        Q1_Statistics["KNN Defense Success"].append(np.sum(tempDataFrame["Poisoning Successful on Target Image"] == False) / float(tempDataFrame.shape[0]))

Q1_Plot = pd.DataFrame.from_dict(Q1_Statistics)
ax = sns.swarmplot(x="K Value", y="KNN Defense Success", hue="Model Architecture", data=Q1_Plot)
plt.title("Is the Effectiveness of kNN Defense Model Agnostic? (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_ModelAgnostic.png")
plt.clf()

#Q2: (Fixed Class Balance) Precision of All Models?
Q2_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "Precision"]]
ax = sns.pointplot(x="K Value", y="Precision", hue="Model Architecture", ci=None, data=Q2_dataFrame)

plt.title("Precision of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_Precision.png")
plt.clf()

#Q3: (Fixed Class Balance) Recall of All Models?
Q3_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "Recall"]]
ax = sns.pointplot(x="K Value", y="Recall", hue="Model Architecture", ci=None, data=Q3_dataFrame)

plt.title("Recall of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_Recall.png")
plt.clf()

#Q4: (Fixed Class Balance) F1 of All Models?
Q4_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "F1"]]
ax = sns.pointplot(x="K Value", y="F1", hue="Model Architecture", ci=None, data=Q4_dataFrame)

plt.title("F1 of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_F1.png")
plt.clf()

#Q5: (Fixed Class Balance) Specificity of All Models?
Q5_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "True Negative Rate"]]
ax = sns.pointplot(x="K Value", y="True Negative Rate", hue="Model Architecture", ci=None, data=Q5_dataFrame)

plt.title("Specificity of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_Specificity.png")
plt.clf()

#Q6: (Fixed Class Balance) Negative Predictive Value of All Models?
Q6_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "Negative Predictive Value"]]
ax = sns.pointplot(x="K Value", y="Negative Predictive Value", hue="Model Architecture", ci=None, data=Q6_dataFrame)

plt.title("Negative Predictive Value of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_NegativePredictionValue.png")
plt.clf()

#Q7: (Fixed Class Balance) False Discovery Rate of All Models?
Q7_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "False Discovery Rate"]]
ax = sns.pointplot(x="K Value", y="False Discovery Rate", hue="Model Architecture", ci=None, data=Q7_dataFrame)

plt.title("False Discovery Rate of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_FalseDiscoveryRate.png")
plt.clf()

#Q8: (Fixed Class Balance) False Omission Rate of All Models?
Q8_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "False Omission Rate"]]
ax = sns.pointplot(x="K Value", y="False Omission Rate", hue="Model Architecture", ci=None, data=Q8_dataFrame)

plt.title("False Omission Rate of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_FalseOmissionRate.png")
plt.clf()

#Q9: (Fixed Class Balance) Critical Success Index of All Models?
Q9_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "Critical Success Index"]]
ax = sns.pointplot(x="K Value", y="Critical Success Index", hue="Model Architecture", ci=None, data=Q9_dataFrame)

plt.title("Critical Success Index of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_CriticalSuccessIndex.png")
plt.clf()

#Q10: (Fixed Class Balance) Matthews Correlation Coefficient of All Models?
Q10_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["K Value", "Model Architecture", "Matthews Correlation Coefficient"]]
ax = sns.pointplot(x="K Value", y="Matthews Correlation Coefficient", hue="Model Architecture", ci=None, data=Q10_dataFrame)

plt.title("Matthews Correlation Coefficient of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_CorrelationCoefficient.png")
plt.clf()

#Q11: (Fixed Class Balance) ROC of All Models
Q11_dataFrame = dataFrame[(dataFrame["Class Balance"] == fixedClassBalance)][["Model Architecture", "K Value", "True Positive Rate", "False Positive Rate"]]
Q11_Statistics = {"Model Architecture" : [],
                 "K Value" : [],
                 "True Positive Rate" : [],
                 "False Positive Rate" : []}

for modelName in architecture:
    for K in KValues:
        Q11_Statistics["Model Architecture"].append(modelName)
        Q11_Statistics["K Value"].append(K)
        tempDataFrame = Q11_dataFrame[(Q11_dataFrame["Model Architecture"] == modelName) & (Q11_dataFrame["K Value"] == K)]
        Q11_Statistics["True Positive Rate"].append(np.mean(tempDataFrame["True Positive Rate"]))
        Q11_Statistics["False Positive Rate"].append(np.mean(tempDataFrame["False Positive Rate"]))

Q11_Plot = pd.DataFrame.from_dict(Q11_Statistics)
ax = sns.lineplot(x="False Positive Rate", y="True Positive Rate", hue="Model Architecture", data=Q11_Plot)
plt.title("ROC of All Models (Class Balance = {})".format(fixedClassBalance))
plt.savefig(figureDirectory + "fixedClassBalance_ROC.png")
plt.clf()


#Q12: (Fixed K) Is the effectiveness of KNN Defense Model Specific?
Q12_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Model Architecture", "Class Balance", "Replicate Imbalance", "Poisoning Successful on Target Image"]]
Q12_Statistics = {"Model Architecture" : [],
                 "Class Balance" : [],
                 "Replicate Imbalance" : [],
                 "KNN Defense Success" : []}

for modelName in architecture:
    for targetWeight in classBalance:
        for replicate in replicateImbalance:
            Q12_Statistics["Model Architecture"].append(modelName)
            Q12_Statistics["Class Balance"].append(targetWeight)
            Q12_Statistics["Replicate Imbalance"].append(replicate)
            tempDataFrame = Q12_dataFrame[(Q12_dataFrame["Model Architecture"] == modelName) & (Q12_dataFrame["Replicate Imbalance"] == replicate) & (Q12_dataFrame["Class Balance"] == targetWeight)]
            Q12_Statistics["KNN Defense Success"].append(np.sum(tempDataFrame["Poisoning Successful on Target Image"] == False) / float(tempDataFrame.shape[0]))

Q12_Plot = pd.DataFrame.from_dict(Q12_Statistics)
ax = sns.catplot(x="Class Balance", y="KNN Defense Success", hue="Model Architecture",  data=Q12_Plot, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="swarm")
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Is the Effectiveness of KNN Defense Model Agnostic? (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_modelAgnostic.png")

#Q13: (Fixed K) Precision of All Models?
Q13_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "Precision"]]
ax = sns.catplot(x="Class Balance", y="Precision", hue="Model Architecture",  data=Q13_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Precision of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_Precision.png")
plt.clf()


#Q14: (Fixed K) Recall of All Models?
Q14_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "Recall"]]
ax = sns.catplot(x="Class Balance", y="Recall", hue="Model Architecture",  data=Q14_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Recall of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_Recall.png")
plt.clf()


#Q15: (Fixed K) F1 of All Models?
Q15_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "F1"]]
ax = sns.catplot(x="Class Balance", y="F1", hue="Model Architecture",  data=Q15_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("F1 of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_F1.png")
plt.clf()

#Q16: (Fixed K) Specificity of All Models?
Q16_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "True Negative Rate"]]
ax = sns.catplot(x="Class Balance", y="True Negative Rate", hue="Model Architecture",  data=Q16_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Specificity of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_Specificity.png")
plt.clf()


#Q17: (Fixed K) Negative Predictive Value of All Models?
Q17_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "Negative Predictive Value"]]
ax = sns.catplot(x="Class Balance", y="Negative Predictive Value", hue="Model Architecture",  data=Q17_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Negative Predictive Value of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_NegativePredictiveValue.png")
plt.clf()

#Q18: (Fixed K) False Discovery Rate of All Models?
Q18_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "False Discovery Rate"]]
ax = sns.catplot(x="Class Balance", y="False Discovery Rate", hue="Model Architecture",  data=Q18_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("False Discovery Rate of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_FalseDiscoveryRate.png")
plt.clf()


#Q19: (Fixed K) False Omission Rate of All Models?
Q19_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "False Omission Rate"]]
ax = sns.catplot(x="Class Balance", y="False Omission Rate", hue="Model Architecture",  data=Q19_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("False Omission Rate of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_FalseOmissionRate.png")
plt.clf()

#Q20: (Fixed K) Critical Success Index of All Models?
Q20_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "Critical Success Index"]]
ax = sns.catplot(x="Class Balance", y="Critical Success Index", hue="Model Architecture",  data=Q20_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Critical Success Index of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_CriticalSuccessIndex.png")
plt.clf()

#Q21: (Fixed K) Matthews Correlation Coefficient of All Models?
Q21_dataFrame = dataFrame[(dataFrame["K Value"] == fixedK)][["Class Balance", "Replicate Imbalance", "Model Architecture", "Matthews Correlation Coefficient"]]
ax = sns.catplot(x="Class Balance", y="Matthews Correlation Coefficient", hue="Model Architecture",  data=Q21_dataFrame, col="Replicate Imbalance", order=[5, 10, 25, 50], kind="point", ci=None)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle("Matthews Correlation Coefficient of All Models (K Value = {})".format(fixedK))
plt.savefig(figureDirectory + "fixedK_CorrelationCoefficient.png")
plt.clf()
