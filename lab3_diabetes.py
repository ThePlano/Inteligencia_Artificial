import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff


data,meta = arff.loadarff('./CriterioDiabetes.arff')

attributes = meta.names()
data_value = np.asarray(data)


glicose = np.asarray(data['Glicose']).reshape(-1,1)
peso = np.asarray(data['Peso']).reshape(-1,1)
features = np.concatenate((glicose,peso),axis=1)
target = data['resultado']


Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

plt.figure(figsize=(10, 6.5))
tree.plot_tree(Arvore,feature_names=['Glicose','Peso'],class_names=['Diabetico','NaoDiabetico'],
                   filled=True, rounded=True)
plt.show()

fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(Arvore,features,target,display_labels=['Diabetico','NaoDiabetico'], values_format='d', ax=ax)

plt.show()