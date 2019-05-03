# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 12:32:48 2019

@author: ELİF NUR
"""
#This .py file gives an example of usage.
import data 
import model
import model_ANN
import featureSelection

# In[] 
fromPath="dataset.csv"
LabelColumnName=' Label'
dataset,featureList=data.loadData(fromPath,LabelColumnName,2)
X,y=data.datasetSplit(dataset,LabelColumnName)
X_train, X_test, y_train, y_test= data.train_test_dataset(X,y)

# In[]
fromPath="dataset.csv"
LabelColumnName=' Label'
dataset,featureList=data.loadData(fromPath,LabelColumnName,4)
X,y=data.datasetSplit(dataset,LabelColumnName)
X_train, X_test, y_train, y_test= data.train_test_dataset(dataset)

# In[]
from sklearn.tree import DecisionTreeClassifier
decisionTree =model.Model(X_train, X_test, y_train, y_test, DecisionTreeClassifier(random_state = 0), ['Anormal','Normal'])
decisionTree.confusionMatrix()
decisionTree.roc()
decisionTree.prfTable()

# In[]

epochCount=100
batchSize=128
input_dim=78
unit1=78
unit2=4
unit3=4
targetNames=['Normal','DDoS','DoS','PortScan']
fileName="78_4_ann_4_128"
model_ANN.createANN(X_train, X_test, y_train, y_test,epochCount,batchSize,input_dim,unit1,unit2,unit3,targetNames,fileName)

# In[]
model_ANN.loadANN("78_8_ann_4_64.json","78_8_ann_4_64.h5",X_test,y_test,batchSize,targetNames)
# In[]
featureScoreDataFrame,featureDataFrame=featureSelection.randomForestSelection(X,y,featureList,' Label',dataset,15)
# In[]
featureScoreDataFrame,featureDataFrame=univariateSelection(X,y,featureList,' Label',dataset,15)