# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:35:42 2019

@author: ELİF NUR
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

def correlationSelection(X,y,featureList,LabelColumnName,dataset,featureNumber):#This method makes the feature selection by correlation calculation
    corrmat = dataset.corr() #corr() calculates the correlation.
    top_corr_features = corrmat.index
    plt.figure(figsize=(77,77))
    g=sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")   
    corr_=corrmat.iloc[78,:]
    corr_scores=corr_.tolist()
    unsorted_scores=list(zip(featureList, corr_scores))
    featureScoreDataFrame= pd.DataFrame(unsorted_scores, columns = ['Özellik', 'Skor']).sort_values(by = 'Skor', ascending = False)
    featureNumber=featureNumber+1
    selectedFeature=featureScoreDataFrame.head(featureNumber).sort_values(by = 'Skor')['Özellik'].tolist()
    seriesFeature=pd.Series(featureScoreDataFrame['Skor'].values, featureScoreDataFrame['Özellik'].values)
    plt.figure(figsize=(16,16))
    seriesFeature.nlargest(77).plot(kind='barh')
    plt.show()
    featureDataFrame=dataset.loc[:,selectedFeature]
    return featureDataFrame
def univariateSelection(X,y,featureList,LabelColumnName,dataset,featureNumber):#this method selects the feature with Anova-F
    model=SelectKBest(k = 'all')
    model.fit(X,y)
    unsorted_scores = list(zip(featureList[1:], model.scores_))
    featureScoreDataFrame=pd.DataFrame(unsorted_scores, columns = ['Özellik', 'Skor']).sort_values(by = 'Skor', ascending = False)
    selectedFeature=featureScoreDataFrame.head(featureNumber)['Özellik'].tolist() + [LabelColumnName]
    featureDataFrame=dataset.loc[:,selectedFeature]
    seriesFeature=pd.Series(featureScoreDataFrame['Skor'].values, featureScoreDataFrame['Özellik'].values)
    plt.figure(figsize=(16,16))
    seriesFeature.nlargest(77).plot(kind='barh')
    plt.show()
    return featureScoreDataFrame,featureDataFrame
def randomForestSelection(X,y,featureList,LabelColumnName,dataset, featureNumber):#this method selects the feature with random forest
    model=RandomForestClassifier(n_estimators=20)
    model.fit(X,y)
    importances = list(model.feature_importances_)
    featureImportances = [(feature, round(importance, 4)) for feature, importance in zip(featureList, importances)]
    featureImportances= sorted(featureImportances, key = lambda x: x[1], reverse = True)
    unsorted_scores = list(zip(featureList, model.feature_importances_))
    featureScoreDataFrame=pd.DataFrame(unsorted_scores, columns = ['Özellik', 'Skor']).sort_values(by = 'Skor', ascending = False)
    selectedFeature=featureScoreDataFrame.head(featureNumber)['Özellik'].tolist() + [LabelColumnName]
    featureDataFrame=dataset.loc[:,selectedFeature]
    seriesFeature=pd.Series(model.feature_importances_, index=dataset.drop([LabelColumnName],axis=1).columns)
    plt.figure(figsize=(16,16))
    seriesFeature.nlargest(77).plot(kind='barh')
    plt.show()
    return featureScoreDataFrame,featureDataFrame

def extraTreesSelection(X,y,featureList,LabelColumnName,dataset,featureNumber): #this method selects the feature with Extra Trees
    model=ExtraTreesClassifier(n_estimators=250,random_state=0)
    model.fit(X,y)
    seriesFeature=pd.Series(model.feature_importances_, index=featureList)
    plt.figure(figsize=(16,16))
    seriesFeature.nlargest(77).plot(kind='barh')
    plt.show()
    unsorted_scores = list(zip(featureList, model.feature_importances_))
    featureScoreDataFrame=pd.DataFrame(unsorted_scores, columns = ['Özellik', 'Skor']).sort_values(by = 'Skor', ascending = False)
    selectedFeature=featureScoreDataFrame.head(featureNumber)['Özellik'].tolist() + [LabelColumnName]
    featureDataFrame=dataset.loc[:,selectedFeature]
    return featureScoreDataFrame,featureDataFrame
