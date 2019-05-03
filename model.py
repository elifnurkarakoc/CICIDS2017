# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:34:44 2019

@author: ELİF NUR
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,classification_report

class Model():#This class create a Model and calculate Accuracy
    def __init__(self, X_train, X_test, y_train, y_test,modelName,classNames):
        self.X_train=X_train 
        self.X_test=X_test 
        self.y_train=y_train 
        self.y_test=y_test
        self.model=modelName
        self.classNames=classNames
        self.model.fit(self.X_train,self.y_train) 
        self.score=self.model.score(self.X_test,self.y_test)
        self.predict=self.model.predict(self.X_test)
        self.y_true=self.y_test
        print("Accuracy: "+str(self.score))
        
    
    def confusionMatrix(self):#Calculating the confusion matrix of the model
        cm=confusion_matrix(self.y_true,self.predict)
        f,ax=plt.subplots(figsize=(6,6))
        ax.imshow(cm, interpolation='none', cmap=plt.cm.terrain)
        plt.title(" Confusion Matrix - Test Data")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        tick_marks = np.arange(len(self.classNames))
        plt.xticks(tick_marks, self.classNames, rotation=0)
        plt.yticks(tick_marks, self.classNames)
        s = [['TN','FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                plt.text(j,i, str(s[i][j])+"="+str(cm[i][j]))
        plt.show()
    
    def roc(self):#Calculating the ROC of the model
        self.probs=self.model.predict_proba(self.X_test)[:,1]
        self.roc_auc = roc_auc_score(self.y_test, self.probs) 
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.probs)
        plt.plot(self.fpr, self.tpr, label='RF (auc = %0.8f)' % self.roc_auc, color='navy')
        plt.plot([0, 1], [0, 1],'r--')
        plt.title('ROC')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()
    def prfTable(self):# Calculating  precision,recall and F1-score of the model
        print(classification_report(self.y_test, self.predict, digits=5, target_names=self.classNames))