# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:01:26 2019

@author: ELİF NUR
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve,confusion_matrix


def createANN( X_train, X_test, y_train, y_test,epochCount,batchSize,input_dim,unit1,unit2,unit3,targetNames,fileName):#this method create ANN  model , save model and calculate Confusion Matrix.
    y_ = to_categorical(y_train)
    y_t= to_categorical(y_test)
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = unit1, kernel_initializer = 'uniform', activation = 'relu', input_dim =input_dim))
    classifier.add(Dense(units = unit2, kernel_initializer = 'uniform', activation = 'relu'))   
    classifier.add(Dense(units = unit3, kernel_initializer = 'uniform', activation = 'softmax'))
    adam0=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    classifier.compile(optimizer =adam0, loss = 'categorical_crossentropy', metrics = ['acc'])
    hist=classifier.fit(X_train,y_,epochs=epochCount,batch_size=batchSize,validation_data=(X_test,y_t) ,verbose=2)
    
    y_pred = classifier.predict(X_test, batch_size=batchSize, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_test, y_pred_bool, digits=5, target_names= targetNames))
    print("Accuracy: "+str(accuracy_score(y_test, y_pred_bool)))
    
    model_json =classifier.to_json()
    with open(fileName+".json", "w") as json_file:
        json_file.write(model_json)
    classifier.save_weights(fileName+".h5")
    print("Model kayıt edildi || Saved model")
    cm=confusion_matrix(y_test,y_pred_bool)
    f,ax=plt.subplots(figsize=(8,8))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="blue",fmt=".0f",ax=ax)
    classNames = targetNames
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=0)
    plt.yticks(tick_marks, classNames)
    plt.show()
    
    if unit3==2:
      probs=classifier.predict_proba(X_test)[:,1]
      roc_auc = roc_auc_score(y_test, probs) 
      fpr, tpr, thresholds = roc_curve(y_test,probs)
      plt.plot(fpr, tpr, label='RF (auc = %0.8f)' % roc_auc, color='navy')
      plt.plot([0, 1], [0, 1],'r--')
      plt.title('ROC')
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.legend(loc="lower right")
      plt.show()
    

def loadANN(jsonFile,h5file,X_test,y_test,batchSize,targetNames):#this method load saved ANN  model and  calculate  Confusion Matrix.

    json_file = open(jsonFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5file)
    print("Model yüklendi || Loaded model")
    adam0=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    loaded_model.compile(optimizer =adam0, loss = 'categorical_crossentropy', metrics = ['acc'])
    y_pred = loaded_model.predict(X_test, batch_size=batchSize, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_test, y_pred_bool, digits=5, target_names=targetNames))
    print(accuracy_score(y_test, y_pred_bool,))
    
    cm=confusion_matrix(y_test,y_pred_bool)
    f,ax=plt.subplots(figsize=(8,8))
    sns.heatmap(cm,annot=True,linewidth=0.5,linecolor="blue",fmt=".0f",ax=ax)
    classNames =targetNames
    plt.title('Confusion Matrix - Test Data')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=0)
    plt.yticks(tick_marks, classNames)
    plt.show()


















