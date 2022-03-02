#!/usr/bin/python
#-*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from .Processing import Processing
class Result:
    """This class take Processing Object and then compute accuracies for each model.
    
    Attribute:
        List_score(DataFrame): Include score for each model
        KNN/SVM/RF/NV_score(list): Ap,As,Cm,F1 for model
        Ap,As,Cm,F1(float): score for model you call
    """
    def __init__(self,Processing):

        self.y_true = Processing.y_test
        self.RF_y_pred = Processing.RF.y_pred
        self.KNN_y_pred = Processing.KNN.y_pred
        self.SVM_y_pred = Processing.SVM.y_pred
        self.NV_y_pred = Processing.NV.y_pred
        
    def confusionmatrix(self,type='RF'):
        """ Compute confusion matrix to evaluate the accuracy of a classification.
        
        Args:
            y_true(array-like of shape (n_samples,)): Ground truth (correct) target values.
            y_pred(array-like of shape (n_samples,)): Estimated targets as returned by a classifier.
            
        Returns:
            C(ndarray of shape (n_classes, n_classes)):Confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.

        """

        if type == 'RF':
            self.Cm = confusion_matrix(self.y_true,self.RF_y_pred)
        elif type == 'KNN':
            self.Cm = confusion_matrix(self.y_true,self.KNN_y_pred)
        elif type == 'SVM':
            self.Cm = confusion_matrix(self.y_true,self.SVM_y_pred)
        elif type == 'NV':
            self.Cm = confusion_matrix(self.y_true,self.NV_y_pred)
        return self.Cm
  
    def accuracyscore(self, type = 'RF'):
        """ Accuracy classification score.
        
        Args:
            
            y_true(array-like of shape (n_samples,)): Ground truth (correct) target values.
            y_pred(array-like of shape (n_samples,)): Estimated targets as returned by a classifier.
            
        Returns:
            Score(float):Predict score
        """

        if type == 'RF':
            self.As = accuracy_score(self.y_true,self.RF_y_pred)
        elif type == 'KNN':
            self.As = accuracy_score(self.y_true,self.KNN_y_pred)
        elif type == 'SVM':
            self.As = accuracy_score(self.y_true,self.SVM_y_pred)
        elif type == 'NV':
            self.As = accuracy_score(self.y_true,self.NV_y_pred)
        return self.As

    def averageprecisionscore(self, type = 'RF'):
        """ Compute average precision (AP) from prediction scores.
        
        Args:
            y_true(array-like of shape (n_samples,)): Ground truth (correct) target values.
            y_pred(array-like of shape (n_samples,)): Estimated targets as returned by a classifier.
            
        Returns:
            average_precision(float) : AP accuracy
        """

        if type == 'RF':
            self.Ap = average_precision_score(self.y_true,self.RF_y_pred)
        elif type == 'KNN':
            self.Ap = average_precision_score(self.y_true,self.KNN_y_pred)
        elif type == 'SVM':
            self.Ap = average_precision_score(self.y_true,self.SVM_y_pred)
        elif type == 'NV':
            self.Ap = average_precision_score(self.y_true,self.NV_y_pred)
        return self.Ap

    def f1score(self, type = "RF"):
        """ Compute the F1 score, also known as balanced F-score or F-measure.
        
        Args:
            
            y_true(array-like of shape (n_samples,)): Ground truth (correct) target values.
            y_pred(array-like of shape (n_samples,)): Estimated targets as returned by a classifier.
        
        Returns:
            f1_score(float or array of float)
        """
        
        if type == 'RF':
            self.F1 = f1_score(self.y_true,self.RF_y_pred)
        elif type == 'KNN':
            self.F1 = f1_score(self.y_true,self.KNN_y_pred)
        elif type == 'SVM':
            self.F1 = f1_score(self.y_true,self.SVM_y_pred)
        elif type == 'NV':
            self.F1 = f1_score(self.y_true,self.NV_y_pred)
        return self.F1
    
    def model_score(self,type='RF'):
        """Compute 'type' model score
        
        Args:
            type(string): RF: RandomForest, KNN: Kneighboor, SVM: SVMLinearSVC, NV: Navie Bayes
        
        Returns:
            list : tupe(score's type(string), float )
        """
        if type == 'RF':
            self.F1 = self.f1score('RF')
            self.Ap = self.averageprecisionscore('RF')
            self.As = self.accuracyscore('RF')
            self.Cm = self.confusionmatrix('RF')
            self.RF_score = [("CM",self.Cm),("F1",self.F1),("Ap",self.Ap),("Cm",self.Cm)]
            return self.RF_score
        elif type == 'KNN':
            self.F1 = self.f1score('KNN')
            self.Ap = self.averageprecisionscore('KNN')
            self.As = self.accuracyscore('KNN')
            self.Cm = self.confusionmatrix('KNN')
            self.KNN_score = [("CM",self.Cm),("F1",self.F1),("Ap",self.Ap),("Cm",self.Cm)]
            return self.KNN_score
        elif type == 'SVM':
            self.F1 = self.f1score('SVM')
            self.Ap = self.averageprecisionscore('SVM')
            self.As = self.accuracyscore('SVM')
            self.Cm = self.confusionmatrix('SVM')
            self.SVM_score = [("CM",self.Cm),("F1",self.F1),("Ap",self.Ap),("Cm",self.Cm)]
            return self.SVM_score
        elif type == 'NV':
            self.F1 = self.f1score('NV')
            self.Ap = self.averageprecisionscore('NV')
            self.As = self.accuracyscore('NV')
            self.Cm = self.confusionmatrix('NV')
            self.NV_score = [("CM",self.Cm),("F1",self.F1),("Ap",self.Ap),("Cm",self.Cm)]
            return self.NV_score

        
        
    def full_score(self):
        """ Compute full score for algorithm
        
        Returns:
            List_score(DataFrame)

        """

        y_pred_list = [self.RF_y_pred,self.KNN_y_pred,self.SVM_y_pred,self.NV_y_pred ]
        List_score = []
        for i in y_pred_list:
            Cm = confusion_matrix(self.y_true,i)
            As = accuracy_score(self.y_true,i)
            Ap = average_precision_score(self.y_true,i)
            F1 = f1_score(self.y_true , i)
            List_score.append([Cm,As,Ap,F1])
        self.List_score = pd.DataFrame(List_score,columns = ['Cm','As','Ap','F1'],index = ['RF','KNN','SVM','NV'])
        self.List_score.to_csv('data\list_score.csv')
        return self.List_score
        

