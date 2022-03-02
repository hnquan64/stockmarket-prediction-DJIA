#!/usr/bin/python
#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer

class Processing:
    def __init__(self,x_train,x_test,y_train,y_test):
        """ Process data with sklearn function:
            1.TFIDF Vectorize
            2.Count Vectorize
            3.Random Forest Model
            4.KNN Classifier Model
            5.SVM Model
            6.Naive Bayes Bernoulli Model
    
        Args:
            x_train(list)
            x_test(list)
            y_train(list)
            y_test(list)
    
        """
        self.x_train = TFIDFVectorizer(max_features = 10000).fit_transform(x_train)
        self.x_test = TFIDFVectorizer(max_features = 10000).fit_transform(x_test)
        self.y_train = y_train
        self.y_test = y_test
    def RandomForest(self,**kwargs):
        self.RF = RandomForest(self.x_train,self.y_train,self.x_test,self.y_test,**kwargs).random_process()
        return self.RF
    def KNNClassifier(self,**kwargs):
        self.KNN = KNNClassifier(self.x_train,self.y_train,self.x_test,self.y_test,**kwargs).knn_process()
        return self.KNN
    def SVMLinearSVC(self,**kwargs):
        self.SVM = SVMLinearSVC(self.x_train,self.y_train,self.x_test,self.y_test,**kwargs).svm_process()
        return self.SVM
    def Naive(self,**kwargs):
        self.NV = Naive(self.x_train,self.y_train,self.x_test,self.y_test,**kwargs).naive_process()
        return self.NV
    
class TFIDFVectorizer(TfidfVectorizer):
    def __init__(self,*args,**kwargs):
        super(TFIDFVectorizer,self).__init__(*args,**kwargs)
    def fit_transform(self,data):
        """ Change to tf-idf vector
        Args: 
            data (list) : path to your data 
        
        Returns:
            sparse matrix of (n_samples, n_features)
        """
        vectorized = super().fit_transform(data)
        return vectorized

class Countvectorizer(CountVectorizer):
    def __init__(self,*args,**kwargs):
        super(Countvectorizer,self).__init__(*args,**kwargs)
    def fit_transform(self,data):
        """ Change to tf-idf vector
        Args: 
            data (list) : path to your data 
        
        Returns:
            sparse matrix of (n_samples, n_features)
        """
        vectorized = super().fit_transform(data)
        return vectorized


class RandomForest(RandomForestClassifier):
    def __init__(self,x_train,y_train,x_test, y_test,**kwargs):
        super(RandomForest,self).__init__(**kwargs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def fit(self,X_train,y_train):
        """ Build a forest of trees from the training set (X, y).
        
        Args:
            X_train(array-like, sparse matrix of shape (n_samples, n_features)): Train dataset
            y_train(array-like, sparse matrix of shape (n_samples,) or (n_samples, n_outputs)): Train labels

        Returns:
            self(object) : class

        """
        object = super().fit(X_train,y_train)
        return object

    def score(self,X_test,y_test):
        """ Return the mean accuracy on the given test data and labels.
        Args:
            X_test(array-like of shape (n_samples, n_features)): Test dataset
            y_test(array-like of shape (n_samples,) or (n_samples, n_outputs)): Test labels

        Return:
            Score(float) : Accuracy of model
        """
        predict_score = super().score(X_test,y_test)
        return predict_score

    def predict(self,X):
        """ Predict the class labels for the provided data.
        Args:
            X(array-like of shape (n_queries, n_features), or (n_queries, n_indexed)): Data which want to predict

        Results:
            y(ndarray of shape (n_queries,) or (n_queries, n_outputs)): Result from predict on X    
        """
        y = super().predict(X)
        return y
    
    def random_process(self):
        """ Save results into variables of class
        
        """
        self.ob = self.fit(self.x_train, self.y_train)
        self.y_pred = self.ob.predict(self.x_test)
        self.score = self.ob.score(self.x_test, self.y_test)
        return self.ob
        

class KNNClassifier(KNeighborsClassifier):
        
    def __init__(self,x_train,y_train,x_test,y_test, **kwargs):
        super(KNNClassifier,self).__init__(**kwargs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
    def fit(self,X_train,y_train):
        """ Fit the k-nearest neighbors classifier from the training dataset.
        
        Args:
            X_train(array-like, sparse matrix of shape (n_samples, n_features)): Train dataset
            y_train(array-like, sparse matrix of shape (n_samples,) or (n_samples, n_outputs)): Train labels

        Returns:
            self(object) : class

        """
        object = super().fit(X_train,y_train)
        return object
    def score(self,X_test,y_test):
        """ Return the mean accuracy on the given test data and labels.
        Args:
            X_test(array-like of shape (n_samples, n_features)): Test dataset
            y_test(array-like of shape (n_samples,) or (n_samples, n_outputs)): Test labels

        Return:
            Score(float) : Accuracy of model
        """
        predict_score = super().score(X_test,y_test)
        return predict_score
    def predict(self,X):
        """ Predict the class labels for the provided data.
        Args:
            X(array-like of shape (n_queries, n_features), or (n_queries, n_indexed)): Data which want to predict

        Results:
            y(ndarray of shape (n_queries,) or (n_queries, n_outputs)): Result from predict on X 
        """
        y = super().predict(X)
        return y
    
    def knn_process(self):
        """ Save results into variables of class
        
        """
        self.ob = self.fit(self.x_train, self.y_train)
        self.y_pred = self.ob.predict(self.x_test)
        self.score = self.ob.score(self.x_test, self.y_test)
        return self.ob

class SVMLinearSVC(LinearSVC):
    def __init__(self,x_train,y_train,x_test,y_test,**kwargs):
        super(SVMLinearSVC,self).__init__(**kwargs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def fit(self,X_train,y_train):
        """ Fit the k-nearest neighbors classifier from the training dataset.
        
        Args:
            X_train(array-like, sparse matrix of shape (n_samples, n_features)): Train dataset
            y_train(array-like, sparse matrix of shape (n_samples,) or (n_samples, n_outputs)): Train labels

        Returns:
            self(object) : class

        """
        object = super().fit(X_train,y_train)
        return object

    def score(self,X_test,y_test):
        """ Return the mean accuracy on the given test data and labels.
        Args:
            X_test(array-like of shape (n_samples, n_features)): Test dataset
            y_test(array-like of shape (n_samples,) or (n_samples, n_outputs)): Test labels

        Return:
            Score(float) : Accuracy of model
        """
        predict_score = super().score(X_test,y_test)
        return predict_score

    def predict(self,X):
        """ Predict the class labels for the provided data.
        Args:
            X(array-like of shape (n_queries, n_features), or (n_queries, n_indexed)): Data which want to predict

        Results:
            y(ndarray of shape (n_queries,) or (n_queries, n_outputs)): Result from predict on X 
        """
        y = super().predict(X)
        return y
    def svm_process(self):
        """ Save results into variables of class
        
        """
        self.ob = self.fit(self.x_train, self.y_train)
        self.y_pred = self.ob.predict(self.x_test)
        self.score = self.ob.score(self.x_test, self.y_test)
        return self.ob

class Naive(BernoulliNB):
    def __init__(self,x_train,y_train,x_test,y_test,**kwargs):
        super(Naive,self).__init__(**kwargs)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def fit(self,X_train,y_train):
        """ Fit the k-nearest neighbors classifier from the training dataset.
        
        Args:
            X_train(array-like, sparse matrix of shape (n_samples, n_features)): Train dataset
            y_train(array-like, sparse matrix of shape (n_samples,) or (n_samples, n_outputs)): Train labels

        Returns:
            self(object) : class

        """
        object = super().fit(X_train,y_train)
        return object

    def score(self,X_test,y_test):
        """ Return the mean accuracy on the given test data and labels.
        Args:
            X_test(array-like of shape (n_samples, n_features)): Test dataset
            y_test(array-like of shape (n_samples,) or (n_samples, n_outputs)): Test labels

        Returns:
            Score(float) : Accuracy of model
        """
        predict_score = super().score(X_test,y_test)
        return predict_score

    def predict(self,X):
        """ Predict the class labels for the provided data.
        Args:
            X(array-like of shape (n_queries, n_features), or (n_queries, n_indexed)): Data which want to predict

        Returns:
            y(ndarray of shape (n_queries,) or (n_queries, n_outputs)): Result from predict on X 
        """
        y = super().predict(X)
        return y
    def naive_process(self):
        """ Save results into variables of class
        
        """
        self.ob = self.fit(self.x_train, self.y_train)
        self.y_pred = self.ob.predict(self.x_test)
        self.score = self.ob.score(self.x_test, self.y_test)
        return self.ob
                
