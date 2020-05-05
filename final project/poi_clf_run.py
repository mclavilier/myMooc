# -*- coding: utf-8 -*-
"""
Created on Mon May 04 04:05:19 2020

@author: MCLAVILIER
"""
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from time import time
from sklearn.cross_validation import StratifiedShuffleSplit
import numpy as np
"""Import for KBest  """
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

""" """
def Evaluate_NB(my_dataset,features,feature,test_size):
    print
    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    
    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html
    
    # Provided to give you a starting point. Try a variety of classifiers.

    
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=test_size, random_state=42)
    """ 
    
    """    
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    print ' For the feature ',feature , 'The GaussianNB results are : '
    print 'The prediction is : ', pred
    accuracy = accuracy_score(labels_test, pred)
    print 'The accuracy is : ', accuracy  
    print "The precision score is :", precision_score(labels_test,clf.predict(features_test))
    print "the recall score is :", recall_score(labels_test,clf.predict(features_test))
    precision = precision_score(labels_test,clf.predict(features_test))
    recall = recall_score(labels_test,clf.predict(features_test))
    
    

    return  accuracy, precision, recall


def Evaluate_DT(dataset,features,feature,split,size):
    print
    ### Extract features and labels from dataset for local testing
    data = featureFormat(dataset, features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    #features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=size, random_state=42)
    """ 
    In order to maximize the split efficiency, we will use StratifiedShuffleSplit instead of train_test_split
    """
    

    X = np.array(features)
    y = np.array(labels)
    sss = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.3, random_state=42)      
    for train_index, test_index in sss:
        features_train, features_test = X[train_index], X[test_index]
        labels_train, labels_test = y[train_index], y[test_index]
        
    ### Create the Classifier
    clf = tree.DecisionTreeClassifier(min_samples_split=split)
    print ' For the features: ',feature ,' and a split of: ',split, 'The DecisionTree results are : '
    ### define t0
    t0 = time()
    clf = clf.fit(features_train, labels_train)
    print "training Time:", round(time()-t0,3), "s"
    
    ### define t1
    t1 = time()
    pred = clf.predict(features_test) 
    ### print the results                         
    print "Prediction time:", round(time()-t1,3),"s"
    print 'The prediction is : ', pred
    
    accuracy = accuracy_score(labels_test, pred)
    print 'The accuracy is : ', accuracy
    print "The precision score is :", precision_score(labels_test,pred)
    print "the recall score is :", recall_score(labels_test,pred)
    precision = precision_score(labels_test,pred)
    recall = recall_score(labels_test,pred)
    importances = clf.feature_importances_
    
    result = []
    for i in range(len(importances)):
        t = [features[i], importances[i]]
        result.append(t)

    result = sorted(result, key=lambda x: x[1], reverse=True)

    print '# The important features are:'
    print result
    print "\n"
    
    return  accuracy, precision, recall
    
    
    
def Evaluate_KN(dataset, features, feature,n,size):
    print
    ### Extract features and labels from dataset for local testing
    data = featureFormat(dataset, features, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    #features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=size, random_state=42)
    """ 
    In order to maximize the split efficiency, we will use StratifiedShuffleSplit instead of train_test_split
    """
    

    X = np.array(features)
    y = np.array(labels)
    sss = StratifiedShuffleSplit(labels, n_iter=1000, test_size=0.3, random_state=42)      
    for train_index, test_index in sss:
        features_train, features_test = X[train_index], X[test_index]
        labels_train, labels_test = y[train_index], y[test_index]
        
    ### Create the Classifier
    print ' For the features: ',feature ,' and a n neighbors: ',n, 'The KNeighbors Classifier results are : '
    clf = KNeighborsClassifier(n_neighbors=n)  
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    print 'The prediction is: ',clf.predict(features_test)
    accuracy = accuracy_score(labels_test, pred)
    print 'The accuracy is : ', accuracy
    print "The precision score is :", precision_score(labels_test,pred)
    print "the recall score is :", recall_score(labels_test,pred)
    
    precision = precision_score(labels_test,pred)
    recall = recall_score(labels_test,pred)
    
    
    return  accuracy, precision, recall
    
def featureKbest (data, features_all,k):
    """
    In this function we will try to figure out which are the 10 most important
    feature. for this we will apply a SelectKBest model
    """
    print '=============================='
    print 'Selecting features with kbest!'
    print '=============================='
    
     # Keep only the values from features_list
    data = featureFormat(data, features_all, sort_keys=True)
    # Split between labels (poi) and the rest of features
    labels, features = targetFeatureSplit(data)
    print features_all
    # Set up the scaler
    scaler = preprocessing.MinMaxScaler()
    features_minmax = scaler.fit_transform(features)

    # Use SelectKBest to tune for k
    k_best = SelectKBest(chi2, k=k)

    # Use the instance to extract the k best features
    features_kbest = k_best.fit_transform(features_minmax, labels)
    print
    print '=============================================='
    print 'feature kbest rigth after the fit & transform '
    print features_kbest
    print '=============================================='
    print
    feature_scores = ['%.2f' % elem for elem in k_best.scores_]

    # Get SelectKBest pvalues, rounded to 3 decimal places (remember the scaler values are in the interval [0;1])
    feature_scores_pvalues = ['%.3f' % elem for elem in k_best.pvalues_]

    # Get SelectKBest feature names, from 'K_best.get_support',
    # Create an array of feature names, scores and pvalues
    k_features = [(features_all[i+1],
                   feature_scores[i],
                   feature_scores_pvalues[i]) for i in k_best.get_support(indices=True)]
    
    # Sort the array by score
    k_features = sorted(k_features, key=lambda f: float(f[1]))
    print
    print '=============================================='
    print 'The KBest Features are:'
    print k_features
#    for f in k_best.get_support(indices=True):
#        print 'The features name is: ',f
    print "\n"
    print '=============================================='
    print
    return