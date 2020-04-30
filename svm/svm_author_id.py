#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################

from sklearn.svm import SVC
from sklearn.metrics import classification_report

print "start"
#classifier = SVC(kernel='linear',gamma='auto')
classifier = SVC(C=10000 ,gamma='auto', kernel='rbf')
print "Classifier ok"

#Accelerate but which price ?
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

### define T0
t0 = time()
print " Classifier under training ... "
classifier.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### define T1
t1 = time()

print "Prediction on going... "
predicter = classifier.predict(features_test)

print "Prediction time:", round(time()-t1, 3), "s"
#pred1 = predicter[10]
#pred2 = predicter[26]
#pred3 = predicter[50]

#print pred1
#print pred2
#print pred3

#print classifier.class_weight_
#print classifier.shape_fit_
#print classifier
#print classifier.classes_
#print predicter

print classification_report(labels_test, predicter)
print "No. of predicted to be in the 'Chris'(1): %r" % sum(predicter)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, predicter)

print accuracy


