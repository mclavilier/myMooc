#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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

from sklearn.naive_bayes import GaussianNB
#create the classifier
clf = GaussianNB()

#Define T0
t0 = time()
### t0 will measure the training time

# populate with the training set
clf.fit(features_train,labels_train)

print "the training time is: ", round(time()-t0,3), "s"

#Define t1 to measure predicting time
t1 = time()
# the prediction should predict the label
pred = clf.predict(features_test)
print "the predict time is: ", round(time()-t1,3), "s"
print "the prediction is: ",pred

#Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print accuracy  
print "another way to do it with score function"  
print clf.score(features_test, labels_test)