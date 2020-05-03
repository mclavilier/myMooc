#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
### add more features to features_list!
from sklearn.tree import DecisionTreeClassifier
from sklearn import cross_validation
features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels,test_size=0.3,
                                                                                          random_state=42)
clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
clf.score(features_test,labels_test)

# Prediction
clf.predict(features_test)
print "the number of predicted POI is : "
print len([e for e in labels_test if e == 1.0])
print "the number of person in my test set is :"    
print len(labels_test)

#Precision and Recall may help the investigation and the quality of the classifier
# Need to import sklearn.metrics
from sklearn.metrics import *
print "The precision score is :"
print precision_score(labels_test,clf.predict(features_test))
print "the recall score is :"
print recall_score(labels_test,clf.predict(features_test))

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

pop = len(true_labels)
print "total set lenght is: ", pop
i = 0
true_pos = false_pos = true_neg = false_neg = 0

for p in predictions:
    if p == true_labels[i] and p == 1.0:
        true_pos += 1
    if p == true_labels[i] and p == 0.0:
        true_neg += 1
    if p == 1.0 and true_labels[i] == 0.0:
        false_pos += 1
    if p == 0.0 and true_labels[i] == 1.0:
        false_neg +=1
    
    i += 1


#Number of true pos
print "the number of true pos is : ",true_pos
#Number of true neg
print "the number of true neg is : ",true_neg
#Number of false neg is :
print "the number of false neg is : ",false_neg    
#Number of false neg is :
print "the number of false pos is : ",false_pos

 
precision = precision_score(true_labels,predictions)
print "The precision of this classifier is: ", precision
recall =  recall_score(true_labels, predictions)
print "The Recall of this classifier is: ", recall
# The F1 score is : F1 = 2 * (precision * recall) / (precision + recall)
F1 = f1_score(true_labels,predictions)
print "The F1 score of this classifier is: ", F1
