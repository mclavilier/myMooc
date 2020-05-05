#!/usr/bin/python
import poi_myfunctions as fn
import pprint
pr = pprint.PrettyPrinter()
import sys
import pickle
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
### Task 1: Select what features you'll use.
 
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
### In order to complete my previous exploration I will use financial categories and email behavioural as well.
features_all = ['poi','salary','deferral_payments','total_payments','loan_advances','bonus','restricted_stock_deferred',
                'deferred_income','total_stock_value','expenses','exercised_stock_options','other','long_term_incentive',
                'restricted_stock','director_fees','to_messages','from_poi_to_this_person','from_messages',
                'from_this_person_to_poi','shared_receipt_with_poi','ratio_to_poi','ratio_from_poi']

### Next set of features will be only the financial feature.
features_0 = ['poi','salary','deferral_payments','total_payments','loan_advances','bonus','restricted_stock_deferred','deferred_income',
              'total_stock_value','expenses','exercised_stock_options','other','long_term_incentive','restricted_stock','director_fees']

# Fourth feature set is a finance one + 2 enginered features
features_1 = ['poi','bonus','exercised_stock_options','ratio_to_poi','loan_advances', 'total_stock_value']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Number of records
print 'The number of record in this dataset is: ', fn.get_dataset_size(data_dict)
### Number of POi already identified
print 'The number of poi in this dataset is: ', fn.count_poi(data_dict)

### In order to identify which features we will use, we will do a bit of investigation and cleaning.
## Check the NaN features :
pr.pprint(fn.NaN_count(data_dict))
## In the financiales category, it's better to improve the data_dict by replacing NaN by 0.
data_dict = fn.NaN_Replace(data_dict)
    
    
### Task 2: Remove outliers
'''
### As seen during the cursus and the eploration (notebook) we will remove record 'Total'
'''
# Remove outlier TOTAL
data_dict.pop('TOTAL', None)

### We will then remove all person with incomplete information
'''
As mention in the notebook, this correspond to :
 'WHALEY DAVID A',
 'WROBEL BRUCE',
 'LOCKHART EUGENE E',
 'THE TRAVEL AGENCY IN THE PARK',
 'GRAMM WENDY L'
 
'''
incomplete = fn.get_incomplete(data_dict, 0.90)
print 'The following person (incomplete) will be removed :'
pr.pprint(incomplete)
for person in incomplete:
    print ' The person : ', person, ' is removed from the dataset.'
    data_dict.pop(person,None)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
print
print ' ============= Task 3 / 4 : 5 '
print
my_dataset = data_dict


# Generate new features
for person, features in my_dataset.iteritems():
    features['ratio_from_poi'] = fn.computefeature(features['from_poi_to_this_person'], features['to_messages'])
    features['ratio_to_poi'] = fn.computefeature(features['from_this_person_to_poi'], features['from_messages'])


### Define & Import an include that will be call several times.
import poi_clf_run as clf

# 
###features selection :
"""
Feature selection process.
For this we will use a KBest with the all features as input.
The aim is not to include the feature in our dataset.
"""
print
print 'Use of feature kbest to select the best features'
print
clf.featureKbest(my_dataset,features_all,5)
print
print '==============================================='
print

# We will start by using the Naive_Baise model.
'''
We will do a manual tunning for the Naive_Bayse model
'''
size_range = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
results = []
for s in size_range:
    accuracy, precision, recall = clf.Evaluate_NB(my_dataset,features_list,'features_list',s)
    # We only check Tunning for the first list to not confuse the reading of the result.
    results.append([s,accuracy,precision,recall])
    accuracy, precision, recall = clf.Evaluate_NB(my_dataset,features_all,'features_all',s)
    accuracy, precision, recall = clf.Evaluate_NB(my_dataset,features_0,'features_0',s)
    accuracy, precision, recall = clf.Evaluate_NB(my_dataset,features_1,'features_1',s)
    
nb_results = pd.DataFrame(results, columns= ['Size', 'accuracy' , 'Precision' , 'Recall'])
print 'The result of the tunning is: '
print nb_results
''' 
The Best accuracy is obtained for a size of 0.1 (10%) of the dataset;
but then it did not feet the expected 0.3 to get it we need a size of 0.2
'''


data = featureFormat(my_dataset, features_1, sort_keys = True)
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
    

'''
let's perform an automatique tuning with the KNeighborsClassifier
the parameter we will want to evaluate :algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                         metric='minkowski', metric_params=None, n_jobs=None, **kwargs

'''

from sklearn.model_selection import GridSearchCV
# do varying number of neighbors
k = np.arange(10)+1
# varying leaf_size for algorithm Ball_tree and kd_tree
leaf = np.arange(30)+1
# varying algorithm
params = {'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute'),'leaf_size': leaf,'n_neighbors': k}

cltf_params = GridSearchCV(KNeighborsClassifier(), params, cv=5)
cltf_params.fit(features_train,labels_train)

print 'The best estimator is :'
print cltf_params.best_estimator_
'''================================================'''




results = []
# DecisionTree
accuracy, precision, recall =clf.Evaluate_DT(my_dataset,features_list,'features_list',30,0.2)
results.append(['DT model','Feature_list',accuracy,precision,recall])
accuracy, precision, recall =clf.Evaluate_DT(my_dataset,features_all,'features_all',40,0.2)
results.append(['DT model','features_all',accuracy,precision,recall])
accuracy, precision, recall =clf.Evaluate_DT(my_dataset,features_0,'features_0',40,0.2)
results.append(['DT model','features_0',accuracy,precision,recall])
accuracy, precision, recall =clf.Evaluate_DT(my_dataset,features_1,'features_1',40,0.2)
results.append(['DT model','features_1',accuracy,precision,recall])

#KNeighborsClassifier Model
accuracy, precision, recall =clf.Evaluate_KN(my_dataset,features_list,'features_list',2,0.2)
results.append(['KNeighborsClassifier model','Feature_list',accuracy,precision,recall])
accuracy, precision, recall =clf.Evaluate_KN(my_dataset,features_all,'features_all',2,0.2)
results.append(['KNeighborsClassifier model','features_all',accuracy,precision,recall])
accuracy, precision, recall =clf.Evaluate_KN(my_dataset,features_0,'features_0',2,0.2)
results.append(['KNeighborsClassifier model','features_0',accuracy,precision,recall])
accuracy, precision, recall =clf.Evaluate_KN(my_dataset,features_1,'features_1',2,0.2)
results.append(['KNeighborsClassifier model','features_1',accuracy,precision,recall])

clf_results = pd.DataFrame(results, columns= ['Model','Feature Liste', 'accuracy' , 'Precision' , 'Recall'])
print
print 'The result is: '
print clf_results

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
### Extract features and labels from dataset for local testing

### Create the Classifier with the best parameters value
clf = KNeighborsClassifier(n_neighbors=2, leaf_size = 1)  
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf, my_dataset, features)
