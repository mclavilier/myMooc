# -*- coding: utf-8 -*-
"""
@author: 2CLAVILIER Mathieu
In this file I store my functions
"""

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

""" This function will return the lenght of the data set """
def get_dataset_size(dataset):
    """ Return the lenght"""
    return len(dataset)

""" This function will return the number of poi """
def count_poi(dataset):
    """ return the number of POI """
    n = 0
    for p in dataset:
        if dataset[p]["poi"]:
            n += 1
    return n

""" This fn Returns a dictionary with these key-value pairs:
        key = feature name
        value = number of NaNs the feature has across the dataset
    """
def NaN_count(dataset):
    
    d = {}
    for person in dataset:
        for key, value in dataset[person].iteritems():
            if value == "NaN":
                if key in d:
                    d[key] += 1
                else:
                    d[key] = 1

    return d

""" This function will return all the financial category with 0 instead of NaN """
def NaN_Replace(dataset):
    """ As financial category should be treated as number, I decided to replace all NaN value """
    
    #Define Financial Category
    finance = [
        "salary",
        "deferral_payments",
        "total_payments",
        "loan_advances",
        "bonus",
        "restricted_stock_deferred",
        "deferred_income",
        "total_stock_value",
        "expenses",
        "exercised_stock_options",
        "other",
        "long_term_incentive",
        "restricted_stock",
        "director_fees"
    ]
    for f in finance:
        for person in dataset:
            if dataset[person][f] == "NaN":
                dataset[person][f] = 0.

    return dataset

""" This function will identify and return a table all the incomplete records"""
def get_incomplete(dataset,ratio):

    incompletes = []
    for person in dataset:
        n = 0
        for key, value in dataset[person].iteritems():
            if value == 'NaN' or value == 0:
                n += 1
        fraction = float(n) / float(21)
        if fraction > ratio:
            incompletes.append(person)

    return incompletes

def scatterplot(dataset, var1, var2):
    """
    Creates and shows a scatterplot given a dataset and two features.
    """
    features_name = [str(var1), str(var2)]
    features = [var1, var2]
    data = featureFormat(dataset, features)

    for point in data:
        var1 = point[0]
        var2 = point[1]
        matplotlib.pyplot.scatter(var1, var2)

    matplotlib.pyplot.xlabel(features_name[0])
    matplotlib.pyplot.ylabel(features_name[1])
    matplotlib.pyplot.show()

    return


### Compute new features
def computefeature(sub,total):
    if sub == 'NaN' or total == 'NaN':
        fraction = 0
    else:
        fraction = float(sub) / float(total)
    return fraction