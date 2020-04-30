#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
max_bonus = 0
nom = ''

for p in data_dict:
    if data_dict[p]['bonus'] != 'NaN':
        if data_dict[p]['bonus'] > max_bonus:
            max_bonus = data_dict[p]['bonus']
            nom = p
            


data_dict.pop(nom,None)

print 'le bonus max est: ', max_bonus
print 'Il est pour: ', nom

max_exercised = 0
min_exercised = 34000000
for p in data_dict:
    if data_dict[p]['exercised_stock_options'] != 'NaN':
        if data_dict[p]['exercised_stock_options'] > max_exercised:
            max_exercised = data_dict[p]['exercised_stock_options']
        if data_dict[p]['exercised_stock_options'] < min_exercised and data_dict[p]['exercised_stock_options'] > 0:
            min_exercised = data_dict[p]['exercised_stock_options']

print min_exercised
print max_exercised

max_salary = 0
min_salary = 34000000
for p in data_dict:
    if data_dict[p]['salary'] != 'NaN':
        if data_dict[p]['salary'] > max_salary:
            max_exercised = data_dict[p]['salary']
        if data_dict[p]['salary'] < min_salary and data_dict[p]['salary'] > 0:
            min_exercised = data_dict[p]['salary']

print min_salary
print max_salary


data = featureFormat(data_dict, features)

### Visualization
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### Outliers with Bonus over 5M & Salari over 1M
for p in data_dict:
    if data_dict[p]['bonus'] != 'NaN':
        if data_dict[p]['bonus'] > 5000000 and data_dict[p]['salary'] > 1000000:
            print p
            
