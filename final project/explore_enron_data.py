#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import poi_myfunctions

import pickle
### include pprint allow a nicer view when using print function
import pprint


"""
==================================================
========== EXPLORATION ===========================
==================================================
"""
def explore_dataset(enron_data):
    
    """
    The number of line in the dataset correspond to the length
    """
    print 'Number of People in Dataset: ', poi_myfunctions.get_dataset_size(enron_data)
    print 'Number of POI in dataset: ', poi_myfunctions.count_poi(enron_data)
    
    print
    """ Bit of cleaning for all the Financial """
    # Replace NaN In finance as number are expected
    print "Replace all NaN value in Financial Category..."
    enron_data = poi_myfunctions.NaN_Replace(enron_data)
    print
    """ Cleaning all records with less than 90 % of completion """
    # Find out , record with missing info
    # After couple of try, 90% of completion is the limit used.
    incomplete = poi_myfunctions.get_incomplete(enron_data, 0.9)
    print 'Table of Incomplete: (less than 90% of completion) ' , incomplete
    print 'The number of personne with less than 90% of completion is: ', len(incomplete)
    print 'Remove the incomplete from the dataset ...'
    #remove these personne from the dataset
    for p in incomplete:
        enron_data.pop(p, None)
    
    """ Sorting the records alphabetically """
    pretty = pprint.PrettyPrinter()
    names = sorted(enron_data.keys())  #sort names of Enron employees in dataset by first letter of last name
    print
    print 'Sorted list of Enron employees by last name'
    pretty.pprint(names) 
    
    """ Print a sample """
    print 'Dictionnary Example : Value : Feature'
    pretty.pprint(enron_data['HORTON STANLEY C']) 
    
    """ How many category per record """
    print "for each there is xx category"
    ### Define a counter to identify the number of category
    l_counter_cat = 0
    for cat in enron_data['HORTON STANLEY C']:
        l_counter_cat = l_counter_cat + 1
                
    print ("Nombre de category")  
    print (l_counter_cat, '. : ' , cat)
    l_counter_cat = 0   
      
    
    """ Include CSV """
    import csv
    ### Write the data into a csv file in order to ease the analysis
    fieldnames = ['name'] + enron_data['HORTON STANLEY C'].keys()
    
    ### Remove outlier (Total)
    with open('enron.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
        writer.writeheader()
        for name in enron_data.keys():
            if name != 'TOTAL':
                n = {'name':name}
                n.update(enron_data[name])
                writer.writerow(n)
    
    
    ### Use a panda dataframe to ease data reading
    import pandas as pd
    ### Use a Panda data frame to read enron csv we generate above
    enron = pd.read_csv('enron.csv')
    
    ### Understand mail behaviour
    #added feature, ratio of e-mails to and from poi
    enron['ratio_to_poi'] = enron['from_this_person_to_poi'].fillna(0.0)/enron['from_messages'].fillna(0.0)
    enron['ratio_from_poi'] = enron['from_poi_to_this_person'].fillna(0.0)/enron['to_messages'].fillna(0.0)
    
    enron.info()
    
    #number of POI in dataset
    print 'There are still 18 POI in our Dataset as you can see by our "True" count'
    enron['poi'].value_counts()
    
    #set a baseline by extracting non-POIs and printing stats
    
    non_poi = enron[enron.poi.isin([False])]
    
    non_poi_money = non_poi[['salary','bonus','exercised_stock_options','total_stock_value',\
                             'total_payments']].describe()
    print non_poi_money
    non_poi_email_behavior = non_poi[['shared_receipt_with_poi','to_messages',\
                                      'from_messages', 'ratio_to_poi','ratio_from_poi']].describe()
    print non_poi_email_behavior
    
    #POI stats
    
    poi_info = enron[enron.poi.isin([True])]
    
    poi_money = poi_info[['salary','bonus','exercised_stock_options','total_stock_value',\
                          'total_payments']].describe()
    print poi_money
    poi_email_behavior = poi_info[['shared_receipt_with_poi','to_messages',\
                                      'from_messages', 'ratio_to_poi', 'ratio_from_poi']].describe()
    print poi_email_behavior
    
    """
    vizualise DATA
    """
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    get_ipython().run_line_magic('matplotlib', 'inline')
    #%matplotlib inline
    #%pylab inline
    import seaborn 
    
    ## Average Salary
    average_salary = enron.groupby('poi').mean()['salary']
    print average_salary
    sns1 = seaborn()
    sns.boxplot(x='poi',y='salary',data=enron)
    
    # Who is the out of range
    print enron[(enron['salary']>600000)][['name','salary', 'bonus','poi']]
    
    ### Average Bonus
    average_bonus = enron.groupby('poi').mean()['bonus']
    print average_bonus
    
    sns2 = seaborn()
    sns2.boxplot(x='poi',y='bonus',data=enron)
    
    print enron[(enron['bonus']>4000000)][['name','salary','bonus','exercised_stock_options','restricted_stock','total_stock_value','poi']]
    
    ### Average Total Payments
    average_total_payments = enron.groupby('poi').mean()['total_payments']
    print average_total_payments
    
    sns3 = seaborn()
    sn3.boxplot(x='poi',y='total_payments',data=enron)
    # Who are the out of range
    print enron[(enron['total_payments']>40000000)][['name','salary', 'bonus', 'total_stock_value','total_payments','poi']]
    
    
    ### Exclude Kenneth from this comparison
    enron_k_excluded = enron[(enron['total_payments']<40000000)]
    sns4 = seaborn()
    sns4.boxplot(x='poi',y='total_payments',data=enron_k_excluded)
    
    ### Exercised Stock Options
    average_optionsvalue = enron.groupby('poi').mean()['exercised_stock_options']
    print average_optionsvalue
    print
    sns5 = seaborn()
    sns5.boxplot(x='poi',y='exercised_stock_options',data=enron)
    
    ### average Stock Value
    average_stockvalue = enron.groupby('poi').mean()['total_stock_value']
    print 'The average stock value for POI is: ',average_stockvalue
    print 
    # Who are the out of range
    print 'The list of person with a total stock value above 20 millions is:'
    print enron[(enron['total_stock_value']>20000000)][['name','salary', 'bonus', 'total_stock_value','total_payments','poi']]
    
    
    ### average Shared Receipt
    print
    average_shared_receipt = enron.groupby('poi').mean()['shared_receipt_with_poi']
    print 'The average shared receipt is: ',average_shared_receipt
    
    ### average To Messages
    print
    average_to = enron.groupby('poi').mean()['to_messages']
    print 'The average to_message for a POI is: ',average_to
    print
    ### average From Message
    average_from = enron.groupby('poi').mean()['from_messages']
    print 'The average from_message for a POI is: ',average_from
    print 
    print 'The POIs with more than 2000 from message are:' 
    print enron[(enron['from_messages']>2000)][['name','salary', 'bonus', 'from_messages','poi']]
    
    ### Ratio to poi
    average_ratio_to = enron.groupby('poi').mean()['ratio_to_poi']
    print 'The average ratio to poi (from poi) is: ',average_ratio_to
    
    ### Ratio from poi
    average_ratio_from = enron.groupby('poi').mean()['ratio_from_poi']
    print 'The average ratio from poi (to poi) is: ',average_ratio_from





