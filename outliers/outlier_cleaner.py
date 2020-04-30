#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    i = 0
    while i < len(ages) :
        error = abs(net_worths[i] - predictions[i])
        age = ages[i]
        net_worth  = net_worths[i]
        cleaned_data.append((age,net_worth,error))
        i += 1
    
    print len(cleaned_data)
    cleaned_data = sorted(cleaned_data, key=lambda tup: tup[2] )[:81]
    
    
    print i
    print len(cleaned_data)
    
    
    return cleaned_data

