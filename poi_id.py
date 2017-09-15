#!/usr/bin/python

### https://github.com/seifip/udacity-data-analyst-nanodegree/blob/master/P5%20-%20Identifying%20Fraud%20from%20Enron%20Emails%20and%20Financial%20Data/project/poi_id.py

import sys
import pickle
import matplotlib

import numpy
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import precision_score, recall_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

#classifiers tested:
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

EXTRA = False

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

'''
financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
                   'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                   'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                   'restricted_stock', 'director_fees'] (all units are in US dollars)
email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] (units are generally number of emails messages; notable exception is ‘email_address’, which is a text string)
POI label: [‘poi’] (boolean, represented as integer)
'''

features_list = ['poi'] # these are features that we will look to include no matter what

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

print "# points: ", len(data_dict)
print "# of features: ", len(data_dict[data_dict.keys()[0]])

### layout: PERSONS (DICT) : FINANCIAL/EMAIL/POI features

cnt = 0
for pers in data_dict.values():
    if pers['poi'] == 1:
        cnt += 1
print "# POI: ", cnt

### check for missing values or NAN

#for pers in data_dict.values():
#    matplotlib.pyplot.scatter(pers['salary'],pers['long_term_incentive'])
#
#matplotlib.pyplot.show()

### extreme outlier was identified earlier as a total row, which will be removed

del data_dict['TOTAL']

### AFTER

# for pers in data_dict.values():
    # matplotlib.pyplot.scatter(pers['salary'],pers['long_term_incentive'])

# matplotlib.pyplot.show()

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for item in my_dataset:
    person = my_dataset[item]
    #verify all information is available from individual
    if (all([person['from_poi_to_this_person'] != 'NaN',
             person['from_this_person_to_poi'] != 'NaN',
             person['to_messages'] != 'NaN',
             person['from_messages'] != 'NaN']
            )):
        #add feature of relation to person of interest
        person["fraction_from_poi"] = float(person["from_poi_to_this_person"]) / float(person["to_messages"])
        person["fraction_to_poi"] = float(person["from_this_person_to_poi"]) / float(person["from_messages"])
    else:
        person["fraction_from_poi"] = person["fraction_to_poi"] = 0

# add wealth feature

for item in my_dataset:
    person = my_dataset[item]
    if (all([person['salary'] != 'NaN',
             person['total_stock_value'] != 'NaN',
             person['exercised_stock_options'] != 'NaN',
             person['bonus'] != 'NaN']
            )):
        person['wealth'] = sum([person[field] for field in ['salary',
                                                            'total_stock_value',
                                                            'exercised_stock_options',
                                                            'bonus']])
    else:
        person['wealth'] = 'NaN'

my_features = features_list + ['fraction_from_poi',
                               'fraction_to_poi',
                               'shared_receipt_with_poi',
                               'expenses',
                               'loan_advances',
                               'long_term_incentive',
                               'other',
                               'restricted_stock',
                               'restricted_stock_deferred',
                               'deferral_payments',
                               'deferred_income',
                               'salary',
                               'total_stock_value',
                               'exercised_stock_options',
                               'total_payments',
                               'bonus',
                               'wealth']

print "\n# of possible features: ", len(my_features)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

print "\nAll features: ", my_features

#as per working with ratios above we will also look to scale on some features

# Scale features
scaler = preprocessing.MinMaxScaler() # this will transform each feature into a range from 0 to 1
features = scaler.fit_transform(features) 

# K-best features
k_best = SelectKBest(k=5)
k_best.fit(features, labels)

results_list = zip(k_best.get_support(), my_features[1:], k_best.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True) # sort by k_best.scores
print "\nK-best features:", results_list

xbar = []
ybar = []
for ele in results_list:
    xbar.append(ele[1])
    ybar.append(ele[2])

matplotlib.pyplot.bar(numpy.arange(len(xbar)),ybar,align='center',alpha=0.5)

matplotlib.pyplot.xticks(numpy.arange(len(xbar)),xbar,rotation=90)
matplotlib.pyplot.ylabel('Score')
matplotlib.pyplot.title('K-Best Scores')

matplotlib.pyplot.show()

## 6 best features chosen by SelectKBest
my_features = features_list + ['exercised_stock_options',
                               'total_stock_value',
                               'bonus',
                               'salary',
                               'fraction_to_poi']

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def test_clf(grid_search, features, labels, parameters, iterations=100):
    
    # Recall: out of all the items that are truly postive, how many were correctly classfied as postive?
    # Precision: Out of all the items labeled as positive, how many truly belong to the positive class?    
    precision, recall = [], []
    
    for iteration in range(iterations):
		features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=iteration)
		grid_search.fit(features_train, labels_train)
		predictions = grid_search.predict(features_test)
		precision = precision + [precision_score(labels_test, predictions)]
		recall = recall + [recall_score(labels_test, predictions)]
		if iteration % 10 == 0:
			sys.stdout.write('.')
    print '\nPrecision:', numpy.mean(precision)
    print 'Recall:', numpy.mean(recall)
    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '%s=%r, ' % (param_name, best_params[param_name])

clf = GaussianNB()
parameters = {}
grid_search = GridSearchCV(clf, parameters)

print '\nGaussianNB:'
test_clf(grid_search, features, labels, parameters)

#GaussianNB:
#Precision: 0.307566313011
#Recall: 0.394829365079

if EXTRA == True:
    clf = tree.DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'],
                  'min_samples_split': [2, 10, 20],
                  'max_depth': [None, 2, 5, 10],
                  'min_samples_leaf': [1, 5, 10],
                  'max_leaf_nodes': [None, 5, 10, 20]}
    grid_search = GridSearchCV(clf, parameters)
    
    print '\nDecisionTree:'
    test_clf(grid_search, features, labels, parameters)

#DecisionTree:
#Precision: 0.263218253968
#Recall: 0.258222222222
#criterion='entropy', 
#max_depth=None, 
#max_leaf_nodes=None, 
#min_samples_leaf=1, 
#min_samples_split=10, 

if EXTRA == True:
    clf = AdaBoostClassifier()
    parameters = {'n_estimators': [5, 10, 20, 40, 80],
                  'algorithm': ['SAMME', 'SAMME.R'],
                  'learning_rate': [.5,.8, 1, 1.2, 1.5]}
    grid_search = GridSearchCV(clf, parameters)
    
    print '\nAdaBoost:'
    test_clf(grid_search, features, labels, parameters)

#AdaBoost:
#Precision: 0.345583333333
#Recall: 0.221150793651
#algorithm='SAMME', 
#learning_rate=0.5, 
#n_estimators=20, 

'''
Classifier using: GaussianNB
'''

clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, my_features)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, my_features)